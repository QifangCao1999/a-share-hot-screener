"""通用评分函数层(Session 5).

设计原则:
  1. 所有函数只做"单指标映射到子分",不做权重聚合(聚合在 AxisScore 中)
  2. 每个子分产出一个 ScoreItem,包含完整的溯源信息(raw_value / derived_value /
     subscore / weight / weighted_score / is_applicable / is_data_available / note)
  3. 数据缺失时:is_data_available=False, subscore=None,不参与加权平均
  4. 百分位型函数不接受单指标截面(必须有同批次 pool),pool 为 None 时退化为 is_data_available=False
  5. 所有 subscore 范围:[0, 1](已归一化),保留 4 位小数(输出时按需格式化)
  6. coverage = applicable & available 的比例(基于 weight 加权)

函数类型:
  score_lower_bound  - 下限型(三段折线): L→0, T→0.70, H→1.0;mid_threshold=None 时退化为两段线性
  score_upper_bound  - 上限型(三段折线): G→1.0, T→0.70, B→0;mid_threshold=None 时退化为两段线性
  score_clamp_linear - 区间线性型(兼容上下限正反方向)
  score_discrete     - 离散型:value → 预设映射表查询
  score_percentile   - 百分位型:value 在 pool 中的分位(升序或降序)
  score_bool         - 布尔型:True→1.0, False→0.0

AxisScore:
  管理一个评分轴的所有 ScoreItem,计算加权平均分和 coverage。
"""

from __future__ import annotations

import json
import logging
import math
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence

logger = logging.getLogger("a_share_hot_screener.scoring")
_scoring_logger = logger  # alias for ScoringPool classmethod/staticmethod context


# ════════════════════════════════════════════════════════
# 数据结构
# ════════════════════════════════════════════════════════

@dataclass
class ScoreItem:
    """单个指标的评分记录,包含完整溯源信息.

    字段说明:
      name              指标名称(英文蛇形命名)
      raw_value         原始输入值(来自 detail 字段)
      derived_value     中间推导值(百分位、比率等,可 None)
      subscore          映射后的标准化子分(0~1),保留 4 位小数
      weight            该指标在评分轴中的权重(原始权重,未归一化)
      weighted_score    subscore * weight(仅当 is_applicable & is_data_available)
      is_applicable     该指标是否适用于当前股票(False = 永远跳过)
      is_data_available 数据是否可用(False = 数据缺失,跳过但影响 coverage)
      note              说明(数据口径、降级、警告等)
    """
    name: str
    raw_value: Any = None
    derived_value: Any = None
    subscore: Optional[float] = None
    weight: float = 1.0
    weighted_score: Optional[float] = None
    is_applicable: bool = True
    is_data_available: bool = True
    note: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "raw_value": _json_safe(self.raw_value),
            "derived_value": _json_safe(self.derived_value),
            "subscore": round(self.subscore, 4) if self.subscore is not None else None,
            "weight": self.weight,
            "weighted_score": round(self.weighted_score, 4) if self.weighted_score is not None else None,
            "is_applicable": self.is_applicable,
            "is_data_available": self.is_data_available,
            "note": self.note,
        }


@dataclass
class AxisScore:
    """一个评分轴(如 hot_theme_score)的计算结果.

    score     加权平均分(0~1),None 表示没有任何可用指标
    coverage  有效指标权重占总权重的比例(0~1)
    items     各指标的 ScoreItem 列表
    """
    axis_name: str
    score: Optional[float] = None
    coverage: float = 0.0
    items: List[ScoreItem] = field(default_factory=list)

    def compute(self) -> "AxisScore":
        """根据 items 计算加权平均分和 coverage.

        逻辑:
          - 只有 is_applicable=True 的指标参与分母(总权重)
          - 只有 is_applicable=True & is_data_available=True 的指标参与分子和 coverage
          - subscore=None 时等价于 is_data_available=False

        Returns:
            self(链式调用)
        """
        total_weight = 0.0
        available_weight = 0.0
        weighted_sum = 0.0

        for item in self.items:
            if not item.is_applicable:
                continue
            total_weight += item.weight
            if item.is_data_available and item.subscore is not None:
                available_weight += item.weight
                ws = item.subscore * item.weight
                item.weighted_score = ws
                weighted_sum += ws
            else:
                item.weighted_score = None

        if total_weight <= 0:
            self.score = None
            self.coverage = 0.0
        elif available_weight <= 0:
            self.score = None
            self.coverage = 0.0
        else:
            self.score = round(weighted_sum / available_weight, 4)
            self.coverage = round(available_weight / total_weight, 4)

        return self

    def to_dict(self) -> Dict[str, Any]:
        return {
            "axis": self.axis_name,
            "score": round(self.score, 4) if self.score is not None else None,
            "coverage": self.coverage,
            "items": [item.to_dict() for item in self.items],
        }


# ════════════════════════════════════════════════════════
# 评分函数
# ════════════════════════════════════════════════════════

def score_lower_bound(
    name: str,
    value: Optional[float],
    *,
    bad_threshold: float,
    good_threshold: float,
    weight: float = 1.0,
    note: str = "",
    is_applicable: bool = True,
    mid_threshold: Optional[float] = None,
) -> ScoreItem:
    """下限型:越高越好(三段式折线)。

    三段映射规则(当 mid_threshold 不为 None 时):
      value <= L (bad_threshold)   → 0.0
      L < value < T (mid_threshold) → 线性 0.0 → 0.70
      T <= value < H (good_threshold) → 线性 0.70 → 1.0
      value >= H                   → 1.0

    两段映射规则(当 mid_threshold 为 None 时,向后兼容):
      value <= bad_threshold  → 0.0
      value >= good_threshold → 1.0
      中间                     → 线性插值

    Args:
        bad_threshold:  L - ≤ 此值得 0 分(最差)
        mid_threshold:  T - L~T 线性 0→0.70, T~H 线性 0.70→1.0(None 则退化为两段)
        good_threshold: H - ≥ 此值得 1.0 分(最优)
    """
    item = ScoreItem(name=name, raw_value=value, weight=weight,
                     is_applicable=is_applicable, note=note)

    if not is_applicable:
        item.is_data_available = False
        return item

    if value is None or (isinstance(value, float) and math.isnan(value)):
        item.is_data_available = False
        item.note = (note + " [data_missing]").strip()
        return item

    item.is_data_available = True
    if good_threshold <= bad_threshold:
        logger.warning("score_lower_bound: good_threshold(%s) <= bad_threshold(%s) for %s",
                       good_threshold, bad_threshold, name)
        item.subscore = 0.5
        return item

    if mid_threshold is not None:
        # 三段式折线: L → 0, T → 0.70, H → 1.0
        L, T, H = bad_threshold, mid_threshold, good_threshold
        if value >= H:
            item.subscore = 1.0
        elif value <= L:
            item.subscore = 0.0
        elif value < T:
            # L ~ T → 0 ~ 0.70
            item.subscore = 0.70 * (value - L) / (T - L) if T > L else 0.70
        else:
            # T ~ H → 0.70 ~ 1.0
            item.subscore = 0.70 + 0.30 * (value - T) / (H - T) if H > T else 1.0
    else:
        # 两段式(向后兼容)
        if value >= good_threshold:
            item.subscore = 1.0
        elif value <= bad_threshold:
            item.subscore = 0.0
        else:
            item.subscore = (value - bad_threshold) / (good_threshold - bad_threshold)

    item.subscore = round(item.subscore, 4)
    return item


def score_upper_bound(
    name: str,
    value: Optional[float],
    *,
    good_threshold: float,
    bad_threshold: float,
    weight: float = 1.0,
    note: str = "",
    is_applicable: bool = True,
    mid_threshold: Optional[float] = None,
) -> ScoreItem:
    """上限型:越低越好(三段式折线)。

    三段映射规则(当 mid_threshold 不为 None 时):
      value <= G (good_threshold)   → 1.0
      G < value <= T (mid_threshold) → 线性 1.0 → 0.70
      T < value < B (bad_threshold)  → 线性 0.70 → 0.0
      value >= B                    → 0.0

    两段映射规则(当 mid_threshold 为 None 时,向后兼容):
      value <= good_threshold → 1.0
      value >= bad_threshold  → 0.0
      中间                     → 线性插值(递减)

    Args:
        good_threshold: G - ≤ 此值得 1.0 分
        mid_threshold:  T - G~T 线性 1.0→0.70, T~B 线性 0.70→0.0(None 则退化为两段)
        bad_threshold:  B - ≥ 此值得 0 分
    """
    item = ScoreItem(name=name, raw_value=value, weight=weight,
                     is_applicable=is_applicable, note=note)

    if not is_applicable:
        item.is_data_available = False
        return item

    if value is None or (isinstance(value, float) and math.isnan(value)):
        item.is_data_available = False
        item.note = (note + " [data_missing]").strip()
        return item

    item.is_data_available = True
    if bad_threshold <= good_threshold:
        item.subscore = 0.5
        return item

    if mid_threshold is not None:
        # 三段式折线: G → 1.0, T → 0.70, B → 0.0
        G, T, B = good_threshold, mid_threshold, bad_threshold
        if value <= G:
            item.subscore = 1.0
        elif value >= B:
            item.subscore = 0.0
        elif value <= T:
            # G ~ T → 1.0 ~ 0.70
            item.subscore = 1.0 - 0.30 * (value - G) / (T - G) if T > G else 0.70
        else:
            # T ~ B → 0.70 ~ 0.0
            item.subscore = 0.70 * (B - value) / (B - T) if B > T else 0.0
    else:
        # 两段式(向后兼容)
        if value <= good_threshold:
            item.subscore = 1.0
        elif value >= bad_threshold:
            item.subscore = 0.0
        else:
            item.subscore = (bad_threshold - value) / (bad_threshold - good_threshold)

    item.subscore = round(item.subscore, 4)
    return item


def score_clamp_linear(
    name: str,
    value: Optional[float],
    *,
    lo: float,
    hi: float,
    reverse: bool = False,
    weight: float = 1.0,
    note: str = "",
    is_applicable: bool = True,
) -> ScoreItem:
    """区间线性型:将 value 映射到 [lo, hi] 之间的线性分。

    reverse=False: lo → 0.0, hi → 1.0(越大越好)
    reverse=True:  lo → 1.0, hi → 0.0(越小越好)
    """
    item = ScoreItem(name=name, raw_value=value, weight=weight,
                     is_applicable=is_applicable, note=note)

    if not is_applicable:
        item.is_data_available = False
        return item

    if value is None or (isinstance(value, float) and math.isnan(value)):
        item.is_data_available = False
        item.note = (note + " [data_missing]").strip()
        return item

    item.is_data_available = True
    rng = hi - lo
    if rng <= 0:
        item.subscore = 0.5
        return item

    clamped = max(lo, min(hi, value))
    raw_score = (clamped - lo) / rng
    item.subscore = round(1.0 - raw_score if reverse else raw_score, 4)
    item.derived_value = round(clamped, 4)
    return item


def score_discrete(
    name: str,
    value: Any,
    *,
    mapping: Mapping[Any, float],
    default_score: Optional[float] = None,
    weight: float = 1.0,
    note: str = "",
    is_applicable: bool = True,
) -> ScoreItem:
    """离散型:按预设映射表查分。

    Args:
        mapping:       {value → subscore},subscore 必须在 [0, 1]
        default_score: 未命中 mapping 时的默认分,None → is_data_available=False

    Example(龙虎榜上榜次数映射):
        mapping={0: 0.0, 1: 0.6, 2: 0.8, 3: 1.0}
        value=2 → subscore=0.8
    """
    item = ScoreItem(name=name, raw_value=value, weight=weight,
                     is_applicable=is_applicable, note=note)

    if not is_applicable:
        item.is_data_available = False
        return item

    if value is None:
        item.is_data_available = False
        item.note = (note + " [data_missing]").strip()
        return item

    # 精确匹配
    if value in mapping:
        item.subscore = round(float(mapping[value]), 4)
        item.is_data_available = True
        return item

    # 对整数型 value,尝试向下取整(处理超界)
    if isinstance(value, int) and mapping:
        max_key = max(k for k in mapping if isinstance(k, int))
        if value >= max_key:
            item.subscore = round(float(mapping[max_key]), 4)
            item.is_data_available = True
            item.note = (note + f" [clamped_to_{max_key}]").strip()
            return item

    if default_score is not None:
        item.subscore = round(float(default_score), 4)
        item.is_data_available = True
        item.note = (note + " [default_score_used]").strip()
    else:
        item.is_data_available = False
        item.note = (note + f" [no_mapping_for_{value!r}]").strip()

    return item


def score_percentile(
    name: str,
    value: Optional[float],
    *,
    pool: Optional[Sequence[float]],
    ascending: bool = True,
    weight: float = 1.0,
    note: str = "",
    is_applicable: bool = True,
    presorted: bool = False,
) -> ScoreItem:
    """百分位型:value 在 pool 中的分位(0~1)。

    Args:
        pool:      当前 run_date scoring pool 中所有股票该指标的值列表
                   (不含 None,调用方负责过滤)
                   pool=None 或 len(pool)<2 时 → is_data_available=False
        ascending: True  → 分位越高越好(value 越大 subscore 越高)
                   False → 分位越低越好(value 越小 subscore 越高)
        presorted: 若为 True 则跳过排序(调用方保证 pool 已升序)。
                   ScoringPool 构建时已预排序,传 True 可避免 O(n log n) 重复排序。

    口径说明:
      使用 bisect_right 分位:subscore = bisect_right(sorted_pool, value) / len(pool)
      即与自身相等的值视为"排在自己前面",使得满分的条件是唯一最高值。
      ascending=False 时取 1 - subscore。
    """
    import bisect as _bisect

    item = ScoreItem(name=name, raw_value=value, weight=weight,
                     is_applicable=is_applicable, note=note)

    if not is_applicable:
        item.is_data_available = False
        return item

    if value is None or (isinstance(value, float) and math.isnan(value)):
        item.is_data_available = False
        item.note = (note + " [data_missing]").strip()
        return item

    if pool is None or len(pool) < 2:
        item.is_data_available = False
        item.note = (note + " [pool_too_small]").strip()
        return item

    sorted_pool = pool if presorted else sorted(pool)
    n = len(sorted_pool)
    idx = _bisect.bisect_right(sorted_pool, value)  # 0 ~ n

    raw_pctile = idx / n  # 0.0 ~ 1.0
    item.derived_value = round(raw_pctile, 4)
    item.subscore = round(raw_pctile if ascending else 1.0 - raw_pctile, 4)
    item.is_data_available = True
    return item


def score_bool(
    name: str,
    value: Optional[bool],
    *,
    true_score: float = 1.0,
    false_score: float = 0.0,
    weight: float = 1.0,
    note: str = "",
    is_applicable: bool = True,
) -> ScoreItem:
    """布尔型:True → true_score, False → false_score."""
    item = ScoreItem(name=name, raw_value=value, weight=weight,
                     is_applicable=is_applicable, note=note)

    if not is_applicable:
        item.is_data_available = False
        return item

    if value is None:
        item.is_data_available = False
        item.note = (note + " [data_missing]").strip()
        return item

    item.is_data_available = True
    item.subscore = round(float(true_score if value else false_score), 4)
    return item


# ════════════════════════════════════════════════════════
# ScoringPool - 横截面百分位所需的 pool 数据容器
# ════════════════════════════════════════════════════════

@dataclass
class ScoringPool:
    """同一 trade_date_used 下所有通过 hard_filter 股票的横截面数据.

    由 pipeline 在并发处理完毕后、调用评分之前构建。
    所有 pool 列表均已过滤 None,保证分母干净。

    字段命名规则:pool_<指标名>
    """
    # 涨跌幅横截面池(百分位评分用)
    pool_return_5d: List[float] = field(default_factory=list)    # 近5日收益率(%)
    pool_return_10d: List[float] = field(default_factory=list)   # 近10日收益率(%)

    # 事件层横截面池
    pool_limit_up_count_10d: List[float] = field(default_factory=list)   # 近10日涨停次数
    pool_strong_pool_3d: List[float] = field(default_factory=list)       # 近3日强势股池次数

    # 行业热度(全行业分位,已由 event_layer 计算,此处为二次分位备用)
    pool_industry_heat_pctile: List[float] = field(default_factory=list)

    # 量价横截面池(trend_flow 用)
    pool_volume_ratio_20d: List[float] = field(default_factory=list)
    pool_amount_ratio_5d_to_20d: List[float] = field(default_factory=list)
    pool_close_position_20d: List[float] = field(default_factory=list)
    pool_clv_latest: List[float] = field(default_factory=list)

    # 流动性横截面池(liquidity_execution 用,Session 6)
    pool_amount_avg_5d: List[float] = field(default_factory=list)   # 近5日均成交额(元,tushare_amount)

    # stock_count: 实际参与构建 pool 的股票数
    stock_count: int = 0

    @classmethod
    def build(cls, details: "List[HotStockDetail]") -> "ScoringPool":  # type: ignore[name-defined]
        """从通过 hard_filter 的 details 列表构建横截面 pool.

        只包含 passed_hard_filter=True 的股票的非 None 值。
        """
        pool = cls()
        eligible = [d for d in details if d.passed_hard_filter]
        pool.stock_count = len(eligible)

        for d in eligible:
            _append_if_not_none(pool.pool_return_5d, d.return_5d)
            _append_if_not_none(pool.pool_return_10d, d.return_10d)
            _append_if_not_none(pool.pool_limit_up_count_10d,
                                float(d.limit_up_count_10d) if d.limit_up_count_10d is not None else None)
            _append_if_not_none(pool.pool_strong_pool_3d,
                                float(d.strong_pool_entry_count_3d) if d.strong_pool_entry_count_3d is not None else None)
            _append_if_not_none(pool.pool_industry_heat_pctile, d.industry_heat_pctile_5d)
            _append_if_not_none(pool.pool_volume_ratio_20d, d.volume_ratio_20d)
            _append_if_not_none(pool.pool_amount_ratio_5d_to_20d, d.amount_ratio_5d_to_20d)
            _append_if_not_none(pool.pool_close_position_20d, d.close_position_20d)
            _append_if_not_none(pool.pool_clv_latest, d.clv_latest)
            # Session 6: 流动性横截面池
            _append_if_not_none(pool.pool_amount_avg_5d, d.amount_avg_5d)

        pool._pre_sort()
        return pool

    def _pre_sort(self) -> None:
        """将所有 pool 列表原地排序（升序），供 score_percentile(presorted=True) 使用."""
        for fname in self._POOL_FIELDS:
            getattr(self, fname).sort()

    # ── 基准 pool 序列化/反序列化（Session 12 P0-2）───────────

    _POOL_FIELDS = [
        "pool_return_5d", "pool_return_10d",
        "pool_limit_up_count_10d", "pool_strong_pool_3d",
        "pool_industry_heat_pctile",
        "pool_volume_ratio_20d", "pool_amount_ratio_5d_to_20d",
        "pool_close_position_20d", "pool_clv_latest",
        "pool_amount_avg_5d",
    ]

    def save_baseline(self, path: str) -> None:
        """将当前 pool 数据序列化为 JSON 文件(供后续单只查询使用).

        只在 stock_count >= MIN_BASELINE_POOL_SIZE 时才有意义。
        """
        data = {
            "stock_count": self.stock_count,
        }
        for fname in self._POOL_FIELDS:
            data[fname] = getattr(self, fname)
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
        _scoring_logger.info("[baseline] 已保存基准 pool → %s (stock_count=%d)", path, self.stock_count)

    @classmethod
    def load_baseline(cls, path: str) -> Optional["ScoringPool"]:
        """从 JSON 文件加载基准 pool.  文件不存在或格式错误返回 None."""
        if not path or not os.path.isfile(path):
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            pool = cls()
            pool.stock_count = data.get("stock_count", 0)
            for fname in cls._POOL_FIELDS:
                setattr(pool, fname, data.get(fname, []))
            pool._pre_sort()
            _scoring_logger.info("[baseline] 已加载基准 pool ← %s (stock_count=%d)", path, pool.stock_count)
            return pool
        except Exception as e:
            _scoring_logger.warning("[baseline] 加载基准 pool 失败(%s): %s", path, e)
            return None

    def merge_with_baseline(self, baseline: "ScoringPool") -> "ScoringPool":
        """合并当前 pool 与基准 pool,返回新 ScoringPool.

        当前 live pool 的数据点会被保留(占比通常很小),
        基准 pool 的数据点追加到列表末尾。
        stock_count = 两者之和(合并后的百分位分母更大、更有代表性)。
        """
        merged = ScoringPool()
        merged.stock_count = self.stock_count + baseline.stock_count
        for fname in self._POOL_FIELDS:
            live_list = getattr(self, fname)
            base_list = getattr(baseline, fname)
            setattr(merged, fname, live_list + base_list)
        merged._pre_sort()
        _scoring_logger.info(
            "[baseline] pool 合并完成: live=%d + baseline=%d → merged=%d",
            self.stock_count, baseline.stock_count, merged.stock_count,
        )
        return merged


# ════════════════════════════════════════════════════════
# 工具
# ════════════════════════════════════════════════════════

def _append_if_not_none(lst: List[float], value: Optional[float]) -> None:
    if value is not None and not (isinstance(value, float) and math.isnan(value)):
        lst.append(value)


def _json_safe(val: Any) -> Any:
    if isinstance(val, float):
        if math.isnan(val) or math.isinf(val):
            return None
    return val
