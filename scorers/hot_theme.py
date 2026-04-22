"""hot_theme_score 计算模块（Session 5; Session 8 P1; Session 11 HT3/HT4 重构; #2 小池百分位修正）.

热点题材强度评分轴，衡量股票在当前热点环境中的参与程度与强度。

──────────────────────────────────────────────────────
指标列表（7 项）：
  HT1  近5日收益率横截面百分位       weight=8   百分位→三段下限型（L=60,T=80,H=95）
  HT2  近10日收益率横截面百分位      weight=6   百分位→三段下限型（L=60,T=80,H=95）
  HT3  近10日大涨天数（涨幅>5%）     weight=8   离散型 0→0, 1→0.40, 2→0.70, 3→0.90, ≥4→1.0
       Session 11 改：从涨停次数改为大涨天数，对蓝筹和题材股均有区分度
  HT4  连续上涨天数                  weight=6   离散型 0→0, 1→0.25, 2→0.50, 3→0.70, 4→0.85, ≥5→1.0
       Session 11 改：从强势股池入选次数改为连续上涨天数，不依赖事件层接口
  HT5  所属行业近5日强度百分位       weight=7   百分位→三段下限型（L=55,T=75,H=90）
  HT6  所属概念板块热度百分位       weight=7   百分位→三段下限型（L=55,T=75,H=90）
       仅在 enable_concept_heat_module=True 且概念模块可用时纳入评分
  HT7  板块轮动动量信号              weight=5   离散型（Session 22）
       rotate_in→1.0, steady_strong→0.85, neutral→0.50, rotate_out→0.15, steady_weak→0.0
       从 flags["sector_momentum_signal"] 读取；未启用时 is_applicable=False

#2 小池百分位修正：
  pool_size >= 30:  正常百分位
  pool_size < 30:   百分位子分上限 0.90（避免小样本满分）
  pool_size < 2:    fallback 到绝对涨幅三段型评分，子分上限 0.70
──────────────────────────────────────────────────────
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

from a_share_hot_screener.scoring import (
    AxisScore,
    ScoreItem,
    ScoringPool,
    score_clamp_linear,
    score_discrete,
    score_lower_bound,
    score_percentile,
)

if TYPE_CHECKING:
    from a_share_hot_screener.models import HotStockDetail

logger = logging.getLogger("a_share_hot_screener.scorers.hot_theme")

# HT3 离散映射（Session 11：大涨天数，与原涨停次数映射表相同但语义不同）
# 0→0, 1→0.40, 2→0.70, 3→0.90, ≥4→1.0
_HT3_DISCRETE_MAP = {0: 0.0, 1: 0.40, 2: 0.70, 3: 0.90, 4: 1.0}

# HT4 离散映射（Session 11：连续上涨天数，梯度更细）
# 0→0, 1→0.25, 2→0.50, 3→0.70, 4→0.85, ≥5→1.0
_HT4_DISCRETE_MAP = {0: 0.0, 1: 0.25, 2: 0.50, 3: 0.70, 4: 0.85, 5: 1.0}

# ── #2: 小池百分位修正阈值 ─────────────────────────────────
SMALL_POOL_THRESHOLD = 30       # 池子 < 此值时限制百分位子分上限
SMALL_POOL_SUBSCORE_CAP = 0.90  # 小池子百分位子分上限
TINY_POOL_SUBSCORE_CAP = 0.70   # 极小池子/绝对涨幅 fallback 子分上限


def _pctile_to_three_segment(
    name: str,
    value: Optional[float],
    pool: Optional[list],
    ascending: bool,
    weight: float,
    L: float,
    T: float,
    H: float,
    note: str = "",
    subscore_cap: Optional[float] = None,
) -> ScoreItem:
    """百分位 → 三段下限型评分.

    先算横截面百分位（0~100），再用三段折线映射：
      pctile <= L → 0
      L < pctile < T → 线性 0→0.70
      T <= pctile < H → 线性 0.70→1.0
      pctile >= H → 1.0

    Args:
        subscore_cap: 若非 None，子分不超过此值（#2: 小池限制）
    """
    # 先算百分位（pool 来自 ScoringPool，已预排序）
    pctile_item = score_percentile(
        name=name,
        value=value,
        pool=pool,
        ascending=ascending,
        weight=weight,
        note=note,
        presorted=True,
    )
    if not pctile_item.is_data_available or pctile_item.subscore is None:
        return pctile_item

    # derived_value 已是 0~1 百分位，转换为 0~100 再做三段映射
    raw_pctile = pctile_item.derived_value * 100.0 if pctile_item.derived_value is not None else pctile_item.subscore * 100.0

    # 三段折线
    if raw_pctile >= H:
        subscore = 1.0
    elif raw_pctile <= L:
        subscore = 0.0
    elif raw_pctile < T:
        subscore = 0.70 * (raw_pctile - L) / (T - L) if T > L else 0.70
    else:
        subscore = 0.70 + 0.30 * (raw_pctile - T) / (H - T) if H > T else 1.0

    # #2: 小池上限
    if subscore_cap is not None and subscore > subscore_cap:
        pctile_item.note += f"; small_pool_cap={subscore_cap}(raw_subscore={round(subscore, 4)})"
        subscore = subscore_cap

    pctile_item.subscore = round(subscore, 4)
    pctile_item.note += f"; L={L}/T={T}/H={H}; pctile_100={round(raw_pctile, 2)}"
    return pctile_item


def _absolute_return_fallback(
    name: str,
    value: Optional[float],
    *,
    weight: float,
    low_pct: float,
    mid_pct: float,
    high_pct: float,
    cap: float = TINY_POOL_SUBSCORE_CAP,
    note: str = "",
) -> ScoreItem:
    """绝对涨幅三段型 fallback（#2: 池子极小时替代百分位）.

    映射：
      return <= low_pct  → 0
      return == mid_pct  → 0.70 * cap（按比例缩放到 cap 范围内）
      return >= high_pct → cap
    线性插值，子分上限 = cap。
    """
    item = ScoreItem(name=name, raw_value=value, weight=weight, note=note)

    if value is None:
        item.is_data_available = False
        item.note += " [data_missing]"
        return item

    item.is_data_available = True
    item.derived_value = round(value, 4)

    # 三段折线映射到 [0, cap]
    if value >= high_pct:
        subscore = cap
    elif value <= low_pct:
        subscore = 0.0
    elif value < mid_pct:
        subscore = (cap * 0.70) * (value - low_pct) / (mid_pct - low_pct) if mid_pct > low_pct else 0.0
    else:
        subscore = (cap * 0.70) + (cap * 0.30) * (value - mid_pct) / (high_pct - mid_pct) if high_pct > mid_pct else cap

    item.subscore = round(min(subscore, cap), 4)
    item.note += f"; abs_fallback(low={low_pct}/mid={mid_pct}/high={high_pct}/cap={cap})"
    return item


def _resolve_pool_strategy(
    pool_list: Optional[list],
    pool: ScoringPool,
) -> tuple:
    """根据 pool 大小决定 HT1/HT2 评分策略.

    Returns:
        (effective_pool, subscore_cap, use_absolute_fallback)
        - effective_pool: 用于百分位的列表（或 None）
        - subscore_cap: 子分上限（或 None=无限制）
        - use_absolute_fallback: True 则跳过百分位，直接用绝对涨幅
    """
    if pool_list is None or len(pool_list) < 2:
        # 池子太小，无法计算百分位 → 绝对涨幅 fallback
        return None, TINY_POOL_SUBSCORE_CAP, True

    pool_size = pool.stock_count
    if pool_size >= SMALL_POOL_THRESHOLD:
        # 大池正常
        return pool_list, None, False
    else:
        # 小池：仍用百分位但加 cap
        return pool_list, SMALL_POOL_SUBSCORE_CAP, False


def compute_hot_theme_score(
    detail: "HotStockDetail",
    pool: ScoringPool,
) -> AxisScore:
    """计算单只股票的 hot_theme_score."""
    axis = AxisScore(axis_name="hot_theme_score")

    # ── HT1: 近5日收益率 ─────────────────────────────────
    pool_5d = pool.pool_return_5d if pool.pool_return_5d else None
    eff_pool_5d, cap_5d, use_abs_5d = _resolve_pool_strategy(pool_5d, pool)

    if use_abs_5d:
        # #2: 绝对涨幅 fallback
        axis.items.append(_absolute_return_fallback(
            name="return_5d_pctile",
            value=detail.return_5d,
            weight=8.0,
            low_pct=5.0, mid_pct=12.0, high_pct=25.0,
            cap=TINY_POOL_SUBSCORE_CAP,
            note="近5日收益率(abs_fallback, pool<2)",
        ))
    else:
        axis.items.append(_pctile_to_three_segment(
            name="return_5d_pctile",
            value=detail.return_5d,
            pool=eff_pool_5d,
            ascending=True,
            weight=8.0,
            L=60, T=80, H=95,
            note=f"近5日收益率在scoring_pool中的百分位→三段型(pool_size={pool.stock_count})",
            subscore_cap=cap_5d,
        ))

    # ── HT2: 近10日收益率 ────────────────────────────────
    pool_10d = pool.pool_return_10d if pool.pool_return_10d else None
    eff_pool_10d, cap_10d, use_abs_10d = _resolve_pool_strategy(pool_10d, pool)

    if use_abs_10d:
        axis.items.append(_absolute_return_fallback(
            name="return_10d_pctile",
            value=detail.return_10d,
            weight=6.0,
            low_pct=8.0, mid_pct=20.0, high_pct=40.0,
            cap=TINY_POOL_SUBSCORE_CAP,
            note="近10日收益率(abs_fallback, pool<2)",
        ))
    else:
        axis.items.append(_pctile_to_three_segment(
            name="return_10d_pctile",
            value=detail.return_10d,
            pool=eff_pool_10d,
            ascending=True,
            weight=6.0,
            L=60, T=80, H=95,
            note=f"近10日收益率横截面百分位→三段型(pool_size={pool.stock_count})",
            subscore_cap=cap_10d,
        ))

    # ── HT3: 近10日大涨天数（Session 11 改：从涨停次数改为大涨天数）weight=8
    # 大涨天数 = 10日内单日涨幅>5%的天数（来自 price_features.big_up_count_10d）
    big_up_val = detail.big_up_count_10d
    axis.items.append(score_discrete(
        name="big_up_count_10d",
        value=big_up_val,
        mapping=_HT3_DISCRETE_MAP,
        default_score=None,
        weight=8.0,
        note=(
            f"近10日大涨天数（单日涨幅>5%，离散映射：0→0/1→40/2→70/3→90/≥4→100）；"
            f"S11改：从涨停次数改为大涨天数，蓝筹/题材均有区分度；"
            f"raw={big_up_val}"
        ),
    ))

    # ── HT4: 连续上涨天数（Session 11 改：从强势股池入选次数改为 consec_up_days）weight=6
    consec_val = detail.consec_up_days
    axis.items.append(score_discrete(
        name="consec_up_days",
        value=consec_val,
        mapping=_HT4_DISCRETE_MAP,
        default_score=None,
        weight=6.0,
        note=(
            f"连续上涨天数（离散映射：0→0/1→25/2→50/3→70/4→85/≥5→100）；"
            f"S11改：从强势股池入选次数改为连续上涨天数；"
            f"raw={consec_val}"
        ),
    ))

    # ── HT5: 行业近5日强度百分位 → 三段型（L=55,T=75,H=90）weight=7
    # P0-E: degraded 模式子分上限 0.80
    industry_pctile_100 = detail.industry_heat_pctile_5d * 100.0 if detail.industry_heat_pctile_5d is not None else None
    ht5_item = score_lower_bound(
        name="industry_heat_pctile_5d",
        value=industry_pctile_100,
        bad_threshold=55.0,
        mid_threshold=75.0,
        good_threshold=90.0,
        weight=7.0,
        note=(
            f"行业近5日热度百分位→三段型（L=55/T=75/H=90）；"
            f"source={detail.industry_heat_source or 'none'}"
            + ("" if detail.industry_heat_pctile_5d is not None
               else "; industry_heat_not_available")
        ),
    )
    # P0-E: degraded cap
    _HT5_DEGRADED_CAP = 0.80
    if (
        detail.industry_heat_source
        and "degraded" in detail.industry_heat_source
        and ht5_item.subscore is not None
        and ht5_item.subscore > _HT5_DEGRADED_CAP
    ):
        ht5_item.note += f"; ht5_degraded_cap={_HT5_DEGRADED_CAP}(raw={ht5_item.subscore:.4f})"
        ht5_item.subscore = _HT5_DEGRADED_CAP
    axis.items.append(ht5_item)

    # ── HT6: 概念板块热度百分位 → 三段下限型（L=55,T=75,H=90）weight=7
    # 仅在 enable_concept_heat_module=True 且概念模块实际可用时纳入评分
    if detail.advanced_concept_module_available and detail.concept_heat_pctile_5d is not None:
        concept_pctile_100 = detail.concept_heat_pctile_5d * 100.0
        axis.items.append(score_lower_bound(
            name="concept_heat_pctile_5d",
            value=concept_pctile_100,
            bad_threshold=55.0,
            mid_threshold=75.0,
            good_threshold=90.0,
            weight=7.0,
            note=(
                f"概念板块热度百分位→三段型（L=55/T=75/H=90）；"
                f"source={detail.concept_heat_source or 'none'}"
            ),
        ))

    # ── HT7: 板块轮动动量信号（离散型）weight=5 ────
    # Session 22 新增：rotate_in=1.0, steady_strong=0.85, neutral=0.50,
    #                 rotate_out=0.15, steady_weak=0.0
    _HT7_MOMENTUM_MAP = {
        "rotate_in": 1.0,
        "steady_strong": 0.85,
        "neutral": 0.50,
        "rotate_out": 0.15,
        "steady_weak": 0.0,
    }
    sector_momentum = detail.flags.get("sector_momentum_signal")
    ht7_applicable = sector_momentum is not None and sector_momentum != "neutral"
    ht7_item = ScoreItem(
        name="sector_momentum_signal",
        weight=5.0,
        is_applicable=sector_momentum is not None,
        note=(
            f"板块轮动动量信号；raw={sector_momentum}；"
            "rotate_in=1.0/steady_strong=0.85/neutral=0.50/"
            "rotate_out=0.15/steady_weak=0.0"
        ),
    )
    if sector_momentum is not None:
        ht7_item.raw_value = sector_momentum
        ht7_item.is_data_available = True
        ht7_item.subscore = _HT7_MOMENTUM_MAP.get(sector_momentum, 0.50)
    else:
        ht7_item.is_data_available = False
    axis.items.append(ht7_item)

    axis.compute()
    return axis
