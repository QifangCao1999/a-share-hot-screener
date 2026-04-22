"""Context Scores 模块 — HT8/HT9/HT10 experimental 评分.

Phase 3 (DESIGN_v2.1 Section 6): 三个 experimental 评分子项，
初期不进入 total_score，用于回测验证。

──────────────────────────────────────────────────────
HT8: 市场确认度 (market_confirmation_score)
  用结构化市场行为数据推断该股近期被市场关注的程度。
  离散型评分，根据信号组合确定确认级别。
  Level 3 缺失时 cap 0.60。

HT9: 板块扩散度 (sector_breadth_score)
  该股所属热门概念板块内，同步走强股票的比例。
  三段下限型 L=10/T=30/H=50。
  小板块 cap / 不适用逻辑。
  辅助输出: sector_amount_breadth_ratio, top5_amount_concentration。

HT10: 板块内辨识度 (sector_position_score)
  该股在所属板块内的领涨/容量地位。
  三子项加权: rank_pctile(0.30) + amount_share(0.35) + first_zt(0.35)。
  Level 2 缺失时重新归一 + cap 0.80。
──────────────────────────────────────────────────────
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from a_share_hot_screener.event_layer import EventLayerContext
    from a_share_hot_screener.models import HotStockDetail

logger = logging.getLogger("a_share_hot_screener.context_scores")


# ════════════════════════════════════════════════════════
# 数据模型
# ════════════════════════════════════════════════════════

@dataclass
class ContextScoresResult:
    """三个 context score 的计算结果容器."""

    # HT8 市场确认度
    ht8_score: Optional[float] = None
    ht8_confirmation_level: str = ""      # 确认级别描述
    ht8_signals: List[str] = field(default_factory=list)
    ht8_degraded: bool = False            # Level 3 缺失降级

    # HT9 板块扩散度
    ht9_score: Optional[float] = None
    ht9_breadth_ratio: Optional[float] = None       # 板块内涨幅>5%股票占比
    ht9_is_applicable: bool = True
    ht9_cap_applied: bool = False
    ht9_sector_name: str = ""
    ht9_sector_size: int = 0
    # 辅助输出
    ht9_amount_breadth_ratio: Optional[float] = None   # 成交额放大>50%的股票占比
    ht9_top5_amount_concentration: Optional[float] = None  # Top5成交额占板块总成交额比例

    # HT10 板块内辨识度
    ht10_score: Optional[float] = None
    ht10_rank_pctile: Optional[float] = None     # 涨幅排名百分位
    ht10_amount_share_score: Optional[float] = None  # 成交额占比得分
    ht10_first_zt_score: Optional[float] = None  # 率先涨停得分
    ht10_is_applicable: bool = True
    ht10_position_type: str = ""          # frontline_like/capacity_core_like/follower_like/unknown
    ht10_confidence: str = "high"         # high/medium
    ht10_degraded: bool = False           # Level 2 缺失降级

    def to_dict(self) -> Dict[str, Any]:
        """序列化为 dict（用于 flags/summary 输出）."""
        return {
            "ht8_score": _round_opt(self.ht8_score),
            "ht8_confirmation_level": self.ht8_confirmation_level,
            "ht8_signals": self.ht8_signals,
            "ht8_degraded": self.ht8_degraded,
            "ht9_score": _round_opt(self.ht9_score),
            "ht9_breadth_ratio": _round_opt(self.ht9_breadth_ratio),
            "ht9_is_applicable": self.ht9_is_applicable,
            "ht9_cap_applied": self.ht9_cap_applied,
            "ht9_sector_name": self.ht9_sector_name,
            "ht9_sector_size": self.ht9_sector_size,
            "ht9_amount_breadth_ratio": _round_opt(self.ht9_amount_breadth_ratio),
            "ht9_top5_amount_concentration": _round_opt(self.ht9_top5_amount_concentration),
            "ht10_score": _round_opt(self.ht10_score),
            "ht10_rank_pctile": _round_opt(self.ht10_rank_pctile),
            "ht10_amount_share_score": _round_opt(self.ht10_amount_share_score),
            "ht10_first_zt_score": _round_opt(self.ht10_first_zt_score),
            "ht10_is_applicable": self.ht10_is_applicable,
            "ht10_position_type": self.ht10_position_type,
            "ht10_confidence": self.ht10_confidence,
            "ht10_degraded": self.ht10_degraded,
        }


def _round_opt(v: Optional[float], n: int = 4) -> Optional[float]:
    return round(v, n) if v is not None else None


# ════════════════════════════════════════════════════════
# HT8: 市场确认度 (market_confirmation_score)
# ════════════════════════════════════════════════════════

def compute_ht8(
    detail: "HotStockDetail",
    event_ctx: "EventLayerContext",
) -> tuple:
    """计算 HT8 市场确认度.

    Returns:
        (score, confirmation_level, signals, degraded)
    """
    signals: List[str] = []

    # 1. 涨停信号
    has_limit_up = False
    lu_5d = detail.limit_up_count_5d
    if lu_5d is not None and lu_5d > 0:
        has_limit_up = True
        signals.append("limit_up")

    # 2. 龙虎榜信号
    has_lhb = False
    lhb_20d = detail.lhb_count_20d
    if lhb_20d is not None and lhb_20d > 0:
        has_lhb = True
        signals.append("lhb")

    # 3. 概念板块活跃度（需 Level 3: ths_daily）
    ths_available = event_ctx.concept_heat_mode not in ("none", "")
    concept_hot = False
    concept_very_hot = False

    if ths_available and detail.concept_heat_pctile_5d is not None:
        pctile = detail.concept_heat_pctile_5d
        if pctile >= 0.90:  # Top 10%
            concept_very_hot = True
            signals.append("concept_top10")
        elif pctile >= 0.80:  # Top 20%
            concept_hot = True
            signals.append("concept_top20")

    # 4. 资金异动信号（成交额放大 + 涨幅）
    has_volume_surge = False
    amt_ratio = detail.amount_ratio_5d_to_20d
    ret_5d = detail.return_5d
    if (amt_ratio is not None and amt_ratio >= 2.0
            and ret_5d is not None and ret_5d >= 5.0):
        has_volume_surge = True
        if "limit_up" not in signals:  # 避免重复
            signals.append("volume_surge")

    # 确认级别判定
    degraded = not ths_available

    if has_lhb and has_limit_up and concept_very_hot:
        score = 1.0
        level = "multi_confirmed"
    elif has_limit_up and (concept_hot or concept_very_hot):
        score = 0.80
        level = "sector_resonance"
    elif (has_lhb and not has_limit_up) or (has_limit_up and not concept_hot and not concept_very_hot):
        score = 0.60
        level = "single_anomaly"
    elif has_volume_surge:
        score = 0.40
        level = "volume_anomaly"
    else:
        score = 0.10
        level = "no_confirmation"

    # Level 3 缺失降级
    if degraded and score > 0.60:
        score = 0.60
        if level in ("multi_confirmed", "sector_resonance"):
            level = "single_anomaly"  # 降级到不需要板块数据的级别

    return score, level, signals, degraded


# ════════════════════════════════════════════════════════
# HT9: 板块扩散度 (sector_breadth_score)
# ════════════════════════════════════════════════════════

def compute_ht9(
    detail: "HotStockDetail",
    event_ctx: "EventLayerContext",
    sector_members_daily: Optional[Dict[str, Dict[str, float]]] = None,
) -> tuple:
    """计算 HT9 板块扩散度.

    Args:
        sector_members_daily: 板块成分股的日线数据
            {ts_code: {"return_5d": float, "amount_ratio": float, "amount_5d": float}}
            如果为 None，HT9 不适用。

    Returns:
        (score, breadth_ratio, is_applicable, cap_applied,
         sector_name, sector_size,
         amount_breadth_ratio, top5_amount_concentration)
    """
    if sector_members_daily is None or len(sector_members_daily) == 0:
        return None, None, False, False, "", 0, None, None

    sector_size = len(sector_members_daily)
    sector_name = ""  # Will be set by caller

    # 板块成分太少
    if sector_size < 10:
        return None, None, False, False, sector_name, sector_size, None, None

    # 计算涨幅>5%的股票占比
    strong_count = 0
    for ts_code, data in sector_members_daily.items():
        ret = data.get("return_5d", 0.0)
        if ret is not None and ret > 5.0:
            strong_count += 1

    breadth_ratio = strong_count / sector_size * 100.0  # 百分比

    # 三段下限型: L=10, T=30, H=50
    if breadth_ratio >= 50.0:
        score = 1.0
    elif breadth_ratio <= 10.0:
        score = 0.0
    elif breadth_ratio < 30.0:
        score = 0.70 * (breadth_ratio - 10.0) / 20.0
    else:
        score = 0.70 + 0.30 * (breadth_ratio - 30.0) / 20.0

    # 小板块 cap
    cap_applied = False
    if 10 <= sector_size < 30:
        cap_applied = True
        score = min(score, 0.80)

    score = round(score, 4)

    # 辅助输出
    amount_breadth_ratio = _calc_amount_breadth_ratio(sector_members_daily)
    top5_concentration = _calc_top5_amount_concentration(sector_members_daily)

    return (score, round(breadth_ratio, 2), True, cap_applied,
            sector_name, sector_size,
            amount_breadth_ratio, top5_concentration)


def _calc_amount_breadth_ratio(members: Dict[str, Dict[str, float]]) -> Optional[float]:
    """成交额放大>50%的股票占比."""
    if not members:
        return None
    surge_count = 0
    total = 0
    for data in members.values():
        ratio = data.get("amount_ratio")
        if ratio is not None:
            total += 1
            if ratio > 1.5:
                surge_count += 1
    if total == 0:
        return None
    return round(surge_count / total * 100.0, 2)


def _calc_top5_amount_concentration(members: Dict[str, Dict[str, float]]) -> Optional[float]:
    """Top5 成交额占板块总成交额比例."""
    amounts = []
    for data in members.values():
        amt = data.get("amount_5d")
        if amt is not None and amt > 0:
            amounts.append(amt)
    if len(amounts) < 5:
        return None
    amounts.sort(reverse=True)
    total = sum(amounts)
    top5 = sum(amounts[:5])
    if total <= 0:
        return None
    return round(top5 / total * 100.0, 2)


# ════════════════════════════════════════════════════════
# HT10: 板块内辨识度 (sector_position_score)
# ════════════════════════════════════════════════════════

def compute_ht10(
    detail: "HotStockDetail",
    event_ctx: "EventLayerContext",
    sector_members_daily: Optional[Dict[str, Dict[str, float]]] = None,
) -> tuple:
    """计算 HT10 板块内辨识度.

    Returns:
        (score, rank_pctile, amount_share_score, first_zt_score,
         is_applicable, position_type, confidence, degraded)
    """
    if sector_members_daily is None or len(sector_members_daily) < 10:
        return None, None, None, None, False, "unknown", "high", False

    sector_size = len(sector_members_daily)
    ts_code = detail.ts_code

    # 1. 涨幅排名百分位 (rank_pctile)
    returns = []
    for code, data in sector_members_daily.items():
        ret = data.get("return_5d")
        if ret is not None:
            returns.append((code, ret))

    if len(returns) < 2:
        return None, None, None, None, False, "unknown", "high", False

    returns.sort(key=lambda x: x[1])
    my_return = None
    for code, ret in returns:
        if code == ts_code:
            my_return = ret
            break

    if my_return is None:
        # 该股可能不在板块成分中，尝试使用 detail.return_5d
        my_return = detail.return_5d
        if my_return is None:
            return None, None, None, None, False, "unknown", "high", False

    # 计算百分位
    import bisect
    return_values = [r for _, r in returns]
    idx = bisect.bisect_right(return_values, my_return)
    rank_pctile = idx / len(return_values)

    # 2. 成交额占比得分 (amount_share_score)
    amounts = []
    my_amount = None
    for code, data in sector_members_daily.items():
        amt = data.get("amount_5d")
        if amt is not None and amt > 0:
            amounts.append((code, amt))
            if code == ts_code:
                my_amount = amt

    if my_amount is None:
        # Fallback to detail
        if detail.amount_avg_5d is not None and detail.amount_avg_5d > 0:
            my_amount = detail.amount_avg_5d * 5  # 近似总额
        else:
            my_amount = 0

    # 成交额在板块 Top10 中的占比
    amounts.sort(key=lambda x: x[1], reverse=True)
    top10_amounts = amounts[:10]
    top10_total = sum(a for _, a in top10_amounts)

    if top10_total > 0 and my_amount > 0:
        my_share = my_amount / top10_total
        # 映射: 0%→0, 10%→0.50, 20%→0.80, ≥30%→1.0
        if my_share >= 0.30:
            amount_share_score = 1.0
        elif my_share <= 0.0:
            amount_share_score = 0.0
        elif my_share < 0.10:
            amount_share_score = 0.50 * (my_share / 0.10)
        elif my_share < 0.20:
            amount_share_score = 0.50 + 0.30 * ((my_share - 0.10) / 0.10)
        else:
            amount_share_score = 0.80 + 0.20 * ((my_share - 0.20) / 0.10)
    else:
        amount_share_score = 0.0

    # 3. 是否率先涨停 (first_zt_score)
    zt_available = event_ctx.zt_pool_available
    first_zt_score = None
    degraded = False

    if zt_available:
        # 检查该股是否在涨停池中且较早（通过 limit_up_count 判断强度）
        lu_5d = detail.limit_up_count_5d or 0
        lu_sector_max = 0
        for code, data in sector_members_daily.items():
            lu = data.get("limit_up_count_5d", 0)
            if lu is not None and lu > lu_sector_max:
                lu_sector_max = lu

        if lu_5d >= 2 and lu_5d >= lu_sector_max:
            first_zt_score = 1.0  # 多次涨停且板块最多
        elif lu_5d >= 1 and lu_5d >= lu_sector_max:
            first_zt_score = 0.80  # 涨停且领先
        elif lu_5d >= 1:
            first_zt_score = 0.50  # 涨停但不领先
        else:
            first_zt_score = 0.0   # 未涨停

        # 完整公式
        score = (
            0.30 * rank_pctile
            + 0.35 * amount_share_score
            + 0.35 * first_zt_score
        )
        confidence = "high"
    else:
        # Level 2 缺失降级
        degraded = True
        score = (
            0.46 * rank_pctile       # 0.30/0.65 归一
            + 0.54 * amount_share_score  # 0.35/0.65 归一
        )
        score = min(score, 0.80)  # cap
        confidence = "medium"

    score = round(score, 4)
    rank_pctile = round(rank_pctile, 4)
    amount_share_score = round(amount_share_score, 4)
    if first_zt_score is not None:
        first_zt_score = round(first_zt_score, 4)

    # 辨识度类型
    position_type = _classify_position_type(
        rank_pctile, amount_share_score, first_zt_score,
    )

    return (score, rank_pctile, amount_share_score, first_zt_score,
            True, position_type, confidence, degraded)


def _classify_position_type(
    rank_pctile: float,
    amount_share_score: float,
    first_zt_score: Optional[float],
) -> str:
    """根据子项得分分类辨识度类型."""
    if first_zt_score is not None and first_zt_score >= 0.80 and rank_pctile >= 0.70:
        return "frontline_like"
    if amount_share_score >= 0.60 and rank_pctile >= 0.40:
        return "capacity_core_like"
    if rank_pctile < 0.40 and amount_share_score < 0.30:
        return "follower_like"
    return "unknown"


# ════════════════════════════════════════════════════════
# 综合计算入口
# ════════════════════════════════════════════════════════

def compute_context_scores(
    detail: "HotStockDetail",
    event_ctx: "EventLayerContext",
    sector_members_daily: Optional[Dict[str, Dict[str, float]]] = None,
    sector_name: str = "",
) -> ContextScoresResult:
    """计算 HT8/HT9/HT10 三个 context score.

    Args:
        detail: 已填充事件层字段的 HotStockDetail
        event_ctx: 事件层上下文
        sector_members_daily: 板块成分股的日线数据
            {ts_code: {"return_5d": float, "amount_ratio": float,
                       "amount_5d": float, "limit_up_count_5d": int}}
        sector_name: 板块名称（用于标注）

    Returns:
        ContextScoresResult
    """
    result = ContextScoresResult()

    # HT8
    try:
        ht8_score, level, signals, degraded = compute_ht8(detail, event_ctx)
        result.ht8_score = ht8_score
        result.ht8_confirmation_level = level
        result.ht8_signals = signals
        result.ht8_degraded = degraded
    except Exception as e:
        logger.error("compute_ht8(%s) 异常: %s", detail.code, e, exc_info=True)

    # HT9
    try:
        (ht9_score, breadth_ratio, is_applicable, cap_applied,
         _, sector_size,
         amount_breadth, top5_conc) = compute_ht9(detail, event_ctx, sector_members_daily)
        result.ht9_score = ht9_score
        result.ht9_breadth_ratio = breadth_ratio
        result.ht9_is_applicable = is_applicable
        result.ht9_cap_applied = cap_applied
        result.ht9_sector_name = sector_name
        result.ht9_sector_size = sector_size
        result.ht9_amount_breadth_ratio = amount_breadth
        result.ht9_top5_amount_concentration = top5_conc
    except Exception as e:
        logger.error("compute_ht9(%s) 异常: %s", detail.code, e, exc_info=True)

    # HT10
    try:
        (ht10_score, rank_pctile, amount_share, first_zt,
         ht10_applicable, position_type, confidence, degraded) = compute_ht10(
            detail, event_ctx, sector_members_daily,
        )
        result.ht10_score = ht10_score
        result.ht10_rank_pctile = rank_pctile
        result.ht10_amount_share_score = amount_share
        result.ht10_first_zt_score = first_zt
        result.ht10_is_applicable = ht10_applicable
        result.ht10_position_type = position_type
        result.ht10_confidence = confidence
        result.ht10_degraded = degraded
    except Exception as e:
        logger.error("compute_ht10(%s) 异常: %s", detail.code, e, exc_info=True)

    return result
