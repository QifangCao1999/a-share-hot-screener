"""liquidity_execution_score 计算模块（Session 6; Session 8 P1 对齐提示词; Session 10 重校准）.

流动性与可执行性评分轴。

──────────────────────────────────────────────────────
指标列表（4 项，总权重 20）：
  LE1  近5日均成交额             weight=9  三段下限型（L=2亿,T=8亿,H=30亿）
       Session 10: 权重 8→9（填补 LE4 降权）
  LE2  近5日均换手率              weight=6  三段下限型（L=5%,T=12%,H=25%）
       Session 10: 权重 5→6（填补 LE4 降权）
  LE3  流通市值分档               weight=4  离散型（大盘友好分档，S10 重校准）
  LE4  近20日龙虎榜上榜次数       weight=1  离散型（enable_lhb=True 时有效；S10 降权 3→1）
──────────────────────────────────────────────────────

Session 10 改动说明：
  LE3 分档重校准（大盘友好）：
    原版：500-1500亿=0.45，>1500亿=0.20（对大盘蓝筹系统性惩罚）
    新版：500-1500亿=0.70，>1500亿=0.50（提升大盘蓝筹天花板约 0.15~0.20）
  LE4 龙虎榜权重降至 1（原 3），避免大盘蓝筹因不上龙虎榜被结构性压低。
  LE1 权重升至 9，LE2 权重升至 6，总权重保持 20（9+6+4+1=20）。
"""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING, Optional

from a_share_hot_screener.scoring import (
    AxisScore,
    ScoringPool,
    score_clamp_linear,
    score_discrete,
    score_lower_bound,
    score_percentile,
    ScoreItem,
)

if TYPE_CHECKING:
    from a_share_hot_screener.models import HotStockDetail

logger = logging.getLogger("a_share_hot_screener.scorers.liquidity_execution")

_1_YI = 100_000_000.0          # 1亿元

# LE3 流通市值分档映射（Session 10 重校准，大盘友好）
# < 15亿=0.20, 15~50亿=1.00, 50~150亿=0.90, 150~500亿=0.70
# 500~1500亿=0.70（原 0.45），>1500亿=0.50（原 0.20）
_LE3_BRACKETS = [
    (0,       15,   0.20),
    (15,      50,   1.00),
    (50,     150,   0.90),
    (150,    500,   0.70),
    (500,   1500,   0.70),   # S10 改: 0.45 → 0.70
    (1500, 99999,   0.50),   # S10 改: 0.20 → 0.50
]

# LE4 龙虎榜离散映射（提示词：0→20, 1→60, 2→80, ≥3→100, /100尺度）
_LE4_LHB_MAP = {0: 0.20, 1: 0.60, 2: 0.80, 3: 1.0}


def compute_liquidity_execution_score(
    detail: "HotStockDetail",
    pool: ScoringPool,
    enable_lhb_module: bool = True,
) -> AxisScore:
    """计算单只股票的 liquidity_execution_score."""
    axis = AxisScore(axis_name="liquidity_execution_score")

    # ── LE1: 近5日均成交额（绝对值三段下限型）weight=9 ───
    # L=2亿, T=8亿, H=30亿；S10 权重 8→9
    axis.items.append(score_lower_bound(
        name="amount_avg_5d",
        value=detail.amount_avg_5d,
        bad_threshold=2.0 * _1_YI,       # 2亿
        mid_threshold=8.0 * _1_YI,       # 8亿
        good_threshold=30.0 * _1_YI,     # 30亿
        weight=9.0,
        note=(
            "近5日均成交额（绝对值三段型：L=2亿/T=8亿/H=30亿）；"
            "weight=9（S10从8升）；"
        ),
    ))

    # ── LE2: 近5日均换手率（推算值, %）weight=6 ───────────
    # L=5%, T=12%, H=25%；S10 权重 5→6
    # #3: 推算换手率子分上限 0.85（不等同于真实 turnover_rate）
    turnover_5d_approx = _calc_turnover_5d_approx(detail)
    le2_note = _build_le2_note(detail, turnover_5d_approx)

    le2_item = score_lower_bound(
        name="turnover_avg_5d_approx",
        value=turnover_5d_approx,
        bad_threshold=5.0,
        mid_threshold=12.0,
        good_threshold=25.0,
        weight=6.0,
        note=le2_note,
    )
    # #3: proxy 子分上限
    if le2_item.subscore is not None and le2_item.subscore > _TURNOVER_PROXY_SUBSCORE_CAP:
        le2_item.note += f" [proxy_capped: {le2_item.subscore:.4f}→{_TURNOVER_PROXY_SUBSCORE_CAP}]"
        le2_item.subscore = _TURNOVER_PROXY_SUBSCORE_CAP
    axis.items.append(le2_item)

    # ── LE3: 流通市值分档 weight=4 ─────────────────────
    axis.items.append(_score_float_market_cap(detail))

    # ── LE4: 近20日龙虎榜上榜次数（离散型）weight=1 ──────
    # S10 降权 3→1：大盘蓝筹几乎不上龙虎榜，避免结构性压低
    lhb_val = detail.lhb_count_20d
    axis.items.append(score_discrete(
        name="lhb_count_20d",
        value=lhb_val,
        mapping=_LE4_LHB_MAP,
        default_score=None,
        weight=1.0,
        is_applicable=enable_lhb_module,
        note=(
            f"近20日龙虎榜上榜次数（离散：0→20/1→60/2→80/≥3→100）；"
            f"weight=1（S10从3降）；"
            f"source={detail.lhb_source or 'none'}；"
            f"raw={detail.lhb_count_20d}"
            + ("" if enable_lhb_module else "；enable_lhb_module=False→is_applicable=False")
        ),
    ))

    axis.compute()
    return axis


# ── 内部工具 ─────────────────────────────────────────────

# #3: 换手率 proxy 子分上限（amount/float_market_cap 推算值不等同于真实换手率）
_TURNOVER_PROXY_SUBSCORE_CAP = 0.85


def _calc_turnover_5d_approx(detail: "HotStockDetail") -> Optional[float]:
    """推算近5日均换手率（%）= amount_avg_5d / float_market_cap * 100."""
    amt = detail.amount_avg_5d
    fmc = detail.float_market_cap
    if amt is None or fmc is None or fmc <= 0:
        return None
    return amt / fmc * 100.0


def _build_le2_note(detail: "HotStockDetail", approx_val: Optional[float]) -> str:
    if approx_val is not None:
        return (
            f"近5日均换手率推算（amount_avg_5d/float_market_cap*100）；"
            f"approx={approx_val:.4f}%；三段型L=5%/T=12%/H=25%；"
            f"weight=6（S10从5升）；proxy_cap={_TURNOVER_PROXY_SUBSCORE_CAP}；"
        )
    return (
        f"近5日均换手率推算；amount_avg_5d={detail.amount_avg_5d}；"
        f"float_market_cap={detail.float_market_cap}；"
        "data_missing"
    )


def _score_float_market_cap(detail: "HotStockDetail") -> ScoreItem:
    """LE3: 流通市值分档评分（Session 10 大盘友好分档）weight=4."""
    item = ScoreItem(
        name="float_market_cap_bracket",
        weight=4.0,
        note=(
            "流通市值分档（亿，S10大盘友好）；"
            "<15亿→0.20, 15~50亿→1.00, 50~150亿→0.90, "
            "150~500亿→0.70, 500~1500亿→0.70, >1500亿→0.50"
        ),
    )

    fmc = detail.float_market_cap or detail.market_cap
    item.raw_value = fmc

    if fmc is None or (isinstance(fmc, float) and math.isnan(fmc)):
        item.is_data_available = False
        item.note += " [data_missing]"
        return item

    fmc_bn = fmc / _1_YI
    item.derived_value = round(fmc_bn, 4)
    item.is_data_available = True

    for lo, hi, subscore in _LE3_BRACKETS:
        if lo < fmc_bn <= hi:
            item.subscore = subscore
            return item

    # 兜底
    if fmc_bn <= 0:
        item.subscore = 0.20
        item.note += " [below_zero]"
    else:
        item.subscore = 0.50   # 超大市值兜底，同 >1500亿

    return item
