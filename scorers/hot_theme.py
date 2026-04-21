"""hot_theme_score 计算模块（Session 5; Session 8 P1; Session 11 HT3/HT4 重构）.

热点题材强度评分轴，衡量股票在当前热点环境中的参与程度与强度。

──────────────────────────────────────────────────────
指标列表（6 项）：
  HT1  近5日收益率横截面百分位       weight=8   百分位→三段下限型（L=60,T=80,H=95）
  HT2  近10日收益率横截面百分位      weight=6   百分位→三段下限型（L=60,T=80,H=95）
  HT3  近10日大涨天数（涨幅>5%）     weight=8   离散型 0→0, 1→0.40, 2→0.70, 3→0.90, ≥4→1.0
       Session 11 改：从涨停次数改为大涨天数，对蓝筹和题材股均有区分度
  HT4  连续上涨天数                  weight=6   离散型 0→0, 1→0.25, 2→0.50, 3→0.70, 4→0.85, ≥5→1.0
       Session 11 改：从强势股池入选次数改为连续上涨天数，不依赖事件层接口
  HT5  所属行业近5日强度百分位       weight=7   百分位→三段下限型（L=55,T=75,H=90）
  HT6  所属概念板块热度百分位       weight=7   百分位→三段下限型（L=55,T=75,H=90）
       仅在 enable_concept_heat_module=True 且概念模块可用时纳入评分
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
) -> ScoreItem:
    """百分位 → 三段下限型评分.

    先算横截面百分位（0~100），再用三段折线映射：
      pctile <= L → 0
      L < pctile < T → 线性 0→0.70
      T <= pctile < H → 线性 0.70→1.0
      pctile >= H → 1.0
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

    pctile_item.subscore = round(subscore, 4)
    pctile_item.note += f"; L={L}/T={T}/H={H}; pctile_100={round(raw_pctile, 2)}"
    return pctile_item


def compute_hot_theme_score(
    detail: "HotStockDetail",
    pool: ScoringPool,
) -> AxisScore:
    """计算单只股票的 hot_theme_score."""
    axis = AxisScore(axis_name="hot_theme_score")

    # ── HT1: 近5日收益率横截面百分位 → 三段型（L=60,T=80,H=95）weight=8
    axis.items.append(_pctile_to_three_segment(
        name="return_5d_pctile",
        value=detail.return_5d,
        pool=pool.pool_return_5d if pool.pool_return_5d else None,
        ascending=True,
        weight=8.0,
        L=60, T=80, H=95,
        note="近5日收益率在scoring_pool中的百分位→三段型",
    ))

    # ── HT2: 近10日收益率横截面百分位 → 三段型（L=60,T=80,H=95）weight=6
    axis.items.append(_pctile_to_three_segment(
        name="return_10d_pctile",
        value=detail.return_10d,
        pool=pool.pool_return_10d if pool.pool_return_10d else None,
        ascending=True,
        weight=6.0,
        L=60, T=80, H=95,
        note="近10日收益率横截面百分位→三段型",
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
    # industry_heat_pctile_5d 已是 0~1 百分位，先转为 0~100 再映射
    industry_pctile_100 = detail.industry_heat_pctile_5d * 100.0 if detail.industry_heat_pctile_5d is not None else None
    axis.items.append(score_lower_bound(
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
    ))

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

    axis.compute()
    return axis
