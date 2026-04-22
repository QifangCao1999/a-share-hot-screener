"""trend_flow_score 计算模块（Session 5; Session 8 P1; Session 22 新增TF6/TF7）.

趋势与资金流向评分轴，衡量股票当前的量价结构与资金流入态势。

──────────────────────────────────────────────────────
指标列表（7 项）：
  TF1  20日区间收盘位置                weight=7   三段下限型（L=0.55,T=0.75,H=0.95）
  TF2  均线多头排列程度                 weight=6   离散型（close>ma5, ma5>ma10, ma10>ma20 三条件）
  TF3  最新成交量/20日均量（量比）       weight=6   三段下限型（L=0.8,T=1.8,H=3.0）clamp=5
  TF4  最新收盘在日内位置（CLV）         weight=5   三段下限型（L=0.40,T=0.70,H=0.90）
  TF5  近5日均成交额/近20日均成交额      weight=6   三段下限型（L=1.0,T=1.5,H=2.5）clamp=4
  TF6  近5日主力净流入占比(%)         weight=6   三段下限型（L=-5,T=3,H=10）
       Session 22 新增：从 flags["net_main_inflow_ratio_5d"] 读取
  TF7  近5日融资净买入占比(%)         weight=4   三段下限型（L=-2,T=1,H=5）
       Session 22 新增：从 flags["margin_buy_net_ratio_5d"] 读取；非两融标的 is_applicable=False
──────────────────────────────────────────────────────
"""

from __future__ import annotations

import logging
import math
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

logger = logging.getLogger("a_share_hot_screener.scorers.trend_flow")

# TF2 均线多头排列条件数 → subscore 映射
# 提示词：0个=0, 1个=35, 2个=70, 3个=100 （/100 尺度）
_TF2_MA_MAP = {0: 0.0, 1: 0.35, 2: 0.70, 3: 1.0}


def compute_trend_flow_score(
    detail: "HotStockDetail",
    pool: ScoringPool,
) -> AxisScore:
    """计算单只股票的 trend_flow_score."""
    axis = AxisScore(axis_name="trend_flow_score")

    # ── TF1: 20日区间收盘位置（三段下限型：L=0.55,T=0.75,H=0.95）weight=7
    axis.items.append(score_lower_bound(
        name="close_position_20d",
        value=detail.close_position_20d,
        bad_threshold=0.55,
        mid_threshold=0.75,
        good_threshold=0.95,
        weight=7.0,
        note="收盘价在近20日高低区间的位置（0=最低，1=最高）；来自Tushare daily；三段型L=0.55/T=0.75/H=0.95",
    ))

    # ── TF2: 均线多头排列程度（离散型，3条件）weight=6
    axis.items.append(_score_ma_bullish_alignment(detail, weight=6.0))

    # ── TF3: 量比（volume_ratio_20d）三段下限型（L=1.1,T=1.8,H=3.0）weight=6
    vr = detail.volume_ratio_20d
    vr_clamped = min(vr, 5.0) if vr is not None else None
    axis.items.append(score_lower_bound(
        name="volume_ratio_20d",
        value=vr_clamped,
        bad_threshold=0.8,
        mid_threshold=1.8,
        good_threshold=3.0,
        weight=6.0,
        note=(
            f"量比（最新量/20日均量）；raw={vr}；clamp_at_5.0；"
            "三段型L=0.8/T=1.8/H=3.0（S10改：L从1.1降至0.8）；NOTE: Tushare vol"
        ),
    ))

    # ── TF4: CLV（三段下限型：L=0.40,T=0.70,H=0.90）weight=5
    axis.items.append(score_lower_bound(
        name="clv_latest",
        value=detail.clv_latest,
        bad_threshold=0.40,
        mid_threshold=0.70,
        good_threshold=0.90,
        weight=5.0,
        note="当日CLV=(close-low)/(high-low)；三段型L=0.40/T=0.70/H=0.90；来自Tushare daily",
    ))

    # ── TF5: 量能比率（三段下限型：L=1.0,T=1.5,H=2.5）weight=6
    ar = detail.amount_ratio_5d_to_20d
    ar_clamped = min(ar, 4.0) if ar is not None else None
    axis.items.append(score_lower_bound(
        name="amount_ratio_5d_to_20d",
        value=ar_clamped,
        bad_threshold=1.0,
        mid_threshold=1.5,
        good_threshold=2.5,
        weight=6.0,
        note=(
            f"近5日均成交额/近20日均成交额；raw={ar}；clamp_at_4.0；"
            "三段型L=1.0/T=1.5/H=2.5；"
        ),
    ))

    # ── TF6: 近5日主力净流入占比（三段下限型：L=-5,T=3,H=10）weight=6 ────
    # Session 22 新增：主力资金(大单+特大单)净流入占总成交额比例
    main_inflow = detail.flags.get("net_main_inflow_ratio_5d")
    axis.items.append(score_lower_bound(
        name="net_main_inflow_ratio_5d",
        value=main_inflow,
        bad_threshold=-5.0,
        mid_threshold=3.0,
        good_threshold=10.0,
        weight=6.0,
        note=(
            f"近5日主力净流入占总成交额比(%)；raw={main_inflow}；"
            "三段型L=-5%/T=3%/H=10%；来源flags[net_main_inflow_ratio_5d]"
        ),
    ))

    # ── TF7: 近5日融资净买入占比（三段下限型：L=-2,T=1,H=5）weight=4 ────
    # Session 22 新增：仅两融标的适用，非两融 is_applicable=False
    margin_ratio = detail.flags.get("margin_buy_net_ratio_5d")
    is_margin = detail.flags.get("is_margin_eligible", False)
    axis.items.append(score_lower_bound(
        name="margin_buy_net_ratio_5d",
        value=margin_ratio,
        bad_threshold=-2.0,
        mid_threshold=1.0,
        good_threshold=5.0,
        weight=4.0,
        is_applicable=bool(is_margin),
        note=(
            f"近5日融资净买入占成交额(%)；raw={margin_ratio}；"
            f"is_margin={is_margin}；"
            "三段型L=-2%/T=1%/H=5%；非两融标的is_applicable=False"
        ),
    ))

    axis.compute()
    return axis


def _score_ma_bullish_alignment(
    detail: "HotStockDetail",
    weight: float = 6.0,
) -> ScoreItem:
    """TF2: 均线多头排列程度（三条件离散型）.

    条件：
      1. close > ma5
      2. ma5 > ma10
      3. ma10 > ma20
    离散映射：0条件=0, 1条件=0.35, 2条件=0.70, 3条件=1.0
    """
    item = ScoreItem(
        name="ma_bullish_alignment",
        weight=weight,
        note=(
            "均线多头排列程度；"
            "条件：close>ma5, ma5>ma10, ma10>ma20；"
            "0=0/1=0.35/2=0.70/3=1.0"
        ),
    )

    close = detail.latest_price
    ma5 = detail.ma5
    ma10 = detail.ma10
    ma20 = detail.ma20

    # 需要 close + 至少 ma5/ma10 才有意义
    if close is None or ma5 is None or ma10 is None:
        item.is_data_available = False
        item.raw_value = None
        item.note += " [data_missing: close/ma5/ma10]"
        return item

    conditions_met = 0
    conditions_detail = []

    if close > ma5:
        conditions_met += 1
        conditions_detail.append("close>ma5")

    if ma5 > ma10:
        conditions_met += 1
        conditions_detail.append("ma5>ma10")

    if ma20 is not None and ma10 > ma20:
        conditions_met += 1
        conditions_detail.append("ma10>ma20")
    elif ma20 is None:
        # ma20 缺失时，只检查前两个条件，但 is_applicable 仍为 True
        item.note += " [ma20_missing, only 2 conditions checked]"

    item.raw_value = conditions_met
    item.derived_value = conditions_met
    item.is_data_available = True

    # 映射
    item.subscore = round(_TF2_MA_MAP.get(conditions_met, 0.0), 4)
    item.note += f"; met={conditions_detail}"
    return item
