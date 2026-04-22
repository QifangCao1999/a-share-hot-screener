"""risk_control_score 计算模块（Session 6; Session 8 P1 对齐提示词; Session 10 重校准; Session 22 新增RC6/7/8）.

风险控制评分轴，高分=低风险。

──────────────────────────────────────────────────────
指标列表（10 项）：
  RC1  近5日一字板次数             weight=4  离散型（0→1.0, 1→0.50, 2→0.20, ≥3→0.0）
       Session 10 改动：从"最新1日一字板"扩展为"近5日一字板次数"，增强区分度
  RC2  近5日均振幅/涨跌停幅度     weight=3  三段上限型（G=0.5,T=1.0,B=1.8）clamp=2.0
  RC3  收盘价偏离10日均线绝对值   weight=4  三段上限型（G=2%,T=6%,B=15%）
       Session 10 改动：从 G=3%/T=10%/B=20% 收紧为 G=2%/T=6%/B=15%
  RC4  近5日长上影/炸板proxy次数  weight=2  离散型（0=1.0,1=0.70,2=0.40,≥3=0.0）
  RC5  近3日累计涨幅/涨跌停幅度   weight=2  三段上限型（G=0.8,T=1.8,B=3.5）clamp=[-4,4]
  RC6  质押比例(%)               weight=2  三段上限型（G=5,T=20,B=40）
       Session 22 新增：从 flags["pledge_ratio_latest"] 读取
  RC7  未来20日解禁占比(%)        weight=2  三段上限型（G=1,T=5,B=15）
       Session 22 新增：从 flags["restricted_shares_unlock_ratio_20d"] 读取
  RC8  近3月股东人数增幅(%)       weight=1  三段上限型（G=0,T=10,B=25）
       Session 22 新增：从 flags["shareholder_net_reduction_ratio_3m"] 读取
  RC9  近30日股东净减持占流通比(%) weight=3  三段上限型（G=0,T=0.5,B=2）
       Session 22 新增：从 flags["net_holder_reduction_ratio_30d"] 读取
  RC10 近5日融券余额变化率(%)   weight=1  三段上限型（G=10,T=50,B=100）
       Session 22 新增：从 flags["short_sell_ratio_change_5d"] 读取；非两融 is_applicable=False
──────────────────────────────────────────────────────
"""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING, Optional

from a_share_hot_screener.limit_rules import infer_limit_pct
from a_share_hot_screener.scoring import (
    AxisScore,
    ScoreItem,
    ScoringPool,
    score_discrete,
    score_upper_bound,
    score_clamp_linear,
)

if TYPE_CHECKING:
    from a_share_hot_screener.models import HotStockDetail

logger = logging.getLogger("a_share_hot_screener.scorers.risk_control")

# RC1 近5日一字板次数离散映射（Session 10 新增）
# 0次=完全无压制=1.0; 1次=轻度=0.50; 2次=较重=0.20; ≥3次=严重=0.0
_RC1_LIMIT_BOARD_5D_MAP = {0: 1.0, 1: 0.50, 2: 0.20, 3: 0.0}

# RC4 离散映射（提示词：0=100, 1=70, 2=40, ≥3=0，/100尺度）
_RC4_SHADOW_MAP = {0: 1.0, 1: 0.70, 2: 0.40, 3: 0.0}


def compute_risk_control_score(
    detail: "HotStockDetail",
    pool: ScoringPool,
) -> AxisScore:
    """计算单只股票的 risk_control_score."""
    axis = AxisScore(axis_name="risk_control_score")

    limit_pct = infer_limit_pct(detail.code)

    # ── RC1: 近5日一字板次数（离散型）weight=4 ───────────
    # Session 10: 从单日改为5日，增加区分度
    axis.items.append(_score_limit_board_5d(detail))

    # ── RC2: 近5日均振幅/涨跌停幅度（三段上限型：G=0.5,T=1.0,B=1.8）weight=3
    amp = detail.amp_norm_avg_5d
    norm_amp = None
    if amp is not None and limit_pct > 0:
        norm_amp = amp / limit_pct
        norm_amp = min(norm_amp, 2.0)

    axis.items.append(score_upper_bound(
        name="amp_norm_5d_ratio",
        value=norm_amp,
        good_threshold=0.5,
        mid_threshold=1.0,
        bad_threshold=1.8,
        weight=3.0,
        note=(
            f"近5日均振幅/涨跌停幅度（{limit_pct}%）；"
            f"raw_amp={amp}；norm={norm_amp}；clamp_at_2.0；"
            "三段型G=0.5/T=1.0/B=1.8"
        ),
    ))

    # ── RC3: 价格偏离 MA10 绝对值（三段上限型：G=2%,T=6%,B=15%）weight=4
    # Session 10 改动: 阈值收紧 G=3%→2%, T=10%→6%, B=20%→15%，提升区分度
    dist = detail.abs_distance_to_ma10
    abs_dist_pct = abs(dist) * 100.0 if dist is not None else None  # 转为%

    axis.items.append(score_upper_bound(
        name="abs_deviation_ma10",
        value=abs_dist_pct,
        good_threshold=2.0,
        mid_threshold=6.0,
        bad_threshold=15.0,
        weight=4.0,
        note=(
            f"abs((close-MA10)/MA10)*100；"
            f"raw_dist={dist}；abs_pct={abs_dist_pct}；"
            "三段型G=2%/T=6%/B=15%（S10收紧）"
        ),
    ))

    # ── RC4: 近5日上影线 proxy 次数（离散型）weight=2
    axis.items.append(score_discrete(
        name="upper_shadow_count_5d",
        value=detail.upper_shadow_count_5d,
        mapping=_RC4_SHADOW_MAP,
        default_score=None,
        weight=2.0,
        note=(
            "近5日长上影/炸板proxy次数；"
            "proxy_rule: (high-close)/(high-low)>=0.45 and amp>=5%；"
            "0=1.0/1=0.70/2=0.40/≥3=0.0"
        ),
    ))

    # ── RC5: 近3日累积涨幅/涨跌停幅度（三段上限型：G=0.8,T=1.8,B=3.5）weight=2
    r3d = detail.return_3d
    norm_r3d = None
    if r3d is not None and limit_pct > 0:
        norm_r3d = r3d / limit_pct
        norm_r3d = max(-4.0, min(norm_r3d, 4.0))

    axis.items.append(score_upper_bound(
        name="return_3d_vs_limit",
        value=norm_r3d,
        good_threshold=0.8,
        mid_threshold=1.8,
        bad_threshold=3.5,
        weight=2.0,
        note=(
            f"近3日涨幅/涨跌停幅度({limit_pct}%)；"
            f"raw_return_3d={r3d}；norm={norm_r3d}；"
            "三段型G=0.8/T=1.8/B=3.5"
        ),
    ))

    # ── RC6: 质押比例（三段上限型：G=5%,T=20%,B=40%）weight=2 ────
    # Session 22 新增：高质押 → 大股东融资压力大，有被强平风险
    pledge_ratio = detail.flags.get("pledge_ratio_latest")
    axis.items.append(score_upper_bound(
        name="pledge_ratio_pct",
        value=pledge_ratio,
        good_threshold=5.0,
        mid_threshold=20.0,
        bad_threshold=40.0,
        weight=2.0,
        note=(
            f"质押占总股本比例(%)；raw={pledge_ratio}；"
            "三段型G=5%/T=20%/B=40%；来源flags[pledge_ratio_latest]"
        ),
    ))

    # ── RC7: 未来20日限售解禁占比（三段上限型：G=1%,T=5%,B=15%）weight=2 ────
    # Session 22 新增：大量解禁 → 抛压风险
    unlock_ratio = detail.flags.get("restricted_shares_unlock_ratio_20d")
    axis.items.append(score_upper_bound(
        name="unlock_ratio_20d",
        value=unlock_ratio,
        good_threshold=1.0,
        mid_threshold=5.0,
        bad_threshold=15.0,
        weight=2.0,
        note=(
            f"未来20日解禁占流通盘比例(%)；raw={unlock_ratio}；"
            "三段型G=1%/T=5%/B=15%；来源flags[restricted_shares_unlock_ratio_20d]"
        ),
    ))

    # ── RC8: 近3月股东人数增幅（三段上限型：G=0%,T=10%,B=25%）weight=1 ────
    # Session 22 新增：人数增加=筹码分散=bearish；人数减少=集中=bullish
    # 正值=增加（差），负值=减少（好）
    holder_change = detail.flags.get("shareholder_net_reduction_ratio_3m")
    axis.items.append(score_upper_bound(
        name="shareholder_increase_3m",
        value=holder_change,
        good_threshold=0.0,
        mid_threshold=10.0,
        bad_threshold=25.0,
        weight=1.0,
        note=(
            f"近3月股东人数变化率(%)；raw={holder_change}；"
            "正值=人数增加=筹码分散；负值=人数减少=集中；"
            "三段型G=0%/T=10%/B=25%；来源flags[shareholder_net_reduction_ratio_3m]"
        ),
    ))

    # ── RC9: 近30日股东净减持占流通比（三段上限型：G=0%,T=0.5%,B=2%）weight=3 ────
    # Session 22 新增：大股东/高管集中减持 = 最直接的跑路预警
    holder_trade = detail.flags.get("net_holder_reduction_ratio_30d")
    axis.items.append(score_upper_bound(
        name="holder_trade_reduction_30d",
        value=holder_trade,
        good_threshold=0.0,
        mid_threshold=0.5,
        bad_threshold=2.0,
        weight=3.0,
        note=(
            f"近30日股东净减持占流通比(%)；raw={holder_trade}；"
            "正值=净减持；负值=净增持；"
            "三段型G=0%/T=0.5%/B=2%；来源flags[net_holder_reduction_ratio_30d]"
        ),
    ))

    # ── RC10: 近5日融券余额变化率（三段上限型：G=10%,T=50%,B=100%）weight=1 ────
    # Session 22 新增：融券余额暴增 = 做空压力；仅两融标的适用
    short_sell_change = detail.flags.get("short_sell_ratio_change_5d")
    is_margin = detail.flags.get("is_margin_eligible", False)
    axis.items.append(score_upper_bound(
        name="short_sell_pressure_5d",
        value=short_sell_change,
        good_threshold=10.0,
        mid_threshold=50.0,
        bad_threshold=100.0,
        weight=1.0,
        is_applicable=bool(is_margin),
        note=(
            f"近5日融券余额变化率(%)；raw={short_sell_change}；"
            f"is_margin={is_margin}；"
            "三段型G=10%/T=50%/B=100%；非两融标的is_applicable=False"
        ),
    ))

    axis.compute()
    return axis


def _score_limit_board_5d(
    detail: "HotStockDetail",
) -> ScoreItem:
    """RC1: 近5日一字板次数（离散型）weight=4.

    Session 10 改动：
      原逻辑（S8）：只看最新1日是否一字板（binary判断）
      新逻辑（S10）：统计近5日内的一字板次数（limit_board_count_5d proxy），
                    离散映射 0→1.0, 1→0.50, 2→0.20, ≥3→0.0

    数据字段：HotStockDetail.limit_board_count_5d（price_features proxy）
    降级：若 limit_board_count_5d 为 None，尝试用 latest_is_limit_board 推断
    """
    item = ScoreItem(
        name="limit_board_count_5d",
        weight=4.0,
        note=(
            "近5日一字板次数（离散：0→1.0/1→0.50/2→0.20/≥3→0.0）；"
            "proxy: limit_board_count_5d（S10改，原为最新日一字板单日判定）"
        ),
    )

    count = detail.limit_board_count_5d

    # 降级：若5日代理值缺失，用最新日判定结果
    if count is None and detail.latest_is_limit_board is not None:
        count = 1 if detail.latest_is_limit_board else 0
        item.note += "; fallback=latest_is_limit_board→count"

    if count is None:
        item.is_data_available = False
        item.raw_value = None
        item.note += " [data_missing]"
        return item

    item.raw_value = count
    item.is_data_available = True

    # 离散映射（≥3 全归 0.0）
    clamped = min(count, 3)
    item.subscore = _RC1_LIMIT_BOARD_5D_MAP[clamped]
    item.note += f"; count={count}"
    return item


# ── 向后兼容别名 ─────────────────────────────────────────
_infer_limit_pct = infer_limit_pct
