"""risk_control_score 计算模块（Session 6; Session 8 P1 对齐提示词; Session 10 重校准）.

风险控制评分轴，高分=低风险。

──────────────────────────────────────────────────────
指标列表（5 项）：
  RC1  近5日一字板次数             weight=4  离散型（0→1.0, 1→0.50, 2→0.20, ≥3→0.0）
       Session 10 改动：从"最新1日一字板"扩展为"近5日一字板次数"，增强区分度
  RC2  近5日均振幅/涨跌停幅度     weight=3  三段上限型（G=0.5,T=1.0,B=1.8）clamp=2.0
  RC3  收盘价偏离10日均线绝对值   weight=4  三段上限型（G=2%,T=6%,B=15%）
       Session 10 改动：从 G=3%/T=10%/B=20% 收紧为 G=2%/T=6%/B=15%
  RC4  近5日长上影/炸板proxy次数  weight=2  离散型（0=1.0,1=0.70,2=0.40,≥3=0.0）
  RC5  近3日累计涨幅/涨跌停幅度   weight=2  三段上限型（G=0.8,T=1.8,B=3.5）clamp=[-4,4]
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
