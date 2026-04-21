"""Stage1 通过判定模块 — 从 pipeline.py Step 8 提取.

负责：
  - data_coverage 阈值淡出检查
  - 高位拥挤风控 total_score cap（#5）
  - 各轴评分阈值判定（AND 关系）
  - blocked_by 字段填充
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, List, Optional

from a_share_hot_screener.models import RejectedRecord

if TYPE_CHECKING:
    from a_share_hot_screener.config import HotScreenerConfig
    from a_share_hot_screener.models import HotStockDetail

logger = logging.getLogger("a_share_hot_screener.stage1_judge")

# ── 高位拥挤 total_score cap 规则（#5）────────────────────
# 防止“强但不可交易”的股票因热度高而通过 pass_stage1。
# 规则应用于 total_score 计算之后、阈值判定之前。
CAP_ONE_WORD_LIMIT_UP = 0.67      # 最新日一字涨停 → cap
CAP_RC_VERY_LOW = 0.66            # risk_control_score < 0.30 → cap
CAP_HIGH_DEVIATION_SHADOW = 0.65  # 偏离MA10>25% 且 上影线≥2 → cap


def judge_pass_stage1(
    detail: "HotStockDetail",
    config: "HotScreenerConfig",
) -> Optional[RejectedRecord]:
    """判定单只股票是否通过 Stage1.

    直接修改 detail 的 pass_stage1 / blocked_by / pass_stage1_reasons 字段。

    Args:
        detail: 已填充四轴评分的 HotStockDetail
        config: 运行配置（含各轴阈值）

    Returns:
        RejectedRecord if rejected by data_coverage, else None
    """
    cfg = config

    # 未通过硬筛的股票不进行 pass_stage1 判定
    if not detail.passed_hard_filter:
        detail.pass_stage1 = False
        return None

    # Step 8a: data_coverage 阈值淡出
    if (
        detail.data_coverage is None
        or detail.data_coverage < cfg.min_data_coverage
    ):
        detail.pass_stage1 = False
        cov_val = detail.data_coverage
        logger.debug(
            "[data_coverage] %s 淡出: coverage=%.4f < %.4f",
            detail.code, cov_val if cov_val is not None else -1, cfg.min_data_coverage,
        )
        return RejectedRecord(
            code=detail.code,
            name=detail.name,
            input_order=detail.input_order,
            reject_stage="data_coverage",
            reject_reason="coverage_below_threshold",
            reject_detail=(
                f"data_coverage={cov_val} < min_data_coverage={cfg.min_data_coverage}"
            ),
            warnings="; ".join(detail.warnings[:5]),
        )

    # Step 8b: 高位拥挤 total_score cap（#5）
    _apply_crowding_caps(detail)

    # Step 8c: pass_stage1 各轴阈值判定
    reasons_pass: List[str] = []
    reasons_fail: List[str] = []

    def _check_axis(
        score_val: Optional[float],
        threshold: float,
        label: str,
    ) -> bool:
        if score_val is None:
            reasons_fail.append(f"{label}=None < {threshold:.2f}")
            return False
        if score_val >= threshold:
            reasons_pass.append(f"{label}={score_val:.4f}>={threshold:.2f}")
            return True
        reasons_fail.append(f"{label}={score_val:.4f} < {threshold:.2f}")
        return False

    ok_ht = _check_axis(detail.hot_theme_score, cfg.min_hot_theme_score, "hot_theme_score")
    ok_tf = _check_axis(detail.trend_flow_score, cfg.min_trend_flow_score, "trend_flow_score")
    ok_le = _check_axis(detail.liquidity_execution_score, cfg.min_liquidity_execution_score, "liquidity_execution_score")
    ok_rc = _check_axis(detail.risk_control_score, cfg.min_risk_control_score, "risk_control_score")
    ok_total = _check_axis(detail.total_score, cfg.min_total_score, "total_score")
    ok_cov = detail.data_coverage >= cfg.min_data_coverage
    if ok_cov:
        reasons_pass.append(f"data_coverage={detail.data_coverage:.4f}>={cfg.min_data_coverage:.2f}")

    detail.pass_stage1 = bool(
        ok_ht and ok_tf and ok_le and ok_rc and ok_total and ok_cov
    )
    detail.pass_stage1_reasons = reasons_pass if detail.pass_stage1 else reasons_fail

    # blocked_by 记录未通过的轴
    blocked = []
    if not ok_ht:
        blocked.append("hot_theme")
    if not ok_tf:
        blocked.append("trend_flow")
    if not ok_le:
        blocked.append("liquidity_execution")
    if not ok_rc:
        blocked.append("risk_control")
    if not ok_total:
        blocked.append("total_score")
    if not ok_cov:
        blocked.append("data_coverage")
    detail.blocked_by = blocked

    if detail.pass_stage1:
        logger.debug(
            "[pass_stage1] %s 通过: total_score=%.4f coverage=%.4f",
            detail.code,
            detail.total_score if detail.total_score is not None else -1,
            detail.data_coverage,
        )

    return None


# ════════════════════════════════════════════════════════
# 高位拥挤风控 cap（#5）
# ════════════════════════════════════════════════════════

def _apply_crowding_caps(detail: "HotStockDetail") -> None:
    """对高位拥挤股票施加 total_score 上限，降低误放风险.

    规则（可叠加，取最低 cap）：
      1. 最新日一字涨停 → cap 0.67
      2. risk_control_score < 0.30 → cap 0.66
      3. 偏离 MA10 > 25% 且 近5日上影线 ≥ 2 → cap 0.65

    修改 detail.total_score 和 detail.crowding_cap_applied。
    """
    if detail.total_score is None:
        return

    cap: Optional[float] = None
    cap_reasons: List[str] = []

    # 规则 1: 最新日一字涨停（OHLC 近似相等 且 涨幅 > 0）
    if (
        detail.latest_is_limit_board is True
        and detail.latest_pct_change is not None
        and detail.latest_pct_change > 0
    ):
        cap = _tighter_cap(cap, CAP_ONE_WORD_LIMIT_UP)
        cap_reasons.append(
            f"one_word_limit_up: pct={detail.latest_pct_change:.2f}% → cap={CAP_ONE_WORD_LIMIT_UP}"
        )

    # 规则 2: risk_control_score 极低
    if detail.risk_control_score is not None and detail.risk_control_score < 0.30:
        cap = _tighter_cap(cap, CAP_RC_VERY_LOW)
        cap_reasons.append(
            f"rc_very_low: rc={detail.risk_control_score:.4f}<0.30 → cap={CAP_RC_VERY_LOW}"
        )

    # 规则 3: 高位偏离 + 上影线
    abs_dev_pct = (
        abs(detail.abs_distance_to_ma10) * 100.0
        if detail.abs_distance_to_ma10 is not None else None
    )
    shadow_count = detail.upper_shadow_count_5d
    if (
        abs_dev_pct is not None and abs_dev_pct > 25.0
        and shadow_count is not None and shadow_count >= 2
    ):
        cap = _tighter_cap(cap, CAP_HIGH_DEVIATION_SHADOW)
        cap_reasons.append(
            f"high_dev_shadow: MA10偏离={abs_dev_pct:.1f}%>25% 且 shadow={shadow_count}≥2 → cap={CAP_HIGH_DEVIATION_SHADOW}"
        )

    # 应用 cap
    if cap is not None and detail.total_score > cap:
        original = detail.total_score
        detail.total_score = round(cap, 4)
        logger.info(
            "[crowding_cap] %s total_score %.4f → %.4f | %s",
            detail.code, original, cap, "; ".join(cap_reasons),
        )

    # 记录到 detail 供输出/调试
    detail.crowding_cap_applied = cap_reasons if cap_reasons else None


def _tighter_cap(current: Optional[float], new: float) -> float:
    """取更严格（更低）的 cap."""
    if current is None:
        return new
    return min(current, new)
