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
    """判定单只股票是否通过 Stage1 + 分池分类.

    P0-C 语义变更:
      pass_stage1      = tradeable only（安全默认）
      pass_stage1_watch = 观察池（分数达标但不可交易）
      pass_stage1_any   = tradeable ∪ watch_only
      candidate_pool_type = tradeable/watch_only/failed_score/insufficient_data/rejected_hard

    直接修改 detail 的上述字段。

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
        detail.candidate_pool_type = "rejected_hard"
        detail.candidate_pool_reason = detail.hard_filter_reason or "hard_filter_failed"
        return None

    # Step 8a: data_coverage 阈值淡出 (P0-D: 优先用 core_data_coverage)
    effective_cov = detail.core_data_coverage if detail.core_data_coverage is not None else detail.data_coverage
    if (
        effective_cov is None
        or effective_cov < cfg.min_data_coverage
    ):
        detail.pass_stage1 = False
        detail.candidate_pool_type = "insufficient_data"
        cov_val = effective_cov
        detail.candidate_pool_reason = f"core_data_coverage={cov_val} < {cfg.min_data_coverage}"
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

    # Step 8b: 保存 pre-cap 分数，然后应用 crowding cap
    pre_cap_total = detail.total_score
    _apply_crowding_caps(detail)

    # Step 8c: pass_stage1 各轴阈值判定
    # P0-C: 同时检查 pre-cap 和 post-cap 两套分数
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
    ok_total_post_cap = _check_axis(detail.total_score, cfg.min_total_score, "total_score")
    ok_cov = (effective_cov is not None and effective_cov >= cfg.min_data_coverage)
    if ok_cov:
        reasons_pass.append(f"core_data_coverage={effective_cov:.4f}>={cfg.min_data_coverage:.2f}")

    # pre-cap total 达标判断（用于 watch_only 判定）
    ok_total_pre_cap = (
        pre_cap_total is not None and pre_cap_total >= cfg.min_total_score
    )
    # post-cap 全部达标
    post_cap_pass = bool(ok_ht and ok_tf and ok_le and ok_rc and ok_total_post_cap and ok_cov)
    # pre-cap 四轴+total 达标（用于判断是否“本来强但被 cap”）
    pre_cap_pass = bool(ok_ht and ok_tf and ok_le and ok_rc and ok_total_pre_cap and ok_cov)

    # blocked_by 记录未通过的轴（基于 post-cap）
    blocked = []
    if not ok_ht:
        blocked.append("hot_theme")
    if not ok_tf:
        blocked.append("trend_flow")
    if not ok_le:
        blocked.append("liquidity_execution")
    if not ok_rc:
        blocked.append("risk_control")
    if not ok_total_post_cap:
        blocked.append("total_score")
    if not ok_cov:
        blocked.append("data_coverage")
    detail.blocked_by = blocked

    # ── P0-C: 分池分类 ──────────────────────────────────
    if post_cap_pass:
        # post-cap 也达标 — 检查 watch 触发器
        watch_reasons = _detect_watch_only_reasons(detail)
        if watch_reasons:
            detail.pass_stage1 = False
            detail.pass_stage1_watch = True
            detail.pass_stage1_any = True
            detail.candidate_pool_type = "watch_only"
            detail.candidate_pool_reason = "; ".join(watch_reasons)
        else:
            detail.pass_stage1 = True
            detail.pass_stage1_watch = False
            detail.pass_stage1_any = True
            detail.candidate_pool_type = "tradeable"
            detail.candidate_pool_reason = "all_criteria_met"
        detail.pass_stage1_reasons = reasons_pass
    elif pre_cap_pass and not post_cap_pass:
        # pre-cap 达标但 post-cap 不达标 — 被 crowding cap 拉下来的强股
        detail.pass_stage1 = False
        detail.pass_stage1_watch = True
        detail.pass_stage1_any = True
        detail.candidate_pool_type = "watch_only"
        cap_reason = "; ".join(detail.crowding_cap_applied) if detail.crowding_cap_applied else "crowding_cap"
        detail.candidate_pool_reason = f"crowding_cap_reduced({cap_reason})"
        detail.pass_stage1_reasons = reasons_pass
    else:
        # pre-cap 也不达标 — 真正的分数不足
        detail.pass_stage1 = False
        detail.pass_stage1_watch = False
        detail.pass_stage1_any = False
        detail.candidate_pool_type = "failed_score"
        detail.candidate_pool_reason = "; ".join(reasons_fail[:3])
        detail.pass_stage1_reasons = reasons_fail

    if detail.pass_stage1:
        logger.debug(
            "[pass_stage1] %s tradeable: total_score=%.4f coverage=%.4f",
            detail.code,
            detail.total_score if detail.total_score is not None else -1,
            detail.data_coverage,
        )
    elif detail.pass_stage1_watch:
        logger.debug(
            "[pass_stage1] %s watch_only: total_score=%.4f reason=%s",
            detail.code,
            detail.total_score if detail.total_score is not None else -1,
            detail.candidate_pool_reason,
        )

    return None


def _detect_watch_only_reasons(detail: "HotStockDetail") -> List[str]:
    """检测分数达标但不可交易的原因.

    watch_only 条件（任一触发）：
      1. 最新日一字涨停（OHLC + 涨幅）
      2. crowding_cap_applied 触发
      3. risk_control_score < 0.45（high risk flag）
    """
    reasons: List[str] = []

    # 一字涨停
    if (
        detail.latest_is_limit_board is True
        and detail.latest_pct_change is not None
        and detail.latest_pct_change > 0
    ):
        reasons.append(f"one_word_limit_up(pct={detail.latest_pct_change:.1f}%)")

    # crowding cap
    if detail.crowding_cap_applied:
        reasons.append(f"crowding_cap({detail.crowding_cap_applied[0].split(':')[0]})")

    # 高风险 RC
    if (
        detail.risk_control_score is not None
        and detail.risk_control_score < 0.45
    ):
        reasons.append(f"high_risk(rc={detail.risk_control_score:.2f})")

    return reasons


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

    # 规则 3: 高位偏离 + 冲高回落（P0-A: 使用 upper_reversal_count_5d）
    abs_dev_pct = (
        abs(detail.abs_distance_to_ma10) * 100.0
        if detail.abs_distance_to_ma10 is not None else None
    )
    reversal_count = detail.upper_reversal_count_5d
    if (
        abs_dev_pct is not None and abs_dev_pct > 25.0
        and reversal_count is not None and reversal_count >= 2
    ):
        cap = _tighter_cap(cap, CAP_HIGH_DEVIATION_SHADOW)
        cap_reasons.append(
            f"high_dev_reversal: MA10偏离={abs_dev_pct:.1f}%>25% 且 reversal={reversal_count}≥2 → cap={CAP_HIGH_DEVIATION_SHADOW}"
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
