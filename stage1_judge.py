"""Stage1 通过判定模块 — 从 pipeline.py Step 8 提取（重构，不改业务逻辑）.

负责：
  - data_coverage 阈值淡出检查
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

    # Step 8b: pass_stage1 各轴阈值判定
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
