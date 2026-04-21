"""四轴评分聚合 + total_score 计算 — 从 pipeline.py 提取（重构，不改业务逻辑）.

负责：
  - 调用四个 scorer 模块，将结果写入 HotStockDetail
  - 计算 total_score 和 data_coverage（加权平均）
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

from a_share_hot_screener.scoring import ScoringPool
from a_share_hot_screener.scorers.hot_theme import compute_hot_theme_score
from a_share_hot_screener.scorers.trend_flow import compute_trend_flow_score
from a_share_hot_screener.scorers.liquidity_execution import compute_liquidity_execution_score
from a_share_hot_screener.scorers.risk_control import compute_risk_control_score

if TYPE_CHECKING:
    from a_share_hot_screener.config import HotScreenerConfig
    from a_share_hot_screener.logger import WarningsCollector
    from a_share_hot_screener.models import HotStockDetail

logger = logging.getLogger("a_share_hot_screener.scoring_aggregator")


def apply_four_axis_scores(
    detail: "HotStockDetail",
    pool: ScoringPool,
    config: "Optional[HotScreenerConfig]",
    warnings: "WarningsCollector",
) -> None:
    """计算全部四轴评分（hot_theme / trend_flow / liquidity_execution / risk_control）并写入 detail."""
    try:
        ht = compute_hot_theme_score(detail, pool)
        detail.hot_theme_score = ht.score
        detail.hot_theme_coverage = ht.coverage
        detail.hot_theme_subscores = ht.to_dict()
    except Exception as e:
        logger.error("hot_theme_score(%s) 异常: %s", detail.code, e, exc_info=True)
        warnings.add(detail.code, f"[scoring] hot_theme_score 计算异常: {e}")

    try:
        tf = compute_trend_flow_score(detail, pool)
        detail.trend_flow_score = tf.score
        detail.trend_flow_coverage = tf.coverage
        detail.trend_flow_subscores = tf.to_dict()
    except Exception as e:
        logger.error("trend_flow_score(%s) 异常: %s", detail.code, e, exc_info=True)
        warnings.add(detail.code, f"[scoring] trend_flow_score 计算异常: {e}")

    enable_lhb = config.enable_lhb_module if config is not None else True
    try:
        le = compute_liquidity_execution_score(detail, pool, enable_lhb_module=enable_lhb)
        detail.liquidity_execution_score = le.score
        detail.liquidity_execution_coverage = le.coverage
        detail.liquidity_execution_subscores = le.to_dict()
    except Exception as e:
        logger.error("liquidity_execution_score(%s) 异常: %s", detail.code, e, exc_info=True)
        warnings.add(detail.code, f"[scoring] liquidity_execution_score 计算异常: {e}")

    try:
        rc = compute_risk_control_score(detail, pool)
        detail.risk_control_score = rc.score
        detail.risk_control_coverage = rc.coverage
        detail.risk_control_subscores = rc.to_dict()
    except Exception as e:
        logger.error("risk_control_score(%s) 异常: %s", detail.code, e, exc_info=True)
        warnings.add(detail.code, f"[scoring] risk_control_score 计算异常: {e}")

    # ── total_score + data_coverage 计算 ──────────────────
    _w_ht = getattr(config, "axis_weight_hot_theme", 35.0)
    _w_tf = getattr(config, "axis_weight_trend_flow", 30.0)
    _w_le = getattr(config, "axis_weight_liquidity_execution", 20.0)
    _w_rc = getattr(config, "axis_weight_risk_control", 15.0)

    try:
        detail.total_score, detail.data_coverage = compute_total_score(
            detail,
            w_hot_theme=_w_ht,
            w_trend_flow=_w_tf,
            w_liquidity_execution=_w_le,
            w_risk_control=_w_rc,
        )
    except Exception as e:
        logger.error("compute_total_score(%s) 异常: %s", detail.code, e, exc_info=True)
        warnings.add(detail.code, f"[scoring] total_score 计算异常: {e}")


def apply_two_axis_scores(
    detail: "HotStockDetail",
    pool: ScoringPool,
    warnings: "WarningsCollector",
) -> None:
    """计算 hot_theme_score + trend_flow_score 并写入 detail（向后兼容入口）."""
    apply_four_axis_scores(detail, pool, None, warnings)


def compute_total_score(
    detail: "HotStockDetail",
    *,
    w_hot_theme: float = 35.0,
    w_trend_flow: float = 30.0,
    w_liquidity_execution: float = 20.0,
    w_risk_control: float = 15.0,
) -> tuple:
    """计算 total_score 和 data_coverage.

    设计原则：
      - total_score 为四轴评分的加权平均（仅对有数据的轴参与）
      - data_coverage 为四轴覆盖率的加权平均（按轴权重加权）
      - 某轴 score=None 时：该轴不进入 total_score 分子和分母并汇报 data_coverage=0
      - 公式：total_score = Σ(score_i * w_i) / Σ(w_i for available axes)
      -           data_coverage = Σ(coverage_i * w_i) / Σ(w_i)

    Returns:
        (total_score, data_coverage)，均保留 4 位小数
        若四轴全部为 None，返回 (None, 0.0)
    """
    axes = [
        (detail.hot_theme_score,            detail.hot_theme_coverage or 0.0,           w_hot_theme),
        (detail.trend_flow_score,            detail.trend_flow_coverage or 0.0,          w_trend_flow),
        (detail.liquidity_execution_score,   detail.liquidity_execution_coverage or 0.0, w_liquidity_execution),
        (detail.risk_control_score,          detail.risk_control_coverage or 0.0,        w_risk_control),
    ]

    total_w = sum(w for _, _, w in axes)
    score_wsum = 0.0
    score_wdenom = 0.0
    cov_wsum = 0.0

    for score_val, cov_val, w in axes:
        if score_val is not None:
            cov_wsum += cov_val * w
        if score_val is not None:
            score_wsum += score_val * w
            score_wdenom += w

    data_coverage = round(cov_wsum / total_w, 4) if total_w > 0 else 0.0

    if score_wdenom <= 0:
        return None, data_coverage

    total_score = round(score_wsum / score_wdenom, 4)
    return total_score, data_coverage
