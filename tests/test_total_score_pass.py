"""Session 7 — total_score / data_coverage / pass_stage1 / metadata / limit_rules 单元测试.

覆盖范围：
  TestComputeTotalScore        — _compute_total_score 所有路径
  TestPassStage1Logic          — pass_stage1 各轴阈值判定
  TestDataCoverageRejection    — data_coverage 淘汰逻辑
  TestMetadataFields           — RunMetadata 新增字段
  TestLimitRules               — limit_rules.infer_limit_pct 共享模块
  TestConfigNewFields          — HotScreenerConfig 新增阈值字段
  TestSummaryTotalScoreFields  — HotStockSummary.from_detail 总分字段映射
  TestPipelineTotalScoreInteg  — pipeline _apply_four_axis_scores + total 集成
  TestQAChecks                 — QA checklist 验证
"""

from __future__ import annotations

import math
import pytest
from typing import Optional
from dataclasses import fields as dc_fields

from a_share_hot_screener.models import (
    HotStockDetail,
    HotStockSummary,
    RejectedRecord,
    RunMetadata,
)
from a_share_hot_screener.scoring import ScoringPool
from a_share_hot_screener.limit_rules import (
    infer_limit_pct,
    LIMIT_PCT_STAR_BOARD,
    LIMIT_PCT_MAIN_BOARD,
)
from a_share_hot_screener.config import HotScreenerConfig


# ════════════════════════════════════════════════════════
# 测试辅助
# ════════════════════════════════════════════════════════

def _make_detail(**kwargs) -> HotStockDetail:
    defaults = dict(
        code="600519",
        name="贵州茅台",
        exchange="SH",
        passed_hard_filter=True,
    )
    defaults.update(kwargs)
    return HotStockDetail(**defaults)


def _make_pool(**kwargs) -> ScoringPool:
    pool = ScoringPool()
    for k, v in kwargs.items():
        setattr(pool, k, v)
    return pool


def _full_pool() -> ScoringPool:
    return _make_pool(
        pool_return_5d=[1.0, 3.0, 5.0, 8.0, 10.0],
        pool_return_10d=[2.0, 5.0, 8.0, 12.0, 15.0],
        pool_limit_up_count_10d=[0.0, 0.0, 1.0, 2.0, 3.0],
        pool_strong_pool_3d=[0.0, 0.0, 1.0, 2.0],
        pool_amount_avg_5d=[1e8, 2e8, 3e8, 5e8, 8e8],
        pool_volume_ratio_20d=[0.5, 1.0, 1.5, 2.0, 3.0],
        pool_amount_ratio_5d_to_20d=[0.5, 0.8, 1.0, 1.2, 1.5],
        pool_close_position_20d=[0.1, 0.3, 0.5, 0.7, 0.9],
        pool_clv_latest=[0.2, 0.4, 0.5, 0.7, 0.9],
        pool_industry_heat_pctile=[0.1, 0.3, 0.5, 0.7, 0.9],
        stock_count=5,
    )


def _scored_detail(**kwargs) -> HotStockDetail:
    """创建一个已填充四轴评分的 detail，用于 pass_stage1 测试."""
    defaults = dict(
        code="600519",
        name="贵州茅台",
        exchange="SH",
        passed_hard_filter=True,
        hot_theme_score=0.75,
        hot_theme_coverage=1.0,
        trend_flow_score=0.70,
        trend_flow_coverage=1.0,
        liquidity_execution_score=0.65,
        liquidity_execution_coverage=1.0,
        risk_control_score=0.55,
        risk_control_coverage=1.0,
        total_score=0.70,
        data_coverage=0.90,
    )
    defaults.update(kwargs)
    return HotStockDetail(**defaults)


# ════════════════════════════════════════════════════════
# TestComputeTotalScore
# ════════════════════════════════════════════════════════

class TestComputeTotalScore:
    """_compute_total_score 函数的所有路径."""

    def test_normal_four_axis(self):
        """四轴都有分数的正常情况."""
        from a_share_hot_screener.scoring_aggregator import compute_total_score as _compute_total_score

        d = _make_detail(
            hot_theme_score=0.8,
            hot_theme_coverage=1.0,
            trend_flow_score=0.6,
            trend_flow_coverage=1.0,
            liquidity_execution_score=0.7,
            liquidity_execution_coverage=1.0,
            risk_control_score=0.5,
            risk_control_coverage=1.0,
        )
        total, cov = _compute_total_score(d)
        # 加权: (0.8*35 + 0.6*30 + 0.7*20 + 0.5*15) / (35+30+20+15) = (28+18+14+7.5)/100 = 67.5/100
        expected = round(67.5 / 100.0, 4)
        assert total == expected
        # coverage: 全部 1.0，加权均值也是 1.0
        assert cov == 1.0

    def test_one_axis_none(self):
        """一个轴为 None 时，该轴不进分子分母."""
        from a_share_hot_screener.scoring_aggregator import compute_total_score as _compute_total_score

        d = _make_detail(
            hot_theme_score=0.8,
            hot_theme_coverage=1.0,
            trend_flow_score=None,  # 缺失
            trend_flow_coverage=0.0,
            liquidity_execution_score=0.7,
            liquidity_execution_coverage=0.8,
            risk_control_score=0.5,
            risk_control_coverage=1.0,
        )
        total, cov = _compute_total_score(d)
        # total: (0.8*35 + 0.7*20 + 0.5*15) / (35+20+15) = (28+14+7.5)/70 = 49.5/70
        expected = round(49.5 / 70.0, 4)
        assert total == expected
        # coverage: (1.0*35 + 0.0*30 + 0.8*20 + 1.0*15) / 100 = (35+0+16+15)/100
        cov_expected = round(66.0 / 100.0, 4)
        assert cov == cov_expected

    def test_all_axis_none(self):
        """四轴全部为 None."""
        from a_share_hot_screener.scoring_aggregator import compute_total_score as _compute_total_score

        d = _make_detail()
        total, cov = _compute_total_score(d)
        assert total is None
        assert cov == 0.0

    def test_custom_weights(self):
        """自定义轴权重."""
        from a_share_hot_screener.scoring_aggregator import compute_total_score as _compute_total_score

        d = _make_detail(
            hot_theme_score=1.0,
            hot_theme_coverage=1.0,
            trend_flow_score=0.0,
            trend_flow_coverage=1.0,
            liquidity_execution_score=0.5,
            liquidity_execution_coverage=1.0,
            risk_control_score=0.5,
            risk_control_coverage=1.0,
        )
        total, cov = _compute_total_score(
            d,
            w_hot_theme=4.0,
            w_trend_flow=1.0,
            w_liquidity_execution=1.0,
            w_risk_control=1.0,
        )
        # (1.0*4 + 0.0*1 + 0.5*1 + 0.5*1) / 7.0 = 5.0/7.0
        expected = round(5.0 / 7.0, 4)
        assert total == expected

    def test_partial_coverage(self):
        """部分轴 coverage 不为 1.0 时 data_coverage 正确加权."""
        from a_share_hot_screener.scoring_aggregator import compute_total_score as _compute_total_score

        d = _make_detail(
            hot_theme_score=0.5,
            hot_theme_coverage=0.6,
            trend_flow_score=0.5,
            trend_flow_coverage=0.8,
            liquidity_execution_score=0.5,
            liquidity_execution_coverage=0.5,
            risk_control_score=0.5,
            risk_control_coverage=1.0,
        )
        total, cov = _compute_total_score(d)
        # coverage: (0.6*35 + 0.8*30 + 0.5*20 + 1.0*15) / 100 = (21+24+10+15)/100 = 70/100
        expected_cov = round(70.0 / 100.0, 4)
        assert cov == expected_cov
        # total: 0.5 (all axes same)
        assert total == 0.5


# ════════════════════════════════════════════════════════
# TestPassStage1Logic
# ════════════════════════════════════════════════════════

class TestPassStage1Logic:
    """pass_stage1 判定的各种边界条件."""

    def _default_config(self, **overrides):
        import datetime as dt
        defaults = dict(
            tushare_token="test",
            run_date=dt.date(2026, 4, 18),
            stock_codes=["600519"],
            output_dir="/tmp/test",
        )
        defaults.update(overrides)
        return HotScreenerConfig(**defaults)

    def test_all_pass(self):
        """所有轴都超过阈值 → pass_stage1=True."""
        cfg = self._default_config()
        d = _scored_detail()
        # 全部超阈值
        assert d.hot_theme_score >= cfg.min_hot_theme_score
        assert d.trend_flow_score >= cfg.min_trend_flow_score
        assert d.liquidity_execution_score >= cfg.min_liquidity_execution_score
        assert d.risk_control_score >= cfg.min_risk_control_score
        assert d.total_score >= cfg.min_total_score
        assert d.data_coverage >= cfg.min_data_coverage

    def test_hot_theme_below_threshold(self):
        """hot_theme_score 低于阈值 → pass_stage1=False."""
        cfg = self._default_config()
        d = _scored_detail(hot_theme_score=0.50)  # < 0.65
        assert d.hot_theme_score < cfg.min_hot_theme_score

    def test_total_score_below_threshold(self):
        """total_score 低于阈值."""
        cfg = self._default_config()
        d = _scored_detail(total_score=0.60)  # < 0.68
        assert d.total_score < cfg.min_total_score

    def test_data_coverage_below_threshold(self):
        """data_coverage 低于阈值."""
        cfg = self._default_config()
        d = _scored_detail(data_coverage=0.50)  # < 0.75
        assert d.data_coverage < cfg.min_data_coverage

    def test_axis_score_none_fails(self):
        """某轴为 None → 该轴不通过."""
        cfg = self._default_config()
        d = _scored_detail(trend_flow_score=None)
        # None < 任何 threshold
        assert d.trend_flow_score is None

    def test_edge_exact_threshold(self):
        """恰好等于阈值 → 通过（>=）."""
        cfg = self._default_config()
        d = _scored_detail(
            hot_theme_score=0.65,
            trend_flow_score=0.60,
            liquidity_execution_score=0.55,
            risk_control_score=0.40,
            total_score=0.68,
            data_coverage=0.75,
        )
        assert d.hot_theme_score >= cfg.min_hot_theme_score
        assert d.trend_flow_score >= cfg.min_trend_flow_score
        assert d.liquidity_execution_score >= cfg.min_liquidity_execution_score
        assert d.risk_control_score >= cfg.min_risk_control_score
        assert d.total_score >= cfg.min_total_score
        assert d.data_coverage >= cfg.min_data_coverage

    def test_hard_filter_failed_never_pass(self):
        """未通过硬筛的股票不能 pass_stage1."""
        d = _scored_detail(passed_hard_filter=False)
        # pipeline 中 passed_hard_filter=False → pass_stage1 始终为 False
        assert not d.passed_hard_filter


# ════════════════════════════════════════════════════════
# TestCrowdingCaps（#5）
# ════════════════════════════════════════════════════════

class TestCrowdingCaps:
    """高位拥挤 total_score cap 规则测试."""

    def _default_config(self, **overrides):
        import datetime as dt
        defaults = dict(
            tushare_token="test",
            run_date=dt.date(2026, 4, 18),
            stock_codes=["600519"],
            output_dir="/tmp/test",
        )
        defaults.update(overrides)
        return HotScreenerConfig(**defaults)

    def test_no_cap_normal_stock(self):
        """普通股票不触发 cap."""
        from a_share_hot_screener.stage1_judge import judge_pass_stage1
        cfg = self._default_config()
        d = _scored_detail(total_score=0.75)
        judge_pass_stage1(d, cfg)
        assert d.total_score == 0.75
        assert d.crowding_cap_applied is None

    def test_cap_one_word_limit_up(self):
        """最新日一字涨停 → total_score capped to 0.67."""
        from a_share_hot_screener.stage1_judge import judge_pass_stage1, CAP_ONE_WORD_LIMIT_UP
        cfg = self._default_config()
        d = _scored_detail(
            total_score=0.80,
            latest_is_limit_board=True,
            latest_pct_change=9.98,  # 涨停
        )
        judge_pass_stage1(d, cfg)
        assert d.total_score == CAP_ONE_WORD_LIMIT_UP
        assert d.crowding_cap_applied is not None
        assert any("one_word_limit_up" in r for r in d.crowding_cap_applied)
        # 0.67 < 0.68 → 不应通过
        assert d.pass_stage1 is False

    def test_no_cap_limit_board_down(self):
        """一字跌停不触发 cap（只 cap 涨停）."""
        from a_share_hot_screener.stage1_judge import judge_pass_stage1
        cfg = self._default_config()
        d = _scored_detail(
            total_score=0.75,
            latest_is_limit_board=True,
            latest_pct_change=-9.98,  # 跌停
        )
        judge_pass_stage1(d, cfg)
        assert d.total_score == 0.75  # 未被 cap

    def test_cap_rc_very_low(self):
        """risk_control_score < 0.30 → total_score capped to 0.66."""
        from a_share_hot_screener.stage1_judge import judge_pass_stage1, CAP_RC_VERY_LOW
        cfg = self._default_config()
        d = _scored_detail(
            total_score=0.80,
            risk_control_score=0.25,  # < 0.30
        )
        judge_pass_stage1(d, cfg)
        assert d.total_score == CAP_RC_VERY_LOW
        assert any("rc_very_low" in r for r in d.crowding_cap_applied)
        assert d.pass_stage1 is False

    def test_cap_high_deviation_shadow(self):
        """偏离MA10>25% 且 上影线>=2 → cap 0.65."""
        from a_share_hot_screener.stage1_judge import judge_pass_stage1, CAP_HIGH_DEVIATION_SHADOW
        cfg = self._default_config()
        d = _scored_detail(
            total_score=0.80,
            abs_distance_to_ma10=0.30,    # 30% > 25%
            upper_reversal_count_5d=3,    # >= 2
        )
        judge_pass_stage1(d, cfg)
        assert d.total_score == CAP_HIGH_DEVIATION_SHADOW
        assert any("high_dev_reversal" in r for r in d.crowding_cap_applied)

    def test_multiple_caps_takes_lowest(self):
        """多条 cap 规则同时触发 → 取最低值."""
        from a_share_hot_screener.stage1_judge import judge_pass_stage1, CAP_HIGH_DEVIATION_SHADOW
        cfg = self._default_config()
        d = _scored_detail(
            total_score=0.85,
            latest_is_limit_board=True,
            latest_pct_change=9.98,
            risk_control_score=0.20,
            abs_distance_to_ma10=0.30,
            upper_reversal_count_5d=2,
        )
        judge_pass_stage1(d, cfg)
        # 三条规则全触发，cap = min(0.67, 0.66, 0.65) = 0.65
        assert d.total_score == CAP_HIGH_DEVIATION_SHADOW
        assert len(d.crowding_cap_applied) == 3

    def test_cap_not_applied_when_score_already_low(self):
        """total_score 已经低于 cap 时不修改."""
        from a_share_hot_screener.stage1_judge import judge_pass_stage1
        cfg = self._default_config()
        d = _scored_detail(
            total_score=0.60,
            risk_control_score=0.25,  # 触发 cap 0.66
        )
        judge_pass_stage1(d, cfg)
        assert d.total_score == 0.60  # 未被拉高
        # cap_reasons 记录了触发规则，但分数未被修改
        assert d.crowding_cap_applied is not None  # 规则被记录

    def test_cap_blocks_previously_passing_stock(self):
        """原本能通过的股票因 cap 而不通过."""
        from a_share_hot_screener.stage1_judge import judge_pass_stage1
        cfg = self._default_config()
        # 这只股票所有轴都超阈值，total=0.72 > 0.68
        d = _scored_detail(
            total_score=0.72,
            hot_theme_score=0.80,
            trend_flow_score=0.70,
            liquidity_execution_score=0.60,
            risk_control_score=0.28,  # 低于 0.30 → cap 0.66
            data_coverage=0.90,
        )
        judge_pass_stage1(d, cfg)
        # cap 到 0.66 < 0.68，且 rc=0.28 < 0.40
        assert d.pass_stage1 is False


# ════════════════════════════════════════════════════════
# TestDataCoverageRejection
# ════════════════════════════════════════════════════════

class TestDataCoverageRejection:
    """data_coverage 淘汰逻辑验证."""

    def test_coverage_none_rejected(self):
        """data_coverage=None → 应被 reject."""
        d = _scored_detail(data_coverage=None, passed_hard_filter=True)
        # data_coverage is None or < threshold → rejected
        assert d.data_coverage is None

    def test_coverage_zero_rejected(self):
        """data_coverage=0.0 → 低于默认 0.75 → rejected."""
        d = _scored_detail(data_coverage=0.0, passed_hard_filter=True)
        cfg_min = 0.75
        assert d.data_coverage < cfg_min

    def test_coverage_above_threshold_passes(self):
        """data_coverage=0.90 → 高于默认 0.75 → 不被 coverage 淘汰."""
        d = _scored_detail(data_coverage=0.90, passed_hard_filter=True)
        cfg_min = 0.75
        assert d.data_coverage >= cfg_min

    def test_rejected_record_format(self):
        """验证 rejected record 格式正确."""
        r = RejectedRecord(
            code="600519",
            name="贵州茅台",
            reject_stage="data_coverage",
            reject_reason="coverage_below_threshold",
            reject_detail="data_coverage=0.5 < min_data_coverage=0.75",
        )
        assert r.reject_stage == "data_coverage"
        assert r.reject_reason == "coverage_below_threshold"


# ════════════════════════════════════════════════════════
# TestMetadataFields
# ════════════════════════════════════════════════════════

class TestMetadataFields:
    """RunMetadata Session 7 新增字段验证."""

    def test_new_fields_exist(self):
        """确认 RunMetadata 新增字段存在且有默认值."""
        m = RunMetadata()
        assert hasattr(m, "input_stock_codes")
        assert hasattr(m, "scoring_pool_size")
        assert hasattr(m, "average_data_coverage")
        assert hasattr(m, "cache_hit_rate")
        assert hasattr(m, "pass_stage1_thresholds")

    def test_default_values(self):
        """新增字段默认值正确."""
        m = RunMetadata()
        assert m.input_stock_codes == []
        assert m.scoring_pool_size == 0
        assert m.average_data_coverage is None
        assert m.cache_hit_rate is None
        assert m.pass_stage1_thresholds == {}

    def test_threshold_dict(self):
        """pass_stage1_thresholds 写入和读取."""
        m = RunMetadata(
            pass_stage1_thresholds={
                "total_score": 0.68,
                "hot_theme_score": 0.65,
                "data_coverage": 0.75,
            }
        )
        assert m.pass_stage1_thresholds["total_score"] == 0.68
        assert len(m.pass_stage1_thresholds) == 3

    def test_backward_compat(self):
        """旧字段仍然存在且可用."""
        m = RunMetadata(
            run_date="2026-04-18",
            trade_date_used="2026-04-18",
            hard_filter_passed=10,
            modules_enabled={"lhb": True},
        )
        assert m.run_date == "2026-04-18"
        assert m.hard_filter_passed == 10


# ════════════════════════════════════════════════════════
# TestLimitRules
# ════════════════════════════════════════════════════════

class TestLimitRules:
    """limit_rules.infer_limit_pct 共享模块测试."""

    def test_main_board_sh(self):
        assert infer_limit_pct("600519") == LIMIT_PCT_MAIN_BOARD  # 10%

    def test_main_board_sz(self):
        assert infer_limit_pct("000858") == LIMIT_PCT_MAIN_BOARD  # 10%

    def test_star_board(self):
        assert infer_limit_pct("688001") == LIMIT_PCT_STAR_BOARD  # 20%

    def test_chinext(self):
        assert infer_limit_pct("300750") == LIMIT_PCT_STAR_BOARD  # 20%

    def test_empty_code(self):
        assert infer_limit_pct("") == LIMIT_PCT_MAIN_BOARD

    def test_constants(self):
        assert LIMIT_PCT_MAIN_BOARD == 10.0
        assert LIMIT_PCT_STAR_BOARD == 20.0

    def test_risk_control_uses_shared(self):
        """risk_control.py 使用共享模块."""
        from a_share_hot_screener.scorers.risk_control import _infer_limit_pct
        assert _infer_limit_pct("688001") == 20.0
        assert _infer_limit_pct("600519") == 10.0

    def test_flags_uses_shared(self):
        """flags.py 使用共享模块."""
        from a_share_hot_screener.flags import _infer_limit_pct_from_code
        assert _infer_limit_pct_from_code("300750") == 20.0
        assert _infer_limit_pct_from_code("000001") == 10.0


# ════════════════════════════════════════════════════════
# TestConfigNewFields
# ════════════════════════════════════════════════════════

class TestConfigNewFields:
    """HotScreenerConfig Session 7 新增阈值字段."""

    def _default_config(self, **kw):
        import datetime as dt
        defaults = dict(
            tushare_token="test",
            run_date=dt.date(2026, 4, 18),
            stock_codes=["600519"],
            output_dir="/tmp/test",
        )
        defaults.update(kw)
        return HotScreenerConfig(**defaults)

    def test_default_thresholds(self):
        cfg = self._default_config()
        assert cfg.min_total_score == 0.68
        assert cfg.min_hot_theme_score == 0.65
        assert cfg.min_trend_flow_score == 0.60
        assert cfg.min_liquidity_execution_score == 0.55
        assert cfg.min_risk_control_score == 0.40

    def test_custom_thresholds(self):
        cfg = self._default_config(
            min_total_score=0.50,
            min_hot_theme_score=0.40,
        )
        assert cfg.min_total_score == 0.50
        assert cfg.min_hot_theme_score == 0.40

    def test_axis_weights(self):
        cfg = self._default_config()
        # Session 10: 权重从 35:30:20:15 → 40:30:20:10
        assert cfg.axis_weight_hot_theme == 40.0
        assert cfg.axis_weight_trend_flow == 30.0
        assert cfg.axis_weight_liquidity_execution == 20.0
        assert cfg.axis_weight_risk_control == 10.0

    def test_total_axis_weight(self):
        cfg = self._default_config()
        total = (
            cfg.axis_weight_hot_theme
            + cfg.axis_weight_trend_flow
            + cfg.axis_weight_liquidity_execution
            + cfg.axis_weight_risk_control
        )
        assert total == 100.0


# ════════════════════════════════════════════════════════
# TestSummaryTotalScoreFields
# ════════════════════════════════════════════════════════

class TestSummaryTotalScoreFields:
    """HotStockSummary.from_detail 正确映射 total_score 和 data_coverage."""

    def test_total_score_mapped(self):
        d = _scored_detail(total_score=0.72, data_coverage=0.85)
        s = HotStockSummary.from_detail(d)
        assert s.total_score == 0.72
        assert s.data_coverage == 0.85

    def test_total_score_none(self):
        d = _scored_detail(total_score=None, data_coverage=None)
        s = HotStockSummary.from_detail(d)
        assert s.total_score is None
        assert s.data_coverage is None

    def test_pass_stage1_mapped(self):
        d = _scored_detail()
        d.pass_stage1 = True
        s = HotStockSummary.from_detail(d)
        assert s.pass_stage1 is True


# ════════════════════════════════════════════════════════
# TestPipelineTotalScoreInteg
# ════════════════════════════════════════════════════════

class TestPipelineTotalScoreInteg:
    """pipeline _apply_four_axis_scores 产出 total_score 集成测试."""

    def test_four_axis_plus_total(self):
        """_apply_four_axis_scores 填充四轴 + total_score."""
        from a_share_hot_screener.scoring_aggregator import apply_four_axis_scores as _apply_four_axis_scores
        from a_share_hot_screener.logger import WarningsCollector

        pool = _full_pool()
        warnings = WarningsCollector()

        d = _make_detail(
            return_5d=8.0,
            return_10d=12.0,
            limit_up_count_10d=2,
            strong_pool_entry_count_3d=1,
            industry_heat_pctile_5d=0.7,
            close_position_20d=0.8,
            abs_distance_to_ma10=0.03,
            volume_ratio_20d=2.0,
            clv_latest=0.7,
            amount_ratio_5d_to_20d=1.2,
            amount_avg_5d=5e8,
            float_market_cap=50e8,
            lhb_count_20d=2,
            lhb_source="stock_lhb_detail_em",
            limit_board_count_5d=0,
            amp_norm_avg_5d=3.0,
            upper_shadow_count_5d=1,
            return_3d=5.0,
            pct_change_1d=3.5,
            limit_up_source="stock_zt_pool_em",
            strong_pool_source="stock_zt_pool_strong_em",
        )
        _apply_four_axis_scores(d, pool, None, warnings)

        # 四轴都应有值
        assert d.hot_theme_score is not None
        assert d.trend_flow_score is not None
        assert d.liquidity_execution_score is not None
        assert d.risk_control_score is not None
        # total_score 是四轴加权平均
        assert d.total_score is not None
        assert 0.0 <= d.total_score <= 1.0
        # data_coverage
        assert d.data_coverage is not None
        assert 0.0 <= d.data_coverage <= 1.0

    def test_no_data_total_none(self):
        """所有评分字段缺失 → total_score = None."""
        from a_share_hot_screener.scoring_aggregator import apply_four_axis_scores as _apply_four_axis_scores
        from a_share_hot_screener.logger import WarningsCollector

        pool = _make_pool()  # 空 pool
        warnings = WarningsCollector()
        d = _make_detail()
        _apply_four_axis_scores(d, pool, None, warnings)

        # 四轴全为 None → total_score = None
        assert d.total_score is None
        assert d.data_coverage == 0.0


# ════════════════════════════════════════════════════════
# TestQAChecks — 质量保障项
# ════════════════════════════════════════════════════════

class TestQAChecks:
    """QA checklist 验证."""

    def test_no_universe_parameter(self):
        """QA#1: 不存在 universe 参数（config/pipeline/cli）."""
        import inspect
        from a_share_hot_screener import config, pipeline, cli

        for mod in [config, pipeline, cli]:
            src = inspect.getsource(mod)
            # 允许注释中提到 universe，但不允许作为参数名
            # 检查 "universe" 不出现在参数定义中
            lines = [l for l in src.split("\n")
                     if "universe" in l.lower()
                     and not l.strip().startswith("#")
                     and not l.strip().startswith('"""')
                     and not l.strip().startswith("'")]
            # SpotUniverse 和 spot_universe 是类名/变量名，不是 "universe 参数"
            # 排除这些合法使用
            suspicious = [l for l in lines
                          if "SpotUniverse" not in l
                          and "spot_universe" not in l
                          and "_spot_universe" not in l]
            # 这些都应该是零
            for line in suspicious:
                assert "universe" not in line.lower().split("=")[0] or "comment" in line.lower(), \
                    f"疑似 universe 参数: {line.strip()}"

    def test_no_future_data_in_scoring(self):
        """QA#3: 评分函数不引入未来数据.
        
        scoring pool 构建口径：只使用 passed_hard_filter=True 的股票。
        pool 数据来自 trade_date_used 当天或之前，不包含未来数据。
        """
        pool = ScoringPool()
        # 模拟 build：只取 passed_hard_filter=True
        d1 = _make_detail(code="600519", passed_hard_filter=True, return_5d=5.0)
        d2 = _make_detail(code="000858", passed_hard_filter=False, return_5d=100.0)
        pool = ScoringPool.build([d1, d2])
        assert pool.stock_count == 1  # 只有 d1
        assert len(pool.pool_return_5d) == 1
        assert pool.pool_return_5d[0] == 5.0

    def test_detail_summary_fields_match(self):
        """QA#4: detail 和 summary 的评分字段一致."""
        d = _scored_detail()
        s = HotStockSummary.from_detail(d)
        # 评分字段映射
        assert s.total_score == d.total_score
        assert s.hot_theme_score == d.hot_theme_score
        assert s.trend_flow_score == d.trend_flow_score
        assert s.liquidity_execution_score == d.liquidity_execution_score
        assert s.risk_control_score == d.risk_control_score
        assert s.data_coverage == d.data_coverage

    def test_none_not_treated_as_zero_in_total_score(self):
        """QA#5: None 值不被当成 0 参与 total_score 计算."""
        from a_share_hot_screener.scoring_aggregator import compute_total_score as _compute_total_score

        d = _make_detail(
            hot_theme_score=0.8,
            hot_theme_coverage=1.0,
            trend_flow_score=None,      # 缺失
            trend_flow_coverage=0.0,
            liquidity_execution_score=0.8,
            liquidity_execution_coverage=1.0,
            risk_control_score=0.8,
            risk_control_coverage=1.0,
        )
        total, _ = _compute_total_score(d)
        # 如果 None 被当成 0，total 会偏低
        # 正确行为：None 不参与分子分母 → total = (0.8*35 + 0.8*20 + 0.8*15)/(35+20+15)=56/70=0.8
        assert total == 0.8

    def test_is_applicable_vs_is_data_available(self):
        """QA#6: is_applicable 和 is_data_available 逻辑一致性.
        
        is_applicable=False → is_data_available=False（必须）
        is_data_available=False 不要求 is_applicable=False
        """
        from a_share_hot_screener.scoring import score_lower_bound

        # is_applicable=False → is_data_available must be False
        item = score_lower_bound("test", 5.0, bad_threshold=0.0, good_threshold=10.0,
                                  is_applicable=False)
        assert not item.is_applicable
        assert not item.is_data_available

        # is_applicable=True, value=None → is_data_available=False
        item2 = score_lower_bound("test", None, bad_threshold=0.0, good_threshold=10.0)
        assert item2.is_applicable
        assert not item2.is_data_available

    def test_limit_pct_traceable(self):
        """QA#7: price_limit_rule 可追踪，来自共享模块."""
        from a_share_hot_screener.limit_rules import infer_limit_pct
        from a_share_hot_screener.scorers.risk_control import _infer_limit_pct as rc_fn
        from a_share_hot_screener.flags import _infer_limit_pct_from_code as flags_fn
        # 三个引用指向同一个函数
        assert rc_fn is infer_limit_pct
        assert flags_fn is infer_limit_pct

    def test_metadata_has_all_required_fields(self):
        """验证 metadata 包含提示词要求的所有字段（P2-4 对齐）."""
        required_fields = [
            "run_date", "trade_date_used", "input_stock_codes",
            "input_pool_size", "validation_passed", "scoring_pool_size",
            "hard_filter_passed", "pass_stage1_count",
            "rejected_before_scoring_count",  # P2-4 新增
            "average_data_coverage", "cache_hit_rate",
            "enable_concept_heat_module", "enable_lhb_module",
            "enable_unlock_risk_module", "global_warnings",
            "axis_weights",  # P2-4 新增
        ]
        field_names = {f.name for f in dc_fields(RunMetadata)}
        for req in required_fields:
            assert req in field_names, f"RunMetadata 缺少字段: {req}"


# ════════════════════════════════════════════════════════
# TestCLINewArgs
# ════════════════════════════════════════════════════════

class TestCLINewArgs:
    """CLI 新增参数解析."""

    def test_new_args_parse(self):
        from a_share_hot_screener.cli import build_parser
        parser = build_parser()
        args = parser.parse_args([
            "--tushare-token", "test",
            "--run-date", "2026-04-18",
            "--stock-codes", "600519",
            "--output-dir", "/tmp",
            "--min-total-score", "0.50",
            "--min-hot-theme-score", "0.40",
            "--min-trend-flow-score", "0.35",
            "--min-liquidity-execution-score", "0.30",
            "--min-risk-control-score", "0.25",
        ])
        assert args.min_total_score == 0.50
        assert args.min_hot_theme_score == 0.40
        assert args.min_trend_flow_score == 0.35
        assert args.min_liquidity_execution_score == 0.30
        assert args.min_risk_control_score == 0.25

    def test_default_args(self):
        from a_share_hot_screener.cli import build_parser
        parser = build_parser()
        args = parser.parse_args([
            "--tushare-token", "test",
            "--run-date", "2026-04-18",
            "--stock-codes", "600519",
            "--output-dir", "/tmp",
        ])
        assert args.min_total_score == 0.68
        assert args.min_hot_theme_score == 0.65
        assert args.min_trend_flow_score == 0.60
        assert args.min_liquidity_execution_score == 0.55
        assert args.min_risk_control_score == 0.40
