"""Phase 0 新功能测试 — P0-A/B/C/D/E 共 25+ 测试.

覆盖:
  P0-A: indicators.py 双指标 + upper_reversal_count_5d + RC4 + crowding cap
  P0-B: LE2 turnover 优先级
  P0-C: candidate_pool_type 分池
  P0-D: core/overall coverage split
  P0-E: HT5 degraded cap
"""

import datetime as dt
from dataclasses import dataclass
from typing import List, Optional

import pytest

from a_share_hot_screener.models import HotStockDetail, HotStockSummary


def _make_detail(**kwargs) -> HotStockDetail:
    defaults = dict(code="600519", name="测试", exchange="SH", passed_hard_filter=True)
    defaults.update(kwargs)
    return HotStockDetail(**defaults)


def _make_pool():
    from a_share_hot_screener.scoring import ScoringPool
    return ScoringPool(
        pool_return_5d=[1.0, 2.0, 5.0, 10.0, 15.0],
        pool_return_10d=[2.0, 4.0, 8.0, 15.0, 25.0],
        stock_count=5,
    )


# ════════════════════════════════════════════════════════
# P0-A: indicators 模块
# ════════════════════════════════════════════════════════

class TestIndicators:
    def test_upper_wick_ratio_normal(self):
        from a_share_hot_screener.indicators import upper_wick_ratio
        # open=10, high=12, low=9, close=10 → wick = 12-10=2, range=3 → 0.667
        assert upper_wick_ratio(10, 12, 9, 10) == pytest.approx(2/3, rel=1e-3)

    def test_upper_wick_ratio_no_wick(self):
        from a_share_hot_screener.indicators import upper_wick_ratio
        # close is the high → no upper wick
        assert upper_wick_ratio(10, 12, 9, 12) == pytest.approx(0.0)

    def test_upper_wick_ratio_flat(self):
        from a_share_hot_screener.indicators import upper_wick_ratio
        # high == low → 0.0
        assert upper_wick_ratio(10, 10, 10, 10) == 0.0

    def test_upper_reversal_ratio_normal(self):
        from a_share_hot_screener.indicators import upper_reversal_ratio
        # high=12, low=9, close=10 → (12-10)/(12-9) = 0.667
        assert upper_reversal_ratio(12, 9, 10) == pytest.approx(2/3, rel=1e-3)

    def test_upper_reversal_ratio_close_at_high(self):
        from a_share_hot_screener.indicators import upper_reversal_ratio
        assert upper_reversal_ratio(12, 9, 12) == pytest.approx(0.0)

    def test_upper_reversal_ratio_close_at_low(self):
        from a_share_hot_screener.indicators import upper_reversal_ratio
        assert upper_reversal_ratio(12, 9, 9) == pytest.approx(1.0)

    def test_upper_reversal_ratio_flat(self):
        from a_share_hot_screener.indicators import upper_reversal_ratio
        assert upper_reversal_ratio(10, 10, 10) == 0.0


class TestUpperReversalCount:
    """P0-A: upper_reversal_count_5d 计算."""

    def test_count_reversal_days(self):
        from a_share_hot_screener.price_features import compute_price_features
        # 构造 6 行数据 (需要 prev_close)
        # 第 2-6 行 = 5 日窗口
        # Day 3: high=11, low=10, close=10 → reversal=1/1=1.0 ≥ 0.45, amp=(11-10)/10*100=10% ≥ 5% → count
        # Day 4: high=10.2, low=10, close=10.1 → reversal=0.1/0.2=0.5 ≥ 0.45, amp=(0.2)/10*100=2% < 5% → no count
        rows = [
            {"trade_date": "20260410", "open": 10, "high": 10.5, "low": 9.8, "close": 10, "vol": 100, "amount": 1000},
            {"trade_date": "20260411", "open": 10, "high": 10.3, "low": 9.9, "close": 10.1, "vol": 100, "amount": 1000},
            {"trade_date": "20260414", "open": 10.1, "high": 11, "low": 10, "close": 10, "vol": 100, "amount": 1000},  # reversal + amp
            {"trade_date": "20260415", "open": 10, "high": 10.2, "low": 10, "close": 10.1, "vol": 100, "amount": 1000},  # reversal but low amp
            {"trade_date": "20260416", "open": 10.1, "high": 10.8, "low": 9.5, "close": 9.6, "vol": 100, "amount": 1000},  # reversal=1.2/1.3≈0.92, amp=(1.3)/10.1*100=12.9% → count
            {"trade_date": "20260417", "open": 9.6, "high": 9.8, "low": 9.5, "close": 9.7, "vol": 100, "amount": 1000},  # reversal=0.1/0.3=0.33 < 0.45
        ]
        feat = compute_price_features(rows, "2026-04-17")
        assert feat.upper_reversal_count_5d == 2  # Day 3 and Day 5


class TestRC4UsesReversalCount:
    """P0-A: RC4 uses upper_reversal_count_5d."""

    def test_rc4_zero_reversal_full_score(self):
        from a_share_hot_screener.scorers.risk_control import compute_risk_control_score
        detail = _make_detail(upper_reversal_count_5d=0)
        axis = compute_risk_control_score(detail, _make_pool())
        rc4 = next(i for i in axis.items if i.name == "upper_reversal_count_5d")
        assert rc4.subscore == pytest.approx(1.0)

    def test_rc4_heavy_reversal_zero_score(self):
        from a_share_hot_screener.scorers.risk_control import compute_risk_control_score
        detail = _make_detail(upper_reversal_count_5d=5)
        axis = compute_risk_control_score(detail, _make_pool())
        rc4 = next(i for i in axis.items if i.name == "upper_reversal_count_5d")
        assert rc4.subscore == pytest.approx(0.0)


# ════════════════════════════════════════════════════════
# P0-B: LE2 turnover 优先级
# ════════════════════════════════════════════════════════

class TestLE2TurnoverPriority:
    """P0-B: turnover_rate_f > turnover_rate > amount_proxy."""

    def test_priority_turnover_rate_f(self):
        """turnover_rate_f 优先使用，无 cap."""
        from a_share_hot_screener.scorers.liquidity_execution import compute_liquidity_execution_score
        detail = _make_detail(
            turnover_rate_f_1d=30.0,   # >= H=25% → 满分
            turnover_rate_1d=5.0,      # 应被忽略
            amount_avg_5d=3e8,
            float_market_cap=10e8,
        )
        axis = compute_liquidity_execution_score(detail, _make_pool(), enable_lhb_module=False)
        le2 = next(i for i in axis.items if i.name == "turnover_avg_5d")
        assert le2.subscore == pytest.approx(1.0)
        assert detail.turnover_method == "turnover_rate_f"
        assert "proxy_capped" not in (le2.note or "")

    def test_priority_turnover_rate(self):
        """turnover_rate_f 不可用时用 turnover_rate，无 cap."""
        from a_share_hot_screener.scorers.liquidity_execution import compute_liquidity_execution_score
        detail = _make_detail(
            turnover_rate_f_1d=None,
            turnover_rate_1d=30.0,
            amount_avg_5d=3e8,
            float_market_cap=10e8,
        )
        axis = compute_liquidity_execution_score(detail, _make_pool(), enable_lhb_module=False)
        le2 = next(i for i in axis.items if i.name == "turnover_avg_5d")
        assert le2.subscore == pytest.approx(1.0)
        assert detail.turnover_method == "turnover_rate"

    def test_fallback_to_proxy_with_cap(self):
        """两个 turnover 都不可用时 fallback 到 proxy，有 cap 0.85."""
        from a_share_hot_screener.scorers.liquidity_execution import (
            compute_liquidity_execution_score, _TURNOVER_PROXY_SUBSCORE_CAP,
        )
        detail = _make_detail(
            turnover_rate_f_1d=None,
            turnover_rate_1d=None,
            amount_avg_5d=3e8,
            float_market_cap=10e8,  # proxy = 30%
        )
        axis = compute_liquidity_execution_score(detail, _make_pool(), enable_lhb_module=False)
        le2 = next(i for i in axis.items if i.name == "turnover_avg_5d")
        assert le2.subscore == pytest.approx(_TURNOVER_PROXY_SUBSCORE_CAP)
        assert detail.turnover_method == "amount_proxy"
        assert "proxy_capped" in (le2.note or "")

    def test_turnover_method_written_to_detail(self):
        from a_share_hot_screener.scorers.liquidity_execution import compute_liquidity_execution_score
        detail = _make_detail(turnover_rate_f_1d=10.0)
        compute_liquidity_execution_score(detail, _make_pool(), enable_lhb_module=False)
        assert detail.turnover_method == "turnover_rate_f"


# ════════════════════════════════════════════════════════
# P0-C: candidate_pool_type 分池
# ════════════════════════════════════════════════════════

class TestCandidatePoolType:
    """P0-C: watch_only / tradeable / failed_score 分类."""

    def _config(self, **overrides):
        from a_share_hot_screener.config import HotScreenerConfig
        defaults = dict(
            tushare_token="test",
            run_date=dt.date(2026, 4, 21),
            stock_codes=["600519"],
            output_dir="/tmp",
        )
        defaults.update(overrides)
        return HotScreenerConfig(**defaults)

    def test_tradeable(self):
        """分数全部达标且无 watch 标记 → tradeable."""
        from a_share_hot_screener.stage1_judge import judge_pass_stage1
        cfg = self._config()
        d = _make_detail(
            total_score=0.80, hot_theme_score=0.70, trend_flow_score=0.65,
            liquidity_execution_score=0.60, risk_control_score=0.50,
            data_coverage=0.85, core_data_coverage=0.85, overall_data_coverage=0.85,
            latest_is_limit_board=False,
        )
        judge_pass_stage1(d, cfg)
        assert d.pass_stage1 is True
        assert d.pass_stage1_watch is False
        assert d.pass_stage1_any is True
        assert d.candidate_pool_type == "tradeable"

    def test_watch_only_limit_up(self):
        """一字涨停但分数达标 → watch_only."""
        from a_share_hot_screener.stage1_judge import judge_pass_stage1
        cfg = self._config()
        d = _make_detail(
            total_score=0.80, hot_theme_score=0.70, trend_flow_score=0.65,
            liquidity_execution_score=0.60, risk_control_score=0.50,
            data_coverage=0.85, core_data_coverage=0.85, overall_data_coverage=0.85,
            latest_is_limit_board=True, latest_pct_change=9.98,
        )
        judge_pass_stage1(d, cfg)
        assert d.pass_stage1 is False  # not tradeable
        assert d.pass_stage1_watch is True
        assert d.pass_stage1_any is True
        assert d.candidate_pool_type == "watch_only"
        assert "one_word_limit_up" in d.candidate_pool_reason

    def test_watch_only_high_risk(self):
        """RC score < 0.45 → watch_only."""
        from a_share_hot_screener.stage1_judge import judge_pass_stage1
        cfg = self._config()
        d = _make_detail(
            total_score=0.80, hot_theme_score=0.70, trend_flow_score=0.65,
            liquidity_execution_score=0.60, risk_control_score=0.42,
            data_coverage=0.85, core_data_coverage=0.85, overall_data_coverage=0.85,
        )
        judge_pass_stage1(d, cfg)
        assert d.pass_stage1 is False
        assert d.pass_stage1_watch is True
        assert d.candidate_pool_type == "watch_only"
        assert "high_risk" in d.candidate_pool_reason

    def test_failed_score(self):
        """分数不够 → failed_score."""
        from a_share_hot_screener.stage1_judge import judge_pass_stage1
        cfg = self._config()
        d = _make_detail(
            total_score=0.50, hot_theme_score=0.40, trend_flow_score=0.30,
            liquidity_execution_score=0.30, risk_control_score=0.20,
            data_coverage=0.80, core_data_coverage=0.80, overall_data_coverage=0.80,
        )
        judge_pass_stage1(d, cfg)
        assert d.pass_stage1 is False
        assert d.pass_stage1_watch is False
        assert d.pass_stage1_any is False
        assert d.candidate_pool_type == "failed_score"

    def test_rejected_hard(self):
        """未通过硬筛 → rejected_hard."""
        from a_share_hot_screener.stage1_judge import judge_pass_stage1
        cfg = self._config()
        d = _make_detail(passed_hard_filter=False, hard_filter_reason="too_cheap")
        judge_pass_stage1(d, cfg)
        assert d.candidate_pool_type == "rejected_hard"

    def test_insufficient_data(self):
        """coverage 不足 → insufficient_data."""
        from a_share_hot_screener.stage1_judge import judge_pass_stage1
        cfg = self._config()
        d = _make_detail(
            total_score=0.80, data_coverage=0.50, core_data_coverage=0.50,
        )
        result = judge_pass_stage1(d, cfg)
        assert d.candidate_pool_type == "insufficient_data"
        assert result is not None  # returns RejectedRecord

    def test_pass_stage1_any_semantics(self):
        """pass_stage1_any = tradeable ∪ watch_only."""
        from a_share_hot_screener.stage1_judge import judge_pass_stage1
        cfg = self._config()
        # tradeable
        d1 = _make_detail(
            total_score=0.80, hot_theme_score=0.70, trend_flow_score=0.65,
            liquidity_execution_score=0.60, risk_control_score=0.50,
            data_coverage=0.85, core_data_coverage=0.85, overall_data_coverage=0.85,
        )
        judge_pass_stage1(d1, cfg)
        assert d1.pass_stage1_any is True
        assert d1.pass_stage1 is True

        # watch_only (limit up)
        d2 = _make_detail(
            total_score=0.80, hot_theme_score=0.70, trend_flow_score=0.65,
            liquidity_execution_score=0.60, risk_control_score=0.50,
            data_coverage=0.85, core_data_coverage=0.85, overall_data_coverage=0.85,
            latest_is_limit_board=True, latest_pct_change=9.98,
        )
        judge_pass_stage1(d2, cfg)
        assert d2.pass_stage1_any is True
        assert d2.pass_stage1 is False

    def test_summary_has_pool_fields(self):
        """Summary 自动投影包含 P0-C 新字段."""
        d = _make_detail(
            pass_stage1=True, pass_stage1_watch=False, pass_stage1_any=True,
            candidate_pool_type="tradeable", candidate_pool_reason="all_criteria_met",
        )
        s = HotStockSummary.from_detail(d)
        assert s.candidate_pool_type == "tradeable"
        assert s.pass_stage1_any is True


# ════════════════════════════════════════════════════════
# P0-D: Core/Overall Coverage Split
# ════════════════════════════════════════════════════════

class TestCoverageSplit:
    """P0-D: core_data_coverage (TF+LE) vs overall_data_coverage."""

    def test_core_coverage_computation(self):
        from a_share_hot_screener.scoring_aggregator import compute_coverage_split
        d = _make_detail(
            hot_theme_coverage=0.50,   # not core
            trend_flow_coverage=0.90,  # core
            liquidity_execution_coverage=0.80,  # core
            risk_control_coverage=0.70,  # not core
        )
        core, overall = compute_coverage_split(
            d, w_hot_theme=40, w_trend_flow=30, w_liquidity_execution=20, w_risk_control=10,
        )
        # core = (0.90*30 + 0.80*20) / (30+20) = (27+16)/50 = 0.86
        assert core == pytest.approx(0.86, abs=1e-3)
        # overall = (0.50*40 + 0.90*30 + 0.80*20 + 0.70*10) / 100 = (20+27+16+7)/100 = 0.70
        assert overall == pytest.approx(0.70, abs=1e-3)

    def test_stage1_uses_core_coverage(self):
        """pass_stage1 的 data_coverage 检查使用 core_data_coverage."""
        from a_share_hot_screener.stage1_judge import judge_pass_stage1
        from a_share_hot_screener.config import HotScreenerConfig
        cfg = HotScreenerConfig(
            tushare_token="test", run_date=dt.date(2026, 4, 21),
            stock_codes=["600519"], output_dir="/tmp",
            min_data_coverage=0.75,
        )
        # overall < 0.75 but core >= 0.75 → should pass coverage check
        d = _make_detail(
            total_score=0.80, hot_theme_score=0.70, trend_flow_score=0.65,
            liquidity_execution_score=0.60, risk_control_score=0.50,
            data_coverage=0.70,           # overall-like
            core_data_coverage=0.85,      # core is good
            overall_data_coverage=0.70,
        )
        result = judge_pass_stage1(d, cfg)
        assert result is None  # not rejected for coverage
        assert d.candidate_pool_type != "insufficient_data"


# ════════════════════════════════════════════════════════
# P0-E: HT5 Degraded Cap
# ════════════════════════════════════════════════════════

class TestHT5DegradedCap:
    """P0-E: degraded 模式子分上限 0.80."""

    def test_degraded_source_capped(self):
        from a_share_hot_screener.scorers.hot_theme import compute_hot_theme_score
        detail = _make_detail(
            industry_heat_pctile_5d=0.95,  # → pctile=95 → would be 1.0
            industry_heat_source="tushare_200_degraded",
            return_5d=10.0,
            return_10d=15.0,
        )
        axis = compute_hot_theme_score(detail, _make_pool())
        ht5 = next(i for i in axis.items if i.name == "industry_heat_pctile_5d")
        assert ht5.subscore is not None
        assert ht5.subscore <= 0.80
        assert "ht5_degraded_cap" in (ht5.note or "")

    def test_full_source_not_capped(self):
        from a_share_hot_screener.scorers.hot_theme import compute_hot_theme_score
        detail = _make_detail(
            industry_heat_pctile_5d=0.95,
            industry_heat_source="ths_daily_full",
            return_5d=10.0,
            return_10d=15.0,
        )
        axis = compute_hot_theme_score(detail, _make_pool())
        ht5 = next(i for i in axis.items if i.name == "industry_heat_pctile_5d")
        assert ht5.subscore is not None
        assert ht5.subscore > 0.80  # should be 1.0 for pctile=95
        assert "ht5_degraded_cap" not in (ht5.note or "")

    def test_degraded_low_value_not_capped(self):
        """degraded 但 subscore 本身 < 0.80 → 不触发 cap."""
        from a_share_hot_screener.scorers.hot_theme import compute_hot_theme_score
        detail = _make_detail(
            industry_heat_pctile_5d=0.60,  # → pctile=60 → ~0.245
            industry_heat_source="tushare_200_degraded",
            return_5d=10.0,
            return_10d=15.0,
        )
        axis = compute_hot_theme_score(detail, _make_pool())
        ht5 = next(i for i in axis.items if i.name == "industry_heat_pctile_5d")
        assert ht5.subscore is not None
        assert ht5.subscore < 0.80
        assert "ht5_degraded_cap" not in (ht5.note or "")
