"""Session 22 新增测试 — HT7 板块轮动动量信号 + 6000pt 特性启用.

覆盖范围：
  1. HT7 板块轮动动量评分（discrete: rotate_in→1.0 ... steady_weak→0.0）
  2. config 默认值变更验证（enable_concept_heat_module=True, enable_sector_rotation=True）
  3. flags.py 保留 Session 22 新增 flags
  4. sector_rotation 动量映射正确性
"""

from __future__ import annotations

import datetime as dt

import pytest


def _make_detail(**kwargs):
    from a_share_hot_screener.models import HotStockDetail
    defaults = {"code": "600519", "name": "贵州茅台", "exchange": "SH"}
    defaults.update(kwargs)
    return HotStockDetail(**defaults)


def _make_pool(**kwargs):
    from a_share_hot_screener.scoring import ScoringPool
    pool = ScoringPool.__new__(ScoringPool)
    pool.pool_return_5d = kwargs.get("pool_return_5d", [1, 2, 3, 4, 5])
    pool.pool_return_10d = kwargs.get("pool_return_10d", [1, 2, 3, 4, 5])
    pool.pool_amount_avg_5d = kwargs.get("pool_amount_avg_5d", [])
    pool.stock_count = kwargs.get("stock_count", 5)
    return pool


def _get_item(axis, name):
    return next((i for i in axis.items if i.name == name), None)


# ════════════════════════════════════════════════════════
# HT7: 板块轮动动量信号
# ════════════════════════════════════════════════════════

class TestHT7SectorMomentum:

    def test_ht7_rotate_in(self):
        """HT7: rotate_in → 1.0."""
        from a_share_hot_screener.scorers.hot_theme import compute_hot_theme_score
        detail = _make_detail(flags={"sector_momentum_signal": "rotate_in"})
        axis = compute_hot_theme_score(detail, _make_pool())
        ht7 = _get_item(axis, "sector_momentum_signal")
        assert ht7 is not None
        assert ht7.subscore == pytest.approx(1.0)
        assert ht7.weight == pytest.approx(5.0)
        assert ht7.is_applicable is True

    def test_ht7_steady_strong(self):
        """HT7: steady_strong → 0.85."""
        from a_share_hot_screener.scorers.hot_theme import compute_hot_theme_score
        detail = _make_detail(flags={"sector_momentum_signal": "steady_strong"})
        axis = compute_hot_theme_score(detail, _make_pool())
        ht7 = _get_item(axis, "sector_momentum_signal")
        assert ht7.subscore == pytest.approx(0.85)

    def test_ht7_neutral(self):
        """HT7: neutral → 0.50."""
        from a_share_hot_screener.scorers.hot_theme import compute_hot_theme_score
        detail = _make_detail(flags={"sector_momentum_signal": "neutral"})
        axis = compute_hot_theme_score(detail, _make_pool())
        ht7 = _get_item(axis, "sector_momentum_signal")
        assert ht7.subscore == pytest.approx(0.50)

    def test_ht7_rotate_out(self):
        """HT7: rotate_out → 0.15."""
        from a_share_hot_screener.scorers.hot_theme import compute_hot_theme_score
        detail = _make_detail(flags={"sector_momentum_signal": "rotate_out"})
        axis = compute_hot_theme_score(detail, _make_pool())
        ht7 = _get_item(axis, "sector_momentum_signal")
        assert ht7.subscore == pytest.approx(0.15)

    def test_ht7_steady_weak(self):
        """HT7: steady_weak → 0.0."""
        from a_share_hot_screener.scorers.hot_theme import compute_hot_theme_score
        detail = _make_detail(flags={"sector_momentum_signal": "steady_weak"})
        axis = compute_hot_theme_score(detail, _make_pool())
        ht7 = _get_item(axis, "sector_momentum_signal")
        assert ht7.subscore == pytest.approx(0.0)

    def test_ht7_none_not_available(self):
        """HT7: None → is_data_available=False."""
        from a_share_hot_screener.scorers.hot_theme import compute_hot_theme_score
        detail = _make_detail(flags={})
        axis = compute_hot_theme_score(detail, _make_pool())
        ht7 = _get_item(axis, "sector_momentum_signal")
        assert ht7 is not None
        assert ht7.is_data_available is False

    def test_ht7_unknown_defaults_to_neutral(self):
        """HT7: 未知信号 → fallback 0.50."""
        from a_share_hot_screener.scorers.hot_theme import compute_hot_theme_score
        detail = _make_detail(flags={"sector_momentum_signal": "unknown_value"})
        axis = compute_hot_theme_score(detail, _make_pool())
        ht7 = _get_item(axis, "sector_momentum_signal")
        assert ht7.subscore == pytest.approx(0.50)


# ════════════════════════════════════════════════════════
# Config 默认值变更
# ════════════════════════════════════════════════════════

class TestConfigDefaults:

    def test_concept_heat_default_true(self):
        """enable_concept_heat_module 默认 True（Session 22 改）."""
        from a_share_hot_screener.config import HotScreenerConfig
        cfg = HotScreenerConfig(
            tushare_token="test",
            run_date=dt.date(2026, 4, 21),
            stock_codes=["600519"],
            output_dir="/tmp/test",
        )
        assert cfg.enable_concept_heat_module is True

    def test_sector_rotation_default_true(self):
        """enable_sector_rotation 默认 True（Session 22 改）."""
        from a_share_hot_screener.config import HotScreenerConfig
        cfg = HotScreenerConfig(
            tushare_token="test",
            run_date=dt.date(2026, 4, 21),
            stock_codes=["600519"],
            output_dir="/tmp/test",
        )
        assert cfg.enable_sector_rotation is True


# ════════════════════════════════════════════════════════
# flags.py 保留 Session 22 flags
# ════════════════════════════════════════════════════════

class TestFlagsPreserveSession22:

    def test_moneyflow_flag_preserved(self):
        """compute_flags 保留 net_main_inflow_ratio_5d."""
        from a_share_hot_screener.flags import compute_flags
        detail = _make_detail()
        detail.flags["net_main_inflow_ratio_5d"] = 8.5
        flags = compute_flags(detail)
        assert flags["net_main_inflow_ratio_5d"] == 8.5

    def test_holdertrade_flag_preserved(self):
        """compute_flags 保留 net_holder_reduction_ratio_30d."""
        from a_share_hot_screener.flags import compute_flags
        detail = _make_detail()
        detail.flags["net_holder_reduction_ratio_30d"] = 0.3
        flags = compute_flags(detail)
        assert flags["net_holder_reduction_ratio_30d"] == 0.3

    def test_margin_flags_preserved(self):
        """compute_flags 保留 margin + is_margin_eligible."""
        from a_share_hot_screener.flags import compute_flags
        detail = _make_detail()
        detail.flags["margin_buy_net_ratio_5d"] = 2.5
        detail.flags["short_sell_ratio_change_5d"] = 15.0
        detail.flags["is_margin_eligible"] = True
        flags = compute_flags(detail)
        assert flags["margin_buy_net_ratio_5d"] == 2.5
        assert flags["short_sell_ratio_change_5d"] == 15.0
        assert flags["is_margin_eligible"] is True

    def test_sector_momentum_preserved(self):
        """compute_flags 保留 sector_momentum_signal."""
        from a_share_hot_screener.flags import compute_flags
        detail = _make_detail()
        detail.flags["sector_momentum_signal"] = "rotate_in"
        flags = compute_flags(detail)
        assert flags["sector_momentum_signal"] == "rotate_in"


# ════════════════════════════════════════════════════════
# sector_rotation 动量分类正确性
# ════════════════════════════════════════════════════════

class TestSectorMomentumClassify:

    def test_rotate_in(self):
        """前期弱(20d<50%) + 近期强(5d>70%) → rotate_in."""
        from a_share_hot_screener.sector_rotation import classify_momentum
        assert classify_momentum(0.80, 0.30) == "rotate_in"

    def test_rotate_out(self):
        """前期强(20d>70%) + 近期弱(5d<40%) → rotate_out."""
        from a_share_hot_screener.sector_rotation import classify_momentum
        assert classify_momentum(0.20, 0.80) == "rotate_out"

    def test_steady_strong(self):
        """5d>70% + 20d>60% → steady_strong."""
        from a_share_hot_screener.sector_rotation import classify_momentum
        assert classify_momentum(0.80, 0.70) == "steady_strong"

    def test_steady_weak(self):
        """5d<30% + 20d<40% → steady_weak."""
        from a_share_hot_screener.sector_rotation import classify_momentum
        assert classify_momentum(0.20, 0.30) == "steady_weak"

    def test_neutral(self):
        """中间值 → neutral."""
        from a_share_hot_screener.sector_rotation import classify_momentum
        assert classify_momentum(0.50, 0.50) == "neutral"

    def test_none_neutral(self):
        """None 值 → neutral."""
        from a_share_hot_screener.sector_rotation import classify_momentum
        assert classify_momentum(None, 0.50) == "neutral"
        assert classify_momentum(0.50, None) == "neutral"
