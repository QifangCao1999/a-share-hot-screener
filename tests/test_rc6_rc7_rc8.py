"""Session 22 新增测试 — RC6/RC7/RC8 接入评分.

覆盖范围：
  RC6  质押比例 pledge_ratio_pct        — G=5%/T=20%/B=40%, weight=2
  RC7  限售解禁 unlock_ratio_20d        — G=1%/T=5%/B=15%, weight=2
  RC8  股东人数增幅 shareholder_increase_3m — G=0%/T=10%/B=25%, weight=1
  + RC 轴总权重从 15 → 20 的验证
"""

from __future__ import annotations

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
    return next(i for i in axis.items if i.name == name)


# ════════════════════════════════════════════════════════
# RC6: 质押比例
# ════════════════════════════════════════════════════════

class TestRC6PledgeRatio:

    def test_rc6_zero_pledge_full_score(self):
        """RC6: pledge=0% ≤ G=5% → 1.0."""
        from a_share_hot_screener.scorers.risk_control import compute_risk_control_score
        detail = _make_detail(flags={"pledge_ratio_latest": 0.0})
        axis = compute_risk_control_score(detail, _make_pool())
        rc6 = _get_item(axis, "pledge_ratio_pct")
        assert rc6.subscore == pytest.approx(1.0)
        assert rc6.weight == pytest.approx(2.0)

    def test_rc6_at_good_threshold(self):
        """RC6: pledge=5% = G → 1.0."""
        from a_share_hot_screener.scorers.risk_control import compute_risk_control_score
        detail = _make_detail(flags={"pledge_ratio_latest": 5.0})
        axis = compute_risk_control_score(detail, _make_pool())
        rc6 = _get_item(axis, "pledge_ratio_pct")
        assert rc6.subscore == pytest.approx(1.0)

    def test_rc6_between_G_and_T(self):
        """RC6: G=5% < pledge=12.5% < T=20% → 线性 1.0→0.70 中点 ≈ 0.85."""
        from a_share_hot_screener.scorers.risk_control import compute_risk_control_score
        detail = _make_detail(flags={"pledge_ratio_latest": 12.5})
        axis = compute_risk_control_score(detail, _make_pool())
        rc6 = _get_item(axis, "pledge_ratio_pct")
        assert rc6.subscore == pytest.approx(0.85, abs=1e-3)

    def test_rc6_at_mid_threshold(self):
        """RC6: pledge=20% = T → 0.70."""
        from a_share_hot_screener.scorers.risk_control import compute_risk_control_score
        detail = _make_detail(flags={"pledge_ratio_latest": 20.0})
        axis = compute_risk_control_score(detail, _make_pool())
        rc6 = _get_item(axis, "pledge_ratio_pct")
        assert rc6.subscore == pytest.approx(0.70, abs=1e-3)

    def test_rc6_between_T_and_B(self):
        """RC6: T=20% < pledge=30% < B=40% → 线性 0.70→0.0 中点 ≈ 0.35."""
        from a_share_hot_screener.scorers.risk_control import compute_risk_control_score
        detail = _make_detail(flags={"pledge_ratio_latest": 30.0})
        axis = compute_risk_control_score(detail, _make_pool())
        rc6 = _get_item(axis, "pledge_ratio_pct")
        assert rc6.subscore == pytest.approx(0.35, abs=1e-3)

    def test_rc6_at_bad_threshold(self):
        """RC6: pledge=40% ≥ B → 0.0."""
        from a_share_hot_screener.scorers.risk_control import compute_risk_control_score
        detail = _make_detail(flags={"pledge_ratio_latest": 40.0})
        axis = compute_risk_control_score(detail, _make_pool())
        rc6 = _get_item(axis, "pledge_ratio_pct")
        assert rc6.subscore == pytest.approx(0.0)

    def test_rc6_above_bad_threshold(self):
        """RC6: pledge=60% > B=40% → 0.0."""
        from a_share_hot_screener.scorers.risk_control import compute_risk_control_score
        detail = _make_detail(flags={"pledge_ratio_latest": 60.0})
        axis = compute_risk_control_score(detail, _make_pool())
        rc6 = _get_item(axis, "pledge_ratio_pct")
        assert rc6.subscore == pytest.approx(0.0)

    def test_rc6_none_data_unavailable(self):
        """RC6: pledge=None → is_data_available=False."""
        from a_share_hot_screener.scorers.risk_control import compute_risk_control_score
        detail = _make_detail(flags={})
        axis = compute_risk_control_score(detail, _make_pool())
        rc6 = _get_item(axis, "pledge_ratio_pct")
        assert rc6.is_data_available is False
        assert rc6.subscore is None


# ════════════════════════════════════════════════════════
# RC7: 限售解禁
# ════════════════════════════════════════════════════════

class TestRC7UnlockRatio:

    def test_rc7_zero_unlock_full_score(self):
        """RC7: unlock=0% ≤ G=1% → 1.0."""
        from a_share_hot_screener.scorers.risk_control import compute_risk_control_score
        detail = _make_detail(flags={"restricted_shares_unlock_ratio_20d": 0.0})
        axis = compute_risk_control_score(detail, _make_pool())
        rc7 = _get_item(axis, "unlock_ratio_20d")
        assert rc7.subscore == pytest.approx(1.0)
        assert rc7.weight == pytest.approx(2.0)

    def test_rc7_at_good_threshold(self):
        """RC7: unlock=1% = G → 1.0."""
        from a_share_hot_screener.scorers.risk_control import compute_risk_control_score
        detail = _make_detail(flags={"restricted_shares_unlock_ratio_20d": 1.0})
        axis = compute_risk_control_score(detail, _make_pool())
        rc7 = _get_item(axis, "unlock_ratio_20d")
        assert rc7.subscore == pytest.approx(1.0)

    def test_rc7_between_G_and_T(self):
        """RC7: G=1% < unlock=3% < T=5% → 线性 1.0→0.70 → ≈ 0.85."""
        from a_share_hot_screener.scorers.risk_control import compute_risk_control_score
        detail = _make_detail(flags={"restricted_shares_unlock_ratio_20d": 3.0})
        axis = compute_risk_control_score(detail, _make_pool())
        rc7 = _get_item(axis, "unlock_ratio_20d")
        assert rc7.subscore == pytest.approx(0.85, abs=1e-3)

    def test_rc7_at_mid_threshold(self):
        """RC7: unlock=5% = T → 0.70."""
        from a_share_hot_screener.scorers.risk_control import compute_risk_control_score
        detail = _make_detail(flags={"restricted_shares_unlock_ratio_20d": 5.0})
        axis = compute_risk_control_score(detail, _make_pool())
        rc7 = _get_item(axis, "unlock_ratio_20d")
        assert rc7.subscore == pytest.approx(0.70, abs=1e-3)

    def test_rc7_between_T_and_B(self):
        """RC7: T=5% < unlock=10% < B=15% → 线性 0.70→0.0 → ≈ 0.35."""
        from a_share_hot_screener.scorers.risk_control import compute_risk_control_score
        detail = _make_detail(flags={"restricted_shares_unlock_ratio_20d": 10.0})
        axis = compute_risk_control_score(detail, _make_pool())
        rc7 = _get_item(axis, "unlock_ratio_20d")
        assert rc7.subscore == pytest.approx(0.35, abs=1e-3)

    def test_rc7_at_bad_threshold(self):
        """RC7: unlock=15% ≥ B → 0.0."""
        from a_share_hot_screener.scorers.risk_control import compute_risk_control_score
        detail = _make_detail(flags={"restricted_shares_unlock_ratio_20d": 15.0})
        axis = compute_risk_control_score(detail, _make_pool())
        rc7 = _get_item(axis, "unlock_ratio_20d")
        assert rc7.subscore == pytest.approx(0.0)

    def test_rc7_none_data_unavailable(self):
        """RC7: unlock=None → is_data_available=False."""
        from a_share_hot_screener.scorers.risk_control import compute_risk_control_score
        detail = _make_detail(flags={})
        axis = compute_risk_control_score(detail, _make_pool())
        rc7 = _get_item(axis, "unlock_ratio_20d")
        assert rc7.is_data_available is False
        assert rc7.subscore is None


# ════════════════════════════════════════════════════════
# RC8: 股东人数增幅
# ════════════════════════════════════════════════════════

class TestRC8ShareholderIncrease:

    def test_rc8_negative_change_full_score(self):
        """RC8: holder_change=-5%（人数减少=集中=好）≤ G=0% → 1.0."""
        from a_share_hot_screener.scorers.risk_control import compute_risk_control_score
        detail = _make_detail(flags={"shareholder_net_reduction_ratio_3m": -5.0})
        axis = compute_risk_control_score(detail, _make_pool())
        rc8 = _get_item(axis, "shareholder_increase_3m")
        assert rc8.subscore == pytest.approx(1.0)
        assert rc8.weight == pytest.approx(1.0)

    def test_rc8_zero_change_full_score(self):
        """RC8: holder_change=0% = G → 1.0."""
        from a_share_hot_screener.scorers.risk_control import compute_risk_control_score
        detail = _make_detail(flags={"shareholder_net_reduction_ratio_3m": 0.0})
        axis = compute_risk_control_score(detail, _make_pool())
        rc8 = _get_item(axis, "shareholder_increase_3m")
        assert rc8.subscore == pytest.approx(1.0)

    def test_rc8_between_G_and_T(self):
        """RC8: G=0% < change=5% < T=10% → 线性 1.0→0.70 → ≈ 0.85."""
        from a_share_hot_screener.scorers.risk_control import compute_risk_control_score
        detail = _make_detail(flags={"shareholder_net_reduction_ratio_3m": 5.0})
        axis = compute_risk_control_score(detail, _make_pool())
        rc8 = _get_item(axis, "shareholder_increase_3m")
        assert rc8.subscore == pytest.approx(0.85, abs=1e-3)

    def test_rc8_at_mid_threshold(self):
        """RC8: change=10% = T → 0.70."""
        from a_share_hot_screener.scorers.risk_control import compute_risk_control_score
        detail = _make_detail(flags={"shareholder_net_reduction_ratio_3m": 10.0})
        axis = compute_risk_control_score(detail, _make_pool())
        rc8 = _get_item(axis, "shareholder_increase_3m")
        assert rc8.subscore == pytest.approx(0.70, abs=1e-3)

    def test_rc8_at_bad_threshold(self):
        """RC8: change=25% ≥ B → 0.0."""
        from a_share_hot_screener.scorers.risk_control import compute_risk_control_score
        detail = _make_detail(flags={"shareholder_net_reduction_ratio_3m": 25.0})
        axis = compute_risk_control_score(detail, _make_pool())
        rc8 = _get_item(axis, "shareholder_increase_3m")
        assert rc8.subscore == pytest.approx(0.0)

    def test_rc8_above_bad_threshold(self):
        """RC8: change=40% > B=25% → 0.0."""
        from a_share_hot_screener.scorers.risk_control import compute_risk_control_score
        detail = _make_detail(flags={"shareholder_net_reduction_ratio_3m": 40.0})
        axis = compute_risk_control_score(detail, _make_pool())
        rc8 = _get_item(axis, "shareholder_increase_3m")
        assert rc8.subscore == pytest.approx(0.0)

    def test_rc8_none_data_unavailable(self):
        """RC8: holder_change=None → is_data_available=False."""
        from a_share_hot_screener.scorers.risk_control import compute_risk_control_score
        detail = _make_detail(flags={})
        axis = compute_risk_control_score(detail, _make_pool())
        rc8 = _get_item(axis, "shareholder_increase_3m")
        assert rc8.is_data_available is False
        assert rc8.subscore is None


# ════════════════════════════════════════════════════════
# RC 轴整体验证
# ════════════════════════════════════════════════════════

class TestRCAxisTotal:

    def test_rc_total_weight_is_24_margin(self):
        """RC 轴 10 项两融标的总权重 = 4+3+4+2+2+2+2+1+3+1 = 24."""
        from a_share_hot_screener.scorers.risk_control import compute_risk_control_score
        detail = _make_detail(
            limit_board_count_5d=0,
            amp_norm_avg_5d=3.0,
            abs_distance_to_ma10=0.03,
            upper_shadow_count_5d=0,
            return_3d=5.0,
            flags={
                "pledge_ratio_latest": 10.0,
                "restricted_shares_unlock_ratio_20d": 2.0,
                "shareholder_net_reduction_ratio_3m": 3.0,
                "net_holder_reduction_ratio_30d": 0.1,
                "short_sell_ratio_change_5d": 5.0,
                "is_margin_eligible": True,
            },
        )
        axis = compute_risk_control_score(detail, _make_pool())
        applicable_w = sum(item.weight for item in axis.items if item.is_applicable)
        assert applicable_w == pytest.approx(24.0)
        assert len(axis.items) == 10

    def test_rc_item_count(self):
        """RC 轴现在有 10 个 ScoreItem（RC1~RC10）."""
        from a_share_hot_screener.scorers.risk_control import compute_risk_control_score
        detail = _make_detail()
        axis = compute_risk_control_score(detail, _make_pool())
        assert len(axis.items) == 10

    def test_rc_all_good_score_near_1(self):
        """所有 RC 指标都是最优值时，axis.score ≈ 1.0."""
        from a_share_hot_screener.scorers.risk_control import compute_risk_control_score
        detail = _make_detail(
            limit_board_count_5d=0,         # RC1 → 1.0
            amp_norm_avg_5d=2.0,            # RC2 → 1.0 (norm=2/10=0.2 < G=0.5)
            abs_distance_to_ma10=0.01,      # RC3 → 1.0 (1% < G=2%)
            upper_shadow_count_5d=0,        # RC4 → 1.0
            return_3d=3.0,                  # RC5 → 1.0 (norm=3/10=0.3 < G=0.8)
            flags={
                "pledge_ratio_latest": 0.0,     # RC6 → 1.0
                "restricted_shares_unlock_ratio_20d": 0.0,  # RC7 → 1.0
                "shareholder_net_reduction_ratio_3m": -5.0,  # RC8 → 1.0
            },
        )
        axis = compute_risk_control_score(detail, _make_pool())
        assert axis.score is not None
        assert axis.score >= 0.95

    def test_rc_new_items_data_missing_reduces_coverage(self):
        """RC6/7/8 全部缺失时 coverage 应低于全量可用时."""
        from a_share_hot_screener.scorers.risk_control import compute_risk_control_score

        # 全量可用
        detail_full = _make_detail(
            limit_board_count_5d=0,
            amp_norm_avg_5d=3.0,
            abs_distance_to_ma10=0.03,
            upper_shadow_count_5d=0,
            return_3d=5.0,
            flags={
                "pledge_ratio_latest": 10.0,
                "restricted_shares_unlock_ratio_20d": 2.0,
                "shareholder_net_reduction_ratio_3m": 3.0,
            },
        )
        axis_full = compute_risk_control_score(detail_full, _make_pool())

        # RC6/7/8 缺失
        detail_partial = _make_detail(
            limit_board_count_5d=0,
            amp_norm_avg_5d=3.0,
            abs_distance_to_ma10=0.03,
            upper_shadow_count_5d=0,
            return_3d=5.0,
            flags={},
        )
        axis_partial = compute_risk_control_score(detail_partial, _make_pool())

        assert axis_full.coverage > axis_partial.coverage
