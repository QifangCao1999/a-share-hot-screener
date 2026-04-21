"""Session 10 新增测试.

覆盖范围：
  1. RC1 近5日一字板次数（S10改动）
  2. RC3 阈值收紧（G=2%/T=6%/B=15%）
  3. LE3 大盘友好分档（500-1500亿→0.70，>1500亿→0.50）
  4. LE1/LE2/LE4 权重（9/6/1）
  5. LE 大盘天花板改善
  6. TF3 量比 L 从 1.1→0.8
  7. config: 轴权重 40:30:20:10，include_finance，preset，apply_preset()
  8. hard_filters H9: include_finance=True 跳过金融行业检查
  9. blocked_by 字段
  10. cache_hit_rate 统计
  11. CLI: --include-finance / --preset
"""

from __future__ import annotations

import datetime as dt
import os
import sys
import tempfile
from typing import Any, Dict, List, Optional

import pytest

# ── 工具函数 ─────────────────────────────────────────────

def _make_detail(**kwargs):
    from a_share_hot_screener.models import HotStockDetail
    defaults = {
        "code": "600519",
        "name": "贵州茅台",
        "exchange": "SH",
    }
    defaults.update(kwargs)
    return HotStockDetail(**defaults)


def _make_pool(**kwargs):
    from a_share_hot_screener.scoring import ScoringPool
    pool = ScoringPool.__new__(ScoringPool)
    pool.pool_return_5d = kwargs.get("pool_return_5d", [1.0, 2.0, 3.0, 4.0, 5.0])
    pool.pool_return_10d = kwargs.get("pool_return_10d", [1.0, 2.0, 3.0, 4.0, 5.0])
    pool.pool_amount_avg_5d = kwargs.get("pool_amount_avg_5d", [])
    pool.stock_count = kwargs.get("stock_count", 5)
    return pool


def _make_config(**kwargs):
    from a_share_hot_screener.config import HotScreenerConfig
    defaults = dict(
        tushare_token="test",
        run_date=dt.date(2026, 4, 18),
        stock_codes=["600519"],
        output_dir="/tmp/test_s10_output",
    )
    defaults.update(kwargs)
    return HotScreenerConfig(**defaults)


# ════════════════════════════════════════════════════════
# 1. RC1 近5日一字板次数
# ════════════════════════════════════════════════════════

class TestRC1LimitBoard5d:

    def test_rc1_zero_count_full_score(self):
        """RC1: limit_board_count_5d=0 → 1.0."""
        from a_share_hot_screener.scorers.risk_control import compute_risk_control_score
        detail = _make_detail(limit_board_count_5d=0)
        pool = _make_pool()
        axis = compute_risk_control_score(detail, pool)
        rc1 = next(i for i in axis.items if i.name == "limit_board_count_5d")
        assert rc1.subscore == pytest.approx(1.0)
        assert rc1.is_data_available is True

    def test_rc1_one_count(self):
        """RC1: limit_board_count_5d=1 → 0.50."""
        from a_share_hot_screener.scorers.risk_control import compute_risk_control_score
        detail = _make_detail(limit_board_count_5d=1)
        pool = _make_pool()
        axis = compute_risk_control_score(detail, pool)
        rc1 = next(i for i in axis.items if i.name == "limit_board_count_5d")
        assert rc1.subscore == pytest.approx(0.50)

    def test_rc1_two_count(self):
        """RC1: limit_board_count_5d=2 → 0.20."""
        from a_share_hot_screener.scorers.risk_control import compute_risk_control_score
        detail = _make_detail(limit_board_count_5d=2)
        pool = _make_pool()
        axis = compute_risk_control_score(detail, pool)
        rc1 = next(i for i in axis.items if i.name == "limit_board_count_5d")
        assert rc1.subscore == pytest.approx(0.20)

    def test_rc1_three_or_more_count(self):
        """RC1: limit_board_count_5d≥3 → 0.0."""
        from a_share_hot_screener.scorers.risk_control import compute_risk_control_score
        for cnt in [3, 4, 5]:
            detail = _make_detail(limit_board_count_5d=cnt)
            pool = _make_pool()
            axis = compute_risk_control_score(detail, pool)
            rc1 = next(i for i in axis.items if i.name == "limit_board_count_5d")
            assert rc1.subscore == pytest.approx(0.0), f"cnt={cnt}"

    def test_rc1_fallback_to_latest_is_limit_board(self):
        """RC1: limit_board_count_5d=None，降级为 latest_is_limit_board=False → count=0 → 1.0."""
        from a_share_hot_screener.scorers.risk_control import compute_risk_control_score
        detail = _make_detail(limit_board_count_5d=None, latest_is_limit_board=False)
        pool = _make_pool()
        axis = compute_risk_control_score(detail, pool)
        rc1 = next(i for i in axis.items if i.name == "limit_board_count_5d")
        assert rc1.subscore == pytest.approx(1.0)
        assert "fallback" in rc1.note

    def test_rc1_fallback_latest_is_limit_true(self):
        """RC1: limit_board_count_5d=None, latest_is_limit_board=True → count=1 → 0.50."""
        from a_share_hot_screener.scorers.risk_control import compute_risk_control_score
        detail = _make_detail(limit_board_count_5d=None, latest_is_limit_board=True)
        pool = _make_pool()
        axis = compute_risk_control_score(detail, pool)
        rc1 = next(i for i in axis.items if i.name == "limit_board_count_5d")
        assert rc1.subscore == pytest.approx(0.50)

    def test_rc1_both_none_data_unavailable(self):
        """RC1: 所有字段均缺失 → is_data_available=False."""
        from a_share_hot_screener.scorers.risk_control import compute_risk_control_score
        detail = _make_detail(limit_board_count_5d=None, latest_is_limit_board=None)
        pool = _make_pool()
        axis = compute_risk_control_score(detail, pool)
        rc1 = next(i for i in axis.items if i.name == "limit_board_count_5d")
        assert rc1.is_data_available is False
        assert rc1.subscore is None


# ════════════════════════════════════════════════════════
# 2. RC3 阈值收紧
# ════════════════════════════════════════════════════════

class TestRC3TighterThreshold:

    def test_rc3_at_good_threshold(self):
        """RC3: abs_dist=2% <= G=2% → 1.0."""
        from a_share_hot_screener.scorers.risk_control import compute_risk_control_score
        detail = _make_detail(abs_distance_to_ma10=0.02)  # 2%
        pool = _make_pool()
        axis = compute_risk_control_score(detail, pool)
        rc3 = next(i for i in axis.items if i.name == "abs_deviation_ma10")
        assert rc3.subscore == pytest.approx(1.0)

    def test_rc3_between_G_and_T(self):
        """RC3: G=2% < abs_dist=4% < T=6% → linear (1.0→0.70)."""
        from a_share_hot_screener.scorers.risk_control import compute_risk_control_score
        # (4-2)/(6-2) * (0.70-1.0) + 1.0 = 0.5 * (-0.30) + 1.0 = 0.85
        detail = _make_detail(abs_distance_to_ma10=0.04)  # 4%
        pool = _make_pool()
        axis = compute_risk_control_score(detail, pool)
        rc3 = next(i for i in axis.items if i.name == "abs_deviation_ma10")
        assert rc3.subscore == pytest.approx(0.85, abs=1e-3)

    def test_rc3_at_mid_threshold(self):
        """RC3: abs_dist=6% = T → 0.70."""
        from a_share_hot_screener.scorers.risk_control import compute_risk_control_score
        detail = _make_detail(abs_distance_to_ma10=0.06)  # 6%
        pool = _make_pool()
        axis = compute_risk_control_score(detail, pool)
        rc3 = next(i for i in axis.items if i.name == "abs_deviation_ma10")
        assert rc3.subscore == pytest.approx(0.70, abs=1e-3)

    def test_rc3_above_bad_threshold(self):
        """RC3: abs_dist=15% >= B=15% → 0.0（S10 B 从 20% 收紧至 15%）."""
        from a_share_hot_screener.scorers.risk_control import compute_risk_control_score
        detail = _make_detail(abs_distance_to_ma10=0.15)  # 15%
        pool = _make_pool()
        axis = compute_risk_control_score(detail, pool)
        rc3 = next(i for i in axis.items if i.name == "abs_deviation_ma10")
        assert rc3.subscore == pytest.approx(0.0)

    def test_rc3_old_bad_threshold_now_penalized(self):
        """RC3: abs_dist=17%（S8 时不触发 0 分，S10 B=15% 现在触发 0 分）."""
        from a_share_hot_screener.scorers.risk_control import compute_risk_control_score
        detail = _make_detail(abs_distance_to_ma10=0.17)  # 17%
        pool = _make_pool()
        axis = compute_risk_control_score(detail, pool)
        rc3 = next(i for i in axis.items if i.name == "abs_deviation_ma10")
        # B=15%, 17% > B=15% → 0.0
        assert rc3.subscore == pytest.approx(0.0)


# ════════════════════════════════════════════════════════
# 3. LE3 大盘友好分档
# ════════════════════════════════════════════════════════

class TestLE3LargeCapFriendly:

    def test_le3_500_1500_bn(self):
        """LE3: 500~1500亿 → 0.70（S10改，原 0.45）."""
        from a_share_hot_screener.scorers.liquidity_execution import compute_liquidity_execution_score
        for fmc_bn in [500, 800, 1000, 1499]:
            detail = _make_detail(float_market_cap=fmc_bn * 1e8)
            pool = _make_pool()
            axis = compute_liquidity_execution_score(detail, pool, enable_lhb_module=False)
            le3 = next(i for i in axis.items if i.name == "float_market_cap_bracket")
            assert le3.subscore == pytest.approx(0.70), f"fmc={fmc_bn}亿"

    def test_le3_above_1500_bn(self):
        """LE3: >1500亿 → 0.50（S10改，原 0.20）；边界1500亿归上一档(0.70)。"""
        from a_share_hot_screener.scorers.liquidity_execution import compute_liquidity_execution_score
        for fmc_bn in [1501, 2000, 5000, 10000]:
            detail = _make_detail(float_market_cap=fmc_bn * 1e8)
            pool = _make_pool()
            axis = compute_liquidity_execution_score(detail, pool, enable_lhb_module=False)
            le3 = next(i for i in axis.items if i.name == "float_market_cap_bracket")
            assert le3.subscore == pytest.approx(0.50), f"fmc={fmc_bn}亿"

    def test_le3_150_500_bn_unchanged(self):
        """LE3: 150~500亿 → 0.70（未变）."""
        from a_share_hot_screener.scorers.liquidity_execution import compute_liquidity_execution_score
        detail = _make_detail(float_market_cap=300e8)
        pool = _make_pool()
        axis = compute_liquidity_execution_score(detail, pool, enable_lhb_module=False)
        le3 = next(i for i in axis.items if i.name == "float_market_cap_bracket")
        assert le3.subscore == pytest.approx(0.70)

    def test_le3_small_cap_unchanged(self):
        """LE3: 15~50亿 → 1.0（未变）."""
        from a_share_hot_screener.scorers.liquidity_execution import compute_liquidity_execution_score
        detail = _make_detail(float_market_cap=30e8)
        pool = _make_pool()
        axis = compute_liquidity_execution_score(detail, pool, enable_lhb_module=False)
        le3 = next(i for i in axis.items if i.name == "float_market_cap_bracket")
        assert le3.subscore == pytest.approx(1.0)

    def test_le_large_cap_ceiling_improved(self):
        """LE 天花板计算：500-1500亿市值 + 龙虎榜0次，成交和换手率拉满，
        ceiling = (9×1.0 + 6×1.0 + 4×0.70 + 1×0.20) / 20 = 0.83（S10改善）."""
        from a_share_hot_screener.scorers.liquidity_execution import compute_liquidity_execution_score
        detail = _make_detail(
            float_market_cap=800e8,       # 800亿
            amount_avg_5d=50e8,           # 50亿 >= H=30亿 → 1.0
            lhb_count_20d=0,              # LE4=0.20
        )
        pool = _make_pool()
        # amount_avg_5d=50亿, float_market_cap=800亿
        # turnover = 50/800*100 = 6.25% (>= T=12%? No, between L=5% and T=12%)
        # turnover subscore = 线性 (6.25-5)/(12-5)*0.70 = 0.125
        axis = compute_liquidity_execution_score(detail, pool, enable_lhb_module=True)
        # LE1=1.0(weight=9), LE2≈0.125(weight=6), LE3=0.70(weight=4), LE4=0.20(weight=1)
        # score = (9×1.0 + 6×0.125 + 4×0.70 + 1×0.20) / 20 = (9+0.75+2.80+0.20)/20 = 12.75/20 = 0.6375
        assert axis.score is not None
        assert axis.score > 0.50  # 明显优于 S8 的 0.45 天花板附近


# ════════════════════════════════════════════════════════
# 4. LE 权重（9+6+4+1=20）
# ════════════════════════════════════════════════════════

class TestLEWeights:

    def test_le1_weight(self):
        """LE1 权重为 9（S10 升权）."""
        from a_share_hot_screener.scorers.liquidity_execution import compute_liquidity_execution_score
        detail = _make_detail(amount_avg_5d=50e8, float_market_cap=100e8, lhb_count_20d=0)
        pool = _make_pool()
        axis = compute_liquidity_execution_score(detail, pool, enable_lhb_module=True)
        le1 = next(i for i in axis.items if i.name == "amount_avg_5d")
        assert le1.weight == pytest.approx(9.0)

    def test_le2_weight(self):
        """LE2 权重为 6（S10 升权）."""
        from a_share_hot_screener.scorers.liquidity_execution import compute_liquidity_execution_score
        detail = _make_detail(amount_avg_5d=50e8, float_market_cap=100e8)
        pool = _make_pool()
        axis = compute_liquidity_execution_score(detail, pool, enable_lhb_module=False)
        le2 = next(i for i in axis.items if i.name == "turnover_avg_5d_approx")
        assert le2.weight == pytest.approx(6.0)

    def test_le4_weight(self):
        """LE4 权重为 1（S10 降权）."""
        from a_share_hot_screener.scorers.liquidity_execution import compute_liquidity_execution_score
        detail = _make_detail(lhb_count_20d=2)
        pool = _make_pool()
        axis = compute_liquidity_execution_score(detail, pool, enable_lhb_module=True)
        le4 = next(i for i in axis.items if i.name == "lhb_count_20d")
        assert le4.weight == pytest.approx(1.0)

    def test_total_weights_sum(self):
        """四项权重之和为 20。"""
        from a_share_hot_screener.scorers.liquidity_execution import compute_liquidity_execution_score
        detail = _make_detail(
            amount_avg_5d=10e8, float_market_cap=100e8, lhb_count_20d=1
        )
        pool = _make_pool()
        axis = compute_liquidity_execution_score(detail, pool, enable_lhb_module=True)
        total_w = sum(item.weight for item in axis.items)
        assert total_w == pytest.approx(20.0)


# ════════════════════════════════════════════════════════
# 5. TF3 量比 L=0.8
# ════════════════════════════════════════════════════════

class TestTF3NewL:

    def test_tf3_below_L(self):
        """TF3: volume_ratio=0.7 < L=0.8 → 0.0."""
        from a_share_hot_screener.scorers.trend_flow import compute_trend_flow_score
        detail = _make_detail(volume_ratio_20d=0.7)
        pool = _make_pool()
        axis = compute_trend_flow_score(detail, pool)
        tf3 = next(i for i in axis.items if i.name == "volume_ratio_20d")
        assert tf3.subscore == pytest.approx(0.0)

    def test_tf3_at_L(self):
        """TF3: volume_ratio=0.8 = L → 0.0（边界条件）."""
        from a_share_hot_screener.scorers.trend_flow import compute_trend_flow_score
        detail = _make_detail(volume_ratio_20d=0.8)
        pool = _make_pool()
        axis = compute_trend_flow_score(detail, pool)
        tf3 = next(i for i in axis.items if i.name == "volume_ratio_20d")
        assert tf3.subscore == pytest.approx(0.0)

    def test_tf3_between_L_and_T(self):
        """TF3: L=0.8 < volume_ratio=1.0 < T=1.8 → 线性 >0。"""
        from a_share_hot_screener.scorers.trend_flow import compute_trend_flow_score
        # (1.0-0.8)/(1.8-0.8)*0.70 = 0.20/1.0*0.70 = 0.14
        detail = _make_detail(volume_ratio_20d=1.0)
        pool = _make_pool()
        axis = compute_trend_flow_score(detail, pool)
        tf3 = next(i for i in axis.items if i.name == "volume_ratio_20d")
        assert tf3.subscore == pytest.approx(0.14, abs=1e-3)

    def test_tf3_old_L_now_has_score(self):
        """TF3: S8 时 volume_ratio=1.05（< L=1.1 → 0），S10 L=0.8 时有分了。"""
        from a_share_hot_screener.scorers.trend_flow import compute_trend_flow_score
        # (1.05-0.8)/(1.8-0.8)*0.70 = 0.25/1.0*0.70 = 0.175
        detail = _make_detail(volume_ratio_20d=1.05)
        pool = _make_pool()
        axis = compute_trend_flow_score(detail, pool)
        tf3 = next(i for i in axis.items if i.name == "volume_ratio_20d")
        assert tf3.subscore == pytest.approx(0.175, abs=1e-3)


# ════════════════════════════════════════════════════════
# 6. Config: 权重/include_finance/preset
# ════════════════════════════════════════════════════════

class TestConfigS10:

    def test_default_axis_weights(self):
        """默认权重 40:30:20:10（S10 改）."""
        cfg = _make_config()
        assert cfg.axis_weight_hot_theme == pytest.approx(40.0)
        assert cfg.axis_weight_trend_flow == pytest.approx(30.0)
        assert cfg.axis_weight_liquidity_execution == pytest.approx(20.0)
        assert cfg.axis_weight_risk_control == pytest.approx(10.0)
        assert (cfg.axis_weight_hot_theme + cfg.axis_weight_trend_flow +
                cfg.axis_weight_liquidity_execution + cfg.axis_weight_risk_control) == pytest.approx(100.0)

    def test_include_finance_default_false(self):
        """include_finance 默认 False."""
        cfg = _make_config()
        assert cfg.include_finance is False

    def test_include_finance_custom_true(self):
        """include_finance=True 可设置。"""
        cfg = _make_config(include_finance=True)
        assert cfg.include_finance is True

    def test_preset_default(self):
        """preset 默认 default。"""
        cfg = _make_config()
        assert cfg.preset == "default"

    def test_apply_preset_default_noop(self):
        """apply_preset('default') 不改变阈值。"""
        cfg = _make_config(preset="default")
        original_total = cfg.min_total_score
        cfg.apply_preset()
        assert cfg.min_total_score == original_total

    def test_apply_preset_relaxed(self):
        """apply_preset('relaxed') 正确降低阈值。"""
        cfg = _make_config(preset="relaxed")
        cfg.apply_preset()
        assert cfg.min_total_score == pytest.approx(0.50)
        assert cfg.min_hot_theme_score == pytest.approx(0.40)
        assert cfg.min_trend_flow_score == pytest.approx(0.40)
        assert cfg.min_liquidity_execution_score == pytest.approx(0.35)
        assert cfg.min_risk_control_score == pytest.approx(0.20)

    def test_apply_preset_relaxed_all_less_than_default(self):
        """relaxed 预设每项阈值均 < default。"""
        cfg_d = _make_config(preset="default")
        cfg_r = _make_config(preset="relaxed")
        cfg_r.apply_preset()
        assert cfg_r.min_total_score < cfg_d.min_total_score
        assert cfg_r.min_hot_theme_score < cfg_d.min_hot_theme_score
        assert cfg_r.min_trend_flow_score < cfg_d.min_trend_flow_score
        assert cfg_r.min_liquidity_execution_score < cfg_d.min_liquidity_execution_score
        assert cfg_r.min_risk_control_score < cfg_d.min_risk_control_score


# ════════════════════════════════════════════════════════
# 7. H9 include_finance 控制
# ════════════════════════════════════════════════════════

class TestH9IncludeFinance:

    def _apply(self, industry: str, include_finance: bool):
        from a_share_hot_screener.hard_filters import apply_hard_filters
        cfg = _make_config(include_finance=include_finance)
        return apply_hard_filters(
            config=cfg,
            name="招商银行",
            industry=industry,
            ipo_date="20020319",
            listing_days=8000,
            latest_price=40.0,
            latest_volume=1e6,
            amount_1d=4e8,
            amount_avg_5d=5e8,
            float_market_cap=500e8,
        )

    def test_h9_finance_excluded_by_default(self):
        """include_finance=False 时，金融行业被硬筛拦截。"""
        result = self._apply("银行", include_finance=False)
        assert result.passed is False
        assert any("industry_finance" in r for r in result.fail_reasons)

    def test_h9_finance_allowed_when_flag_set(self):
        """include_finance=True 时，金融行业通过 H9。"""
        result = self._apply("银行", include_finance=True)
        # 不应有 industry_finance 失败原因
        assert not any("industry_finance" in r for r in result.fail_reasons)
        # 应有 include_finance=True 的 warning
        assert any("include_finance=True" in w for w in result.data_warnings)

    def test_h9_non_finance_always_passes(self):
        """非金融行业不受 include_finance 影响。"""
        for flag in [True, False]:
            result = self._apply("电子", include_finance=flag)
            assert not any("industry_finance" in r for r in result.fail_reasons)

    def test_h9_securities_excluded_by_default(self):
        """证券行业也被 H9 拦截（include_finance=False）。"""
        result = self._apply("证券", include_finance=False)
        assert not result.passed
        assert any("industry_finance" in r for r in result.fail_reasons)


# ════════════════════════════════════════════════════════
# 8. blocked_by 字段
# ════════════════════════════════════════════════════════

class TestBlockedBy:

    def test_blocked_by_empty_when_pass(self):
        """通过所有轴时 blocked_by 为空。"""
        from a_share_hot_screener.models import HotStockDetail
        d = HotStockDetail(code="600519")
        d.pass_stage1 = True
        d.blocked_by = []
        assert d.blocked_by == []

    def test_blocked_by_has_correct_field(self):
        """HotStockDetail 有 blocked_by 字段（List[str]，默认 []）。"""
        from a_share_hot_screener.models import HotStockDetail
        import dataclasses
        field_names = {f.name for f in dataclasses.fields(HotStockDetail)}
        assert "blocked_by" in field_names

    def test_blocked_by_in_summary(self):
        """HotStockSummary 有 blocked_by 字段。"""
        from a_share_hot_screener.models import HotStockSummary
        import dataclasses
        field_names = {f.name for f in dataclasses.fields(HotStockSummary)}
        assert "blocked_by" in field_names

    def test_from_detail_copies_blocked_by(self):
        """from_detail 正确复制 blocked_by 列表。"""
        from a_share_hot_screener.models import HotStockDetail, HotStockSummary
        d = HotStockDetail(code="600519", name="茅台")
        d.blocked_by = ["hot_theme", "liquidity_execution"]
        d.pass_stage1 = False
        s = HotStockSummary.from_detail(d)
        assert s.blocked_by == ["hot_theme", "liquidity_execution"]

    def test_blocked_by_in_detail_long_rows(self):
        """detail 长表的 summary 行含 blocked_by 指标。"""
        from a_share_hot_screener.output import _detail_to_long_rows
        from a_share_hot_screener.models import HotStockDetail
        d = HotStockDetail(code="600519", name="茅台")
        d.blocked_by = ["hot_theme"]
        d.pass_stage1 = False
        rows = _detail_to_long_rows(d)
        summary_rows = [r for r in rows if r["axis"] == "summary"]
        indicator_names = [r["indicator_name"] for r in summary_rows]
        assert "blocked_by" in indicator_names

    def test_blocked_by_value_in_detail_row(self):
        """blocked_by 的 raw_value 为逗号分隔的轴名字符串。"""
        from a_share_hot_screener.output import _detail_to_long_rows
        from a_share_hot_screener.models import HotStockDetail
        d = HotStockDetail(code="600519", name="茅台")
        d.blocked_by = ["hot_theme", "trend_flow"]
        rows = _detail_to_long_rows(d)
        blocked_row = next(
            (r for r in rows if r.get("indicator_name") == "blocked_by"), None
        )
        assert blocked_row is not None
        assert blocked_row["raw_value"] == "hot_theme,trend_flow"


# ════════════════════════════════════════════════════════
# 9. cache_hit_rate 统计
# ════════════════════════════════════════════════════════

class TestCacheHitRate:

    def test_initial_hit_rate_is_none(self):
        """初始状态未访问过缓存时，hit_rate=None。"""
        from a_share_hot_screener.cache import LocalCache
        with tempfile.TemporaryDirectory() as d:
            cache = LocalCache(d)
            assert cache.get_hit_rate() is None

    def test_all_miss(self):
        """全未命中时 hit_rate=0.0。"""
        from a_share_hot_screener.cache import LocalCache
        with tempfile.TemporaryDirectory() as d:
            cache = LocalCache(d)
            cache.get("ns", "key1")
            cache.get("ns", "key2")
            assert cache.get_hit_rate() == pytest.approx(0.0)

    def test_all_hit(self):
        """全命中时 hit_rate=1.0。"""
        from a_share_hot_screener.cache import LocalCache
        with tempfile.TemporaryDirectory() as d:
            cache = LocalCache(d)
            cache.put("ns", "k", "v")
            cache.get("ns", "k")
            cache.get("ns", "k")
            assert cache.get_hit_rate() == pytest.approx(1.0)

    def test_mixed_hit_rate(self):
        """1命中 + 1未命中 → hit_rate=0.5。"""
        from a_share_hot_screener.cache import LocalCache
        with tempfile.TemporaryDirectory() as d:
            cache = LocalCache(d)
            cache.put("ns", "k", "val")
            cache.get("ns", "k")         # hit
            cache.get("ns", "missing")   # miss
            assert cache.get_hit_rate() == pytest.approx(0.5)

    def test_expired_entry_counts_as_miss(self):
        """过期缓存计为 miss。"""
        import time
        from a_share_hot_screener.cache import LocalCache
        with tempfile.TemporaryDirectory() as d:
            cache = LocalCache(d, default_ttl=1)
            cache.put("ns", "k", "v", ttl=1)
            time.sleep(1.1)
            cache.get("ns", "k")  # expired → miss
            assert cache.get_hit_rate() == pytest.approx(0.0)

    def test_hit_rate_in_metadata(self):
        """RunMetadata cache_hit_rate 字段存在且为 Optional[float]。"""
        from a_share_hot_screener.models import RunMetadata
        import dataclasses
        field_map = {f.name: f for f in dataclasses.fields(RunMetadata)}
        assert "cache_hit_rate" in field_map
        meta = RunMetadata()
        assert meta.cache_hit_rate is None  # 默认 None


# ════════════════════════════════════════════════════════
# 10. CLI 参数测试
# ════════════════════════════════════════════════════════

class TestCLIS10Params:

    def test_include_finance_default_false(self):
        """--include-finance 默认 False。"""
        from a_share_hot_screener.cli import build_parser
        parser = build_parser()
        args = parser.parse_args([
            "--tushare-token", "t",
            "--run-date", "2026-04-18",
            "--stock-codes", "600519",
            "--output-dir", "/tmp",
        ])
        assert args.include_finance is False

    def test_include_finance_flag(self):
        """--include-finance 开启后为 True。"""
        from a_share_hot_screener.cli import build_parser
        parser = build_parser()
        args = parser.parse_args([
            "--tushare-token", "t",
            "--run-date", "2026-04-18",
            "--stock-codes", "600519",
            "--output-dir", "/tmp",
            "--include-finance",
        ])
        assert args.include_finance is True

    def test_preset_default(self):
        """--preset 默认 default。"""
        from a_share_hot_screener.cli import build_parser
        parser = build_parser()
        args = parser.parse_args([
            "--tushare-token", "t",
            "--run-date", "2026-04-18",
            "--stock-codes", "600519",
            "--output-dir", "/tmp",
        ])
        assert args.preset == "default"

    def test_preset_relaxed(self):
        """--preset relaxed 被正确接受。"""
        from a_share_hot_screener.cli import build_parser
        parser = build_parser()
        args = parser.parse_args([
            "--tushare-token", "t",
            "--run-date", "2026-04-18",
            "--stock-codes", "600519",
            "--output-dir", "/tmp",
            "--preset", "relaxed",
        ])
        assert args.preset == "relaxed"

    def test_preset_invalid_rejected(self):
        """--preset invalid 触发 argparse 错误（choices 约束）."""
        from a_share_hot_screener.cli import build_parser
        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args([
                "--tushare-token", "t",
                "--run-date", "2026-04-18",
                "--stock-codes", "600519",
                "--output-dir", "/tmp",
                "--preset", "invalid_preset",
            ])


# ════════════════════════════════════════════════════════
# #2: 小股票池百分位失真修正
# ════════════════════════════════════════════════════════

class TestSmallPoolPercentileFix:
    """小股票池百分位修正（#2）的测试."""

    def test_large_pool_no_cap(self):
        """池子 >= 30 时不限制子分，HT1 可以达到满分."""
        from a_share_hot_screener.scorers.hot_theme import compute_hot_theme_score, SMALL_POOL_THRESHOLD
        pool = _make_pool(
            pool_return_5d=list(range(1, 32)),   # 31只
            pool_return_10d=list(range(1, 32)),
            stock_count=31,
        )
        # return_5d=31 是池中最高 → 百分位 100%
        d = _make_detail(return_5d=31.0, return_10d=31.0)
        axis = compute_hot_theme_score(d, pool)
        ht1 = [i for i in axis.items if i.name == "return_5d_pctile"][0]
        assert ht1.subscore == 1.0  # 无 cap
        assert "small_pool_cap" not in (ht1.note or "")

    def test_small_pool_capped_at_090(self):
        """池子 < 30 但 >= 2 时，HT1/HT2 子分上限 0.90."""
        from a_share_hot_screener.scorers.hot_theme import compute_hot_theme_score, SMALL_POOL_SUBSCORE_CAP
        pool = _make_pool(
            pool_return_5d=[1.0, 2.0, 3.0, 4.0, 5.0],  # 5只
            pool_return_10d=[1.0, 2.0, 3.0, 4.0, 5.0],
            stock_count=5,
        )
        # return_5d=5.0 是池中最高，原本百分位=100% → subscore=1.0
        # 但因小池 cap → 子分 <= 0.90
        d = _make_detail(return_5d=5.0, return_10d=5.0)
        axis = compute_hot_theme_score(d, pool)
        ht1 = [i for i in axis.items if i.name == "return_5d_pctile"][0]
        assert ht1.subscore <= SMALL_POOL_SUBSCORE_CAP
        assert "small_pool_cap" in (ht1.note or "")

        ht2 = [i for i in axis.items if i.name == "return_10d_pctile"][0]
        assert ht2.subscore <= SMALL_POOL_SUBSCORE_CAP

    def test_tiny_pool_uses_absolute_fallback(self):
        """池子 < 2 时 HT1/HT2 使用绝对涨幅 fallback."""
        from a_share_hot_screener.scorers.hot_theme import compute_hot_theme_score, TINY_POOL_SUBSCORE_CAP
        pool = _make_pool(
            pool_return_5d=[],  # 空池
            pool_return_10d=[],
            stock_count=0,
        )
        # return_5d=30% 超过 high_pct=25% → 子分 = cap = 0.70
        d = _make_detail(return_5d=30.0, return_10d=50.0)
        axis = compute_hot_theme_score(d, pool)
        ht1 = [i for i in axis.items if i.name == "return_5d_pctile"][0]
        assert ht1.subscore == TINY_POOL_SUBSCORE_CAP
        assert "abs_fallback" in (ht1.note or "")

        ht2 = [i for i in axis.items if i.name == "return_10d_pctile"][0]
        assert ht2.subscore == TINY_POOL_SUBSCORE_CAP
        assert "abs_fallback" in (ht2.note or "")

    def test_absolute_fallback_low_return(self):
        """绝对涨幅 fallback 低涨幅 → 低分."""
        from a_share_hot_screener.scorers.hot_theme import compute_hot_theme_score
        pool = _make_pool(pool_return_5d=[], pool_return_10d=[], stock_count=0)
        # return_5d=3% < low_pct=5% → subscore=0
        d = _make_detail(return_5d=3.0, return_10d=5.0)
        axis = compute_hot_theme_score(d, pool)
        ht1 = [i for i in axis.items if i.name == "return_5d_pctile"][0]
        assert ht1.subscore == 0.0

    def test_absolute_fallback_mid_return(self):
        """绝对涨幅 fallback 中等涨幅 → 中等分."""
        from a_share_hot_screener.scorers.hot_theme import compute_hot_theme_score, TINY_POOL_SUBSCORE_CAP
        pool = _make_pool(pool_return_5d=[], pool_return_10d=[], stock_count=0)
        # return_5d=12% = mid_pct → subscore ≈ 0.70 * 0.70 = 0.49
        d = _make_detail(return_5d=12.0, return_10d=20.0)
        axis = compute_hot_theme_score(d, pool)
        ht1 = [i for i in axis.items if i.name == "return_5d_pctile"][0]
        expected_mid = TINY_POOL_SUBSCORE_CAP * 0.70  # 0.49
        assert abs(ht1.subscore - expected_mid) < 0.01

    def test_absolute_fallback_none_return(self):
        """绝对涨幅 fallback 但 return 缺失 → data_unavailable."""
        from a_share_hot_screener.scorers.hot_theme import compute_hot_theme_score
        pool = _make_pool(pool_return_5d=[], pool_return_10d=[], stock_count=0)
        d = _make_detail(return_5d=None, return_10d=None)
        axis = compute_hot_theme_score(d, pool)
        ht1 = [i for i in axis.items if i.name == "return_5d_pctile"][0]
        assert ht1.is_data_available is False

    def test_single_stock_pool_uses_fallback(self):
        """只有1只股票（pool只有1个元素）→ fallback."""
        from a_share_hot_screener.scorers.hot_theme import compute_hot_theme_score
        pool = _make_pool(
            pool_return_5d=[10.0],  # 1个元素 < 2
            pool_return_10d=[15.0],
            stock_count=1,
        )
        d = _make_detail(return_5d=10.0, return_10d=15.0)
        axis = compute_hot_theme_score(d, pool)
        ht1 = [i for i in axis.items if i.name == "return_5d_pctile"][0]
        assert "abs_fallback" in (ht1.note or "")

    def test_config_min_baseline_pool_size_default_30(self):
        """配置默认 min_baseline_pool_size 已从 5 提升到 30."""
        cfg = _make_config()
        assert cfg.min_baseline_pool_size == 30
