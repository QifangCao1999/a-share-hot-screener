"""Session 6 — liquidity_execution_score / risk_control_score / flags 单元测试.

覆盖范围：
  TestLiquidityExecutionScore  — LE1~LE4 指标完整路径
  TestRiskControlScore         — RC1~RC5 指标完整路径
  TestComputeFlags             — 20项 flags 完整路径
  TestScoringPoolS6            — pool_amount_avg_5d 新字段构建
  TestFourAxisPipeline         — pipeline _apply_four_axis_scores 集成
  TestSummaryFlagsFields       — HotStockSummary.from_detail flags 字段
"""

from __future__ import annotations

import math
import pytest
from typing import Optional

from a_share_hot_screener.models import HotStockDetail, HotStockSummary
from a_share_hot_screener.scoring import ScoringPool


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
        pool_volume_ratio_20d=[0.5, 1.0, 1.5, 2.0, 2.5],
        pool_amount_ratio_5d_to_20d=[0.8, 1.0, 1.2, 1.5],
        pool_close_position_20d=[0.2, 0.4, 0.6, 0.8],
        pool_clv_latest=[0.3, 0.5, 0.7, 0.9],
    )


# ════════════════════════════════════════════════════════
# TestLiquidityExecutionScore
# ════════════════════════════════════════════════════════

class TestLiquidityExecutionScore:

    def test_all_available_with_lhb(self):
        from a_share_hot_screener.scorers.liquidity_execution import compute_liquidity_execution_score
        detail = _make_detail(
            amount_avg_5d=3e8,
            float_market_cap=30e8,    # 30亿 → LE3 区间 20~50亿 → 0.80
            lhb_count_20d=2,
            lhb_source="lhb_detail_em",
        )
        pool = _full_pool()
        axis = compute_liquidity_execution_score(detail, pool, enable_lhb_module=True)
        assert axis.score is not None
        assert 0.0 <= axis.score <= 1.0
        assert axis.coverage == pytest.approx(1.0)
        assert len(axis.items) == 4

    def test_le1_high_amount(self):
        """LE1 使用绝对值三段下限型（L=2亿,T=8亿,H=30亿）."""
        from a_share_hot_screener.scorers.liquidity_execution import compute_liquidity_execution_score
        detail = _make_detail(amount_avg_5d=30e8)   # 30亿 >= H → 1.0
        pool = _make_pool()
        axis = compute_liquidity_execution_score(detail, pool, enable_lhb_module=False)
        le1 = next(i for i in axis.items if i.name == "amount_avg_5d")
        assert le1.subscore == pytest.approx(1.0)

    def test_le1_low_amount(self):
        """LE1 成交额 <= L → 0 分."""
        from a_share_hot_screener.scorers.liquidity_execution import compute_liquidity_execution_score
        detail = _make_detail(amount_avg_5d=1e8)   # 1亿 < L=2亿 → 0
        pool = _make_pool()
        axis = compute_liquidity_execution_score(detail, pool, enable_lhb_module=False)
        le1 = next(i for i in axis.items if i.name == "amount_avg_5d")
        assert le1.subscore == pytest.approx(0.0)

    def test_le2_turnover_calc(self):
        """LE2: turnover_avg_5d = amount_avg_5d / float_market_cap * 100.

        #3: proxy 子分上限 0.85，即使换手率达到满分条件也不超过 0.85。
        """
        from a_share_hot_screener.scorers.liquidity_execution import (
            compute_liquidity_execution_score, _TURNOVER_PROXY_SUBSCORE_CAP,
        )
        # amount_avg_5d = 3e8, float_market_cap = 10e8 → turnover = 30% >= H=25%
        # 原本应该满分 1.0，但 proxy cap → 0.85
        detail = _make_detail(amount_avg_5d=3e8, float_market_cap=10e8)
        pool = _make_pool()
        axis = compute_liquidity_execution_score(detail, pool, enable_lhb_module=False)
        le2 = next(i for i in axis.items if i.name == "turnover_avg_5d")
        assert le2.subscore == pytest.approx(_TURNOVER_PROXY_SUBSCORE_CAP)
        assert "proxy_capped" in (le2.note or "")

    def test_le2_low_turnover(self):
        """LE2: 低换手率得低分."""
        from a_share_hot_screener.scorers.liquidity_execution import compute_liquidity_execution_score
        # turnover = 0.1e8 / 100e8 * 100 = 0.1% → < L=5% → score=0
        detail = _make_detail(amount_avg_5d=0.1e8, float_market_cap=100e8)
        pool = _make_pool()
        axis = compute_liquidity_execution_score(detail, pool, enable_lhb_module=False)
        le2 = next(i for i in axis.items if i.name == "turnover_avg_5d")
        assert le2.subscore == pytest.approx(0.0)

    def test_le2_float_mc_missing(self):
        """LE2: float_market_cap 缺失 → is_data_available=False."""
        from a_share_hot_screener.scorers.liquidity_execution import compute_liquidity_execution_score
        detail = _make_detail(amount_avg_5d=3e8, float_market_cap=None)
        pool = _make_pool()
        axis = compute_liquidity_execution_score(detail, pool, enable_lhb_module=False)
        le2 = next(i for i in axis.items if i.name == "turnover_avg_5d")
        assert le2.is_data_available is False

    def test_le3_bracket_20_50_bn(self):
        """LE3: 20~50亿 → subscore=0.80（最优操作区间）."""
        from a_share_hot_screener.scorers.liquidity_execution import compute_liquidity_execution_score
        detail = _make_detail(float_market_cap=35e8)   # 35亿
        pool = _make_pool()
        axis = compute_liquidity_execution_score(detail, pool, enable_lhb_module=False)
        le3 = next(i for i in axis.items if i.name == "float_market_cap_bracket")
        assert le3.subscore == pytest.approx(1.0)  # 15~50亿 → 1.0

    def test_le3_bracket_50_150_bn(self):
        """LE3: 50~150亿 → subscore=0.90."""
        from a_share_hot_screener.scorers.liquidity_execution import compute_liquidity_execution_score
        detail = _make_detail(float_market_cap=75e8)   # 75亿
        pool = _make_pool()
        axis = compute_liquidity_execution_score(detail, pool, enable_lhb_module=False)
        le3 = next(i for i in axis.items if i.name == "float_market_cap_bracket")
        assert le3.subscore == pytest.approx(0.90)

    def test_le3_bracket_micro_cap(self):
        """LE3: <15亿 → subscore=0.20."""
        from a_share_hot_screener.scorers.liquidity_execution import compute_liquidity_execution_score
        detail = _make_detail(float_market_cap=5e8)   # 5亿
        pool = _make_pool()
        axis = compute_liquidity_execution_score(detail, pool, enable_lhb_module=False)
        le3 = next(i for i in axis.items if i.name == "float_market_cap_bracket")
        assert le3.subscore == pytest.approx(0.20)

    def test_le3_bracket_large_cap(self):
        """LE3: 500~1500亿 → subscore=0.70（Session 10 大盘友好，原 0.45）."""
        from a_share_hot_screener.scorers.liquidity_execution import compute_liquidity_execution_score
        detail = _make_detail(float_market_cap=800e8)   # 800亿
        pool = _make_pool()
        axis = compute_liquidity_execution_score(detail, pool, enable_lhb_module=False)
        le3 = next(i for i in axis.items if i.name == "float_market_cap_bracket")
        assert le3.subscore == pytest.approx(0.70)

    def test_le3_fallback_to_market_cap(self):
        """LE3: float_market_cap 缺失时使用 market_cap 兜底."""
        from a_share_hot_screener.scorers.liquidity_execution import compute_liquidity_execution_score
        detail = _make_detail(float_market_cap=None, market_cap=75e8)
        pool = _make_pool()
        axis = compute_liquidity_execution_score(detail, pool, enable_lhb_module=False)
        le3 = next(i for i in axis.items if i.name == "float_market_cap_bracket")
        assert le3.subscore == pytest.approx(0.90)   # 75亿 → 50~150亿 → 0.90

    def test_le3_both_missing(self):
        """LE3: float_market_cap 和 market_cap 均缺失 → is_data_available=False."""
        from a_share_hot_screener.scorers.liquidity_execution import compute_liquidity_execution_score
        detail = _make_detail(float_market_cap=None, market_cap=None)
        pool = _make_pool()
        axis = compute_liquidity_execution_score(detail, pool, enable_lhb_module=False)
        le3 = next(i for i in axis.items if i.name == "float_market_cap_bracket")
        assert le3.is_data_available is False

    def test_le4_applicable_when_lhb_enabled(self):
        """LE4: enable_lhb_module=True → is_applicable=True."""
        from a_share_hot_screener.scorers.liquidity_execution import compute_liquidity_execution_score
        detail = _make_detail(lhb_count_20d=2)
        pool = _make_pool()
        axis = compute_liquidity_execution_score(detail, pool, enable_lhb_module=True)
        le4 = next(i for i in axis.items if i.name == "lhb_count_20d")
        assert le4.is_applicable is True
        assert le4.subscore is not None

    def test_le4_not_applicable_when_lhb_disabled(self):
        """LE4: enable_lhb_module=False → is_applicable=False."""
        from a_share_hot_screener.scorers.liquidity_execution import compute_liquidity_execution_score
        detail = _make_detail(lhb_count_20d=5)
        pool = _make_pool()
        axis = compute_liquidity_execution_score(detail, pool, enable_lhb_module=False)
        le4 = next(i for i in axis.items if i.name == "lhb_count_20d")
        assert le4.is_applicable is False
        # 当 lhb 不适用时，total applicable weight = 3+2+2 = 7.0
        # coverage 不受 LE4 影响

    def test_le4_lhb_high(self):
        """LE4: lhb_count_20d >= 3 → 1.0（离散映射）."""
        from a_share_hot_screener.scorers.liquidity_execution import compute_liquidity_execution_score
        detail = _make_detail(lhb_count_20d=5)
        pool = _make_pool()
        axis = compute_liquidity_execution_score(detail, pool, enable_lhb_module=True)
        le4 = next(i for i in axis.items if i.name == "lhb_count_20d")
        assert le4.subscore == pytest.approx(1.0)

    def test_le4_zero_lhb(self):
        """LE4: lhb_count_20d=0 → score=0.20."""
        from a_share_hot_screener.scorers.liquidity_execution import compute_liquidity_execution_score
        detail = _make_detail(lhb_count_20d=0)
        pool = _make_pool()
        axis = compute_liquidity_execution_score(detail, pool, enable_lhb_module=True)
        le4 = next(i for i in axis.items if i.name == "lhb_count_20d")
        assert le4.subscore == pytest.approx(0.20)

    def test_coverage_without_lhb(self):
        """关闭 LHB 时 applicable_weight=7.0，coverage 计算正确."""
        from a_share_hot_screener.scorers.liquidity_execution import compute_liquidity_execution_score
        # 只有 LE2(缺失) 不可用，其余可用
        detail = _make_detail(
            amount_avg_5d=3e8,
            float_market_cap=None,  # LE2 缺失, LE3 也缺失
            market_cap=None,        # LE3 也缺失
        )
        pool = _make_pool(pool_amount_avg_5d=[1e8, 2e8, 3e8])
        axis = compute_liquidity_execution_score(detail, pool, enable_lhb_module=False)
        # S10权重: LE1=9,LE2=6,LE3=4,LE4=1（disabled）
        # applicable: LE1(9) + LE2(6) + LE3(4) = 19, available: LE1(9) = 9
        assert axis.coverage == pytest.approx(9.0 / 19.0, abs=1e-3)

    def test_all_missing(self):
        from a_share_hot_screener.scorers.liquidity_execution import compute_liquidity_execution_score
        detail = _make_detail()  # 所有评分字段均 None
        pool = _make_pool()
        axis = compute_liquidity_execution_score(detail, pool, enable_lhb_module=False)
        assert axis.score is None
        assert axis.coverage == 0.0


# ════════════════════════════════════════════════════════
# TestRiskControlScore
# ════════════════════════════════════════════════════════

class TestRiskControlScore:

    def test_all_available_low_risk(self):
        """低风险股票得高分."""
        from a_share_hot_screener.scorers.risk_control import compute_risk_control_score
        detail = _make_detail(
            code="600519",
            latest_is_limit_board=False,   # RC1: 非一字板 → 1.0
            latest_pct_change=2.0,
            amp_norm_avg_5d=1.5,           # RC2: 1.5/10=0.15 < G=0.5 → 1.0
            abs_distance_to_ma10=0.02,     # RC3: abs=2% <= G=2% → 1.0（S10阈值）
            upper_reversal_count_5d=0,     # RC4: 无冲高回落 → 1.0
            return_3d=2.0,                 # RC5: 2.0/10=0.2 < G=0.8 → 1.0
            flags={                        # RC6~RC10: Session 22 新增
                "pledge_ratio_latest": 0.0,                      # RC6 → 1.0
                "restricted_shares_unlock_ratio_20d": 0.0,       # RC7 → 1.0
                "shareholder_net_reduction_ratio_3m": -5.0,      # RC8 → 1.0
                "net_holder_reduction_ratio_30d": -0.5,          # RC9 → 1.0
                "short_sell_ratio_change_5d": 0.0,               # RC10 → 1.0
                "is_margin_eligible": True,                      # RC10 applicable
            },
        )
        pool = _full_pool()
        axis = compute_risk_control_score(detail, pool)
        assert axis.score is not None
        assert 0.0 <= axis.score <= 1.0
        assert axis.coverage == pytest.approx(1.0)
        assert len(axis.items) == 10
        assert axis.axis_name == "risk_control_score"

    def test_rc1_not_limit_board(self):
        """RC1: limit_board_count_5d=0 → subscore=1.0（Session 10: 5日次数）."""
        from a_share_hot_screener.scorers.risk_control import compute_risk_control_score
        # limit_board_count_5d=0: 近5日无一字板 → 1.0
        detail = _make_detail(limit_board_count_5d=0, latest_is_limit_board=False, latest_pct_change=3.0)
        pool = _make_pool()
        axis = compute_risk_control_score(detail, pool)
        rc1 = next(i for i in axis.items if i.name == "limit_board_count_5d")
        assert rc1.subscore == pytest.approx(1.0)

    def test_rc1_limit_up(self):
        """RC1: limit_board_count_5d=1 → subscore=0.50（Session 10: 5日次数）."""
        from a_share_hot_screener.scorers.risk_control import compute_risk_control_score
        # 1次一字板 → 0.50
        detail = _make_detail(limit_board_count_5d=1, latest_is_limit_board=True, latest_pct_change=10.0)
        pool = _make_pool()
        axis = compute_risk_control_score(detail, pool)
        rc1 = next(i for i in axis.items if i.name == "limit_board_count_5d")
        assert rc1.subscore == pytest.approx(0.50)

    def test_rc1_limit_down(self):
        """RC1: limit_board_count_5d=3 → subscore=0.0（Session 10: 5日次数）."""
        from a_share_hot_screener.scorers.risk_control import compute_risk_control_score
        # ≥3次一字板 → 0.0
        detail = _make_detail(limit_board_count_5d=3, latest_is_limit_board=True, latest_pct_change=-10.0)
        pool = _make_pool()
        axis = compute_risk_control_score(detail, pool)
        rc1 = next(i for i in axis.items if i.name == "limit_board_count_5d")
        assert rc1.subscore == pytest.approx(0.0)

    def test_rc2_low_amp(self):
        """RC2: 低振幅 → 高分（安全）."""
        from a_share_hot_screener.scorers.risk_control import compute_risk_control_score
        detail = _make_detail(code="600519", amp_norm_avg_5d=1.0)  # 1.0/10=0.1 < G=0.5 → 1.0
        pool = _make_pool()
        axis = compute_risk_control_score(detail, pool)
        rc2 = next(i for i in axis.items if i.name == "amp_norm_5d_ratio")
        assert rc2.subscore == pytest.approx(1.0)

    def test_rc2_high_amp(self):
        """RC2: 振幅 >= B=1.8 * limit_pct → 0分."""
        from a_share_hot_screener.scorers.risk_control import compute_risk_control_score
        detail = _make_detail(code="600519", amp_norm_avg_5d=20.0)  # 20.0/10=2.0 >= B=1.8 → 0.0
        pool = _make_pool()
        axis = compute_risk_control_score(detail, pool)
        rc2 = next(i for i in axis.items if i.name == "amp_norm_5d_ratio")
        assert rc2.subscore == pytest.approx(0.0)

    def test_rc2_chuangye_limit_pct(self):
        """RC2: 创业板 limit_pct=20%."""
        from a_share_hot_screener.scorers.risk_control import compute_risk_control_score, _infer_limit_pct
        assert _infer_limit_pct("300750") == pytest.approx(20.0)
        # 振幅 = 5.0%，limit_pct=20% → norm=0.25，bad=0.8 → 高分
        detail = _make_detail(code="300750", amp_norm_avg_5d=5.0)
        pool = _make_pool()
        axis = compute_risk_control_score(detail, pool)
        rc2 = next(i for i in axis.items if i.name == "amp_norm_5d_ratio")
        assert rc2.subscore is not None
        assert rc2.subscore > 0.5

    def test_rc2_kechuang_limit_pct(self):
        """RC2: 科创板 limit_pct=20%."""
        from a_share_hot_screener.scorers.risk_control import _infer_limit_pct
        assert _infer_limit_pct("688001") == pytest.approx(20.0)

    def test_rc3_small_deviation(self):
        """RC3: abs(dist)*100 <= G=2% → 1.0（Session 10 收紧阈值）."""
        from a_share_hot_screener.scorers.risk_control import compute_risk_control_score
        # abs_dist = 0.02 → 2% <= G=2% → 1.0（S10阈值）
        detail = _make_detail(abs_distance_to_ma10=0.02)
        pool = _make_pool()
        axis = compute_risk_control_score(detail, pool)
        rc3 = next(i for i in axis.items if i.name == "abs_deviation_ma10")
        assert rc3.subscore == pytest.approx(1.0)

    def test_rc3_large_deviation(self):
        """RC3: abs(dist)*100 >= B=20% → 0分."""
        from a_share_hot_screener.scorers.risk_control import compute_risk_control_score
        detail = _make_detail(abs_distance_to_ma10=0.20)  # 偏离 20%
        pool = _make_pool()
        axis = compute_risk_control_score(detail, pool)
        rc3 = next(i for i in axis.items if i.name == "abs_deviation_ma10")
        assert rc3.subscore == pytest.approx(0.0)

    def test_rc3_negative_deviation(self):
        """RC3: 负值偏离取绝对值，同等风险."""
        from a_share_hot_screener.scorers.risk_control import compute_risk_control_score
        detail = _make_detail(abs_distance_to_ma10=-0.20)  # 跌破MA10 20%
        pool = _make_pool()
        axis = compute_risk_control_score(detail, pool)
        rc3 = next(i for i in axis.items if i.name == "abs_deviation_ma10")
        assert rc3.subscore == pytest.approx(0.0)

    def test_rc4_no_shadow(self):
        """RC4: upper_reversal_count_5d=0 → subscore=1.0."""
        from a_share_hot_screener.scorers.risk_control import compute_risk_control_score
        detail = _make_detail(upper_reversal_count_5d=0)
        pool = _make_pool()
        axis = compute_risk_control_score(detail, pool)
        rc4 = next(i for i in axis.items if i.name == "upper_reversal_count_5d")
        assert rc4.subscore == pytest.approx(1.0)

    def test_rc4_heavy_shadow(self):
        """RC4: upper_reversal_count_5d>=3 → clamp → subscore=0.0."""
        from a_share_hot_screener.scorers.risk_control import compute_risk_control_score
        detail = _make_detail(upper_reversal_count_5d=4)
        pool = _make_pool()
        axis = compute_risk_control_score(detail, pool)
        rc4 = next(i for i in axis.items if i.name == "upper_reversal_count_5d")
        assert rc4.subscore == pytest.approx(0.0)

    def test_rc5_low_return(self):
        """RC5: return_3d 接近 0 → 高分（norm < G=0.8）."""
        from a_share_hot_screener.scorers.risk_control import compute_risk_control_score
        # return_3d=0.0, limit_pct=10% → norm=0.0 <= G=0.8 → 1.0
        detail = _make_detail(code="600519", return_3d=0.0)
        pool = _make_pool()
        axis = compute_risk_control_score(detail, pool)
        rc5 = next(i for i in axis.items if i.name == "return_3d_vs_limit")
        assert rc5.subscore == pytest.approx(1.0)

    def test_rc5_high_return(self):
        """RC5: return_3d 超过 3.5 倍涨跌停 → 0分."""
        from a_share_hot_screener.scorers.risk_control import compute_risk_control_score
        # return_3d=35.0, limit_pct=10% → norm=3.5 >= B=3.5 → 0.0
        detail = _make_detail(code="600519", return_3d=35.0)
        pool = _make_pool()
        axis = compute_risk_control_score(detail, pool)
        rc5 = next(i for i in axis.items if i.name == "return_3d_vs_limit")
        assert rc5.subscore == pytest.approx(0.0)

    def test_rc5_negative_return_gets_high_score(self):
        """RC5: 近3日下跌（return_3d < 0）→ norm < G=0.8 → 1.0."""
        from a_share_hot_screener.scorers.risk_control import compute_risk_control_score
        detail = _make_detail(code="600519", return_3d=-5.0)
        pool = _make_pool()
        axis = compute_risk_control_score(detail, pool)
        rc5 = next(i for i in axis.items if i.name == "return_3d_vs_limit")
        # norm=-5/10=-0.5 <= G=0.8 → 1.0
        assert rc5.subscore == pytest.approx(1.0)

    def test_all_missing(self):
        from a_share_hot_screener.scorers.risk_control import compute_risk_control_score
        detail = _make_detail()
        pool = _make_pool()
        axis = compute_risk_control_score(detail, pool)
        assert axis.score is None
        assert axis.coverage == 0.0


# ════════════════════════════════════════════════════════
# TestComputeFlags
# ════════════════════════════════════════════════════════

class TestComputeFlags:

    def _make_full_detail(self) -> HotStockDetail:
        return _make_detail(
            code="600519",
            pct_change_1d=9.8,          # 接近10%涨停
            clv_latest=0.5,             # CLV=0.5（一字板特征）
            limit_board_count_5d=1,
            limit_up_count_5d=1,
            limit_up_count_10d=2,
            max_consecutive_limit_up_10d=1,
            strong_pool_entry_count_3d=1,
            lhb_count_20d=2,
            amount_avg_5d=3e8,
            float_market_cap=50e8,
            amp_norm_avg_5d=5.0,
            upper_shadow_count_5d=1,
            industry_heat_pctile_5d=0.75,
            concept_heat_pctile_5d=0.65,
            listing_days=100,           # < 120 → 次新股
        )

    def test_all_flags_present(self):
        from a_share_hot_screener.flags import compute_flags
        detail = self._make_full_detail()
        flags = compute_flags(detail, enable_lhb_module=True,
                              enable_concept_heat_module=True)
        expected_keys = [
            "limit_up_count_5d", "limit_up_count_10d", "max_consecutive_limit_up_10d",
            "strong_pool_entry_count_3d", "lhb_count_20d",
            "one_word_limit_up_latest", "one_word_limit_down_latest",
            "amount_avg_5d", "amp_norm_avg_5d", "upper_shadow_count_5d",
            "turnover_avg_5d", "industry_heat_pctile_5d", "concept_heat_pctile_5d",
            "new_stock_flag",
            "shareholder_net_reduction_ratio_3m", "shareholder_reduction_flag_3m",
            "restricted_shares_unlock_ratio_20d", "unlock_risk_flag_20d",
            "pledge_ratio_latest", "pledge_ratio_flag",
        ]
        for k in expected_keys:
            assert k in flags, f"flags 缺失: {k}"

    def test_flags_count(self):
        """flags 应包含 20 个 key."""
        from a_share_hot_screener.flags import compute_flags
        detail = self._make_full_detail()
        flags = compute_flags(detail, enable_lhb_module=True,
                              enable_concept_heat_module=True)
        assert len(flags) == 26  # 20 original + 6 Session 22 (moneyflow/holdertrade/margin/sector)

    def test_lhb_disabled(self):
        """enable_lhb_module=False → lhb_count_20d=None."""
        from a_share_hot_screener.flags import compute_flags
        detail = self._make_full_detail()
        flags = compute_flags(detail, enable_lhb_module=False)
        assert flags["lhb_count_20d"] is None

    def test_concept_disabled(self):
        """enable_concept_heat_module=False → concept_heat_pctile_5d=None."""
        from a_share_hot_screener.flags import compute_flags
        detail = self._make_full_detail()
        flags = compute_flags(detail, enable_concept_heat_module=False)
        assert flags["concept_heat_pctile_5d"] is None

    def test_new_stock_flag_true(self):
        """listing_days < 120 → new_stock_flag=True."""
        from a_share_hot_screener.flags import compute_flags
        detail = _make_detail(listing_days=100)
        flags = compute_flags(detail)
        assert flags["new_stock_flag"] is True

    def test_new_stock_flag_false(self):
        """listing_days >= 120 → new_stock_flag=False."""
        from a_share_hot_screener.flags import compute_flags
        detail = _make_detail(listing_days=200)
        flags = compute_flags(detail)
        assert flags["new_stock_flag"] is False

    def test_new_stock_flag_none(self):
        """listing_days=None → new_stock_flag=None."""
        from a_share_hot_screener.flags import compute_flags
        detail = _make_detail(listing_days=None)
        flags = compute_flags(detail)
        assert flags["new_stock_flag"] is None

    def test_turnover_approx_calc(self):
        """turnover_avg_5d = amount_avg_5d / float_market_cap * 100."""
        from a_share_hot_screener.flags import compute_flags
        detail = _make_detail(amount_avg_5d=3e8, float_market_cap=100e8)
        flags = compute_flags(detail)
        assert flags["turnover_avg_5d"] == pytest.approx(3.0, abs=1e-3)

    def test_turnover_none_when_missing(self):
        from a_share_hot_screener.flags import compute_flags
        detail = _make_detail(amount_avg_5d=None)
        flags = compute_flags(detail)
        assert flags["turnover_avg_5d"] is None

    def test_one_word_limit_up_detected(self):
        """疑似一字涨停检测."""
        from a_share_hot_screener.flags import compute_flags
        detail = _make_detail(
            code="600519",          # 10% 板
            pct_change_1d=9.9,      # 接近 10%
            clv_latest=0.5,         # 一字板
            limit_board_count_5d=1,
        )
        flags = compute_flags(detail)
        assert flags["one_word_limit_up_latest"] is True

    def test_one_word_limit_down_detected(self):
        """疑似一字跌停检测."""
        from a_share_hot_screener.flags import compute_flags
        detail = _make_detail(
            code="600519",
            pct_change_1d=-9.9,
            clv_latest=0.5,
            limit_board_count_5d=1,
        )
        flags = compute_flags(detail)
        assert flags["one_word_limit_down_latest"] is True

    def test_no_limit_up_normal_day(self):
        """普通日不触发一字板标记."""
        from a_share_hot_screener.flags import compute_flags
        detail = _make_detail(
            code="600519",
            pct_change_1d=3.0,      # 普通涨幅
            clv_latest=0.7,
            limit_board_count_5d=0,
        )
        flags = compute_flags(detail)
        assert flags["one_word_limit_up_latest"] is False

    def test_todo_fields_are_none(self):
        """尚未实现的 flag 字段应为 None（不编造）."""
        from a_share_hot_screener.flags import compute_flags
        detail = _make_detail()
        flags = compute_flags(detail)
        for key in [
            "shareholder_net_reduction_ratio_3m", "shareholder_reduction_flag_3m",
            "restricted_shares_unlock_ratio_20d", "unlock_risk_flag_20d",
            "pledge_ratio_latest", "pledge_ratio_flag",
        ]:
            assert flags[key] is None, f"{key} 应为 None"


# ════════════════════════════════════════════════════════
# TestScoringPoolS6
# ════════════════════════════════════════════════════════

class TestScoringPoolS6:

    def test_pool_amount_avg_5d_built(self):
        """ScoringPool.build 正确构建 pool_amount_avg_5d."""
        details = [
            HotStockDetail(code="A", passed_hard_filter=True, amount_avg_5d=3e8),
            HotStockDetail(code="B", passed_hard_filter=True, amount_avg_5d=5e8),
            HotStockDetail(code="C", passed_hard_filter=True, amount_avg_5d=None),  # 过滤
            HotStockDetail(code="D", passed_hard_filter=False, amount_avg_5d=10e8),  # 未通过
        ]
        pool = ScoringPool.build(details)
        assert sorted(pool.pool_amount_avg_5d) == pytest.approx([3e8, 5e8])

    def test_pool_amount_avg_5d_empty(self):
        """无有效 amount_avg_5d 时 pool_amount_avg_5d 为空."""
        details = [
            HotStockDetail(code="A", passed_hard_filter=True, amount_avg_5d=None),
        ]
        pool = ScoringPool.build(details)
        assert pool.pool_amount_avg_5d == []


# ════════════════════════════════════════════════════════
# TestFourAxisPipeline
# ════════════════════════════════════════════════════════

class TestFourAxisPipeline:

    def test_four_axes_written_to_detail(self):
        """_apply_four_axis_scores 正确写入全部四轴字段."""
        from a_share_hot_screener.scoring_aggregator import apply_four_axis_scores as _apply_four_axis_scores
        from a_share_hot_screener.logger import WarningsCollector

        detail = HotStockDetail(
            code="600519",
            passed_hard_filter=True,
            return_5d=8.0, return_10d=12.0,
            close_position_20d=0.75,
            abs_distance_to_ma10=0.04,
            volume_ratio_20d=1.8,
            clv_latest=0.7,
            amount_ratio_5d_to_20d=1.3,
            limit_up_count_10d=2,
            limit_up_source="zt_pool_em",
            strong_pool_entry_count_3d=1,
            strong_pool_source="zt_pool_strong_em",
            industry_heat_pctile_5d=0.8,
            industry_heat_source="hist_em_5d",
            amount_avg_5d=3e8,
            float_market_cap=50e8,
            lhb_count_20d=2,
            lhb_source="lhb_detail_em",
            limit_board_count_5d=0,
            amp_norm_avg_5d=3.0,
            upper_shadow_count_5d=1,
            return_3d=5.0,
        )
        pool = _full_pool()
        pool.pool_amount_avg_5d = [1e8, 2e8, 3e8, 5e8, 8e8]
        wc = WarningsCollector()
        _apply_four_axis_scores(detail, pool, None, wc)

        # 验证四轴均已填充
        assert detail.hot_theme_score is not None
        assert detail.hot_theme_coverage is not None
        assert isinstance(detail.hot_theme_subscores, dict)

        assert detail.trend_flow_score is not None
        assert detail.trend_flow_coverage is not None

        assert detail.liquidity_execution_score is not None
        assert detail.liquidity_execution_coverage is not None

        assert detail.risk_control_score is not None
        assert detail.risk_control_coverage is not None

    def test_all_scores_in_01_range(self):
        from a_share_hot_screener.scoring_aggregator import apply_four_axis_scores as _apply_four_axis_scores
        from a_share_hot_screener.logger import WarningsCollector

        detail = HotStockDetail(
            code="300750",
            passed_hard_filter=True,
            return_5d=5.0, return_10d=8.0,
            close_position_20d=0.6,
            abs_distance_to_ma10=0.03,
            volume_ratio_20d=1.5,
            clv_latest=0.6,
            amount_ratio_5d_to_20d=1.2,
            limit_up_count_10d=1,
            strong_pool_entry_count_3d=1,
            industry_heat_pctile_5d=0.6,
            amount_avg_5d=2e8,
            float_market_cap=30e8,
            lhb_count_20d=1,
            limit_board_count_5d=1,
            amp_norm_avg_5d=4.0,
            upper_shadow_count_5d=1,
            return_3d=3.0,
        )
        pool = _full_pool()
        pool.pool_amount_avg_5d = [1e8, 2e8, 3e8]
        wc = WarningsCollector()
        _apply_four_axis_scores(detail, pool, None, wc)

        for score in [
            detail.hot_theme_score,
            detail.trend_flow_score,
            detail.liquidity_execution_score,
            detail.risk_control_score,
        ]:
            if score is not None:
                assert 0.0 <= score <= 1.0

    def test_exception_in_scorer_does_not_crash(self):
        from a_share_hot_screener.scoring_aggregator import apply_four_axis_scores as _apply_four_axis_scores
        from a_share_hot_screener.logger import WarningsCollector
        from unittest.mock import patch

        detail = HotStockDetail(code="600519", passed_hard_filter=True)
        pool = ScoringPool()
        wc = WarningsCollector()

        with patch(
            "a_share_hot_screener.scoring_aggregator.compute_liquidity_execution_score",
            side_effect=RuntimeError("mock le error"),
        ):
            _apply_four_axis_scores(detail, pool, None, wc)

        assert detail.liquidity_execution_score is None
        warns = wc.get("600519")
        assert any("liquidity_execution_score" in w for w in warns)


# ════════════════════════════════════════════════════════
# TestSummaryFlagsFields
# ════════════════════════════════════════════════════════

class TestSummaryFlagsFields:

    def test_summary_has_all_s6_fields(self):
        """HotStockSummary.from_detail 正确继承所有 Session 6 新增字段."""
        import dataclasses
        summary_fields = {f.name for f in dataclasses.fields(HotStockSummary)}
        expected_s6_fields = [
            "one_word_limit_up_latest",
            "one_word_limit_down_latest",
            "turnover_avg_5d",
            "amp_norm_avg_5d",
            "upper_shadow_count_5d",
            "new_stock_flag",
            "hot_theme_coverage",
            "trend_flow_coverage",
            "liquidity_execution_coverage",
            "risk_control_coverage",
        ]
        for f in expected_s6_fields:
            assert f in summary_fields, f"HotStockSummary 缺失字段: {f}"

    def test_from_detail_maps_flags(self):
        """from_detail 正确从 detail.flags dict 映射 flags 值."""
        detail = HotStockDetail(
            code="600519",
            amp_norm_avg_5d=3.0,
            upper_shadow_count_5d=1,
            hot_theme_score=0.7,
            hot_theme_coverage=0.9,
            trend_flow_score=0.6,
            trend_flow_coverage=1.0,
            liquidity_execution_score=0.75,
            liquidity_execution_coverage=0.8,
            risk_control_score=0.8,
            risk_control_coverage=1.0,
            flags={
                "one_word_limit_up_latest": True,
                "one_word_limit_down_latest": False,
                "turnover_avg_5d": 2.5,
                "new_stock_flag": False,
            },
        )
        summary = HotStockSummary.from_detail(detail)
        assert summary.one_word_limit_up_latest is True
        assert summary.one_word_limit_down_latest is False
        assert summary.turnover_avg_5d == pytest.approx(2.5)
        assert summary.amp_norm_avg_5d == pytest.approx(3.0)
        assert summary.upper_shadow_count_5d == 1
        assert summary.new_stock_flag is False
        assert summary.hot_theme_coverage == pytest.approx(0.9)
        assert summary.liquidity_execution_score == pytest.approx(0.75)
        assert summary.risk_control_coverage == pytest.approx(1.0)

    def test_summary_serializable(self):
        """HotStockSummary 可以 dataclasses.asdict 序列化（用于 CSV 写入）."""
        import dataclasses, json
        detail = HotStockDetail(
            code="600519",
            flags={
                "one_word_limit_up_latest": True,
                "turnover_avg_5d": 2.5,
                "new_stock_flag": False,
            },
        )
        summary = HotStockSummary.from_detail(detail)
        d = dataclasses.asdict(summary)
        # 不应抛出异常
        json.dumps(d, default=str)
        assert "one_word_limit_up_latest" in d
        assert "turnover_avg_5d" in d
