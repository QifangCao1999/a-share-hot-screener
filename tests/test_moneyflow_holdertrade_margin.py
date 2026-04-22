"""Session 22 新增测试 — moneyflow/holdertrade/margin 三大数据源接入评分.

覆盖范围：
  1. TF6 主力净流入占比 — L=-5%/T=3%/H=10%, weight=6
  2. TF7 融资净买入占比 — L=-2%/T=1%/H=5%, weight=4, 非两融 is_applicable=False
  3. RC9 股东净减持占流通比 — G=0%/T=0.5%/B=2%, weight=3
  4. RC10 融券余额变化率 — G=10%/T=50%/B=100%, weight=1, 非两融 is_applicable=False
  5. moneyflow.py 计算模块单元测试
  6. holdertrade.py 计算模块单元测试
  7. margin.py 计算模块单元测试
  8. TF/RC 轴总权重验证
"""

from __future__ import annotations

import datetime as dt
from unittest.mock import MagicMock

import pandas as pd
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
# TF6: 主力净流入占比
# ════════════════════════════════════════════════════════

class TestTF6MainInflow:

    def test_tf6_strong_inflow(self):
        """TF6: ratio=12% >= H=10% → 1.0."""
        from a_share_hot_screener.scorers.trend_flow import compute_trend_flow_score
        detail = _make_detail(flags={"net_main_inflow_ratio_5d": 12.0})
        axis = compute_trend_flow_score(detail, _make_pool())
        tf6 = _get_item(axis, "net_main_inflow_ratio_5d")
        assert tf6.subscore == pytest.approx(1.0)
        assert tf6.weight == pytest.approx(6.0)

    def test_tf6_at_good(self):
        """TF6: ratio=10% = H → 1.0."""
        from a_share_hot_screener.scorers.trend_flow import compute_trend_flow_score
        detail = _make_detail(flags={"net_main_inflow_ratio_5d": 10.0})
        axis = compute_trend_flow_score(detail, _make_pool())
        tf6 = _get_item(axis, "net_main_inflow_ratio_5d")
        assert tf6.subscore == pytest.approx(1.0)

    def test_tf6_at_mid(self):
        """TF6: ratio=3% = T → 0.70."""
        from a_share_hot_screener.scorers.trend_flow import compute_trend_flow_score
        detail = _make_detail(flags={"net_main_inflow_ratio_5d": 3.0})
        axis = compute_trend_flow_score(detail, _make_pool())
        tf6 = _get_item(axis, "net_main_inflow_ratio_5d")
        assert tf6.subscore == pytest.approx(0.70, abs=1e-3)

    def test_tf6_net_outflow(self):
        """TF6: ratio=-5% <= L → 0.0."""
        from a_share_hot_screener.scorers.trend_flow import compute_trend_flow_score
        detail = _make_detail(flags={"net_main_inflow_ratio_5d": -5.0})
        axis = compute_trend_flow_score(detail, _make_pool())
        tf6 = _get_item(axis, "net_main_inflow_ratio_5d")
        assert tf6.subscore == pytest.approx(0.0)

    def test_tf6_none_data_unavailable(self):
        """TF6: None → is_data_available=False."""
        from a_share_hot_screener.scorers.trend_flow import compute_trend_flow_score
        detail = _make_detail(flags={})
        axis = compute_trend_flow_score(detail, _make_pool())
        tf6 = _get_item(axis, "net_main_inflow_ratio_5d")
        assert tf6.is_data_available is False


# ════════════════════════════════════════════════════════
# TF7: 融资净买入占比
# ════════════════════════════════════════════════════════

class TestTF7MarginBuy:

    def test_tf7_strong_margin_buy(self):
        """TF7: ratio=6% >= H=5% → 1.0, is_margin=True."""
        from a_share_hot_screener.scorers.trend_flow import compute_trend_flow_score
        detail = _make_detail(flags={
            "margin_buy_net_ratio_5d": 6.0,
            "is_margin_eligible": True,
        })
        axis = compute_trend_flow_score(detail, _make_pool())
        tf7 = _get_item(axis, "margin_buy_net_ratio_5d")
        assert tf7.subscore == pytest.approx(1.0)
        assert tf7.weight == pytest.approx(4.0)
        assert tf7.is_applicable is True

    def test_tf7_at_mid(self):
        """TF7: ratio=1% = T → 0.70."""
        from a_share_hot_screener.scorers.trend_flow import compute_trend_flow_score
        detail = _make_detail(flags={
            "margin_buy_net_ratio_5d": 1.0,
            "is_margin_eligible": True,
        })
        axis = compute_trend_flow_score(detail, _make_pool())
        tf7 = _get_item(axis, "margin_buy_net_ratio_5d")
        assert tf7.subscore == pytest.approx(0.70, abs=1e-3)

    def test_tf7_not_margin_not_applicable(self):
        """TF7: 非两融标的 → is_applicable=False，不影响评分."""
        from a_share_hot_screener.scorers.trend_flow import compute_trend_flow_score
        detail = _make_detail(flags={
            "margin_buy_net_ratio_5d": None,
            "is_margin_eligible": False,
        })
        axis = compute_trend_flow_score(detail, _make_pool())
        tf7 = _get_item(axis, "margin_buy_net_ratio_5d")
        assert tf7.is_applicable is False

    def test_tf7_no_flag_not_applicable(self):
        """TF7: flags 中无 is_margin_eligible → is_applicable=False."""
        from a_share_hot_screener.scorers.trend_flow import compute_trend_flow_score
        detail = _make_detail(flags={})
        axis = compute_trend_flow_score(detail, _make_pool())
        tf7 = _get_item(axis, "margin_buy_net_ratio_5d")
        assert tf7.is_applicable is False


# ════════════════════════════════════════════════════════
# RC9: 股东净减持
# ════════════════════════════════════════════════════════

class TestRC9HolderTrade:

    def test_rc9_no_reduction_full_score(self):
        """RC9: ratio=0% ≤ G=0% → 1.0."""
        from a_share_hot_screener.scorers.risk_control import compute_risk_control_score
        detail = _make_detail(flags={"net_holder_reduction_ratio_30d": 0.0})
        axis = compute_risk_control_score(detail, _make_pool())
        rc9 = _get_item(axis, "holder_trade_reduction_30d")
        assert rc9.subscore == pytest.approx(1.0)
        assert rc9.weight == pytest.approx(3.0)

    def test_rc9_net_increase_full_score(self):
        """RC9: ratio=-0.5%（净增持）≤ G=0% → 1.0."""
        from a_share_hot_screener.scorers.risk_control import compute_risk_control_score
        detail = _make_detail(flags={"net_holder_reduction_ratio_30d": -0.5})
        axis = compute_risk_control_score(detail, _make_pool())
        rc9 = _get_item(axis, "holder_trade_reduction_30d")
        assert rc9.subscore == pytest.approx(1.0)

    def test_rc9_at_mid(self):
        """RC9: ratio=0.5% = T → 0.70."""
        from a_share_hot_screener.scorers.risk_control import compute_risk_control_score
        detail = _make_detail(flags={"net_holder_reduction_ratio_30d": 0.5})
        axis = compute_risk_control_score(detail, _make_pool())
        rc9 = _get_item(axis, "holder_trade_reduction_30d")
        assert rc9.subscore == pytest.approx(0.70, abs=1e-3)

    def test_rc9_heavy_reduction(self):
        """RC9: ratio=2% ≥ B=2% → 0.0."""
        from a_share_hot_screener.scorers.risk_control import compute_risk_control_score
        detail = _make_detail(flags={"net_holder_reduction_ratio_30d": 2.0})
        axis = compute_risk_control_score(detail, _make_pool())
        rc9 = _get_item(axis, "holder_trade_reduction_30d")
        assert rc9.subscore == pytest.approx(0.0)

    def test_rc9_none_unavailable(self):
        """RC9: None → is_data_available=False."""
        from a_share_hot_screener.scorers.risk_control import compute_risk_control_score
        detail = _make_detail(flags={})
        axis = compute_risk_control_score(detail, _make_pool())
        rc9 = _get_item(axis, "holder_trade_reduction_30d")
        assert rc9.is_data_available is False


# ════════════════════════════════════════════════════════
# RC10: 融券余额变化率
# ════════════════════════════════════════════════════════

class TestRC10ShortSell:

    def test_rc10_low_change(self):
        """RC10: change=5% ≤ G=10% → 1.0."""
        from a_share_hot_screener.scorers.risk_control import compute_risk_control_score
        detail = _make_detail(flags={
            "short_sell_ratio_change_5d": 5.0,
            "is_margin_eligible": True,
        })
        axis = compute_risk_control_score(detail, _make_pool())
        rc10 = _get_item(axis, "short_sell_pressure_5d")
        assert rc10.subscore == pytest.approx(1.0)
        assert rc10.weight == pytest.approx(1.0)

    def test_rc10_at_mid(self):
        """RC10: change=50% = T → 0.70."""
        from a_share_hot_screener.scorers.risk_control import compute_risk_control_score
        detail = _make_detail(flags={
            "short_sell_ratio_change_5d": 50.0,
            "is_margin_eligible": True,
        })
        axis = compute_risk_control_score(detail, _make_pool())
        rc10 = _get_item(axis, "short_sell_pressure_5d")
        assert rc10.subscore == pytest.approx(0.70, abs=1e-3)

    def test_rc10_extreme_increase(self):
        """RC10: change=100% ≥ B → 0.0."""
        from a_share_hot_screener.scorers.risk_control import compute_risk_control_score
        detail = _make_detail(flags={
            "short_sell_ratio_change_5d": 100.0,
            "is_margin_eligible": True,
        })
        axis = compute_risk_control_score(detail, _make_pool())
        rc10 = _get_item(axis, "short_sell_pressure_5d")
        assert rc10.subscore == pytest.approx(0.0)

    def test_rc10_not_margin_not_applicable(self):
        """RC10: 非两融 → is_applicable=False."""
        from a_share_hot_screener.scorers.risk_control import compute_risk_control_score
        detail = _make_detail(flags={
            "short_sell_ratio_change_5d": 80.0,
            "is_margin_eligible": False,
        })
        axis = compute_risk_control_score(detail, _make_pool())
        rc10 = _get_item(axis, "short_sell_pressure_5d")
        assert rc10.is_applicable is False


# ════════════════════════════════════════════════════════
# TF 轴整体验证
# ════════════════════════════════════════════════════════

class TestTFAxisTotal:

    def test_tf_item_count_7(self):
        """TF 轴现在有 7 个 ScoreItem."""
        from a_share_hot_screener.scorers.trend_flow import compute_trend_flow_score
        detail = _make_detail(flags={"is_margin_eligible": True})
        axis = compute_trend_flow_score(detail, _make_pool())
        assert len(axis.items) == 7

    def test_tf_total_weight_margin_stock(self):
        """两融标的：TF 轴 7 项总权重 = 7+6+6+5+6+6+4 = 40."""
        from a_share_hot_screener.scorers.trend_flow import compute_trend_flow_score
        detail = _make_detail(
            close_position_20d=0.8,
            volume_ratio_20d=2.0,
            clv_latest=0.7,
            amount_ratio_5d_to_20d=1.5,
            latest_price=50.0, ma5=49.0, ma10=48.0, ma20=47.0,
            flags={
                "net_main_inflow_ratio_5d": 5.0,
                "margin_buy_net_ratio_5d": 2.0,
                "is_margin_eligible": True,
            },
        )
        axis = compute_trend_flow_score(detail, _make_pool())
        total_w = sum(i.weight for i in axis.items if i.is_applicable)
        assert total_w == pytest.approx(40.0)

    def test_tf_total_weight_non_margin(self):
        """非两融标的：TF7 不参与，总权重 = 36."""
        from a_share_hot_screener.scorers.trend_flow import compute_trend_flow_score
        detail = _make_detail(
            close_position_20d=0.8,
            flags={
                "net_main_inflow_ratio_5d": 5.0,
                "is_margin_eligible": False,
            },
        )
        axis = compute_trend_flow_score(detail, _make_pool())
        applicable_w = sum(i.weight for i in axis.items if i.is_applicable)
        assert applicable_w == pytest.approx(36.0)


# ════════════════════════════════════════════════════════
# RC 轴整体验证（含 RC9/RC10）
# ════════════════════════════════════════════════════════

class TestRCAxisWithNewItems:

    def test_rc_item_count_10(self):
        """RC 轴现在有 10 个 ScoreItem."""
        from a_share_hot_screener.scorers.risk_control import compute_risk_control_score
        detail = _make_detail(flags={"is_margin_eligible": True})
        axis = compute_risk_control_score(detail, _make_pool())
        assert len(axis.items) == 10

    def test_rc_total_weight_margin(self):
        """两融标的 RC 轴总权重 = 4+3+4+2+2+2+2+1+3+1 = 24."""
        from a_share_hot_screener.scorers.risk_control import compute_risk_control_score
        detail = _make_detail(
            limit_board_count_5d=0,
            amp_norm_avg_5d=3.0,
            abs_distance_to_ma10=0.03,
            upper_shadow_count_5d=0,
            return_3d=5.0,
            flags={
                "pledge_ratio_latest": 5.0,
                "restricted_shares_unlock_ratio_20d": 0.5,
                "shareholder_net_reduction_ratio_3m": 2.0,
                "net_holder_reduction_ratio_30d": 0.1,
                "short_sell_ratio_change_5d": 5.0,
                "is_margin_eligible": True,
            },
        )
        axis = compute_risk_control_score(detail, _make_pool())
        applicable_w = sum(i.weight for i in axis.items if i.is_applicable)
        assert applicable_w == pytest.approx(24.0)

    def test_rc_total_weight_non_margin(self):
        """非两融标的 RC 轴 RC10 不参与，总权重 = 23."""
        from a_share_hot_screener.scorers.risk_control import compute_risk_control_score
        detail = _make_detail(
            limit_board_count_5d=0,
            flags={
                "pledge_ratio_latest": 5.0,
                "restricted_shares_unlock_ratio_20d": 0.5,
                "shareholder_net_reduction_ratio_3m": 2.0,
                "net_holder_reduction_ratio_30d": 0.1,
                "is_margin_eligible": False,
            },
        )
        axis = compute_risk_control_score(detail, _make_pool())
        applicable_w = sum(i.weight for i in axis.items if i.is_applicable)
        assert applicable_w == pytest.approx(23.0)


# ════════════════════════════════════════════════════════
# moneyflow.py 单元测试
# ════════════════════════════════════════════════════════

class TestMoneyFlowCompute:

    def test_normal_calculation(self):
        """正常5天数据 → 计算 ratio."""
        from a_share_hot_screener.moneyflow import compute_moneyflow_ratio
        mock_client = MagicMock()
        mock_client.get_moneyflow.return_value = pd.DataFrame([
            {"trade_date": f"2026041{i}", "buy_sm_amount": 100, "buy_md_amount": 200,
             "buy_lg_amount": 300, "buy_elg_amount": 400,
             "sell_sm_amount": 100, "sell_md_amount": 200,
             "sell_lg_amount": 250, "sell_elg_amount": 350}
            for i in range(5)
        ])
        result = compute_moneyflow_ratio("600519", mock_client, "600519.SH", "20260410", "20260420")
        assert result.net_main_inflow_ratio_5d is not None
        assert result.data_days == 5
        assert result.source == "tushare_moneyflow"
        # net_main = (300+400-250-350)*5 = 100*5 = 500
        # total_buy = (100+200+300+400)*5 = 1000*5 = 5000
        # ratio = 500/5000*100 = 10%
        assert result.net_main_inflow_ratio_5d == pytest.approx(10.0, abs=0.1)

    def test_empty_returns_none(self):
        """空数据 → None."""
        from a_share_hot_screener.moneyflow import compute_moneyflow_ratio
        mock_client = MagicMock()
        mock_client.get_moneyflow.return_value = pd.DataFrame()
        result = compute_moneyflow_ratio("600519", mock_client, "600519.SH", "20260410", "20260420")
        assert result.net_main_inflow_ratio_5d is None

    def test_api_exception_handled(self):
        """API 异常 → 正常返回、有 warning."""
        from a_share_hot_screener.moneyflow import compute_moneyflow_ratio
        mock_client = MagicMock()
        mock_client.get_moneyflow.side_effect = Exception("timeout")
        warns = []
        result = compute_moneyflow_ratio("600519", mock_client, "600519.SH", "20260410", "20260420", warnings=warns)
        assert result.net_main_inflow_ratio_5d is None
        assert len(warns) > 0


# ════════════════════════════════════════════════════════
# holdertrade.py 单元测试
# ════════════════════════════════════════════════════════

class TestHolderTradeCompute:

    def test_no_trades_zero(self):
        """无增减持记录 → 0.0."""
        from a_share_hot_screener.holdertrade import compute_holder_trade_reduction
        mock_client = MagicMock()
        mock_client.get_stk_holdertrade.return_value = pd.DataFrame()
        result = compute_holder_trade_reduction(
            "600519", mock_client, dt.date(2026, 4, 20), "600519.SH"
        )
        assert result.net_holder_reduction_ratio_30d == pytest.approx(0.0)
        assert result.source == "tushare_stk_holdertrade"

    def test_net_reduction(self):
        """有减持无增持 → 正值."""
        from a_share_hot_screener.holdertrade import compute_holder_trade_reduction
        mock_client = MagicMock()
        mock_client.get_stk_holdertrade.return_value = pd.DataFrame([
            {"ann_date": "20260415", "in_de": "DE", "change_ratio": 0.5},
            {"ann_date": "20260418", "in_de": "DE", "change_ratio": 0.3},
        ])
        result = compute_holder_trade_reduction(
            "600519", mock_client, dt.date(2026, 4, 20), "600519.SH"
        )
        assert result.net_holder_reduction_ratio_30d == pytest.approx(0.8, abs=0.01)
        assert result.decrease_count == 2

    def test_net_increase(self):
        """有增持无减持 → 负值."""
        from a_share_hot_screener.holdertrade import compute_holder_trade_reduction
        mock_client = MagicMock()
        mock_client.get_stk_holdertrade.return_value = pd.DataFrame([
            {"ann_date": "20260415", "in_de": "IN", "change_ratio": 1.0},
        ])
        result = compute_holder_trade_reduction(
            "600519", mock_client, dt.date(2026, 4, 20), "600519.SH"
        )
        assert result.net_holder_reduction_ratio_30d == pytest.approx(-1.0, abs=0.01)
        assert result.increase_count == 1

    def test_old_records_filtered(self):
        """超过30天的记录被过滤."""
        from a_share_hot_screener.holdertrade import compute_holder_trade_reduction
        mock_client = MagicMock()
        mock_client.get_stk_holdertrade.return_value = pd.DataFrame([
            {"ann_date": "20260201", "in_de": "DE", "change_ratio": 5.0},  # 超60天
            {"ann_date": "20260415", "in_de": "DE", "change_ratio": 0.3},  # 5天内
        ])
        result = compute_holder_trade_reduction(
            "600519", mock_client, dt.date(2026, 4, 20), "600519.SH"
        )
        assert result.net_holder_reduction_ratio_30d == pytest.approx(0.3, abs=0.01)
        assert result.decrease_count == 1


# ════════════════════════════════════════════════════════
# margin.py 单元测试
# ════════════════════════════════════════════════════════

class TestMarginCompute:

    def test_not_margin_eligible(self):
        """空数据 = 非两融标的."""
        from a_share_hot_screener.margin import compute_margin_metrics
        mock_client = MagicMock()
        mock_client.get_margin_detail.return_value = pd.DataFrame()
        result = compute_margin_metrics("600519", mock_client, "600519.SH", "20260410", "20260420")
        assert result.is_margin_eligible is False
        assert result.margin_buy_net_ratio_5d is None

    def test_normal_margin_data(self):
        """正常两融数据 → 计算 TF7 和 RC10."""
        from a_share_hot_screener.margin import compute_margin_metrics
        mock_client = MagicMock()
        mock_client.get_margin_detail.return_value = pd.DataFrame([
            {"trade_date": "20260414", "rzmre": 1e8, "rzche": 5e7, "rqye": 1e7},
            {"trade_date": "20260415", "rzmre": 1.2e8, "rzche": 6e7, "rqye": 1.1e7},
            {"trade_date": "20260416", "rzmre": 1.1e8, "rzche": 5.5e7, "rqye": 1.2e7},
            {"trade_date": "20260417", "rzmre": 1.3e8, "rzche": 7e7, "rqye": 1.15e7},
            {"trade_date": "20260418", "rzmre": 1.0e8, "rzche": 4e7, "rqye": 1.3e7},
        ])
        result = compute_margin_metrics(
            "600519", mock_client, "600519.SH", "20260410", "20260420",
            amount_avg_5d=5e8,
        )
        assert result.is_margin_eligible is True
        assert result.margin_buy_net_ratio_5d is not None
        assert result.short_sell_ratio_change_5d is not None
        assert result.source == "tushare_margin_detail"
        # rqye from 1e7 to 1.3e7 = 30% increase
        assert result.short_sell_ratio_change_5d == pytest.approx(30.0, abs=1.0)

    def test_api_exception_handled(self):
        """API 异常 → 正常返回."""
        from a_share_hot_screener.margin import compute_margin_metrics
        mock_client = MagicMock()
        mock_client.get_margin_detail.side_effect = Exception("timeout")
        result = compute_margin_metrics("600519", mock_client, "600519.SH", "20260410", "20260420")
        assert result.is_margin_eligible is False


# ════════════════════════════════════════════════════════
# tushare_client 新 API 方法存在性验证
# ════════════════════════════════════════════════════════

class TestTushareClientNewAPIs:

    def test_get_moneyflow_exists(self):
        """TushareClient 有 get_moneyflow 方法."""
        from a_share_hot_screener.clients.tushare_client import TushareClient
        assert hasattr(TushareClient, "get_moneyflow")

    def test_get_stk_holdertrade_exists(self):
        """TushareClient 有 get_stk_holdertrade 方法."""
        from a_share_hot_screener.clients.tushare_client import TushareClient
        assert hasattr(TushareClient, "get_stk_holdertrade")

    def test_get_margin_detail_exists(self):
        """TushareClient 有 get_margin_detail 方法."""
        from a_share_hot_screener.clients.tushare_client import TushareClient
        assert hasattr(TushareClient, "get_margin_detail")

    def test_prefetch_flow_data_exists(self):
        """TushareClient 有 prefetch_flow_data 方法."""
        from a_share_hot_screener.clients.tushare_client import TushareClient
        assert hasattr(TushareClient, "prefetch_flow_data")
