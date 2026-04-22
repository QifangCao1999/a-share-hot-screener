"""P0-1~P0-5 修复验证测试."""

from __future__ import annotations

import dataclasses
import datetime as dt
from unittest.mock import MagicMock, patch

import pytest

from a_share_hot_screener.models import (
    HotStockDetail,
    HotStockSummary,
    RejectedRecord,
)
from a_share_hot_screener.price_features import PriceFeatures
from a_share_hot_screener.stock_processor import apply_price_features
from a_share_hot_screener.flags import compute_flags
from a_share_hot_screener.hard_filters import apply_hard_filters


# ════════════════════════════════════════════════════════
# P0-1: SpotUniverse 使用 trade_date_used
# ════════════════════════════════════════════════════════


class TestP01TradeDate:
    """P0-1: 确认 pipeline Step 1 使用解析后的 trade_date_used."""

    def test_spot_universe_called_with_trade_date_used(self):
        """当 run_date 是周末时，SpotUniverse.load 应该收到解析后的交易日."""
        import inspect
        from a_share_hot_screener import pipeline

        source = inspect.getsource(pipeline.Stage1HotPipeline._run_data_collection)
        # 确认不再使用 cfg.run_date_str 调用 spot_universe.load
        assert "self._spot_universe.load(cfg.run_date_str)" not in source
        # 确认使用 self._trade_date_str
        assert "self._spot_universe.load(self._trade_date_str)" in source


# ════════════════════════════════════════════════════════
# P0-2: prefetch 窗口与消费者一致
# ════════════════════════════════════════════════════════


class TestP02PrefetchWindow:
    """P0-2: 确认 pipeline Step 2.8 预取窗口与 stock_processor 消费窗口一致."""

    def test_prefetch_window_matches_consumer(self):
        """pipeline 和 stock_processor 都应使用 15 天窗口."""
        import inspect
        from a_share_hot_screener import pipeline, stock_processor

        # D6: 预取窗口移到 _prefetch_and_enrich_risk_flow
        prefetch_src = inspect.getsource(pipeline.Stage1HotPipeline._prefetch_and_enrich_risk_flow)
        # 确认使用 days=15（而非 days=60）
        assert "timedelta(days=15)" in prefetch_src
        assert "timedelta(days=60)" not in prefetch_src

        # stock_processor Step 6.9: 消费窗口 (现在在 enrich_risk_flow_data 中)
        sp_src = inspect.getsource(stock_processor.enrich_risk_flow_data)
        assert "timedelta(days=15)" in sp_src


# ════════════════════════════════════════════════════════
# P0-3: hard filter rejected 进入 rejected 体系
# ════════════════════════════════════════════════════════


class TestP03HardFilterRejected:
    """P0-3: hard filter 失败的股票应进入 rejected 列表."""

    def test_hard_filter_rejected_record_created(self):
        """确认 pipeline 中有将 hard filter fail 加入 self._rejected 的逻辑."""
        import inspect
        from a_share_hot_screener import pipeline

        source = inspect.getsource(pipeline.Stage1HotPipeline.run)
        assert "reject_stage=\"hard_filter\"" in source

    def test_rejected_record_fields(self):
        """RejectedRecord 应包含 hard_filter 所需的所有字段."""
        rec = RejectedRecord(
            code="000001",
            name="平安银行",
            reject_stage="hard_filter",
            reject_reason="insufficient_amount_avg_5d",
            reject_detail="amount_avg_5d=50000000 < 100000000",
            warnings="[data_warn] 某个警告",
        )
        assert rec.reject_stage == "hard_filter"
        assert rec.reject_reason == "insufficient_amount_avg_5d"
        assert "50000000" in rec.reject_detail
        assert rec.name == "平安银行"


# ════════════════════════════════════════════════════════
# P0-4: pct_change_1d / amount_1d 字段传播
# ════════════════════════════════════════════════════════


class TestP04FieldPropagation:
    """P0-4: price_features 的 pct_change / amount 应回填到 detail 标准字段."""

    def test_pct_change_1d_propagated(self):
        """apply_price_features 后 detail.pct_change_1d 不应为 None."""
        detail = HotStockDetail(code="600519")
        assert detail.pct_change_1d is None

        feat = PriceFeatures()
        feat.latest_pct_change = 5.23
        feat.latest_amount = 1_500_000_000.0

        apply_price_features(detail, feat)
        assert detail.pct_change_1d == 5.23

    def test_amount_1d_propagated(self):
        """apply_price_features 后 detail.amount_1d 不应为 None."""
        detail = HotStockDetail(code="600519")
        assert detail.amount_1d is None

        feat = PriceFeatures()
        feat.latest_amount = 2_000_000_000.0

        apply_price_features(detail, feat)
        assert detail.amount_1d == 2_000_000_000.0

    def test_no_overwrite_existing_values(self):
        """如果 detail 已有值，不应被 price_features 覆盖."""
        detail = HotStockDetail(code="600519", pct_change_1d=3.0, amount_1d=1e9)
        feat = PriceFeatures()
        feat.latest_pct_change = 5.0
        feat.latest_amount = 2e9

        apply_price_features(detail, feat)
        assert detail.pct_change_1d == 3.0  # 保持原值
        assert detail.amount_1d == 1e9      # 保持原值

    def test_none_feat_does_not_crash(self):
        """feat 字段为 None 时不应崩溃."""
        detail = HotStockDetail(code="600519")
        feat = PriceFeatures()
        feat.latest_pct_change = None
        feat.latest_amount = None

        apply_price_features(detail, feat)
        assert detail.pct_change_1d is None
        assert detail.amount_1d is None

    def test_flags_use_propagated_pct_change(self):
        """flags 中的一字涨停检测应能读取到 pct_change_1d."""
        detail = HotStockDetail(code="600519")
        detail.pct_change_1d = 9.98  # 接近 10% 涨停
        detail.limit_board_count_5d = 1
        detail.clv_latest = 0.50
        detail.board_limit_pct = 10.0

        flags = compute_flags(detail)
        # 应该能计算出一字涨停标记（而非 None）
        assert flags["one_word_limit_up_latest"] is not None

    def test_hard_filter_receives_amount_1d(self):
        """hard filter 中的停牌检测应能接收到 amount_1d."""
        from a_share_hot_screener.config import HotScreenerConfig

        cfg = HotScreenerConfig(
            tushare_token="test",
            run_date=dt.date(2026, 4, 22),
            stock_codes=[],
            output_dir="",
        )
        result = apply_hard_filters(
            config=cfg,
            name="测试股票",
            industry="软件服务",
            ipo_date="20200101",
            listing_days=2000,
            latest_price=10.0,
            latest_volume=1000000,
            amount_1d=0,  # 零成交额 → 停牌
            amount_avg_5d=200_000_000.0,
            float_market_cap=2_000_000_000.0,
        )
        # amount_1d=0 应触发停牌检测
        assert not result.passed or any("suspension" in r for r in result.fail_reasons)


# ════════════════════════════════════════════════════════
# P0-5: summary 暴露更多 flags
# ════════════════════════════════════════════════════════


class TestP05SummaryFlags:
    """P0-5: summary 应包含资金流/股东/融资/板块轮动等关键 flags."""

    def test_summary_has_new_flag_fields(self):
        """HotStockSummary 应包含新增的 flags 字段."""
        fields = {f.name for f in dataclasses.fields(HotStockSummary)}
        expected_new_fields = [
            "net_main_inflow_ratio_5d",
            "net_holder_reduction_ratio_30d",
            "margin_buy_net_ratio_5d",
            "short_sell_ratio_change_5d",
            "is_margin_eligible",
            "sector_momentum_signal",
        ]
        for fname in expected_new_fields:
            assert fname in fields, f"HotStockSummary 缺少字段: {fname}"

    def test_from_detail_propagates_new_flags(self):
        """from_detail 应将 flags 中的新字段传播到 summary."""
        detail = HotStockDetail(code="600519", name="贵州茅台")
        detail.flags = {
            "net_main_inflow_ratio_5d": 0.05,
            "net_holder_reduction_ratio_30d": -0.02,
            "margin_buy_net_ratio_5d": 0.03,
            "short_sell_ratio_change_5d": 0.1,
            "is_margin_eligible": True,
            "sector_momentum_signal": "rotate_in",
            # 其他必要的 flags
            "one_word_limit_up_latest": None,
            "one_word_limit_down_latest": None,
            "turnover_avg_5d": None,
            "new_stock_flag": False,
            "shareholder_net_reduction_ratio_3m": None,
            "shareholder_reduction_flag_3m": None,
            "restricted_shares_unlock_ratio_20d": None,
            "unlock_risk_flag_20d": None,
            "pledge_ratio_latest": None,
            "pledge_ratio_flag": None,
        }

        summary = HotStockSummary.from_detail(detail)
        assert summary.net_main_inflow_ratio_5d == 0.05
        assert summary.net_holder_reduction_ratio_30d == -0.02
        assert summary.margin_buy_net_ratio_5d == 0.03
        assert summary.short_sell_ratio_change_5d == 0.1
        assert summary.is_margin_eligible is True
        assert summary.sector_momentum_signal == "rotate_in"

    def test_from_detail_handles_missing_flags(self):
        """当 flags 中没有新字段时，summary 应有默认值."""
        detail = HotStockDetail(code="600519", name="贵州茅台")
        detail.flags = {}

        summary = HotStockSummary.from_detail(detail)
        assert summary.net_main_inflow_ratio_5d is None
        assert summary.is_margin_eligible is None
        # flags.get("sector_momentum_signal") 返回 None，而非默认 ""
        assert summary.sector_momentum_signal is None or summary.sector_momentum_signal == ""

    def test_flags_merge_preserves_step6_values(self):
        """Step 7.5 flags merge 不应丢失 Step 6 写入的自定义 flags."""
        detail = HotStockDetail(code="600519", name="贵州茅台")
        # 模拟 Step 6 写入
        detail.flags["net_main_inflow_ratio_5d"] = 0.08
        detail.flags["custom_step6_flag"] = "test_value"

        # 模拟 Step 7.5 compute_flags + merge
        computed = compute_flags(detail)
        detail.flags.update(computed)

        # Step 6 写入的值应保留（compute_flags 会从 detail.flags.get 读取并写回）
        assert detail.flags.get("net_main_inflow_ratio_5d") == 0.08
        # 自定义 flag 不应被丢失（因为用的是 update 而非覆盖）
        assert detail.flags.get("custom_step6_flag") == "test_value"
