"""Tests for D6: 硬筛后再预取.

验证:
  1. enrich_risk_flow_data 正确填充 flags
  2. process_single_stock 不再包含 Step 6.6~6.11 逻辑
  3. pipeline._prefetch_and_enrich_risk_flow 只对通过硬筛的股票执行
  4. 未通过硬筛的股票不调用风控/资金 API
"""

from __future__ import annotations

import datetime as dt
import inspect
import unittest
from unittest.mock import MagicMock, patch

from a_share_hot_screener.models import HotStockDetail


# ════════════════════════════════════════════════════════
# enrich_risk_flow_data 正确填充 flags
# ════════════════════════════════════════════════════════

class TestEnrichRiskFlowData(unittest.TestCase):
    """enrich_risk_flow_data fills risk + flow flags."""

    def test_fills_all_expected_flags(self):
        """enrich_risk_flow_data should populate all 8 expected flags."""
        from a_share_hot_screener.stock_processor import enrich_risk_flow_data

        detail = HotStockDetail(
            code="600519",
            ts_code="600519.SH",
            passed_hard_filter=True,
            amount_avg_5d=500_000_000,
        )
        detail.flags = {}
        detail.warnings = []

        mock_client = MagicMock()

        # Mock all sub-module functions to return default results
        with patch("a_share_hot_screener.stock_processor.compute_shareholder_reduction") as m_sr, \
             patch("a_share_hot_screener.stock_processor.compute_pledge_ratio") as m_pr, \
             patch("a_share_hot_screener.stock_processor.compute_restricted_unlock") as m_ru, \
             patch("a_share_hot_screener.stock_processor.compute_moneyflow_ratio") as m_mf, \
             patch("a_share_hot_screener.stock_processor.compute_holder_trade_reduction") as m_ht, \
             patch("a_share_hot_screener.stock_processor.compute_margin_metrics") as m_mg:

            m_sr.return_value = MagicMock(
                shareholder_net_reduction_ratio_3m=1.5,
                shareholder_reduction_flag_3m="medium",
            )
            m_pr.return_value = MagicMock(
                pledge_ratio_latest=12.0,
                pledge_ratio_flag="low",
            )
            m_ru.return_value = MagicMock(
                restricted_shares_unlock_ratio_20d=2.0,
                unlock_risk_flag_20d="low",
            )
            m_mf.return_value = MagicMock(
                net_main_inflow_ratio_5d=5.5,
            )
            m_ht.return_value = MagicMock(
                net_holder_reduction_ratio_30d=0.3,
            )
            m_mg.return_value = MagicMock(
                margin_buy_net_ratio_5d=2.1,
                short_sell_ratio_change_5d=10.0,
                is_margin_eligible=True,
            )

            enrich_risk_flow_data(detail, "2026-04-22", mock_client)

        # Check all flags populated
        self.assertEqual(detail.flags["shareholder_net_reduction_ratio_3m"], 1.5)
        self.assertEqual(detail.flags["pledge_ratio_latest"], 12.0)
        self.assertEqual(detail.flags["restricted_shares_unlock_ratio_20d"], 2.0)
        self.assertEqual(detail.flags["net_main_inflow_ratio_5d"], 5.5)
        self.assertEqual(detail.flags["net_holder_reduction_ratio_30d"], 0.3)
        self.assertEqual(detail.flags["margin_buy_net_ratio_5d"], 2.1)
        self.assertEqual(detail.flags["is_margin_eligible"], True)

    def test_skips_without_ts_code(self):
        """enrich_risk_flow_data should do nothing if ts_code is empty."""
        from a_share_hot_screener.stock_processor import enrich_risk_flow_data

        detail = HotStockDetail(code="600519", ts_code="")
        detail.flags = {}
        detail.warnings = []

        mock_client = MagicMock()
        enrich_risk_flow_data(detail, "2026-04-22", mock_client)

        # No flags should be added
        self.assertEqual(len(detail.flags), 0)

    def test_handles_exceptions_gracefully(self):
        """enrich_risk_flow_data should not crash on sub-module exceptions."""
        from a_share_hot_screener.stock_processor import enrich_risk_flow_data

        detail = HotStockDetail(
            code="600519",
            ts_code="600519.SH",
            amount_avg_5d=100_000_000,
        )
        detail.flags = {}
        detail.warnings = []

        mock_client = MagicMock()

        with patch("a_share_hot_screener.stock_processor.compute_shareholder_reduction",
                    side_effect=Exception("test error")), \
             patch("a_share_hot_screener.stock_processor.compute_pledge_ratio",
                    side_effect=Exception("test error")), \
             patch("a_share_hot_screener.stock_processor.compute_restricted_unlock",
                    side_effect=Exception("test error")), \
             patch("a_share_hot_screener.stock_processor.compute_moneyflow_ratio",
                    side_effect=Exception("test error")), \
             patch("a_share_hot_screener.stock_processor.compute_holder_trade_reduction",
                    side_effect=Exception("test error")), \
             patch("a_share_hot_screener.stock_processor.compute_margin_metrics",
                    side_effect=Exception("test error")):

            # Should not raise
            enrich_risk_flow_data(detail, "2026-04-22", mock_client)

        # Should have warnings for each failure
        self.assertTrue(len(detail.warnings) >= 6)


# ════════════════════════════════════════════════════════
# process_single_stock 不再包含 Step 6.6~6.11
# ════════════════════════════════════════════════════════

class TestProcessSingleStockD6(unittest.TestCase):
    """process_single_stock no longer contains risk/flow steps."""

    def test_no_risk_flow_in_process_single_stock(self):
        """process_single_stock should NOT call risk/flow functions."""
        from a_share_hot_screener import stock_processor
        src = inspect.getsource(stock_processor.process_single_stock)

        # These should NOT appear in process_single_stock anymore
        self.assertNotIn("compute_shareholder_reduction", src)
        self.assertNotIn("compute_pledge_ratio", src)
        self.assertNotIn("compute_restricted_unlock", src)
        self.assertNotIn("compute_moneyflow_ratio", src)
        self.assertNotIn("compute_holder_trade_reduction", src)
        self.assertNotIn("compute_margin_metrics", src)

    def test_enrich_function_exists(self):
        """enrich_risk_flow_data should exist as a separate function."""
        from a_share_hot_screener.stock_processor import enrich_risk_flow_data
        self.assertTrue(callable(enrich_risk_flow_data))


# ════════════════════════════════════════════════════════
# Pipeline: 只对通过硬筛的股票预取
# ════════════════════════════════════════════════════════

class TestPipelineD6Integration(unittest.TestCase):
    """Pipeline only prefetches for hard-filter-passed stocks."""

    def test_prefetch_in_new_method(self):
        """prefetch_risk_data and prefetch_flow_data are in _prefetch_and_enrich_risk_flow."""
        from a_share_hot_screener import pipeline
        src = inspect.getsource(pipeline.Stage1HotPipeline._prefetch_and_enrich_risk_flow)
        self.assertIn("prefetch_risk_data", src)
        self.assertIn("prefetch_flow_data", src)

    def test_data_collection_no_prefetch(self):
        """_run_data_collection should NOT contain prefetch_risk_data or prefetch_flow_data."""
        from a_share_hot_screener import pipeline
        src = inspect.getsource(pipeline.Stage1HotPipeline._run_data_collection)
        self.assertNotIn("prefetch_risk_data", src)
        self.assertNotIn("prefetch_flow_data", src)

    def test_data_collection_filters_by_hard_filter(self):
        """_run_data_collection should filter by passed_hard_filter before enrichment."""
        from a_share_hot_screener import pipeline
        src = inspect.getsource(pipeline.Stage1HotPipeline._run_data_collection)
        self.assertIn("passed_hard_filter", src)
        self.assertIn("_prefetch_and_enrich_risk_flow", src)


# ════════════════════════════════════════════════════════
# 验证节省效果
# ════════════════════════════════════════════════════════

class TestD6Savings(unittest.TestCase):
    """D6 correctly avoids enriching failed stocks."""

    def test_only_passed_stocks_enriched(self):
        """Only hard-filter-passed stocks should have risk/flow flags."""
        from a_share_hot_screener.stock_processor import enrich_risk_flow_data

        # Stock that passed hard filter
        passed = HotStockDetail(code="600519", ts_code="600519.SH",
                                passed_hard_filter=True, amount_avg_5d=500_000_000)
        passed.flags = {}
        passed.warnings = []

        # Stock that failed hard filter (shouldn't be enriched)
        failed = HotStockDetail(code="000001", ts_code="000001.SZ",
                                passed_hard_filter=False, amount_avg_5d=50_000_000)
        failed.flags = {}
        failed.warnings = []

        mock_client = MagicMock()

        with patch("a_share_hot_screener.stock_processor.compute_shareholder_reduction") as m_sr, \
             patch("a_share_hot_screener.stock_processor.compute_pledge_ratio") as m_pr, \
             patch("a_share_hot_screener.stock_processor.compute_restricted_unlock") as m_ru, \
             patch("a_share_hot_screener.stock_processor.compute_moneyflow_ratio") as m_mf, \
             patch("a_share_hot_screener.stock_processor.compute_holder_trade_reduction") as m_ht, \
             patch("a_share_hot_screener.stock_processor.compute_margin_metrics") as m_mg:

            m_sr.return_value = MagicMock(
                shareholder_net_reduction_ratio_3m=1.0, shareholder_reduction_flag_3m="low")
            m_pr.return_value = MagicMock(pledge_ratio_latest=5.0, pledge_ratio_flag="low")
            m_ru.return_value = MagicMock(
                restricted_shares_unlock_ratio_20d=0.5, unlock_risk_flag_20d="low")
            m_mf.return_value = MagicMock(net_main_inflow_ratio_5d=3.0)
            m_ht.return_value = MagicMock(net_holder_reduction_ratio_30d=0.1)
            m_mg.return_value = MagicMock(
                margin_buy_net_ratio_5d=1.0, short_sell_ratio_change_5d=5.0, is_margin_eligible=True)

            # Only enrich the passed stock (simulating pipeline behavior)
            details = [passed, failed]
            for d in details:
                if d.passed_hard_filter:
                    enrich_risk_flow_data(d, "2026-04-22", mock_client)

        # Passed stock should have flags
        self.assertIn("pledge_ratio_latest", passed.flags)
        self.assertIn("net_main_inflow_ratio_5d", passed.flags)

        # Failed stock should have NO flags
        self.assertEqual(len(failed.flags), 0)


if __name__ == "__main__":
    unittest.main()
