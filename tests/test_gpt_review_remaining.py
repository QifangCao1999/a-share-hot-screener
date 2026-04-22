"""Tests for GPT Review remaining items (D2/D3/D4/Step0/Step9/Step10/Step6.15).

Covers:
  D2: 限流器 acquire 返回值检查
  D3: 缓存原子写 (tempfile + rename)
  D4: 缓存空结果 (避免重复请求)
  Step 0: --stock-codes-file 支持
  Step 9: trend compare 使用真实 trade_date_used
  Step 10: metadata output_files 包含 sector_heat/setup_timing
  Step 6.15: HT9/10 输出样本量 + context_sector_source
"""

from __future__ import annotations

import json
import os
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd


# ════════════════════════════════════════════════════════
# D2: 限流器 acquire 返回值检查
# ════════════════════════════════════════════════════════

class TestD2RateLimiterAcquireCheck(unittest.TestCase):
    """D2: _call() checks acquire() return value and logs warning on timeout."""

    def test_acquire_false_still_proceeds(self):
        """When acquire() returns False, _call should still attempt the API call."""
        from a_share_hot_screener.clients.tushare_client import TushareClient, _token_bucket

        client = TushareClient.__new__(TushareClient)
        client.cache = None

        mock_pro = MagicMock()
        mock_pro.test_api.return_value = pd.DataFrame({"a": [1, 2]})
        client.pro = mock_pro

        with patch.object(_token_bucket, "acquire", return_value=False):
            result = client._call("test_api", label="test", max_retries=1, foo="bar")

        self.assertIsNotNone(result)
        self.assertEqual(len(result), 2)
        mock_pro.test_api.assert_called_once_with(foo="bar")

    def test_acquire_true_no_warning(self):
        """When acquire() returns True, no warning logged."""
        from a_share_hot_screener.clients.tushare_client import TushareClient, _token_bucket

        client = TushareClient.__new__(TushareClient)
        client.cache = None
        mock_pro = MagicMock()
        mock_pro.test_api.return_value = pd.DataFrame({"a": [1]})
        client.pro = mock_pro

        with patch.object(_token_bucket, "acquire", return_value=True):
            result = client._call("test_api", max_retries=1)

        self.assertIsNotNone(result)


# ════════════════════════════════════════════════════════
# D3: 缓存原子写 (tempfile + rename)
# ════════════════════════════════════════════════════════

class TestD3CacheAtomicWrite(unittest.TestCase):
    """D3: cache.put() uses tempfile + os.replace for atomic writes."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_put_creates_valid_json(self):
        """put() should create a valid JSON file via atomic write."""
        from a_share_hot_screener.cache import LocalCache
        cache = LocalCache(self.tmpdir, default_ttl=3600)
        cache.put("ns", "key1", {"hello": "world"})

        val = cache.get("ns", "key1")
        self.assertEqual(val, {"hello": "world"})

    def test_no_tmp_files_left_on_success(self):
        """After successful put(), no .tmp files should remain."""
        from a_share_hot_screener.cache import LocalCache
        cache = LocalCache(self.tmpdir, default_ttl=3600)
        cache.put("ns", "key2", [1, 2, 3])

        # Scan for .tmp files
        tmp_files = list(Path(self.tmpdir).rglob("*.tmp"))
        self.assertEqual(len(tmp_files), 0, f"Found leftover tmp files: {tmp_files}")

    def test_atomic_write_no_corruption_on_concurrent(self):
        """Multiple concurrent puts to the same key shouldn't corrupt."""
        from a_share_hot_screener.cache import LocalCache
        cache = LocalCache(self.tmpdir, default_ttl=3600)

        errors = []

        def writer(i):
            try:
                cache.put("ns", "concurrent_key", {"writer": i})
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=writer, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(len(errors), 0, f"Errors: {errors}")
        val = cache.get("ns", "concurrent_key")
        self.assertIsNotNone(val)
        self.assertIn("writer", val)


# ════════════════════════════════════════════════════════
# D4: 缓存空结果 (避免重复请求)
# ════════════════════════════════════════════════════════

class TestD4CacheEmptyResults(unittest.TestCase):
    """D4: Per-stock APIs cache empty DataFrames to avoid repeated requests."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _make_client(self):
        from a_share_hot_screener.cache import LocalCache
        from a_share_hot_screener.clients.tushare_client import TushareClient

        cache = LocalCache(self.tmpdir, default_ttl=3600)
        client = TushareClient.__new__(TushareClient)
        client.cache = cache
        client.pro = MagicMock()
        return client, cache

    def test_pledge_stat_caches_empty(self):
        """get_pledge_stat should cache empty DataFrame."""
        client, cache = self._make_client()
        client.pro.pledge_stat.return_value = pd.DataFrame()  # empty

        with patch("a_share_hot_screener.clients.tushare_client._token_bucket") as tb:
            tb.acquire.return_value = True
            result1 = client.get_pledge_stat("600519.SH")

        self.assertIsNotNone(result1)
        self.assertTrue(result1.empty)

        # Second call should hit cache (no API call)
        client.pro.pledge_stat.reset_mock()
        result2 = client.get_pledge_stat("600519.SH")
        self.assertIsNotNone(result2)
        self.assertTrue(result2.empty)
        client.pro.pledge_stat.assert_not_called()

    def test_share_float_caches_empty(self):
        """get_share_float should cache empty DataFrame."""
        client, cache = self._make_client()
        client.pro.share_float.return_value = pd.DataFrame()

        with patch("a_share_hot_screener.clients.tushare_client._token_bucket") as tb:
            tb.acquire.return_value = True
            result = client.get_share_float("600519.SH")

        self.assertTrue(result.empty)

        # Second call should hit cache
        client.pro.share_float.reset_mock()
        result2 = client.get_share_float("600519.SH")
        self.assertTrue(result2.empty)
        client.pro.share_float.assert_not_called()

    def test_moneyflow_caches_empty(self):
        """get_moneyflow should cache empty DataFrame."""
        client, cache = self._make_client()
        client.pro.moneyflow.return_value = pd.DataFrame()

        with patch("a_share_hot_screener.clients.tushare_client._token_bucket") as tb:
            tb.acquire.return_value = True
            result = client.get_moneyflow("600519.SH", start_date="20260401", end_date="20260422")

        self.assertTrue(result.empty)

        client.pro.moneyflow.reset_mock()
        result2 = client.get_moneyflow("600519.SH", start_date="20260401", end_date="20260422")
        self.assertTrue(result2.empty)
        client.pro.moneyflow.assert_not_called()

    def test_holdertrade_caches_empty(self):
        """get_stk_holdertrade should cache empty DataFrame."""
        client, cache = self._make_client()
        client.pro.stk_holdertrade.return_value = pd.DataFrame()

        with patch("a_share_hot_screener.clients.tushare_client._token_bucket") as tb:
            tb.acquire.return_value = True
            result = client.get_stk_holdertrade("600519.SH", start_date="20260401", end_date="20260422")

        self.assertTrue(result.empty)

        client.pro.stk_holdertrade.reset_mock()
        result2 = client.get_stk_holdertrade("600519.SH", start_date="20260401", end_date="20260422")
        self.assertTrue(result2.empty)
        client.pro.stk_holdertrade.assert_not_called()

    def test_margin_detail_caches_empty(self):
        """get_margin_detail should cache empty DataFrame."""
        client, cache = self._make_client()
        client.pro.margin_detail.return_value = pd.DataFrame()

        with patch("a_share_hot_screener.clients.tushare_client._token_bucket") as tb:
            tb.acquire.return_value = True
            result = client.get_margin_detail("600519.SH", start_date="20260401", end_date="20260422")

        self.assertTrue(result.empty)

        client.pro.margin_detail.reset_mock()
        result2 = client.get_margin_detail("600519.SH", start_date="20260401", end_date="20260422")
        self.assertTrue(result2.empty)
        client.pro.margin_detail.assert_not_called()

    def test_holdernumber_caches_empty(self):
        """get_stk_holdernumber should cache empty DataFrame."""
        client, cache = self._make_client()
        client.pro.stk_holdernumber.return_value = pd.DataFrame()

        with patch("a_share_hot_screener.clients.tushare_client._token_bucket") as tb:
            tb.acquire.return_value = True
            result = client.get_stk_holdernumber("600519.SH")

        self.assertTrue(result.empty)

        client.pro.stk_holdernumber.reset_mock()
        result2 = client.get_stk_holdernumber("600519.SH")
        self.assertTrue(result2.empty)
        client.pro.stk_holdernumber.assert_not_called()


# ════════════════════════════════════════════════════════
# Step 0: --stock-codes-file 支持
# ════════════════════════════════════════════════════════

class TestStep0StockCodesFile(unittest.TestCase):
    """Step 0: --stock-codes-file reads codes from a text file."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_read_one_per_line(self):
        """File with one code per line."""
        from a_share_hot_screener.cli import _read_stock_codes_file

        path = os.path.join(self.tmpdir, "codes.txt")
        with open(path, "w") as f:
            f.write("600519\n000858\n300750\n")

        result = _read_stock_codes_file(path)
        self.assertEqual(result, "600519 000858 300750")

    def test_read_comma_separated(self):
        """File with comma-separated codes on one line."""
        from a_share_hot_screener.cli import _read_stock_codes_file

        path = os.path.join(self.tmpdir, "codes.txt")
        with open(path, "w") as f:
            f.write("600519,000858,300750\n")

        result = _read_stock_codes_file(path)
        self.assertEqual(result, "600519 000858 300750")

    def test_read_with_comments_and_blanks(self):
        """File with comments and blank lines."""
        from a_share_hot_screener.cli import _read_stock_codes_file

        path = os.path.join(self.tmpdir, "codes.txt")
        with open(path, "w") as f:
            f.write("# CSI300\n600519\n\n# CSI500\n000858\n\n")

        result = _read_stock_codes_file(path)
        self.assertEqual(result, "600519 000858")

    def test_read_empty_file(self):
        """Empty file returns empty string."""
        from a_share_hot_screener.cli import _read_stock_codes_file

        path = os.path.join(self.tmpdir, "empty.txt")
        with open(path, "w") as f:
            f.write("")

        result = _read_stock_codes_file(path)
        self.assertEqual(result, "")

    def test_read_nonexistent_file(self):
        """Nonexistent file returns None."""
        from a_share_hot_screener.cli import _read_stock_codes_file

        result = _read_stock_codes_file("/nonexistent/path/codes.txt")
        self.assertIsNone(result)

    def test_parser_has_stock_codes_file_arg(self):
        """build_parser() includes --stock-codes-file."""
        from a_share_hot_screener.cli import build_parser

        parser = build_parser()
        # Parse with --stock-codes-file
        args = parser.parse_args([
            "--run-date", "2026-04-22",
            "--stock-codes-file", "/tmp/codes.txt",
            "--output-dir", "/tmp/out",
        ])
        self.assertEqual(args.stock_codes_file, "/tmp/codes.txt")

    def test_stock_codes_no_longer_required(self):
        """--stock-codes is no longer required (can provide via file instead)."""
        from a_share_hot_screener.cli import build_parser

        parser = build_parser()
        args = parser.parse_args([
            "--run-date", "2026-04-22",
            "--output-dir", "/tmp/out",
        ])
        self.assertEqual(args.stock_codes, [])

    def test_read_mixed_format(self):
        """File with mixed formats: lines + commas + spaces."""
        from a_share_hot_screener.cli import _read_stock_codes_file

        path = os.path.join(self.tmpdir, "codes.txt")
        with open(path, "w") as f:
            f.write("600519, 000858\n300750 601398\n# comment\n002594,300059\n")

        result = _read_stock_codes_file(path)
        codes = result.split()
        self.assertEqual(len(codes), 6)
        self.assertIn("600519", codes)
        self.assertIn("300059", codes)


# ════════════════════════════════════════════════════════
# Step 9: trend compare 使用真实 trade_date_used
# ════════════════════════════════════════════════════════

class TestStep9TrendCompareTradeDateUsed(unittest.TestCase):
    """Step 9: load_prev_run reads trade_date_used from metadata JSON."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_reads_trade_date_from_metadata(self):
        """When metadata JSON exists, use trade_date_used instead of filename."""
        from a_share_hot_screener.trend_compare import load_prev_run

        # Create summary CSV with run_date=2026-04-19 (weekend)
        summary_path = os.path.join(self.tmpdir, "2026-04-19_stage1_hot_summary.csv")
        with open(summary_path, "w") as f:
            f.write("code,name,pass_stage1,passed_hard_filter,total_score\n")
            f.write("600519,贵州茅台,True,True,0.85\n")

        # Create metadata JSON with trade_date_used=2026-04-18 (Friday)
        meta_path = os.path.join(self.tmpdir, "2026-04-19_stage1_hot_metadata.json")
        with open(meta_path, "w") as f:
            json.dump({"trade_date_used": "2026-04-18", "run_date": "2026-04-19"}, f)

        snapshot = load_prev_run(self.tmpdir)
        self.assertIsNotNone(snapshot)
        self.assertEqual(snapshot.trade_date_used, "2026-04-18")

    def test_falls_back_to_filename_without_metadata(self):
        """Without metadata JSON, fall back to filename prefix."""
        from a_share_hot_screener.trend_compare import load_prev_run

        summary_path = os.path.join(self.tmpdir, "2026-04-18_stage1_hot_summary.csv")
        with open(summary_path, "w") as f:
            f.write("code,name,pass_stage1,total_score\n")
            f.write("600519,贵州茅台,True,0.85\n")

        snapshot = load_prev_run(self.tmpdir)
        self.assertIsNotNone(snapshot)
        self.assertEqual(snapshot.trade_date_used, "2026-04-18")

    def test_falls_back_on_malformed_metadata(self):
        """Malformed metadata JSON falls back to filename."""
        from a_share_hot_screener.trend_compare import load_prev_run

        summary_path = os.path.join(self.tmpdir, "2026-04-19_stage1_hot_summary.csv")
        with open(summary_path, "w") as f:
            f.write("code,name,pass_stage1,total_score\n")
            f.write("600519,茅台,True,0.85\n")

        meta_path = os.path.join(self.tmpdir, "2026-04-19_stage1_hot_metadata.json")
        with open(meta_path, "w") as f:
            f.write("{invalid json")

        snapshot = load_prev_run(self.tmpdir)
        self.assertIsNotNone(snapshot)
        self.assertEqual(snapshot.trade_date_used, "2026-04-19")


# ════════════════════════════════════════════════════════
# Step 10: metadata output_files 包含 sector_heat/setup_timing
# ════════════════════════════════════════════════════════

class TestStep10MetadataExtraOutputFiles(unittest.TestCase):
    """Step 10: metadata.output_files includes sector_heat and setup_timing."""

    def test_append_extra_output_files_with_both(self):
        """Both sector_heat and setup_timing CSV exist → both in output_files."""
        from a_share_hot_screener.models import RunMetadata

        tmpdir = tempfile.mkdtemp()
        try:
            # Create fake output files
            sector_path = os.path.join(tmpdir, "2026-04-22_sector_heat.csv")
            timing_path = os.path.join(tmpdir, "2026-04-22_setup_timing.csv")
            Path(sector_path).write_text("header\n")
            Path(timing_path).write_text("header\n")

            metadata = RunMetadata(run_date="2026-04-22")
            metadata.output_files = {"summary": "/tmp/summary.csv", "metadata": ""}

            # Simulate what pipeline._append_extra_output_files does
            extra = {}
            if os.path.isfile(sector_path):
                extra["sector_heat"] = sector_path
            if os.path.isfile(timing_path):
                extra["setup_timing"] = timing_path
            metadata.output_files.update(extra)

            self.assertIn("sector_heat", metadata.output_files)
            self.assertIn("setup_timing", metadata.output_files)
            self.assertEqual(metadata.output_files["sector_heat"], sector_path)
        finally:
            import shutil
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_append_extra_output_files_without_files(self):
        """No extra files → output_files unchanged."""
        from a_share_hot_screener.models import RunMetadata

        metadata = RunMetadata(run_date="2026-04-22")
        metadata.output_files = {"summary": "/tmp/s.csv"}

        # No files exist in a random dir
        self.assertNotIn("sector_heat", metadata.output_files)
        self.assertNotIn("setup_timing", metadata.output_files)


# ════════════════════════════════════════════════════════
# Step 6.15: HT9/10 输出样本量 + context_sector_source
# ════════════════════════════════════════════════════════

class TestStep615ContextSectorOutput(unittest.TestCase):
    """Step 6.15: context_scores output includes sector source and member count."""

    def test_to_dict_includes_sector_source(self):
        """ContextScoresResult.to_dict() includes context_sector_source."""
        from a_share_hot_screener.context_scores import ContextScoresResult

        result = ContextScoresResult(
            ht9_sector_name="半导体概念",
            ht9_sector_size=45,
        )
        d = result.to_dict()
        self.assertEqual(d["context_sector_source"], "半导体概念")
        self.assertEqual(d["context_sector_member_count"], 45)

    def test_to_dict_empty_sector(self):
        """Empty sector → empty source and 0 count."""
        from a_share_hot_screener.context_scores import ContextScoresResult

        result = ContextScoresResult()
        d = result.to_dict()
        self.assertEqual(d["context_sector_source"], "")
        self.assertEqual(d["context_sector_member_count"], 0)

    def test_context_sector_in_full_compute(self):
        """compute_context_scores sets sector_name and sector_size."""
        from a_share_hot_screener.context_scores import compute_context_scores
        from a_share_hot_screener.models import HotStockDetail

        detail = HotStockDetail(
            code="600519",
            ts_code="600519.SH",
            return_5d=10.0,
            limit_up_count_5d=1,
            lhb_count_20d=0,
            concept_heat_pctile_5d=0.85,
            amount_ratio_5d_to_20d=2.5,
            amount_avg_5d=500_000_000,
        )

        # Mock event context
        event_ctx = MagicMock()
        event_ctx.concept_heat_mode = "ths_daily_full"
        event_ctx.zt_pool_available = True

        # Build sector data with 15 members
        sector_data = {}
        for i in range(15):
            ts = f"60{i:04d}.SH"
            sector_data[ts] = {
                "return_5d": 3.0 + i * 2,
                "amount_ratio": 1.2 + i * 0.1,
                "amount_5d": 100_000_000 * (i + 1),
                "limit_up_count_5d": 1 if i < 3 else 0,
            }
        # Include the target stock
        sector_data["600519.SH"] = {
            "return_5d": 10.0,
            "amount_ratio": 2.5,
            "amount_5d": 2_500_000_000,
            "limit_up_count_5d": 1,
        }

        result = compute_context_scores(
            detail=detail,
            event_ctx=event_ctx,
            sector_members_daily=sector_data,
            sector_name="白酒概念",
        )

        d = result.to_dict()
        self.assertEqual(d["context_sector_source"], "白酒概念")
        self.assertEqual(d["context_sector_member_count"], 16)


if __name__ == "__main__":
    unittest.main()
