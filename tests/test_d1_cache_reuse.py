"""Tests for D1: 缓存键跨日复用重构.

验证:
  1. _align_date_range 对齐逻辑正确
  2. _slice_df_by_date 切片逻辑正确
  3. get_daily / get_moneyflow / get_stk_holdertrade / get_margin_detail 跨日共享缓存
  4. CACHE_SCHEMA_VERSION 已升级到 v3
  5. 对齐后返回的数据范围正确（不会多也不会少）
"""

from __future__ import annotations

import tempfile
import unittest
from unittest.mock import MagicMock, patch

import pandas as pd

from a_share_hot_screener.clients.tushare_client import (
    _align_date_range,
    _slice_df_by_date,
)


# ════════════════════════════════════════════════════════
# _align_date_range 对齐逻辑
# ════════════════════════════════════════════════════════

class TestAlignDateRange(unittest.TestCase):
    """_align_date_range produces stable aligned boundaries."""

    def test_adjacent_days_same_key(self):
        """相邻两个交易日应该产生相同的对齐范围 (90天周期)."""
        aligned1 = _align_date_range("20260110", "20260421", align_days=90)
        aligned2 = _align_date_range("20260111", "20260422", align_days=90)
        self.assertEqual(aligned1, aligned2,
                         "Adjacent trading days should produce the same aligned range")

    def test_same_day_different_start(self):
        """相同 end_date 但不同 start_date 应产生相同对齐范围."""
        aligned1 = _align_date_range("20260101", "20260422", align_days=90)
        aligned2 = _align_date_range("20260110", "20260422", align_days=90)
        self.assertEqual(aligned1, aligned2,
                         "Same end_date with different starts should align to same range")

    def test_aligned_range_covers_original(self):
        """对齐后的范围必须覆盖原始范围."""
        start, end = "20260301", "20260422"
        a_start, a_end = _align_date_range(start, end, align_days=90)
        self.assertLessEqual(a_start, start,
                             "Aligned start should be <= original start")
        self.assertGreaterEqual(a_end, end,
                                "Aligned end should be >= original end")

    def test_30_day_alignment(self):
        """30天对齐周期也应正确工作."""
        aligned1 = _align_date_range("20260407", "20260421", align_days=30)
        aligned2 = _align_date_range("20260408", "20260422", align_days=30)
        self.assertEqual(aligned1, aligned2,
                         "30-day alignment: adjacent days same key")

    def test_30_day_covers_original(self):
        """30天对齐范围覆盖原始范围."""
        start, end = "20260410", "20260422"
        a_start, a_end = _align_date_range(start, end, align_days=30)
        self.assertLessEqual(a_start, start)
        self.assertGreaterEqual(a_end, end)

    def test_invalid_date_returns_original(self):
        """无效日期格式应 fallback 返回原始值."""
        result = _align_date_range("invalid", "20260422", align_days=90)
        self.assertEqual(result, ("invalid", "20260422"))

    def test_stability_over_week(self):
        """同一周内的请求应产生相同对齐范围."""
        results = set()
        for day in range(18, 23):  # April 18-22
            r = _align_date_range(f"202601{10}", f"202604{day:02d}", align_days=90)
            results.add(r)
        self.assertEqual(len(results), 1,
                         f"All weekdays should produce same aligned range, got {len(results)} distinct")

    def test_different_quarter_different_key(self):
        """相距较远的日期应产生不同的对齐范围."""
        aligned1 = _align_date_range("20260101", "20260301", align_days=90)
        aligned2 = _align_date_range("20260401", "20260601", align_days=90)
        self.assertNotEqual(aligned1, aligned2,
                            "Dates 3 months apart should have different aligned ranges")


# ════════════════════════════════════════════════════════
# _slice_df_by_date 切片逻辑
# ════════════════════════════════════════════════════════

class TestSliceDfByDate(unittest.TestCase):
    """_slice_df_by_date correctly filters date range."""

    def _make_df(self, dates):
        return pd.DataFrame({
            "trade_date": dates,
            "close": range(len(dates)),
        })

    def test_slice_within_range(self):
        """正常切片."""
        df = self._make_df(["20260418", "20260419", "20260420", "20260421", "20260422"])
        result = _slice_df_by_date(df, "20260419", "20260421")
        self.assertEqual(len(result), 3)
        self.assertEqual(result["trade_date"].tolist(), ["20260419", "20260420", "20260421"])

    def test_slice_returns_copy(self):
        """切片应返回 copy，不影响原始 DataFrame."""
        df = self._make_df(["20260420", "20260421", "20260422"])
        result = _slice_df_by_date(df, "20260421", "20260422")
        result["close"] = 999
        self.assertNotEqual(df.iloc[1]["close"], 999)

    def test_none_input(self):
        """None 输入返回 None."""
        self.assertIsNone(_slice_df_by_date(None, "20260420", "20260422"))

    def test_empty_input(self):
        """Empty DataFrame 返回 empty."""
        result = _slice_df_by_date(pd.DataFrame(), "20260420", "20260422")
        self.assertTrue(result.empty)

    def test_no_matching_dates(self):
        """无匹配日期返回空 DataFrame."""
        df = self._make_df(["20260101", "20260102"])
        result = _slice_df_by_date(df, "20260420", "20260422")
        self.assertTrue(result.empty)

    def test_custom_date_col(self):
        """支持自定义日期列名."""
        df = pd.DataFrame({
            "ann_date": ["20260418", "20260420", "20260422"],
            "value": [1, 2, 3],
        })
        result = _slice_df_by_date(df, "20260419", "20260421", date_col="ann_date")
        self.assertEqual(len(result), 1)
        self.assertEqual(result.iloc[0]["ann_date"], "20260420")

    def test_missing_date_col_returns_full(self):
        """日期列不存在时返回完整 DataFrame."""
        df = pd.DataFrame({"value": [1, 2]})
        result = _slice_df_by_date(df, "20260420", "20260422")
        self.assertEqual(len(result), 2)


# ════════════════════════════════════════════════════════
# get_daily 跨日缓存复用
# ════════════════════════════════════════════════════════

class TestGetDailyCacheReuse(unittest.TestCase):
    """get_daily uses aligned cache key so adjacent days share cache."""

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
        return client

    def test_adjacent_days_share_cache(self):
        """Day 1 populates cache; Day 2 hits cache (no API call)."""
        client = self._make_client()

        # Generate mock data covering wide range
        dates = [f"2026{m:02d}{d:02d}" for m in range(1, 7) for d in range(1, 29)]
        mock_df = pd.DataFrame({
            "trade_date": dates,
            "ts_code": "600519.SH",
            "open": range(len(dates)),
            "close": range(len(dates)),
            "high": range(len(dates)),
            "low": range(len(dates)),
        })
        client.pro.daily.return_value = mock_df

        with patch("a_share_hot_screener.clients.tushare_client._token_bucket") as tb:
            tb.acquire.return_value = True

            # Day 1 request
            result1 = client.get_daily("600519.SH", "20260110", "20260421")
            self.assertIsNotNone(result1)
            self.assertTrue(len(result1) > 0)
            api_call_count_1 = client.pro.daily.call_count

            # Day 2 request (shifted by 1 day) — should hit cache
            client.pro.daily.reset_mock()
            result2 = client.get_daily("600519.SH", "20260111", "20260422")
            self.assertIsNotNone(result2)
            self.assertEqual(client.pro.daily.call_count, 0,
                             "Day 2 should hit cache (no API call)")

    def test_result_only_contains_requested_range(self):
        """Returned DataFrame only contains rows within requested date range."""
        client = self._make_client()

        dates = [f"202604{d:02d}" for d in range(1, 29)]
        mock_df = pd.DataFrame({
            "trade_date": dates,
            "ts_code": "600519.SH",
            "close": range(len(dates)),
        })
        client.pro.daily.return_value = mock_df

        with patch("a_share_hot_screener.clients.tushare_client._token_bucket") as tb:
            tb.acquire.return_value = True
            result = client.get_daily("600519.SH", "20260415", "20260422")

        # Should only contain Apr 15-22
        for _, row in result.iterrows():
            self.assertGreaterEqual(row["trade_date"], "20260415")
            self.assertLessEqual(row["trade_date"], "20260422")


# ════════════════════════════════════════════════════════
# get_moneyflow 跨日缓存复用
# ════════════════════════════════════════════════════════

class TestGetMoneyflowCacheReuse(unittest.TestCase):
    """get_moneyflow uses aligned cache key (30-day alignment)."""

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
        return client

    def test_adjacent_days_share_cache(self):
        """Moneyflow: adjacent days share cache."""
        client = self._make_client()

        dates = [f"202604{d:02d}" for d in range(1, 29)]
        mock_df = pd.DataFrame({
            "trade_date": dates,
            "ts_code": "600519.SH",
            "buy_sm_amount": range(len(dates)),
        })
        client.pro.moneyflow.return_value = mock_df

        with patch("a_share_hot_screener.clients.tushare_client._token_bucket") as tb:
            tb.acquire.return_value = True

            result1 = client.get_moneyflow("600519.SH", "20260407", "20260421")
            self.assertIsNotNone(result1)

            client.pro.moneyflow.reset_mock()
            result2 = client.get_moneyflow("600519.SH", "20260408", "20260422")
            self.assertEqual(client.pro.moneyflow.call_count, 0,
                             "Day 2 moneyflow should hit cache")


# ════════════════════════════════════════════════════════
# get_stk_holdertrade 跨日缓存复用
# ════════════════════════════════════════════════════════

class TestGetHoldertradeCacheReuse(unittest.TestCase):
    """get_stk_holdertrade uses aligned cache with ann_date slicing."""

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
        return client

    def test_holdertrade_cross_day_cache(self):
        """Holdertrade: adjacent days share cache."""
        client = self._make_client()

        mock_df = pd.DataFrame({
            "ann_date": ["20260320", "20260401", "20260415"],
            "ts_code": "600519.SH",
            "in_de": ["DE", "IN", "DE"],
        })
        client.pro.stk_holdertrade.return_value = mock_df

        with patch("a_share_hot_screener.clients.tushare_client._token_bucket") as tb:
            tb.acquire.return_value = True

            result1 = client.get_stk_holdertrade("600519.SH", "20260301", "20260421")
            self.assertIsNotNone(result1)

            client.pro.stk_holdertrade.reset_mock()
            result2 = client.get_stk_holdertrade("600519.SH", "20260302", "20260422")
            self.assertEqual(client.pro.stk_holdertrade.call_count, 0,
                             "Day 2 holdertrade should hit cache")

    def test_holdertrade_slices_by_ann_date(self):
        """Holdertrade slices by ann_date, not trade_date."""
        client = self._make_client()

        mock_df = pd.DataFrame({
            "ann_date": ["20260301", "20260315", "20260401", "20260420"],
            "ts_code": "600519.SH",
            "in_de": ["DE", "IN", "DE", "IN"],
        })
        client.pro.stk_holdertrade.return_value = mock_df

        with patch("a_share_hot_screener.clients.tushare_client._token_bucket") as tb:
            tb.acquire.return_value = True
            result = client.get_stk_holdertrade("600519.SH", "20260310", "20260415")

        # Should only include rows with ann_date in [20260310, 20260415]
        self.assertEqual(len(result), 2)  # 20260315, 20260401
        self.assertIn("20260315", result["ann_date"].tolist())
        self.assertIn("20260401", result["ann_date"].tolist())


# ════════════════════════════════════════════════════════
# get_margin_detail 跨日缓存复用
# ════════════════════════════════════════════════════════

class TestGetMarginDetailCacheReuse(unittest.TestCase):
    """get_margin_detail uses aligned cache for range mode, unchanged for single-date."""

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
        return client

    def test_range_mode_cross_day_cache(self):
        """Margin range mode: adjacent days share cache."""
        client = self._make_client()

        dates = [f"202604{d:02d}" for d in range(5, 25)]
        mock_df = pd.DataFrame({
            "trade_date": dates,
            "ts_code": "600519.SH",
            "rzye": range(len(dates)),
        })
        client.pro.margin_detail.return_value = mock_df

        with patch("a_share_hot_screener.clients.tushare_client._token_bucket") as tb:
            tb.acquire.return_value = True

            result1 = client.get_margin_detail("600519.SH", start_date="20260407", end_date="20260421")
            self.assertIsNotNone(result1)

            client.pro.margin_detail.reset_mock()
            result2 = client.get_margin_detail("600519.SH", start_date="20260408", end_date="20260422")
            self.assertEqual(client.pro.margin_detail.call_count, 0,
                             "Day 2 margin should hit cache")

    def test_single_date_mode_unchanged(self):
        """Single date mode doesn't use alignment (cache key is date-specific)."""
        client = self._make_client()

        mock_df = pd.DataFrame({
            "trade_date": ["20260421"],
            "ts_code": "600519.SH",
            "rzye": [100],
        })
        client.pro.margin_detail.return_value = mock_df

        with patch("a_share_hot_screener.clients.tushare_client._token_bucket") as tb:
            tb.acquire.return_value = True

            result1 = client.get_margin_detail("600519.SH", trade_date="20260421")
            self.assertIsNotNone(result1)

            # Different date → different cache key → API call
            client.pro.margin_detail.reset_mock()
            mock_df2 = pd.DataFrame({
                "trade_date": ["20260422"],
                "ts_code": "600519.SH",
                "rzye": [101],
            })
            client.pro.margin_detail.return_value = mock_df2
            result2 = client.get_margin_detail("600519.SH", trade_date="20260422")
            self.assertEqual(client.pro.margin_detail.call_count, 1,
                             "Different single date should be a cache miss")


# ════════════════════════════════════════════════════════
# CACHE_SCHEMA_VERSION 升级
# ════════════════════════════════════════════════════════

class TestCacheSchemaVersion(unittest.TestCase):
    """CACHE_SCHEMA_VERSION has been bumped for D1."""

    def test_version_is_v3(self):
        from a_share_hot_screener.cache import CACHE_SCHEMA_VERSION
        self.assertEqual(CACHE_SCHEMA_VERSION, "v3")


if __name__ == "__main__":
    unittest.main()
