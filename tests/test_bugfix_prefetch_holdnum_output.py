"""Tests for 3 bug fixes:
  1. prefetch_flow_data key alignment (Step 2.8)
  2. holdernumber cache key alignment (Step 2.7)
  3. output filename uses trade_date_used (Step 10)
"""

from __future__ import annotations

import datetime as dt
import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest

from a_share_hot_screener.clients.tushare_client import (
    TushareClient,
    _align_date_range,
)


# ═══════════════════════════════════════════════════════
# Bug #1: prefetch_flow_data key alignment
# ═══════════════════════════════════════════════════════


class TestPrefetchFlowKeyAlignment:
    """prefetch_flow_data should use aligned keys matching get_moneyflow etc."""

    def _make_client_with_cache(self, cached_keys: set):
        """Create a TushareClient mock whose cache.get returns data for cached_keys."""
        cache = MagicMock()

        def mock_get(namespace, key, ttl=None):
            if key in cached_keys:
                return []  # non-None → cache hit
            return None  # cache miss

        cache.get = mock_get
        cache.put = MagicMock()

        client = MagicMock(spec=TushareClient)
        client.cache = cache
        client.get_moneyflow = MagicMock()
        client.get_stk_holdertrade = MagicMock()
        client.get_margin_detail = MagicMock()

        # Use the real prefetch_flow_data
        client.prefetch_flow_data = TushareClient.prefetch_flow_data.__get__(client)
        return client

    def test_skip_when_aligned_key_cached(self):
        """If aligned keys exist in cache, prefetch should skip (0 tasks)."""
        start_date = "20260408"
        end_date = "20260422"

        # Compute aligned keys the same way get_moneyflow etc. do
        mf_aligned_s, mf_aligned_e = _align_date_range(start_date, end_date, align_days=30)
        ht_aligned_s, ht_aligned_e = _align_date_range(start_date, end_date, align_days=90)
        mg_aligned_s, mg_aligned_e = _align_date_range(start_date, end_date, align_days=30)

        ts_code = "000001.SZ"
        cached_keys = {
            f"moneyflow_{ts_code}_{mf_aligned_s}_{mf_aligned_e}",
            f"holdertrade_{ts_code}_{ht_aligned_s}_{ht_aligned_e}",
            f"margin_{ts_code}_{mg_aligned_s}_{mg_aligned_e}",
        }

        client = self._make_client_with_cache(cached_keys)
        stats = client.prefetch_flow_data([ts_code], start_date, end_date)

        # All 3 should be skipped
        assert stats == {'moneyflow': 0, 'holdertrade': 0, 'margin': 0}
        client.get_moneyflow.assert_not_called()
        client.get_stk_holdertrade.assert_not_called()
        client.get_margin_detail.assert_not_called()

    def test_old_unaligned_key_not_found(self):
        """Raw (unaligned) keys in cache should NOT cause skip — proves alignment is used."""
        start_date = "20260408"
        end_date = "20260422"

        ts_code = "000001.SZ"
        # Put raw (unaligned) keys — these should NOT be found by prefetch
        old_keys = {
            f"moneyflow_{ts_code}_{start_date}_{end_date}",
            f"holdertrade_{ts_code}_{start_date}_{end_date}",
            f"margin_{ts_code}_{start_date}_{end_date}",
        }

        client = self._make_client_with_cache(old_keys)
        stats = client.prefetch_flow_data([ts_code], start_date, end_date, max_workers=1)

        # Should NOT skip — raw keys don't match aligned keys
        total = stats['moneyflow'] + stats['holdertrade'] + stats['margin']
        assert total == 3, f"Expected 3 API calls, got {stats}"

    def test_adjacent_days_share_aligned_key(self):
        """Two adjacent days should produce the same aligned keys."""
        day1_start = "20260407"
        day1_end = "20260421"
        day2_start = "20260408"
        day2_end = "20260422"

        mf1 = _align_date_range(day1_start, day1_end, align_days=30)
        mf2 = _align_date_range(day2_start, day2_end, align_days=30)
        assert mf1 == mf2, f"moneyflow alignment differs: {mf1} vs {mf2}"

        ht1 = _align_date_range(day1_start, day1_end, align_days=90)
        ht2 = _align_date_range(day2_start, day2_end, align_days=90)
        assert ht1 == ht2, f"holdertrade alignment differs: {ht1} vs {ht2}"


# ═══════════════════════════════════════════════════════
# Bug #2: holdernumber cache key alignment
# ═══════════════════════════════════════════════════════


class TestHoldnumKeyAlignment:
    """get_stk_holdernumber should align start_date to 180-day boundary."""

    def test_adjacent_days_same_cache_key(self):
        """start_date differing by 1 day should produce the same aligned key."""
        # Two adjacent run_dates → start_date = run_date - 365d
        run1 = dt.date(2026, 4, 21)
        run2 = dt.date(2026, 4, 22)
        start1 = (run1 - dt.timedelta(days=365)).strftime("%Y%m%d")
        start2 = (run2 - dt.timedelta(days=365)).strftime("%Y%m%d")

        # The aligned keys should be identical
        anchor = dt.date(2020, 1, 1)

        def align_holdnum(start_date_str):
            sd = dt.datetime.strptime(start_date_str, "%Y%m%d").date()
            days_since = (sd - anchor).days
            remainder = days_since % 180
            if remainder > 0:
                sd = sd - dt.timedelta(days=remainder)
            return sd.strftime("%Y%m%d")

        aligned1 = align_holdnum(start1)
        aligned2 = align_holdnum(start2)
        assert aligned1 == aligned2, f"holdnum keys differ: {aligned1} vs {aligned2}"

    def test_cache_hit_with_aligned_key(self):
        """get_stk_holdernumber should find cache with aligned key."""
        cache = MagicMock()
        found_keys = set()

        def mock_get(namespace, key, ttl=None):
            found_keys.add(key)
            if "holdnum_000001.SZ_" in key:
                return [{"ts_code": "000001.SZ", "holder_num": 100000}]
            return None

        cache.get = mock_get
        cache.put = MagicMock()

        with patch("a_share_hot_screener.clients.tushare_client.TushareClient.__init__", return_value=None):
            client = TushareClient.__new__(TushareClient)
            client.cache = cache
            client.pro = MagicMock()

            result = client.get_stk_holdernumber("000001.SZ", start_date="20250422")

        # Should have looked up an aligned key, not the raw "20250422"
        holdnum_keys = [k for k in found_keys if k.startswith("holdnum_")]
        assert len(holdnum_keys) == 1
        key = holdnum_keys[0]
        # The aligned start should NOT be "20250422" (it should be aligned to 180-day boundary)
        parts = key.split("_")
        aligned_start = parts[-1]
        # Verify alignment: (aligned_start - anchor) % 180 == 0
        anchor = dt.date(2020, 1, 1)
        asd = dt.datetime.strptime(aligned_start, "%Y%m%d").date()
        assert (asd - anchor).days % 180 == 0, f"Key not aligned: {key}"

    def test_prefetch_risk_data_uses_aligned_holdnum_key(self):
        """prefetch_risk_data should use aligned holdnum key for skip check."""
        cache = MagicMock()
        checked_keys = []

        def mock_get(namespace, key, ttl=None):
            checked_keys.append((namespace, key))
            # Return cache hit for everything
            return [{"dummy": 1}]

        cache.get = mock_get

        with patch("a_share_hot_screener.clients.tushare_client.TushareClient.__init__", return_value=None):
            client = TushareClient.__new__(TushareClient)
            client.cache = cache
            client.pro = MagicMock()

            stats = client.prefetch_risk_data(["000001.SZ"], run_date_str="2026-04-22")

        # All skipped (cache hit)
        assert stats == {'pledge': 0, 'float': 0, 'holdnum': 0}

        # Verify holdnum key is aligned
        holdnum_checks = [k for ns, k in checked_keys if k.startswith("holdnum_")]
        assert len(holdnum_checks) == 1
        parts = holdnum_checks[0].split("_")
        aligned_start = parts[-1]
        anchor = dt.date(2020, 1, 1)
        asd = dt.datetime.strptime(aligned_start, "%Y%m%d").date()
        assert (asd - anchor).days % 180 == 0, f"prefetch holdnum key not aligned: {holdnum_checks[0]}"


# ═══════════════════════════════════════════════════════
# Bug #3: output filename uses trade_date_used
# ═══════════════════════════════════════════════════════


class TestOutputFilenameTradeDateUsed:
    """OutputWriter.write_all should use trade_date_used as filename prefix."""

    def test_filename_uses_trade_date_used(self):
        """When run_date != trade_date_used, files should be named by trade_date_used."""
        from a_share_hot_screener.output import OutputWriter
        from a_share_hot_screener.models import RunMetadata

        with tempfile.TemporaryDirectory() as tmpdir:
            writer = OutputWriter(output_dir=tmpdir)

            # Create a metadata where run_date != trade_date_used
            metadata = RunMetadata(
                run_date="2026-04-20",        # Sunday
                trade_date_used="2026-04-18",  # Friday (actual trade date)
                generated_at="2026-04-20T01:00:00Z",
                version="0.1.0",
                input_pool_size=0,
                elapsed_seconds=0.0,
            )

            writer.write_all(details=[], rejected=[], metadata=metadata)

            # Files should be prefixed with trade_date_used, NOT run_date
            files = os.listdir(tmpdir)
            assert any("2026-04-18" in f for f in files), (
                f"Expected files with '2026-04-18' prefix, got: {files}"
            )
            assert not any("2026-04-20" in f for f in files), (
                f"Found files with run_date '2026-04-20' prefix — should use trade_date_used: {files}"
            )

    def test_filename_fallback_to_run_date(self):
        """If trade_date_used is empty, fall back to run_date."""
        from a_share_hot_screener.output import OutputWriter
        from a_share_hot_screener.models import RunMetadata

        with tempfile.TemporaryDirectory() as tmpdir:
            writer = OutputWriter(output_dir=tmpdir)

            metadata = RunMetadata(
                run_date="2026-04-22",
                trade_date_used="",  # empty
                generated_at="2026-04-22T01:00:00Z",
                version="0.1.0",
                input_pool_size=0,
                elapsed_seconds=0.0,
            )

            writer.write_all(details=[], rejected=[], metadata=metadata)

            files = os.listdir(tmpdir)
            assert any("2026-04-22" in f for f in files), (
                f"Expected files with '2026-04-22' prefix (fallback), got: {files}"
            )
