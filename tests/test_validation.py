"""test_validation.py – 测试 StockValidator + SpotUniverse 校验逻辑.

使用 mock 替代真实 Tushare 网络调用。
"""

from __future__ import annotations

import pandas as pd
import pytest
from unittest.mock import MagicMock

from a_share_hot_screener.cache import LocalCache
from a_share_hot_screener.logger import WarningsCollector
from a_share_hot_screener.models import RejectedRecord, ValidatedHotStock
from a_share_hot_screener.validation import SpotUniverse, StockValidator


# ── Fixtures ─────────────────────────────────────────────

def make_daily_basic_df() -> pd.DataFrame:
    """构造 daily_basic 全市场 DataFrame（模拟 Tushare 返回格式）."""
    return pd.DataFrame([
        {
            "ts_code": "600519.SH", "trade_date": "20260418",
            "close": 1800.0, "turnover_rate": 0.35, "volume_ratio": 1.2,
            "pe": 28.0, "pe_ttm": 28.0, "pb": 12.0,
            "total_mv": 22600000.0,   # 万元 = 2.26万亿
            "circ_mv": 22600000.0,
        },
        {
            "ts_code": "000858.SZ", "trade_date": "20260418",
            "close": 130.0, "turnover_rate": 0.1, "volume_ratio": 0.8,
            "pe": 20.0, "pe_ttm": 20.0, "pb": 6.0,
            "total_mv": 5000000.0,    # 万元 = 5000亿
            "circ_mv": 5000000.0,
        },
        {
            "ts_code": "300750.SZ", "trade_date": "20260418",
            "close": 200.0, "turnover_rate": 0.5, "volume_ratio": 1.5,
            "pe": 35.0, "pe_ttm": 35.0, "pb": 8.0,
            "total_mv": 4400000.0,
            "circ_mv": 4000000.0,
        },
    ])


def make_stock_basic_df() -> pd.DataFrame:
    """构造 stock_basic DataFrame."""
    return pd.DataFrame([
        {"ts_code": "600519.SH", "symbol": "600519", "name": "贵州茅台",
         "industry": "白酒", "list_date": "20010827", "market": "主板"},
        {"ts_code": "000858.SZ", "symbol": "000858", "name": "五粮液",
         "industry": "白酒", "list_date": "19980427", "market": "主板"},
        {"ts_code": "300750.SZ", "symbol": "300750", "name": "宁德时代",
         "industry": "电池", "list_date": "20180611", "market": "创业板"},
    ])


def _make_tushare_mock():
    """创建一个 mock TushareClient."""
    ts = MagicMock()
    ts.get_daily_basic_by_date.return_value = make_daily_basic_df()
    ts.get_stock_basic.return_value = make_stock_basic_df()
    return ts


@pytest.fixture
def mock_spot_universe(tmp_path):
    cache = LocalCache(str(tmp_path / "cache"))
    warnings = WarningsCollector()
    ts = _make_tushare_mock()
    universe = SpotUniverse(ts, cache, warnings)
    universe.load("2026-04-18")
    return universe


@pytest.fixture
def validator(mock_spot_universe):
    warnings = WarningsCollector()
    return StockValidator(
        spot_universe=mock_spot_universe,
        warnings=warnings,
        include_beijing=False,
    )


# ════════════════════════════════════════════════════════
# SpotUniverse 测试
# ════════════════════════════════════════════════════════

class TestSpotUniverse:
    def test_load_builds_index(self, mock_spot_universe):
        assert mock_spot_universe.loaded is True
        assert mock_spot_universe.is_fallback is False
        assert mock_spot_universe.contains("600519")
        assert mock_spot_universe.contains("000858")
        assert mock_spot_universe.contains("300750")

    def test_unknown_code_not_in_universe(self, mock_spot_universe):
        assert not mock_spot_universe.contains("999999")

    def test_get_spot_returns_fields(self, mock_spot_universe):
        spot = mock_spot_universe.get_spot("600519")
        assert spot is not None
        assert spot["name"] == "贵州茅台"
        assert spot["latest_price"] == 1800.0
        assert spot["turnover_rate"] == 0.35
        # market_cap = total_mv(万元) * 10000
        assert spot["market_cap"] == pytest.approx(22600000.0 * 10000)

    def test_get_spot_missing_returns_none(self, mock_spot_universe):
        assert mock_spot_universe.get_spot("999999") is None

    def test_cache_hit_avoids_refetch(self, tmp_path):
        """第二次 load 应命中缓存，不再调用 Tushare."""
        cache = LocalCache(str(tmp_path / "cache"))
        warnings = WarningsCollector()
        ts = _make_tushare_mock()
        u1 = SpotUniverse(ts, cache, warnings)
        u1.load("2026-04-18")
        assert ts.get_daily_basic_by_date.call_count == 1

        u2 = SpotUniverse(ts, cache, warnings)
        u2.load("2026-04-18")
        assert ts.get_daily_basic_by_date.call_count == 1

    def test_fallback_to_stock_basic(self, tmp_path):
        """daily_basic 失败时降级到 stock_basic."""
        cache = LocalCache(str(tmp_path / "cache2"))
        warnings = WarningsCollector()
        ts = MagicMock()
        ts.get_daily_basic_by_date.return_value = None
        ts.get_stock_basic.return_value = make_stock_basic_df()

        universe = SpotUniverse(ts, cache, warnings)
        ok = universe.load("2026-04-19")
        assert ok is True
        assert universe.is_fallback is True
        assert universe.contains("600519")
        assert universe.contains("000858")


# ════════════════════════════════════════════════════════
# StockValidator 测试
# ════════════════════════════════════════════════════════

class TestStockValidator:
    def test_valid_codes_pass(self, validator):
        validated, rejected = validator.validate(
            valid_codes=["600519", "000858", "300750"],
            invalid_codes=[],
        )
        assert len(validated) == 3
        assert len(rejected) == 0

    def test_invalid_format_goes_to_rejected(self, validator):
        validated, rejected = validator.validate(
            valid_codes=[],
            invalid_codes=["abc", "12345"],
        )
        assert len(validated) == 0
        assert len(rejected) == 2

    def test_not_in_universe_goes_to_rejected(self, validator):
        validated, rejected = validator.validate(
            valid_codes=["688999"],
            invalid_codes=[],
        )
        assert len(validated) == 0
        assert len(rejected) == 1
        assert rejected[0].reject_reason == "not_in_a_share_spot_universe"

    def test_beijing_excluded_by_default(self, validator):
        validated, rejected = validator.validate(
            valid_codes=["833979"],
            invalid_codes=[],
        )
        assert len(validated) == 0
        assert rejected[0].reject_reason == "beijing_exchange_excluded"

    def test_beijing_included_when_flag_set(self, mock_spot_universe):
        warnings = WarningsCollector()
        v = StockValidator(
            spot_universe=mock_spot_universe,
            warnings=warnings,
            include_beijing=True,
        )
        validated, _ = v.validate(["833979"], [])
        assert len(validated) == 1
        assert validated[0].exchange == "BJ"

    def test_validated_stock_has_spot_fields(self, validator):
        validated, _ = validator.validate(["600519"], [])
        assert len(validated) == 1
        vs = validated[0]
        assert vs.code == "600519"
        assert vs.name == "贵州茅台"
        assert vs.exchange == "SH"
        assert vs.ts_code == "600519.SH"
        assert vs.latest_price == 1800.0
        assert vs.turnover_rate == 0.35

    def test_input_order_preserved(self, validator):
        validated, _ = validator.validate(
            ["600519", "000858", "300750"], []
        )
        orders = [v.input_order for v in validated]
        assert orders == [0, 1, 2]

    def test_mixed_valid_invalid(self, validator):
        validated, rejected = validator.validate(
            valid_codes=["600519", "688999"],
            invalid_codes=["abc"],
        )
        assert len(validated) == 1
        assert validated[0].code == "600519"
        assert len(rejected) == 2
