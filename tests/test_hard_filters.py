"""test_hard_filters.py – 测试硬筛除规则（H1-H9）."""

import datetime as dt
import pytest

from a_share_hot_screener.config import HotScreenerConfig
from a_share_hot_screener.hard_filters import (
    HardFilterResult,
    _check_amount_avg_5d,
    _check_delisted,
    _check_float_market_cap,
    _check_industry,
    _check_ipo_date,
    _check_listing_days,
    _check_min_price,
    _check_st,
    _check_suspended,
    apply_hard_filters,
)


@pytest.fixture
def default_config():
    return HotScreenerConfig(
        tushare_token="test",
        run_date=dt.date(2026, 4, 17),
        stock_codes=["600519"],
        output_dir="/tmp/test_out",
        min_price=3.0,
        min_amount_avg_5d=200_000_000.0,
        min_float_market_cap=1_500_000_000.0,
        min_trading_days=20,
    )


# ── 单规则测试 ────────────────────────────────────────────

class TestCheckSt:
    def test_st_stock(self):
        assert _check_st("ST 新宏泰") is not None
        assert "st_stock" in _check_st("ST 新宏泰")

    def test_star_st(self):
        assert _check_st("*ST 金力泰") is not None

    def test_delist_warning(self):
        assert _check_st("退市整理 XX") is not None

    def test_normal_stock(self):
        assert _check_st("贵州茅台") is None

    def test_empty_name_no_fail(self):
        assert _check_st("") is None


class TestCheckDelisted:
    def test_has_delist_keyword(self):
        assert _check_delisted("XX退市", None) is not None

    def test_normal(self):
        assert _check_delisted("宁德时代", "汽车") is None


class TestCheckListingDays:
    def test_too_young(self):
        # min_days=20 → 日历天等效 = 30
        r = _check_listing_days(25, 20)
        assert r is not None
        assert "listing_too_young" in r

    def test_ok(self):
        assert _check_listing_days(100, 20) is None

    def test_none_warns(self):
        assert _check_listing_days(None, 20) == "_warn_"


class TestCheckMinPrice:
    def test_below(self):
        r = _check_min_price(2.5, 3.0)
        assert "price_too_low" in r

    def test_equal(self):
        assert _check_min_price(3.0, 3.0) is None

    def test_above(self):
        assert _check_min_price(100.0, 3.0) is None

    def test_none_warns(self):
        assert _check_min_price(None, 3.0) == "_warn_"


class TestCheckSuspended:
    def test_zero_volume(self):
        r = _check_suspended(0, None, 100)
        assert "suspended" in r

    def test_zero_amount(self):
        r = _check_suspended(None, 0, 100)
        assert "suspended" in r

    def test_new_stock_exempt(self):
        # 上市 ≤ 30 天，不淘汰
        assert _check_suspended(0, 0, 20) is None

    def test_normal(self):
        assert _check_suspended(500000, 5e8, 365) is None

    def test_both_none_warns(self):
        assert _check_suspended(None, None, 365) == "_warn_"


class TestCheckAmountAvg5d:
    def test_below(self):
        r = _check_amount_avg_5d(1e8, 2e8)
        assert "amount_too_low" in r

    def test_equal(self):
        assert _check_amount_avg_5d(2e8, 2e8) is None

    def test_none_warns(self):
        assert _check_amount_avg_5d(None, 2e8) == "_warn_"


class TestCheckFloatMC:
    def test_below(self):
        r = _check_float_market_cap(1e9, 1.5e9)
        assert "float_mc_too_small" in r

    def test_ok(self):
        assert _check_float_market_cap(2e9, 1.5e9) is None

    def test_none_warns(self):
        assert _check_float_market_cap(None, 1.5e9) == "_warn_"


class TestCheckIpoDate:
    def test_missing(self):
        r = _check_ipo_date(None)
        assert "ipo_date_missing" in r

    def test_empty_string(self):
        r = _check_ipo_date("")
        assert "ipo_date_missing" in r

    def test_valid(self):
        assert _check_ipo_date("20100101") is None


class TestCheckIndustry:
    def test_bank(self):
        r = _check_industry("银行")
        assert "industry_finance" in r

    def test_securities(self):
        r = _check_industry("证券")
        assert "industry_finance" in r

    def test_insurance(self):
        r = _check_industry("保险")
        assert "industry_finance" in r

    def test_normal(self):
        assert _check_industry("半导体") is None
        assert _check_industry("新能源汽车") is None

    def test_none_warns(self):
        assert _check_industry(None) == "_warn_"


# ── apply_hard_filters 集成测试 ──────────────────────────

class TestApplyHardFilters:
    def test_all_pass(self, default_config):
        result = apply_hard_filters(
            config=default_config,
            name="贵州茅台",
            industry="食品饮料",
            ipo_date="20010827",
            listing_days=9000,
            latest_price=1800.0,
            latest_volume=3_000_000,
            amount_1d=5e9,
            amount_avg_5d=4e9,
            float_market_cap=2e12,
        )
        assert result.passed is True
        assert result.fail_reasons == []

    def test_st_fails(self, default_config):
        result = apply_hard_filters(
            config=default_config,
            name="*ST 某某",
            industry="医药",
            ipo_date="20100101",
            listing_days=5000,
            latest_price=5.0,
            latest_volume=1_000_000,
            amount_1d=3e8,
            amount_avg_5d=3e8,
            float_market_cap=2e9,
        )
        assert result.passed is False
        assert any("st_stock" in r for r in result.fail_reasons)

    def test_multiple_failures(self, default_config):
        result = apply_hard_filters(
            config=default_config,
            name="ST 低价",
            industry="银行",
            ipo_date="20230101",    # 上市天数不足
            listing_days=20,        # < 30 = 20*1.5
            latest_price=1.5,       # < 3.0
            latest_volume=1_000_000,
            amount_1d=5e7,
            amount_avg_5d=5e7,      # < 2e8
            float_market_cap=5e8,   # < 1.5e9
        )
        assert result.passed is False
        # 应有多条失败原因
        assert len(result.fail_reasons) >= 4
        reasons_str = " ".join(result.fail_reasons)
        assert "st_stock" in reasons_str
        assert "industry_finance" in reasons_str
        assert "price_too_low" in reasons_str

    def test_ipo_date_missing_hard_fail(self, default_config):
        result = apply_hard_filters(
            config=default_config,
            name="某某股份",
            industry="半导体",
            ipo_date=None,          # 强制 hard_fail
            listing_days=None,
            latest_price=50.0,
            latest_volume=1_000_000,
            amount_1d=5e8,
            amount_avg_5d=5e8,
            float_market_cap=5e9,
        )
        assert result.passed is False
        assert any("ipo_date_missing" in r for r in result.fail_reasons)

    def test_low_amount_fails(self, default_config):
        result = apply_hard_filters(
            config=default_config,
            name="正常股",
            industry="电子",
            ipo_date="20100101",
            listing_days=5000,
            latest_price=10.0,
            latest_volume=50_000,
            amount_1d=1e7,
            amount_avg_5d=1e7,     # < 2e8
            float_market_cap=5e9,
        )
        assert result.passed is False
        assert any("amount_too_low" in r for r in result.fail_reasons)

    def test_data_warnings_when_fields_none(self, default_config):
        """关键字段缺失时产生 data_warnings 而非 fail."""
        result = apply_hard_filters(
            config=default_config,
            name="某某",
            industry=None,          # 无行业 → H9 warning
            ipo_date="20100101",
            listing_days=None,      # 无上市天数 → H3 warning
            latest_price=None,      # 无价格 → H4 warning
            latest_volume=None,
            amount_1d=None,
            amount_avg_5d=None,     # 无成交额 → H6 warning
            float_market_cap=None,  # 无市值 → H7 warning
        )
        # 没有强制失败原因（ipo_date 有值），只有 warnings
        assert result.passed is True
        assert len(result.data_warnings) >= 3

    def test_primary_reason_is_first(self, default_config):
        result = apply_hard_filters(
            config=default_config,
            name="ST 低价小盘",
            industry="银行",
            ipo_date="20240101",
            listing_days=100,
            latest_price=1.0,
            latest_volume=1_000_000,
            amount_1d=1e7,
            amount_avg_5d=1e7,
            float_market_cap=5e8,
        )
        assert result.primary_reason == result.fail_reasons[0]
