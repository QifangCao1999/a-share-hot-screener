"""test_trade_calendar.py – 测试 TradeCalendar（Session 3 完整实现）."""

import datetime as dt
import os
import tempfile
import pytest

from a_share_hot_screener.cache import LocalCache
from a_share_hot_screener.trade_calendar import TradeCalendar, _bisect_left, _bisect_right


def make_cal_with_dates(dates, run_date):
    """构造一个已加载指定交易日列表的 TradeCalendar（mock）."""
    cal = TradeCalendar(run_date)
    cal._trade_dates = sorted(dates)
    cal._trade_date_set = set(dates)
    cal._loaded = True
    cal._fallback = False
    return cal


class TestTradeCalendarResolve:
    def test_trade_day_no_warning(self):
        dates = [dt.date(2026, 4, 15), dt.date(2026, 4, 16), dt.date(2026, 4, 17)]
        cal = make_cal_with_dates(dates, dt.date(2026, 4, 17))
        cal.resolve()
        assert cal.get_trade_date_used() == dt.date(2026, 4, 17)
        assert cal.get_warnings() == []

    def test_non_trade_day_rolls_back(self):
        # 2026-04-18 是周六，应回退到 2026-04-17（周五，在 dates 中）
        dates = [dt.date(2026, 4, 15), dt.date(2026, 4, 16), dt.date(2026, 4, 17)]
        cal = make_cal_with_dates(dates, dt.date(2026, 4, 18))
        cal.resolve()
        assert cal.get_trade_date_used() == dt.date(2026, 4, 17)
        assert len(cal.get_warnings()) == 1
        assert "自动回退" in cal.get_warnings()[0]

    def test_idempotent(self):
        dates = [dt.date(2026, 4, 17)]
        cal = make_cal_with_dates(dates, dt.date(2026, 4, 17))
        cal.resolve()
        cal.resolve()  # 第二次调用不改变结果
        assert cal.get_trade_date_used() == dt.date(2026, 4, 17)


class TestIsTradeDate:
    def test_in_list(self):
        dates = [dt.date(2026, 4, 17)]
        cal = make_cal_with_dates(dates, dt.date(2026, 4, 17))
        assert cal.is_trade_date(dt.date(2026, 4, 17)) is True

    def test_not_in_list(self):
        dates = [dt.date(2026, 4, 17)]
        cal = make_cal_with_dates(dates, dt.date(2026, 4, 17))
        assert cal.is_trade_date(dt.date(2026, 4, 18)) is False

    def test_fallback_weekend(self):
        cal = TradeCalendar(dt.date(2026, 4, 17))
        cal._loaded = True
        cal._fallback = True
        assert cal.is_trade_date(dt.date(2026, 4, 17)) is True   # 周五
        assert cal.is_trade_date(dt.date(2026, 4, 18)) is False  # 周六


class TestPrevTradeDate:
    def test_prev_from_weekday(self):
        dates = [dt.date(2026, 4, 15), dt.date(2026, 4, 16), dt.date(2026, 4, 17)]
        cal = make_cal_with_dates(dates, dt.date(2026, 4, 17))
        assert cal.prev_trade_date(dt.date(2026, 4, 17)) == dt.date(2026, 4, 16)

    def test_prev_from_date_not_in_list(self):
        # 2026-04-18（周六）→ 最近前一个交易日 = 2026-04-17
        dates = [dt.date(2026, 4, 15), dt.date(2026, 4, 16), dt.date(2026, 4, 17)]
        cal = make_cal_with_dates(dates, dt.date(2026, 4, 18))
        assert cal.prev_trade_date(dt.date(2026, 4, 18)) == dt.date(2026, 4, 17)

    def test_prev_fallback(self):
        cal = TradeCalendar(dt.date(2026, 4, 20))  # 周一
        cal._loaded = True
        cal._fallback = True
        assert cal.prev_trade_date(dt.date(2026, 4, 20)) == dt.date(2026, 4, 17)


class TestNTradeDatesBefore:
    def test_basic(self):
        # 提供至少 n+1 个日期（含 date 本身），向前5个：idx=4, 4-5=-1 → 不足
        # 正确用法：日期列表需要 > n 个才能向前 n 个
        dates = [dt.date(2026, 4, d) for d in [13, 14, 15, 16, 17, 18, 19, 20]]
        cal = make_cal_with_dates(dates, dt.date(2026, 4, 17))
        # date=2026-04-17 在列表 idx=4（0-based），向前5个 → idx-5=-1 → 不足
        # 改为向前4个：idx=4, 4-4=0 → 2026-04-13
        result = cal.n_trade_dates_before(dt.date(2026, 4, 17), 4)
        assert result == dt.date(2026, 4, 13)

    def test_3_before(self):
        dates = [dt.date(2026, 4, d) for d in [14, 15, 16, 17]]
        cal = make_cal_with_dates(dates, dt.date(2026, 4, 17))
        result = cal.n_trade_dates_before(dt.date(2026, 4, 17), 3)
        assert result == dt.date(2026, 4, 14)


class TestEodStartDate:
    def test_returns_date(self):
        dates = [dt.date(2026, 4, d) for d in range(1, 18)]  # 1~17，17个交易日
        cal = make_cal_with_dates(dates, dt.date(2026, 4, 17))
        start = cal.eod_start_date(dt.date(2026, 4, 17), window=5, extra=2)
        # 向前 7 个交易日
        assert start == dt.date(2026, 4, 10)


class TestCountTradeDays:
    def test_both_inclusive(self):
        dates = [dt.date(2026, 4, d) for d in [14, 15, 16, 17]]
        cal = make_cal_with_dates(dates, dt.date(2026, 4, 17))
        count = cal.count_trade_days_between(dt.date(2026, 4, 14), dt.date(2026, 4, 17))
        assert count == 4

    def test_left_inclusive(self):
        dates = [dt.date(2026, 4, d) for d in [14, 15, 16, 17]]
        cal = make_cal_with_dates(dates, dt.date(2026, 4, 17))
        count = cal.count_trade_days_between(
            dt.date(2026, 4, 14), dt.date(2026, 4, 17), inclusive="left"
        )
        assert count == 3  # 不含 17


class TestTushareLoad:
    @pytest.mark.skipif(not os.environ.get("TUSHARE_TOKEN"), reason="TUSHARE_TOKEN not set")
    def test_load_from_tushare(self, tmp_path):
        """实际调用 Tushare（integration test，需网络 + TUSHARE_TOKEN）."""
        import pytest
        pytest.importorskip("tushare")
        cache = LocalCache(str(tmp_path / "cal_cache"))
        cal = TradeCalendar(dt.date(2026, 4, 17), cache=cache)
        ok = cal.load()
        assert ok is True
        assert len(cal._trade_dates) > 5000
        assert dt.date(2026, 4, 17) in cal._trade_date_set  # 周四是交易日
        assert dt.date(2026, 4, 18) not in cal._trade_date_set  # 周六不是

    @pytest.mark.skipif(not os.environ.get("TUSHARE_TOKEN"), reason="TUSHARE_TOKEN not set")
    def test_load_uses_cache(self, tmp_path):
        """第二次 load 命中缓存."""
        import pytest
        pytest.importorskip("tushare")
        cache = LocalCache(str(tmp_path / "cal_cache2"))
        c1 = TradeCalendar(dt.date(2026, 4, 17), cache=cache)
        c1.load()
        n = len(c1._trade_dates)
        c2 = TradeCalendar(dt.date(2026, 4, 17), cache=cache)
        c2.load()
        assert len(c2._trade_dates) == n


class TestBisect:
    def test_bisect_left(self):
        dates = [dt.date(2026, 4, d) for d in [14, 15, 16, 17]]
        assert _bisect_left(dates, dt.date(2026, 4, 15)) == 1
        assert _bisect_left(dates, dt.date(2026, 4, 18)) == 4

    def test_bisect_right(self):
        dates = [dt.date(2026, 4, d) for d in [14, 15, 16, 17]]
        assert _bisect_right(dates, dt.date(2026, 4, 17)) == 4
