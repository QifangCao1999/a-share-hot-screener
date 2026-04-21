"""test_date_utils.py – 测试日期工具函数."""

import datetime as dt
import pytest

from a_share_hot_screener.date_utils import (
    eod_lookback_start,
    ensure_not_future,
    is_on_or_before,
    lookback_dates,
    parse_run_date,
    str_to_date,
    date_to_str,
)


class TestParseRunDate:
    def test_yyyy_mm_dd(self):
        assert parse_run_date("2026-04-18") == dt.date(2026, 4, 18)

    def test_yyyymmdd(self):
        assert parse_run_date("20260418") == dt.date(2026, 4, 18)

    def test_today(self):
        result = parse_run_date("today")
        assert result == dt.date.today()

    def test_invalid(self):
        with pytest.raises(ValueError):
            parse_run_date("2026/04/18")

    def test_strips_whitespace(self):
        assert parse_run_date("  2026-04-18  ") == dt.date(2026, 4, 18)


class TestIsOnOrBefore:
    def test_same_date(self):
        d = dt.date(2026, 4, 18)
        assert is_on_or_before(d, d) is True

    def test_before(self):
        assert is_on_or_before(dt.date(2026, 4, 17), dt.date(2026, 4, 18)) is True

    def test_after(self):
        assert is_on_or_before(dt.date(2026, 4, 19), dt.date(2026, 4, 18)) is False


class TestEnsureNotFuture:
    def test_not_future(self):
        result = ensure_not_future("test", dt.date(2026, 4, 17), dt.date(2026, 4, 18))
        assert result is None

    def test_future(self):
        result = ensure_not_future("price", dt.date(2026, 4, 19), dt.date(2026, 4, 18))
        assert result is not None
        assert "as-of violation" in result
        assert "price" in result


class TestEodLookbackStart:
    def test_default_60_days(self):
        run_date = dt.date(2026, 4, 18)
        start = eod_lookback_start(run_date, calendar_days=60)
        assert start == dt.date(2026, 2, 17)

    def test_custom_days(self):
        run_date = dt.date(2026, 4, 18)
        start = eod_lookback_start(run_date, calendar_days=30)
        assert start == run_date - dt.timedelta(days=30)


class TestLookbackDates:
    def test_n_days(self):
        run_date = dt.date(2026, 4, 18)
        dates = lookback_dates(run_date, 5)
        assert len(dates) == 5
        assert dates[0] == run_date
        assert dates[-1] == run_date - dt.timedelta(days=4)


class TestDateConversions:
    def test_date_to_str(self):
        assert date_to_str(dt.date(2026, 4, 18)) == "2026-04-18"

    def test_str_to_date(self):
        assert str_to_date("2026-04-18") == dt.date(2026, 4, 18)
