"""Tests for daily_runner.py v2 — 全流程集成.

覆盖:
  - resolve_complete_trade_date (17:30 CST 策略)
  - parse_summary_csv (tradeable/watch_only 分离)
  - parse_setup_timing_csv
  - Discord 消息格式化 (overview/tradeable/watch_only/timing)
  - DailyRunResult 数据模型
"""

from __future__ import annotations

import datetime
import os
import tempfile
import pytest
from unittest.mock import patch, MagicMock

from a_share_hot_screener.daily_runner import (
    TradeDate,
    DailyRunResult,
    resolve_complete_trade_date,
    _find_prev_weekday,
    is_trading_day,
    parse_summary_csv,
    parse_setup_timing_csv,
    format_overview_embed,
    format_tradeable_embed,
    format_watch_only_embed,
    format_setup_timing_embed,
    CST,
)


# ════════════════════════════════════════════════════════
# 交易日解析
# ════════════════════════════════════════════════════════

class TestResolveCompleteTradeDate:
    def test_explicit_override(self):
        result = resolve_complete_trade_date(override_date="2026-04-22")
        assert result.trade_date_used == "2026-04-22"
        assert result.data_ready_policy == "explicit"
        assert result.partial_data_risk is False

    def test_after_data_ready_time(self):
        """17:30 CST 之后 → 当天."""
        # Wednesday 18:00 CST
        mock_now = datetime.datetime(2026, 4, 22, 10, 0, 0, tzinfo=datetime.timezone.utc)  # 18:00 CST
        with patch("a_share_hot_screener.daily_runner.datetime") as mock_dt:
            mock_dt.datetime.now.return_value = mock_now
            mock_dt.timezone = datetime.timezone
            mock_dt.timedelta = datetime.timedelta
            mock_dt.date = datetime.date
            result = resolve_complete_trade_date()
            assert result.trade_date_used == "2026-04-22"
            assert result.data_ready_policy == "default_17:30"
            assert result.partial_data_risk is False

    def test_between_close_and_ready_default(self):
        """15:30 CST (收盘后, 数据未就绪), allow_partial=False → 前一天."""
        # Wednesday 15:30 CST = 07:30 UTC
        mock_now = datetime.datetime(2026, 4, 22, 7, 30, 0, tzinfo=datetime.timezone.utc)
        with patch("a_share_hot_screener.daily_runner.datetime") as mock_dt:
            mock_dt.datetime.now.return_value = mock_now
            mock_dt.timezone = datetime.timezone
            mock_dt.timedelta = datetime.timedelta
            mock_dt.date = datetime.date
            result = resolve_complete_trade_date(allow_partial=False)
            assert result.trade_date_used == "2026-04-21"
            assert result.partial_data_risk is False

    def test_between_close_and_ready_partial(self):
        """15:30 CST, allow_partial=True → 当天 + partial_data_risk."""
        mock_now = datetime.datetime(2026, 4, 22, 7, 30, 0, tzinfo=datetime.timezone.utc)
        with patch("a_share_hot_screener.daily_runner.datetime") as mock_dt:
            mock_dt.datetime.now.return_value = mock_now
            mock_dt.timezone = datetime.timezone
            mock_dt.timedelta = datetime.timedelta
            mock_dt.date = datetime.date
            result = resolve_complete_trade_date(allow_partial=True)
            assert result.trade_date_used == "2026-04-22"
            assert result.partial_data_risk is True
            assert result.data_ready_policy == "partial_allowed"


class TestFindPrevWeekday:
    def test_monday_to_friday(self):
        # Monday → previous Friday
        d = datetime.date(2026, 4, 20)  # Monday
        assert _find_prev_weekday(d) == datetime.date(2026, 4, 17)  # Friday

    def test_wednesday_to_tuesday(self):
        d = datetime.date(2026, 4, 22)  # Wednesday
        assert _find_prev_weekday(d) == datetime.date(2026, 4, 21)  # Tuesday

    def test_sunday_to_friday(self):
        d = datetime.date(2026, 4, 19)  # Sunday
        assert _find_prev_weekday(d) == datetime.date(2026, 4, 17)  # Friday


class TestIsTradingDay:
    def test_weekday(self):
        assert is_trading_day("2026-04-22") is True  # Wednesday

    def test_weekend(self):
        assert is_trading_day("2026-04-25") is False  # Saturday


# ════════════════════════════════════════════════════════
# CSV 解析
# ════════════════════════════════════════════════════════

class TestParseSummaryCsv:
    def test_basic_parsing(self, tmp_path):
        csv_content = (
            "code,name,industry,total_score,hot_theme_score,trend_flow_score,"
            "liquidity_execution_score,risk_control_score,return_5d,return_10d,"
            "concept_names_str,candidate_pool_type,candidate_pool_reason,"
            "passed_hard_filter,pass_stage1,pass_stage1_watch,"
            "timing_score,timing_action\n"
            "600519,贵州茅台,白酒,0.82,0.85,0.78,0.71,0.90,5.2,8.1,"
            "白酒,tradeable,,True,True,False,78.5,watch\n"
            "000001,平安银行,银行,0.72,0.70,0.68,0.65,0.85,3.1,4.2,"
            ",watch_only,一字涨停,True,False,True,,\n"
            "002345,中关村,软件,0.45,0.40,0.38,0.35,0.50,1.0,2.0,"
            ",,True,True,False,False,,\n"
        )
        csv_file = tmp_path / "2026-04-22_stage1_hot_summary.csv"
        csv_file.write_text(csv_content, encoding="utf-8-sig")

        tradeable, watch_only, total, hf = parse_summary_csv(str(tmp_path), "2026-04-22")
        assert total == 3
        assert hf == 3
        assert len(tradeable) == 1
        assert tradeable[0]["code"] == "600519"
        assert tradeable[0]["timing_action"] == "watch"
        assert len(watch_only) == 1
        assert watch_only[0]["code"] == "000001"

    def test_missing_file(self, tmp_path):
        tradeable, watch_only, total, hf = parse_summary_csv(str(tmp_path), "2026-04-22")
        assert tradeable == []
        assert watch_only == []
        assert total == 0


class TestParseSetupTimingCsv:
    def test_basic(self, tmp_path):
        csv_content = (
            "code,name,timing_score,action,support_zone_low,support_zone_high,"
            "invalidation_level,resistance_1,ref_reward_risk,level_confidence,"
            "support_basis,reason\n"
            "600519,贵州茅台,82.5,setup_ready,1800.0,1850.0,1750.0,1950.0,2.1,"
            "high,ma10,趋势多头回踩到位\n"
        )
        csv_file = tmp_path / "2026-04-22_setup_timing.csv"
        csv_file.write_text(csv_content, encoding="utf-8-sig")

        results = parse_setup_timing_csv(str(tmp_path), "2026-04-22")
        assert len(results) == 1
        assert results[0]["code"] == "600519"
        assert results[0]["action"] == "setup_ready"
        assert results[0]["timing_score"] == pytest.approx(82.5)

    def test_missing_file(self, tmp_path):
        results = parse_setup_timing_csv(str(tmp_path), "2026-04-22")
        assert results == []


# ════════════════════════════════════════════════════════
# Discord 消息格式化
# ════════════════════════════════════════════════════════

class TestFormatOverviewEmbed:
    def test_basic(self):
        result = DailyRunResult(
            run_date="2026-04-22",
            trade_date_used="2026-04-22",
            data_ready_policy="default_17:30",
            universe_count=1500,
            universe_static_count=1200,
            passed_hard_filter=1000,
            tradeable_count=15,
            watch_only_count=3,
            elapsed_seconds=3000,
            output_dir="/tmp",
        )
        embed = format_overview_embed(result)
        assert "2026-04-22" in embed["title"]
        assert "1500" in embed["description"]
        assert "15" in embed["description"]
        assert embed["color"] == 0xFF6B35  # 有通过股票

    def test_no_tradeable(self):
        result = DailyRunResult(
            run_date="2026-04-22",
            trade_date_used="2026-04-22",
            data_ready_policy="default_17:30",
            tradeable_count=0,
            elapsed_seconds=100,
            output_dir="/tmp",
        )
        embed = format_overview_embed(result)
        assert embed["color"] == 0x808080  # 无通过股票

    def test_partial_risk(self):
        result = DailyRunResult(
            run_date="2026-04-22",
            trade_date_used="2026-04-22",
            data_ready_policy="partial_allowed",
            partial_data_risk=True,
            tradeable_count=5,
            elapsed_seconds=100,
            output_dir="/tmp",
        )
        embed = format_overview_embed(result)
        assert "partial_risk" in embed["description"]


class TestFormatTradeableEmbed:
    def test_with_stocks(self):
        stocks = [
            {"code": "600519", "name": "贵州茅台", "industry": "白酒",
             "total_score": 0.85, "hot_theme_score": 0.9, "trend_flow_score": 0.8,
             "liquidity_execution_score": 0.7, "risk_control_score": 0.9,
             "concept_names_str": "白酒", "timing_action": "watch", "timing_score": "75"},
        ]
        embeds = format_tradeable_embed(stocks)
        assert len(embeds) >= 1
        assert "贵州茅台" in embeds[0].get("description", "") or "贵州茅台" in embeds[-1].get("description", "")

    def test_empty(self):
        assert format_tradeable_embed([]) == []


class TestFormatWatchOnlyEmbed:
    def test_with_stocks(self):
        stocks = [
            {"code": "000001", "name": "平安银行", "total_score": 0.72,
             "candidate_pool_reason": "一字涨停"},
        ]
        embed = format_watch_only_embed(stocks)
        assert embed is not None
        assert "Watch-Only" in embed["title"]
        assert "平安银行" in embed["description"]

    def test_empty(self):
        assert format_watch_only_embed([]) is None


class TestFormatSetupTimingEmbed:
    def test_with_results(self):
        results = [
            {"code": "600519", "name": "贵州茅台", "timing_score": 85.0,
             "action": "setup_ready", "support_zone_low": "1800",
             "support_zone_high": "1850", "support_basis": "ma10",
             "level_confidence": "high", "invalidation_level": "1750",
             "ref_reward_risk": "2.1", "reason": "趋势多头"},
            {"code": "000858", "name": "五粮液", "timing_score": 70.0,
             "action": "watch", "reason": "回踩接近"},
            {"code": "300750", "name": "宁德时代", "timing_score": 50.0,
             "action": "wait", "reason": "位置偏高"},
        ]
        embed = format_setup_timing_embed(results)
        assert embed is not None
        assert "观察时机" in embed["title"]
        assert "🟢" in embed["description"]
        assert "🟡" in embed["description"]
        assert "⏳" in embed["description"]

    def test_empty(self):
        assert format_setup_timing_embed([]) is None


# ════════════════════════════════════════════════════════
# DailyRunResult
# ════════════════════════════════════════════════════════

class TestDailyRunResult:
    def test_default_values(self):
        r = DailyRunResult(
            run_date="2026-04-22",
            trade_date_used="2026-04-22",
            data_ready_policy="default_17:30",
            output_dir="/tmp",
        )
        assert r.tradeable_count == 0
        assert r.watch_only_count == 0
        assert r.error_msg == ""
        assert r.partial_data_risk is False

    def test_with_error(self):
        r = DailyRunResult(
            run_date="2026-04-22",
            trade_date_used="2026-04-22",
            data_ready_policy="default_17:30",
            output_dir="/tmp",
            error_msg="test error",
        )
        assert r.error_msg == "test error"
