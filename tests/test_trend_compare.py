"""Tests for trend_compare module (Session 14 P2-6).

Coverage:
  1. PrevRunSnapshot loading from CSV (load_prev_run / _parse_summary_csv)
  2. Delta computation (compute_trend_delta)
  3. Batch delta computation (compute_all_deltas)
  4. Edge cases: missing prev data, new_entry, same-date skip
  5. CSV parsing edge cases (_parse_float, _parse_bool, _parse_list)
  6. Pipeline integration (trend_delta dict on HotStockDetail)
  7. Summary propagation (HotStockSummary.from_detail)
  8. Output: detail long table includes delta rows
  9. CLI argument parsing (--prev-run-dir, --prev-run-date)
  10. Config fields (prev_run_dir, prev_run_date)
  11. Metadata fields (trend_compare_enabled, trend_compare_prev_run_date)
"""

from __future__ import annotations

import csv
import os
import tempfile
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from unittest.mock import patch, MagicMock

import pytest

from a_share_hot_screener.trend_compare import (
    TrendDelta,
    PrevRunSnapshot,
    compute_trend_delta,
    compute_all_deltas,
    load_prev_run,
    _parse_float,
    _parse_bool,
    _parse_list,
    _safe_delta,
)
from a_share_hot_screener.models import (
    HotStockDetail,
    HotStockSummary,
    RunMetadata,
)
from a_share_hot_screener.config import HotScreenerConfig
from a_share_hot_screener.cli import build_parser


# ════════════════════════════════════════════════════════
# Fixtures
# ════════════════════════════════════════════════════════

@pytest.fixture
def prev_snapshot() -> PrevRunSnapshot:
    """Fixture: a basic prev run snapshot with 3 stocks."""
    return PrevRunSnapshot(
        trade_date_used="2026-04-17",
        stocks={
            "600519": {
                "code": "600519",
                "name": "贵州茅台",
                "pass_stage1": True,
                "passed_hard_filter": True,
                "total_score": 0.82,
                "hot_theme_score": 0.90,
                "trend_flow_score": 0.75,
                "liquidity_execution_score": 0.80,
                "risk_control_score": 0.85,
                "blocked_by": [],
                "return_5d": 5.2,
                "return_10d": 8.1,
                "amount_avg_5d": 3_000_000_000.0,
            },
            "000858": {
                "code": "000858",
                "name": "五粮液",
                "pass_stage1": False,
                "passed_hard_filter": True,
                "total_score": 0.55,
                "hot_theme_score": 0.40,
                "trend_flow_score": 0.50,
                "liquidity_execution_score": 0.70,
                "risk_control_score": 0.90,
                "blocked_by": ["hot_theme", "total_score"],
                "return_5d": 2.1,
                "return_10d": 3.5,
                "amount_avg_5d": 1_500_000_000.0,
            },
            "300750": {
                "code": "300750",
                "name": "宁德时代",
                "pass_stage1": False,
                "passed_hard_filter": False,
                "total_score": None,
                "hot_theme_score": None,
                "trend_flow_score": None,
                "liquidity_execution_score": None,
                "risk_control_score": None,
                "blocked_by": [],
                "return_5d": -1.5,
                "return_10d": -3.2,
                "amount_avg_5d": 500_000_000.0,
            },
        },
    )


def _make_detail(
    code: str,
    name: str = "",
    total_score: Optional[float] = None,
    hot_theme_score: Optional[float] = None,
    trend_flow_score: Optional[float] = None,
    liquidity_execution_score: Optional[float] = None,
    risk_control_score: Optional[float] = None,
    pass_stage1: bool = False,
    passed_hard_filter: bool = True,
    blocked_by: Optional[List[str]] = None,
    return_5d: Optional[float] = None,
    return_10d: Optional[float] = None,
    amount_avg_5d: Optional[float] = None,
) -> HotStockDetail:
    d = HotStockDetail(code=code, name=name)
    d.total_score = total_score
    d.hot_theme_score = hot_theme_score
    d.trend_flow_score = trend_flow_score
    d.liquidity_execution_score = liquidity_execution_score
    d.risk_control_score = risk_control_score
    d.pass_stage1 = pass_stage1
    d.passed_hard_filter = passed_hard_filter
    d.blocked_by = blocked_by or []
    d.return_5d = return_5d
    d.return_10d = return_10d
    d.amount_avg_5d = amount_avg_5d
    return d


# ════════════════════════════════════════════════════════
# 1. CSV Parsing Utilities
# ════════════════════════════════════════════════════════

class TestParseUtilities:
    """Test _parse_float, _parse_bool, _parse_list."""

    def test_parse_float_valid(self):
        assert _parse_float("0.82") == 0.82
        assert _parse_float("3000000000.0") == 3_000_000_000.0
        assert _parse_float("-1.5") == -1.5

    def test_parse_float_empty(self):
        assert _parse_float("") is None
        assert _parse_float("  ") is None

    def test_parse_float_none_string(self):
        assert _parse_float("None") is None
        assert _parse_float("nan") is None
        assert _parse_float("NaN") is None

    def test_parse_float_invalid(self):
        assert _parse_float("abc") is None
        assert _parse_float("[1,2]") is None

    def test_parse_bool_true(self):
        assert _parse_bool("True") is True
        assert _parse_bool("true") is True
        assert _parse_bool("1") is True
        assert _parse_bool("yes") is True

    def test_parse_bool_false(self):
        assert _parse_bool("False") is False
        assert _parse_bool("") is False
        assert _parse_bool("0") is False
        assert _parse_bool("no") is False

    def test_parse_list_empty(self):
        assert _parse_list("") == []
        assert _parse_list("[]") == []
        assert _parse_list("['']") == []
        assert _parse_list('[""]') == []

    def test_parse_list_single(self):
        assert _parse_list("['hot_theme']") == ["hot_theme"]
        assert _parse_list("hot_theme") == ["hot_theme"]

    def test_parse_list_multiple(self):
        result = _parse_list("['hot_theme', 'trend_flow', 'total_score']")
        assert result == ["hot_theme", "trend_flow", "total_score"]

    def test_parse_list_no_quotes(self):
        result = _parse_list("[hot_theme,trend_flow]")
        assert result == ["hot_theme", "trend_flow"]

    def test_parse_list_double_quotes(self):
        result = _parse_list('["hot_theme", "trend_flow"]')
        assert result == ["hot_theme", "trend_flow"]


# ════════════════════════════════════════════════════════
# 2. _safe_delta
# ════════════════════════════════════════════════════════

class TestSafeDelta:
    def test_both_present(self):
        assert _safe_delta(0.85, 0.80) == 0.05

    def test_negative_delta(self):
        assert _safe_delta(0.70, 0.80) == -0.10

    def test_current_none(self):
        assert _safe_delta(None, 0.80) is None

    def test_prev_none(self):
        assert _safe_delta(0.85, None) is None

    def test_both_none(self):
        assert _safe_delta(None, None) is None

    def test_rounding(self):
        result = _safe_delta(0.333333, 0.111111)
        assert result == 0.2222


# ════════════════════════════════════════════════════════
# 3. TrendDelta
# ════════════════════════════════════════════════════════

class TestTrendDelta:
    def test_default_values(self):
        td = TrendDelta()
        assert td.prev_run_date == ""
        assert td.total_score_delta is None
        assert td.pass_stage1_change == ""
        assert td.score_accelerating is None
        assert td.newly_blocked == []
        assert td.newly_unblocked == []

    def test_to_dict(self):
        td = TrendDelta(
            prev_run_date="2026-04-17",
            total_score_delta=0.05,
            pass_stage1_change="keep_pass",
            score_accelerating=True,
            prev_blocked_by=["hot_theme"],
            newly_unblocked=["hot_theme"],
        )
        d = td.to_dict()
        assert d["prev_run_date"] == "2026-04-17"
        assert d["total_score_delta"] == 0.05
        assert d["pass_stage1_change"] == "keep_pass"
        assert d["score_accelerating"] is True
        assert d["newly_unblocked"] == ["hot_theme"]

    def test_to_dict_none_values(self):
        td = TrendDelta()
        d = td.to_dict()
        assert d["total_score_delta"] is None
        assert d["score_accelerating"] is None


# ════════════════════════════════════════════════════════
# 4. compute_trend_delta - Core Logic
# ════════════════════════════════════════════════════════

class TestComputeTrendDelta:
    """Test compute_trend_delta for various scenarios."""

    def test_keep_pass_with_improvement(self, prev_snapshot):
        """Stock that passes both times with score improvement."""
        delta = compute_trend_delta(
            code="600519",
            current_total_score=0.87,
            current_hot_theme_score=0.93,
            current_trend_flow_score=0.78,
            current_liquidity_execution_score=0.82,
            current_risk_control_score=0.88,
            current_pass_stage1=True,
            current_passed_hard_filter=True,
            current_blocked_by=[],
            current_return_5d=6.0,
            current_return_10d=9.5,
            current_amount_avg_5d=3_200_000_000.0,
            prev_snapshot=prev_snapshot,
        )

        assert delta.prev_run_date == "2026-04-17"
        assert delta.pass_stage1_change == "keep_pass"
        assert delta.hard_filter_change == "keep_pass"
        assert delta.total_score_delta == 0.05
        assert delta.hot_theme_score_delta == 0.03
        assert delta.prev_total_score == 0.82
        assert delta.score_accelerating is True
        assert delta.score_decelerating is False

    def test_keep_pass_with_decline(self, prev_snapshot):
        """Stock that passes both times but score declines > 1%."""
        delta = compute_trend_delta(
            code="600519",
            current_total_score=0.80,
            current_hot_theme_score=0.88,
            current_trend_flow_score=0.73,
            current_liquidity_execution_score=0.78,
            current_risk_control_score=0.83,
            current_pass_stage1=True,
            current_passed_hard_filter=True,
            current_blocked_by=[],
            current_return_5d=4.8,
            current_return_10d=7.0,
            current_amount_avg_5d=2_800_000_000.0,
            prev_snapshot=prev_snapshot,
        )

        assert delta.pass_stage1_change == "keep_pass"
        assert delta.total_score_delta == -0.02
        assert delta.score_accelerating is False
        assert delta.score_decelerating is True

    def test_keep_pass_stable(self, prev_snapshot):
        """Score change within 1% threshold -> neither accelerating nor decelerating."""
        delta = compute_trend_delta(
            code="600519",
            current_total_score=0.825,  # +0.005 < 0.01 threshold
            current_hot_theme_score=0.90,
            current_trend_flow_score=0.75,
            current_liquidity_execution_score=0.80,
            current_risk_control_score=0.85,
            current_pass_stage1=True,
            current_passed_hard_filter=True,
            current_blocked_by=[],
            current_return_5d=5.2,
            current_return_10d=8.1,
            current_amount_avg_5d=3_000_000_000.0,
            prev_snapshot=prev_snapshot,
        )

        assert delta.pass_stage1_change == "keep_pass"
        assert delta.total_score_delta == 0.005
        assert delta.score_accelerating is False
        assert delta.score_decelerating is False

    def test_new_pass(self, prev_snapshot):
        """Stock that fails previously, passes now."""
        delta = compute_trend_delta(
            code="000858",
            current_total_score=0.70,
            current_hot_theme_score=0.68,
            current_trend_flow_score=0.65,
            current_liquidity_execution_score=0.72,
            current_risk_control_score=0.92,
            current_pass_stage1=True,
            current_passed_hard_filter=True,
            current_blocked_by=[],
            current_return_5d=4.0,
            current_return_10d=6.5,
            current_amount_avg_5d=1_800_000_000.0,
            prev_snapshot=prev_snapshot,
        )

        assert delta.pass_stage1_change == "new_pass"
        assert delta.hard_filter_change == "keep_pass"
        assert delta.total_score_delta == 0.15  # 0.70 - 0.55
        assert delta.newly_unblocked == ["hot_theme", "total_score"]
        assert delta.newly_blocked == []
        # not keep_pass, so score_accelerating should be None
        assert delta.score_accelerating is None

    def test_lost_pass(self, prev_snapshot):
        """Stock that passes previously, fails now."""
        delta = compute_trend_delta(
            code="600519",
            current_total_score=0.60,
            current_hot_theme_score=0.50,
            current_trend_flow_score=0.55,
            current_liquidity_execution_score=0.65,
            current_risk_control_score=0.70,
            current_pass_stage1=False,
            current_passed_hard_filter=True,
            current_blocked_by=["hot_theme", "trend_flow"],
            current_return_5d=1.0,
            current_return_10d=2.0,
            current_amount_avg_5d=2_500_000_000.0,
            prev_snapshot=prev_snapshot,
        )

        assert delta.pass_stage1_change == "lost_pass"
        assert delta.total_score_delta == -0.22
        assert delta.newly_blocked == ["hot_theme", "trend_flow"]
        assert delta.newly_unblocked == []
        # Not keep_pass, so acceleration signals are None
        assert delta.score_accelerating is None

    def test_keep_fail(self, prev_snapshot):
        """Stock that fails both times."""
        delta = compute_trend_delta(
            code="000858",
            current_total_score=0.58,
            current_hot_theme_score=0.45,
            current_trend_flow_score=0.55,
            current_liquidity_execution_score=0.72,
            current_risk_control_score=0.91,
            current_pass_stage1=False,
            current_passed_hard_filter=True,
            current_blocked_by=["hot_theme"],
            current_return_5d=2.5,
            current_return_10d=4.0,
            current_amount_avg_5d=1_600_000_000.0,
            prev_snapshot=prev_snapshot,
        )

        assert delta.pass_stage1_change == "keep_fail"
        assert delta.prev_blocked_by == ["hot_theme", "total_score"]
        assert delta.newly_unblocked == ["total_score"]
        assert delta.newly_blocked == []

    def test_new_entry(self, prev_snapshot):
        """Stock not in previous run."""
        delta = compute_trend_delta(
            code="601318",
            current_total_score=0.75,
            current_hot_theme_score=0.70,
            current_trend_flow_score=0.68,
            current_liquidity_execution_score=0.78,
            current_risk_control_score=0.92,
            current_pass_stage1=True,
            current_passed_hard_filter=True,
            current_blocked_by=[],
            current_return_5d=3.5,
            current_return_10d=5.0,
            current_amount_avg_5d=2_000_000_000.0,
            prev_snapshot=prev_snapshot,
        )

        assert delta.pass_stage1_change == "new_entry"
        assert delta.hard_filter_change == "new_entry"
        assert delta.total_score_delta is None
        assert delta.prev_total_score is None
        assert delta.score_accelerating is None

    def test_hard_filter_lost_pass(self, prev_snapshot):
        """Stock that was in hard_filter but now fails."""
        delta = compute_trend_delta(
            code="000858",
            current_total_score=None,
            current_hot_theme_score=None,
            current_trend_flow_score=None,
            current_liquidity_execution_score=None,
            current_risk_control_score=None,
            current_pass_stage1=False,
            current_passed_hard_filter=False,
            current_blocked_by=[],
            current_return_5d=None,
            current_return_10d=None,
            current_amount_avg_5d=None,
            prev_snapshot=prev_snapshot,
        )

        assert delta.hard_filter_change == "lost_pass"
        assert delta.pass_stage1_change == "keep_fail"
        assert delta.total_score_delta is None  # current is None

    def test_hard_filter_new_pass(self, prev_snapshot):
        """Stock that previously failed hard_filter now passes."""
        delta = compute_trend_delta(
            code="300750",
            current_total_score=0.60,
            current_hot_theme_score=0.55,
            current_trend_flow_score=0.50,
            current_liquidity_execution_score=0.65,
            current_risk_control_score=0.80,
            current_pass_stage1=False,
            current_passed_hard_filter=True,
            current_blocked_by=["hot_theme", "trend_flow"],
            current_return_5d=1.0,
            current_return_10d=2.0,
            current_amount_avg_5d=800_000_000.0,
            prev_snapshot=prev_snapshot,
        )

        assert delta.hard_filter_change == "new_pass"
        assert delta.total_score_delta is None  # prev was None

    def test_return_deltas(self, prev_snapshot):
        """return_5d / return_10d / amount_avg_5d deltas."""
        delta = compute_trend_delta(
            code="600519",
            current_total_score=0.85,
            current_hot_theme_score=0.92,
            current_trend_flow_score=0.77,
            current_liquidity_execution_score=0.81,
            current_risk_control_score=0.86,
            current_pass_stage1=True,
            current_passed_hard_filter=True,
            current_blocked_by=[],
            current_return_5d=7.0,
            current_return_10d=10.0,
            current_amount_avg_5d=3_500_000_000.0,
            prev_snapshot=prev_snapshot,
        )

        assert delta.return_5d_delta == 1.8  # 7.0 - 5.2
        assert delta.return_10d_delta == 1.9  # 10.0 - 8.1
        assert delta.amount_avg_5d_delta == 500_000_000.0


# ════════════════════════════════════════════════════════
# 5. compute_all_deltas - Batch
# ════════════════════════════════════════════════════════

class TestComputeAllDeltas:
    def test_batch_computation(self, prev_snapshot):
        details = [
            _make_detail("600519", total_score=0.85, pass_stage1=True, blocked_by=[]),
            _make_detail("000858", total_score=0.58, pass_stage1=False, blocked_by=["hot_theme"]),
            _make_detail("601318", total_score=0.75, pass_stage1=True, blocked_by=[]),
        ]
        deltas = compute_all_deltas(details, prev_snapshot)

        assert len(deltas) == 3
        assert "600519" in deltas
        assert deltas["600519"].pass_stage1_change == "keep_pass"
        assert deltas["000858"].pass_stage1_change == "keep_fail"
        assert deltas["601318"].pass_stage1_change == "new_entry"


# ════════════════════════════════════════════════════════
# 6. CSV Loading (load_prev_run)
# ════════════════════════════════════════════════════════

class TestLoadPrevRun:
    def _write_summary_csv(self, tmpdir, run_date="2026-04-17"):
        """Helper: write a minimal summary CSV."""
        path = os.path.join(tmpdir, f"{run_date}_stage1_hot_summary.csv")
        fieldnames = [
            "code", "name", "pass_stage1", "passed_hard_filter",
            "total_score", "hot_theme_score", "trend_flow_score",
            "liquidity_execution_score", "risk_control_score",
            "blocked_by", "return_5d", "return_10d", "amount_avg_5d",
        ]
        rows = [
            {
                "code": "600519", "name": "贵州茅台",
                "pass_stage1": "True", "passed_hard_filter": "True",
                "total_score": "0.82", "hot_theme_score": "0.90",
                "trend_flow_score": "0.75", "liquidity_execution_score": "0.80",
                "risk_control_score": "0.85", "blocked_by": "[]",
                "return_5d": "5.2", "return_10d": "8.1",
                "amount_avg_5d": "3000000000.0",
            },
            {
                "code": "000858", "name": "五粮液",
                "pass_stage1": "False", "passed_hard_filter": "True",
                "total_score": "0.55", "hot_theme_score": "0.40",
                "trend_flow_score": "0.50", "liquidity_execution_score": "0.70",
                "risk_control_score": "0.90",
                "blocked_by": "['hot_theme', 'total_score']",
                "return_5d": "2.1", "return_10d": "3.5",
                "amount_avg_5d": "1500000000.0",
            },
        ]
        with open(path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        return path

    def test_load_with_explicit_date(self, tmp_path):
        self._write_summary_csv(str(tmp_path), "2026-04-17")
        snapshot = load_prev_run(str(tmp_path), prev_run_date="2026-04-17")

        assert snapshot is not None
        assert snapshot.trade_date_used == "2026-04-17"
        assert len(snapshot.stocks) == 2
        assert snapshot.stocks["600519"]["total_score"] == 0.82
        assert snapshot.stocks["600519"]["pass_stage1"] is True
        assert snapshot.stocks["000858"]["blocked_by"] == ["hot_theme", "total_score"]

    def test_load_auto_detect_latest(self, tmp_path):
        """Without explicit date, picks the latest file."""
        self._write_summary_csv(str(tmp_path), "2026-04-16")
        self._write_summary_csv(str(tmp_path), "2026-04-17")

        snapshot = load_prev_run(str(tmp_path))
        assert snapshot is not None
        assert snapshot.trade_date_used == "2026-04-17"

    def test_load_nonexistent_dir(self):
        snapshot = load_prev_run("/nonexistent/path")
        assert snapshot is None

    def test_load_empty_dir(self, tmp_path):
        snapshot = load_prev_run(str(tmp_path))
        assert snapshot is None

    def test_load_wrong_date(self, tmp_path):
        self._write_summary_csv(str(tmp_path), "2026-04-17")
        # Explicit date that doesn't exist falls back to auto-detect
        snapshot = load_prev_run(str(tmp_path), prev_run_date="2026-04-16")
        assert snapshot is not None
        assert snapshot.trade_date_used == "2026-04-17"  # auto-detect kicks in

    def test_load_empty_csv(self, tmp_path):
        """CSV with headers only."""
        path = os.path.join(str(tmp_path), "2026-04-17_stage1_hot_summary.csv")
        with open(path, "w", newline="", encoding="utf-8-sig") as f:
            f.write("code,name,total_score\n")

        snapshot = load_prev_run(str(tmp_path))
        assert snapshot is None  # empty rows

    def test_load_malformed_values(self, tmp_path):
        """CSV with some malformed values should still parse gracefully."""
        path = os.path.join(str(tmp_path), "2026-04-17_stage1_hot_summary.csv")
        fieldnames = ["code", "name", "total_score", "pass_stage1", "passed_hard_filter",
                       "blocked_by", "return_5d"]
        rows = [
            {"code": "600519", "name": "茅台", "total_score": "abc",
             "pass_stage1": "True", "passed_hard_filter": "True",
             "blocked_by": "", "return_5d": ""},
        ]
        with open(path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        snapshot = load_prev_run(str(tmp_path))
        assert snapshot is not None
        assert snapshot.stocks["600519"]["total_score"] is None  # malformed
        assert snapshot.stocks["600519"]["return_5d"] is None
        assert snapshot.stocks["600519"]["pass_stage1"] is True


# ════════════════════════════════════════════════════════
# 7. Model Integration
# ════════════════════════════════════════════════════════

class TestModelIntegration:
    """Test trend_delta field on HotStockDetail + HotStockSummary."""

    def test_detail_trend_delta_default(self):
        d = HotStockDetail(code="600519")
        assert d.trend_delta == {}

    def test_detail_trend_delta_populated(self):
        d = HotStockDetail(code="600519")
        d.trend_delta = {
            "prev_run_date": "2026-04-17",
            "total_score_delta": 0.05,
            "pass_stage1_change": "keep_pass",
            "score_accelerating": True,
        }
        assert d.trend_delta["total_score_delta"] == 0.05

    def test_summary_from_detail_with_delta(self):
        d = HotStockDetail(code="600519", name="贵州茅台")
        d.passed_hard_filter = True
        d.pass_stage1 = True
        d.total_score = 0.85
        d.trend_delta = {
            "prev_run_date": "2026-04-17",
            "total_score_delta": 0.05,
            "hot_theme_score_delta": 0.03,
            "trend_flow_score_delta": 0.02,
            "liquidity_execution_score_delta": 0.01,
            "risk_control_score_delta": 0.04,
            "pass_stage1_change": "keep_pass",
            "score_accelerating": True,
            "score_decelerating": False,
        }
        s = HotStockSummary.from_detail(d)

        assert s.prev_run_date == "2026-04-17"
        assert s.total_score_delta == 0.05
        assert s.hot_theme_score_delta == 0.03
        assert s.pass_stage1_change == "keep_pass"
        assert s.score_accelerating is True
        assert s.score_decelerating is False

    def test_summary_from_detail_without_delta(self):
        d = HotStockDetail(code="600519", name="贵州茅台")
        d.passed_hard_filter = True
        s = HotStockSummary.from_detail(d)

        assert s.prev_run_date == ""
        assert s.total_score_delta is None
        assert s.pass_stage1_change == ""
        assert s.score_accelerating is None


# ════════════════════════════════════════════════════════
# 8. Metadata Integration
# ════════════════════════════════════════════════════════

class TestMetadataIntegration:
    def test_metadata_default(self):
        m = RunMetadata()
        assert m.trend_compare_enabled is False
        assert m.trend_compare_prev_run_date == ""

    def test_metadata_with_trend_compare(self):
        m = RunMetadata(
            trend_compare_enabled=True,
            trend_compare_prev_run_date="2026-04-17",
        )
        assert m.trend_compare_enabled is True
        assert m.trend_compare_prev_run_date == "2026-04-17"


# ════════════════════════════════════════════════════════
# 9. Config Integration
# ════════════════════════════════════════════════════════

class TestConfigIntegration:
    def test_config_defaults(self):
        import datetime as dt
        cfg = HotScreenerConfig(
            tushare_token="test",
            run_date=dt.date(2026, 4, 18),
            stock_codes=["600519"],
            output_dir="/tmp/test",
        )
        assert cfg.prev_run_dir == ""
        assert cfg.prev_run_date == ""

    def test_config_with_prev_run(self):
        import datetime as dt
        cfg = HotScreenerConfig(
            tushare_token="test",
            run_date=dt.date(2026, 4, 18),
            stock_codes=["600519"],
            output_dir="/tmp/test",
            prev_run_dir="/tmp/prev_output",
            prev_run_date="2026-04-17",
        )
        assert cfg.prev_run_dir == "/tmp/prev_output"
        assert cfg.prev_run_date == "2026-04-17"


# ════════════════════════════════════════════════════════
# 10. CLI Integration
# ════════════════════════════════════════════════════════

class TestCLIIntegration:
    def test_cli_prev_run_args(self):
        parser = build_parser()
        args = parser.parse_args([
            "--tushare-token", "test",
            "--run-date", "2026-04-18",
            "--stock-codes", "600519",
            "--output-dir", "/tmp/test",
            "--prev-run-dir", "/tmp/prev_output",
            "--prev-run-date", "2026-04-17",
        ])
        assert args.prev_run_dir == "/tmp/prev_output"
        assert args.prev_run_date == "2026-04-17"

    def test_cli_prev_run_defaults(self):
        parser = build_parser()
        args = parser.parse_args([
            "--tushare-token", "test",
            "--run-date", "2026-04-18",
            "--stock-codes", "600519",
            "--output-dir", "/tmp/test",
        ])
        assert args.prev_run_dir == ""
        assert args.prev_run_date == ""


# ════════════════════════════════════════════════════════
# 11. Output Integration (detail long table)
# ════════════════════════════════════════════════════════

class TestOutputIntegration:
    """Test that detail long table includes trend delta rows."""

    def test_detail_long_rows_with_delta(self):
        from a_share_hot_screener.output import _detail_to_long_rows

        d = HotStockDetail(code="600519", name="贵州茅台")
        d.passed_hard_filter = True
        d.pass_stage1 = True
        d.total_score = 0.85
        d.data_coverage = 1.0
        d.trend_delta = {
            "prev_run_date": "2026-04-17",
            "total_score_delta": 0.05,
            "pass_stage1_change": "keep_pass",
            "score_accelerating": True,
            "score_decelerating": False,
        }

        rows = _detail_to_long_rows(d)
        summary_rows = [r for r in rows if r["axis"] == "summary"]
        indicator_names = [r["indicator_name"] for r in summary_rows]

        # Should include delta summary rows
        assert "prev_run_date" in indicator_names
        assert "total_score_delta" in indicator_names
        assert "pass_stage1_change" in indicator_names
        assert "score_accelerating" in indicator_names
        assert "score_decelerating" in indicator_names

        # Check values
        delta_row = next(r for r in summary_rows if r["indicator_name"] == "total_score_delta")
        assert delta_row["raw_value"] == "0.0500"  # _fmt_val formats float to 4 decimals

        change_row = next(r for r in summary_rows if r["indicator_name"] == "pass_stage1_change")
        assert change_row["raw_value"] == "keep_pass"

    def test_detail_long_rows_without_delta(self):
        from a_share_hot_screener.output import _detail_to_long_rows

        d = HotStockDetail(code="600519", name="贵州茅台")
        d.passed_hard_filter = True
        d.pass_stage1 = True
        d.total_score = 0.85
        d.data_coverage = 1.0
        # no trend_delta set → empty dict

        rows = _detail_to_long_rows(d)
        summary_rows = [r for r in rows if r["axis"] == "summary"]
        indicator_names = [r["indicator_name"] for r in summary_rows]

        # Should NOT include delta rows
        assert "prev_run_date" not in indicator_names
        assert "total_score_delta" not in indicator_names


# ════════════════════════════════════════════════════════
# 12. End-to-End with Pipeline mock
# ════════════════════════════════════════════════════════

class TestPipelineE2E:
    """Test the full pipeline step with trend_compare via mock."""

    def test_trend_delta_populated_on_detail(self, prev_snapshot, tmp_path):
        """Simulate pipeline Step 9: deltas computed and written to details."""
        details = [
            _make_detail("600519", total_score=0.85, pass_stage1=True,
                         hot_theme_score=0.92, trend_flow_score=0.77,
                         liquidity_execution_score=0.81, risk_control_score=0.86,
                         blocked_by=[], return_5d=6.0, return_10d=9.5,
                         amount_avg_5d=3_200_000_000.0),
            _make_detail("000858", total_score=0.58, pass_stage1=False,
                         hot_theme_score=0.45, trend_flow_score=0.55,
                         liquidity_execution_score=0.72, risk_control_score=0.91,
                         blocked_by=["hot_theme"],
                         return_5d=2.5, return_10d=4.0,
                         amount_avg_5d=1_600_000_000.0),
        ]

        deltas = compute_all_deltas(details, prev_snapshot)
        for detail in details:
            delta = deltas.get(detail.code)
            if delta is not None:
                detail.trend_delta = delta.to_dict()

        # Verify deltas on details
        assert details[0].trend_delta["pass_stage1_change"] == "keep_pass"
        assert details[0].trend_delta["total_score_delta"] == 0.03
        assert details[0].trend_delta["score_accelerating"] is True

        assert details[1].trend_delta["pass_stage1_change"] == "keep_fail"
        assert details[1].trend_delta["total_score_delta"] == 0.03  # 0.58 - 0.55
        assert details[1].trend_delta["newly_unblocked"] == ["total_score"]

        # Verify summary propagation
        s0 = HotStockSummary.from_detail(details[0])
        assert s0.prev_run_date == "2026-04-17"
        assert s0.total_score_delta == 0.03
        assert s0.score_accelerating is True

    def test_empty_prev_snapshot(self):
        """All stocks are new_entry when snapshot is empty."""
        snapshot = PrevRunSnapshot(trade_date_used="2026-04-16", stocks={})
        details = [_make_detail("600519", total_score=0.85)]
        deltas = compute_all_deltas(details, snapshot)

        assert deltas["600519"].pass_stage1_change == "new_entry"
        assert deltas["600519"].total_score_delta is None
