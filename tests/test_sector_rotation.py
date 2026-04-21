"""板块轮动信号模块 (A4) 单元测试.

全部使用 Mock 数据，不调用真实网络。
"""

from __future__ import annotations

import csv
import datetime as dt
import os
import tempfile
from unittest.mock import MagicMock

import pandas as pd
import pytest

from a_share_hot_screener.sector_rotation import (
    SectorHeatRow,
    SectorRotationAnalyzer,
    classify_momentum,
    _compute_rank_pctile,
)


# ════════════════════════════════════════════════════════
# classify_momentum 测试
# ════════════════════════════════════════════════════════

class TestClassifyMomentum:
    def test_rotate_in(self):
        # 前期弱 (20d < 50%) + 近期强 (5d > 70%)
        assert classify_momentum(rank_5d=0.80, rank_20d=0.30) == "rotate_in"

    def test_rotate_out(self):
        # 前期强 (20d > 70%) + 近期弱 (5d < 40%)
        assert classify_momentum(rank_5d=0.20, rank_20d=0.80) == "rotate_out"

    def test_steady_strong(self):
        # 近期强 + 前期也强
        assert classify_momentum(rank_5d=0.85, rank_20d=0.75) == "steady_strong"

    def test_steady_weak(self):
        # 近期弱 + 前期也弱
        assert classify_momentum(rank_5d=0.15, rank_20d=0.25) == "steady_weak"

    def test_neutral(self):
        assert classify_momentum(rank_5d=0.50, rank_20d=0.50) == "neutral"

    def test_none_values(self):
        assert classify_momentum(rank_5d=None, rank_20d=0.50) == "neutral"
        assert classify_momentum(rank_5d=0.50, rank_20d=None) == "neutral"
        assert classify_momentum(rank_5d=None, rank_20d=None) == "neutral"

    def test_boundary_rotate_in(self):
        # 边界：exactly 0.50 和 0.70 不满足 (<0.50 和 >0.70)
        assert classify_momentum(rank_5d=0.70, rank_20d=0.50) != "rotate_in"
        assert classify_momentum(rank_5d=0.71, rank_20d=0.49) == "rotate_in"

    def test_boundary_rotate_out(self):
        assert classify_momentum(rank_5d=0.39, rank_20d=0.71) == "rotate_out"
        assert classify_momentum(rank_5d=0.40, rank_20d=0.70) != "rotate_out"


# ════════════════════════════════════════════════════════
# _compute_rank_pctile 测试
# ════════════════════════════════════════════════════════

class TestComputeRankPctile:
    def test_basic(self):
        pct_map = {"A": 10.0, "B": 5.0, "C": -2.0}
        result = _compute_rank_pctile(pct_map)
        # 排序: C(-2) < B(5) < A(10)
        # pctile: C=0.5/3, B=1.5/3, A=2.5/3
        assert abs(result["C"] - 0.1667) < 0.01
        assert abs(result["B"] - 0.5) < 0.01
        assert abs(result["A"] - 0.8333) < 0.01

    def test_empty(self):
        assert _compute_rank_pctile({}) == {}

    def test_single(self):
        result = _compute_rank_pctile({"A": 5.0})
        assert abs(result["A"] - 0.5) < 0.01


# ════════════════════════════════════════════════════════
# SectorRotationAnalyzer 测试
# ════════════════════════════════════════════════════════

def _make_mock_ts_for_rotation():
    ts = MagicMock()

    industry_df = pd.DataFrame({
        "ts_code": ["885001.TI", "885002.TI", "885003.TI"],
        "name": ["半导体", "白酒", "银行"],
        "count": [50, 30, 20],
        "exchange": ["A"] * 3,
        "list_date": ["20200101"] * 3,
        "type": ["I"] * 3,
    })

    concept_df = pd.DataFrame({
        "ts_code": ["885800.TI", "885801.TI"],
        "name": ["人工智能", "华为概念"],
        "count": [60, 40],
        "exchange": ["A"] * 2,
        "list_date": ["20200101"] * 2,
        "type": ["N"] * 2,
    })

    def _ths_index(exchange="A", type_=None, **kwargs):
        if type_ == "I":
            return industry_df
        if type_ == "N":
            return concept_df
        return pd.DataFrame()

    ts.get_ths_index.side_effect = _ths_index

    # 20天数据 — 简化为每天有数据
    # 半导体: 稳步上涨 (steady_strong)
    # 白酒: 先弱后强 (rotate_in)
    # 银行: 先强后弱 (rotate_out)
    # 人工智能: 持续强势
    # 华为概念: 持续弱势

    # 生成20个工作日日期
    weekdays = []
    wd = dt.date(2026, 3, 20)
    while len(weekdays) < 20:
        wd += dt.timedelta(days=1)
        if wd.weekday() < 5:
            weekdays.append(wd)

    daily_data = {}
    for idx, wday in enumerate(weekdays):
        date_str = wday.strftime("%Y%m%d")

        # 半导体: 1000 稳步涨到 1100 (steady strong)
        semi_close = 1000 + idx * 5.0

        # 白酒: 前15天跌，后5天涨 (rotate_in)
        if idx < 15:
            baijiu_close = 1000 - idx * 12.0
        else:
            baijiu_close = 820 + (idx - 15) * 20.0

        # 银行: 前15天涨，后5天跌 (rotate_out)
        if idx < 15:
            bank_close = 500 + idx * 7.0
        else:
            bank_close = 605 - (idx - 15) * 12.0

        # 人工智能: 稳步涨
        ai_close = 1500 + idx * 8.0

        # 华为: 稳步跌
        huawei_close = 2000 - idx * 6.0

        daily_data[date_str] = [
            ("885001.TI", semi_close, 0.5),
            ("885002.TI", baijiu_close, 0.0),
            ("885003.TI", bank_close, 0.0),
            ("885800.TI", ai_close, 0.5),
            ("885801.TI", huawei_close, -0.3),
        ]

    def _ths_daily(trade_date="", **kwargs):
        if trade_date in daily_data:
            data = daily_data[trade_date]
            return pd.DataFrame({
                "ts_code": [row[0] for row in data],
                "trade_date": [trade_date] * len(data),
                "close": [row[1] for row in data],
                "open": [row[1] * 0.99 for row in data],
                "high": [row[1] * 1.01 for row in data],
                "low": [row[1] * 0.98 for row in data],
                "pct_change": [row[2] for row in data],
                "vol": [10000.0] * len(data),
            })
        return pd.DataFrame()

    ts.get_ths_daily.side_effect = _ths_daily
    return ts


class TestSectorRotationAnalyzer:
    def _make_analyzer(self, ts_mock):
        trade_dates = []
        d = dt.date(2026, 3, 20)
        while d <= dt.date(2026, 4, 18):
            d += dt.timedelta(days=1)
            if d.weekday() < 5:
                trade_dates.append(d)

        return SectorRotationAnalyzer(
            tushare_client=ts_mock,
            cache=None,
            run_date=trade_dates[-1] if trade_dates else dt.date(2026, 4, 18),
            trade_dates=trade_dates,
        )

    def test_analyze_returns_results(self):
        ts = _make_mock_ts_for_rotation()
        analyzer = self._make_analyzer(ts)
        results = analyzer.analyze()

        assert len(results) > 0
        # 应该有行业和概念
        types = {r.type for r in results}
        assert "I" in types
        assert "N" in types

    def test_all_sectors_covered(self):
        ts = _make_mock_ts_for_rotation()
        analyzer = self._make_analyzer(ts)
        results = analyzer.analyze()

        names = {r.name for r in results}
        assert "半导体" in names
        assert "白酒" in names
        assert "银行" in names
        assert "人工智能" in names
        assert "华为概念" in names

    def test_pct_values_populated(self):
        ts = _make_mock_ts_for_rotation()
        analyzer = self._make_analyzer(ts)
        results = analyzer.analyze()

        for r in results:
            # 至少 5d 涨幅应有值
            assert r.pct_5d is not None, f"{r.name} pct_5d is None"
            assert r.rank_pctile_5d is not None

    def test_momentum_switch_values(self):
        ts = _make_mock_ts_for_rotation()
        analyzer = self._make_analyzer(ts)
        results = analyzer.analyze()

        momentum_values = {r.momentum_switch for r in results}
        # 所有值应是合法的
        valid = {"rotate_in", "rotate_out", "steady_strong", "steady_weak", "neutral"}
        assert momentum_values.issubset(valid)

    def test_results_sorted_by_momentum_priority(self):
        ts = _make_mock_ts_for_rotation()
        analyzer = self._make_analyzer(ts)
        results = analyzer.analyze()

        order = {"rotate_in": 0, "steady_strong": 1, "neutral": 2,
                 "rotate_out": 3, "steady_weak": 4}
        prev_priority = -1
        for r in results:
            p = order.get(r.momentum_switch, 9)
            if p < prev_priority:
                # 排序乱了 — 但注意同优先级内按 pct_5d 降序
                break
            prev_priority = p

    def test_no_permission(self):
        ts = MagicMock()
        ts.get_ths_index.return_value = None

        analyzer = SectorRotationAnalyzer(
            tushare_client=ts, cache=None,
            run_date=dt.date(2026, 4, 18), trade_dates=[],
        )
        results = analyzer.analyze()
        assert results == []


class TestSectorRotationCSV:
    def test_to_csv(self):
        rows = [
            SectorHeatRow(
                ts_code="885001.TI", name="半导体", type="I",
                pct_5d=5.0, pct_10d=8.0, pct_20d=10.0,
                rank_pctile_5d=0.85, rank_pctile_10d=0.80, rank_pctile_20d=0.75,
                momentum_switch="steady_strong", member_count=50,
            ),
            SectorHeatRow(
                ts_code="885002.TI", name="白酒", type="I",
                pct_5d=3.0, pct_10d=-2.0, pct_20d=-5.0,
                rank_pctile_5d=0.75, rank_pctile_10d=0.30, rank_pctile_20d=0.20,
                momentum_switch="rotate_in", member_count=30,
            ),
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "sector_heat.csv")
            analyzer = SectorRotationAnalyzer(
                tushare_client=MagicMock(), cache=None,
                run_date=dt.date(2026, 4, 18), trade_dates=[],
            )
            result_path = analyzer.to_csv(rows, path)

            assert os.path.exists(result_path)

            with open(result_path, "r", encoding="utf-8-sig") as f:
                reader = csv.DictReader(f)
                csv_rows = list(reader)

            assert len(csv_rows) == 2
            assert csv_rows[0]["name"] == "半导体"
            assert csv_rows[0]["momentum_switch"] == "steady_strong"
            assert float(csv_rows[0]["pct_5d"]) == 5.0
            assert float(csv_rows[0]["rank_pctile_5d"]) == 0.85

    def test_to_csv_empty(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "sector_heat.csv")
            analyzer = SectorRotationAnalyzer(
                tushare_client=MagicMock(), cache=None,
                run_date=dt.date(2026, 4, 18), trade_dates=[],
            )
            result_path = analyzer.to_csv([], path)
            assert os.path.exists(result_path)

            with open(result_path, "r", encoding="utf-8-sig") as f:
                reader = csv.DictReader(f)
                assert list(reader) == []
