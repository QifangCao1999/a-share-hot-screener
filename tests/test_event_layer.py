"""事件层模块单元测试 (Tushare 版).

测试策略：全部使用 Mock 数据，不调用真实网络。
"""

from __future__ import annotations

import datetime as dt
from unittest.mock import MagicMock

import pandas as pd
import pytest

from a_share_hot_screener.event_layer import (
    EventLayerContext,
    EventLayerProcessor,
    EventLayerResult,
    _max_consecutive,
    _bisect_right,
)


def _make_ctx(**kwargs) -> EventLayerContext:
    defaults = dict(
        zt_pool_map={},
        zt_pool_available=False,
        strong_pool_map={},
        strong_pool_available=False,
        lhb_df=None,
        lhb_available=False,
        industry_hist_map={},
        industry_rank_pctile={},
        industry_heat_mode="none",
        concept_heat_mode="none",
        concept_cons_map={},
    )
    defaults.update(kwargs)
    ctx = EventLayerContext()
    for k, v in defaults.items():
        if not k.startswith("_"):
            setattr(ctx, k, v)
    return ctx


class TestMaxConsecutive:
    def test_empty(self):
        assert _max_consecutive([]) == 0

    def test_all_false(self):
        assert _max_consecutive([False, False, False]) == 0

    def test_all_true(self):
        assert _max_consecutive([True, True, True]) == 3

    def test_mixed(self):
        assert _max_consecutive([True, True, False, True]) == 2

    def test_single_true(self):
        assert _max_consecutive([False, True, False]) == 1


class TestBisectRight:
    def test_basic(self):
        dates = [dt.date(2026, 1, d) for d in [1, 2, 3, 4, 5]]
        assert _bisect_right(dates, dt.date(2026, 1, 3)) == 3
        assert _bisect_right(dates, dt.date(2026, 1, 5)) == 5
        assert _bisect_right(dates, dt.date(2025, 12, 31)) == 0


class TestEventLayerProcessorLimitUp:
    def test_normal_count(self):
        ctx = _make_ctx(
            zt_pool_available=True,
            zt_pool_map={
                "20260414": {"600519", "000858"},
                "20260415": {"600519"},
                "20260416": set(),
                "20260417": {"600519"},
                "20260418": {"600519", "300750"},
            },
        )
        proc = EventLayerProcessor(ctx=ctx, tushare_client=MagicMock(), cache=None)
        dates_10d = [dt.date(2026, 4, d) for d in [14, 15, 16, 17, 18]]
        result = proc.process(
            code="600519", industry="白酒", run_date=dt.date(2026, 4, 18),
            trade_dates_10d=dates_10d, trade_dates_3d=dates_10d[-3:],
            trade_dates_20d=dates_10d,
        )
        assert result.limit_up_count_5d == 4  # 14,15,17,18
        assert result.limit_up_count_10d == 4
        assert result.limit_up_source == "tushare_limit_list_d"

    def test_pool_unavailable(self):
        ctx = _make_ctx(zt_pool_available=False)
        proc = EventLayerProcessor(ctx=ctx, tushare_client=MagicMock(), cache=None)
        result = proc.process(
            code="600519", industry="白酒", run_date=dt.date(2026, 4, 18),
            trade_dates_10d=[], trade_dates_3d=[], trade_dates_20d=[],
        )
        assert result.limit_up_count_5d is None
        assert result.limit_up_source == "none"


class TestEventLayerProcessorStrongPool:
    def test_normal_count(self):
        ctx = _make_ctx(
            zt_pool_available=True,
            zt_pool_map={"20260418": {"600519"}},
            strong_pool_available=True,
            strong_pool_map={
                "20260416": {"600519"},
                "20260417": {"600519"},
                "20260418": set(),
            },
        )
        proc = EventLayerProcessor(ctx=ctx, tushare_client=MagicMock(), cache=None)
        dates = [dt.date(2026, 4, d) for d in [16, 17, 18]]
        result = proc.process(
            code="600519", industry="白酒", run_date=dt.date(2026, 4, 18),
            trade_dates_10d=dates, trade_dates_3d=dates, trade_dates_20d=dates,
        )
        assert result.strong_pool_entry_count_3d == 2
        assert result.strong_pool_source == "tushare_limit_list_d_derived"

    def test_fallback_to_zt_count(self):
        ctx = _make_ctx(
            zt_pool_available=True,
            zt_pool_map={"20260418": {"600519"}},
            strong_pool_available=False,
        )
        proc = EventLayerProcessor(ctx=ctx, tushare_client=MagicMock(), cache=None)
        dates = [dt.date(2026, 4, 18)]
        result = proc.process(
            code="600519", industry="白酒", run_date=dt.date(2026, 4, 18),
            trade_dates_10d=dates, trade_dates_3d=dates, trade_dates_20d=dates,
        )
        assert result.strong_pool_source == "fallback_zt_count"


class TestEventLayerProcessorLhb:
    def test_normal_count(self):
        lhb_df = pd.DataFrame({
            "ts_code": ["600519.SH", "600519.SH", "000858.SZ"],
            "trade_date": ["20260415", "20260418", "20260418"],
        })
        ctx = _make_ctx(lhb_available=True, lhb_df=lhb_df)
        proc = EventLayerProcessor(ctx=ctx, tushare_client=MagicMock(), cache=None)
        dates = [dt.date(2026, 4, d) for d in range(1, 19)]
        result = proc.process(
            code="600519", industry="白酒", run_date=dt.date(2026, 4, 18),
            trade_dates_10d=dates[-10:], trade_dates_3d=dates[-3:],
            trade_dates_20d=dates,
        )
        assert result.lhb_count_20d == 2
        assert result.lhb_on_board is True
        assert result.lhb_source == "tushare_top_list"

    def test_lhb_unavailable(self):
        ctx = _make_ctx(lhb_available=False)
        proc = EventLayerProcessor(ctx=ctx, tushare_client=MagicMock(), cache=None, enable_lhb=True)
        result = proc.process(
            code="600519", industry="", run_date=dt.date(2026, 4, 18),
            trade_dates_10d=[], trade_dates_3d=[], trade_dates_20d=[],
        )
        assert result.lhb_count_20d is None
        assert result.lhb_source == "none"


class TestEventLayerProcessorIndustryHeat:
    def test_degraded_mode(self):
        ctx = _make_ctx(industry_heat_mode="tushare_200_degraded")
        proc = EventLayerProcessor(ctx=ctx, tushare_client=MagicMock(), cache=None)
        result = proc.process(
            code="600519", industry="白酒", run_date=dt.date(2026, 4, 18),
            trade_dates_10d=[], trade_dates_3d=[], trade_dates_20d=[],
        )
        assert result.industry_name == "白酒"
        assert result.industry_heat_pctile_5d is None
        assert result.industry_heat_source == "tushare_200_degraded"


class TestEventLayerProcessorConceptHeat:
    def test_concept_unavailable(self):
        ctx = _make_ctx(concept_heat_mode="tushare_200_unavailable")
        proc = EventLayerProcessor(
            ctx=ctx, tushare_client=MagicMock(), cache=None, enable_concept=True,
        )
        result = proc.process(
            code="600519", industry="白酒", run_date=dt.date(2026, 4, 18),
            trade_dates_10d=[], trade_dates_3d=[], trade_dates_20d=[],
        )
        assert result.concept_heat_pctile_5d is None
        assert result.advanced_concept_module_available is False
        assert result.concept_heat_source == "tushare_200_unavailable"
