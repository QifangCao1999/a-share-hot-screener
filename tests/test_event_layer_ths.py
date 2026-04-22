"""事件层 ths_daily 完整版测试 (HT5 行业热度 + HT6 概念热度).

全部使用 Mock 数据，不调用真实网络。
"""

from __future__ import annotations

import datetime as dt
from unittest.mock import MagicMock

import pandas as pd
import pytest

from a_share_hot_screener.event_layer import (
    EventLayerContext,
    EventLayerLoader,
    EventLayerProcessor,
    EventLayerResult,
    _calc_period_pct_change,
)


# ════════════════════════════════════════════════════════
# Fixtures
# ════════════════════════════════════════════════════════

def _make_ths_index_industry():
    """模拟同花顺行业指数列表。"""
    return pd.DataFrame({
        "ts_code": ["885001.TI", "885002.TI", "885003.TI", "885004.TI"],
        "name": ["半导体", "白酒", "新能源", "银行"],
        "count": [50, 30, 40, 20],
        "exchange": ["A"] * 4,
        "list_date": ["20200101"] * 4,
        "type": ["I"] * 4,
    })


def _make_ths_index_concept():
    """模拟同花顺概念指数列表。"""
    return pd.DataFrame({
        "ts_code": ["885800.TI", "885801.TI", "885802.TI"],
        "name": ["华为概念", "人工智能", "芯片"],
        "count": [60, 45, 55],
        "exchange": ["A"] * 3,
        "list_date": ["20200101"] * 3,
        "type": ["N"] * 3,
    })


def _make_ths_daily_day(trade_date, data):
    """构造单日 ths_daily DataFrame。

    data: list of (ts_code, close, pct_change)
    """
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


def _make_index_member_all():
    """模拟 index_member_all 返回。"""
    return pd.DataFrame({
        "l1_name": ["半导体", "白酒", "新能源", "银行"],
        "ts_code": ["300750.SZ", "600519.SH", "002475.SZ", "601398.SH"],
        "name": ["宁德时代", "贵州茅台", "立讯精密", "工商银行"],
        "is_new": ["Y", "Y", "Y", "Y"],
    })


def _make_mock_ts(
    industry_idx=True,
    concept_idx=True,
    permission=True,
):
    """创建 mock TushareClient。"""
    ts = MagicMock()

    if not permission:
        ts.get_ths_index.return_value = None
        ts.get_ths_daily.return_value = None
        ts.get_ths_member.return_value = None
        return ts

    # ths_index
    def _ths_index(exchange="A", type_=None, **kwargs):
        if type_ == "I" and industry_idx:
            return _make_ths_index_industry()
        if type_ == "N" and concept_idx:
            return _make_ths_index_concept()
        return pd.DataFrame()

    ts.get_ths_index.side_effect = _ths_index

    # ths_daily: 5 天数据
    # 行业: 半导体涨最多, 银行跌最多
    # 概念: 人工智能涨最多, 芯片中间, 华为概念最弱
    daily_data = {
        "20260414": [
            ("885001.TI", 1000.0, 0.0), ("885002.TI", 800.0, 0.0),
            ("885003.TI", 600.0, 0.0), ("885004.TI", 500.0, 0.0),
            ("885800.TI", 2000.0, 0.0), ("885801.TI", 1500.0, 0.0),
            ("885802.TI", 1200.0, 0.0),
        ],
        "20260415": [
            ("885001.TI", 1020.0, 2.0), ("885002.TI", 808.0, 1.0),
            ("885003.TI", 606.0, 1.0), ("885004.TI", 495.0, -1.0),
            ("885800.TI", 1990.0, -0.5), ("885801.TI", 1530.0, 2.0),
            ("885802.TI", 1212.0, 1.0),
        ],
        "20260416": [
            ("885001.TI", 1050.0, 2.94), ("885002.TI", 812.0, 0.5),
            ("885003.TI", 612.0, 1.0), ("885004.TI", 490.0, -1.01),
            ("885800.TI", 1980.0, -0.5), ("885801.TI", 1560.0, 1.96),
            ("885802.TI", 1224.0, 1.0),
        ],
        "20260417": [
            ("885001.TI", 1080.0, 2.86), ("885002.TI", 810.0, -0.25),
            ("885003.TI", 615.0, 0.5), ("885004.TI", 485.0, -1.02),
            ("885800.TI", 1970.0, -0.5), ("885801.TI", 1600.0, 2.56),
            ("885802.TI", 1236.0, 0.98),
        ],
        "20260418": [
            ("885001.TI", 1100.0, 1.85), ("885002.TI", 816.0, 0.74),
            ("885003.TI", 618.0, 0.49), ("885004.TI", 480.0, -1.03),
            ("885800.TI", 1960.0, -0.5), ("885801.TI", 1650.0, 3.13),
            ("885802.TI", 1248.0, 0.97),
        ],
    }

    def _ths_daily(trade_date="", ts_code="", **kwargs):
        if trade_date and trade_date in daily_data:
            return _make_ths_daily_day(trade_date, daily_data[trade_date])
        return pd.DataFrame()

    ts.get_ths_daily.side_effect = _ths_daily

    # index_member_all
    ts.get_index_member_all.return_value = _make_index_member_all()

    # ths_member (按 con_code 查询)
    concept_membership = {
        "300750.SZ": [("885802.TI", "300750.SZ", "宁德时代")],  # 芯片
        "600519.SH": [("885800.TI", "600519.SH", "贵州茅台")],  # 华为概念
        "002475.SZ": [
            ("885801.TI", "002475.SZ", "立讯精密"),  # 人工智能
            ("885802.TI", "002475.SZ", "立讯精密"),  # 芯片
        ],
    }

    def _ths_member(con_code="", ts_code="", **kwargs):
        if con_code and con_code in concept_membership:
            rows = concept_membership[con_code]
            return pd.DataFrame(rows, columns=["ts_code", "con_code", "con_name"])
        return pd.DataFrame()

    ts.get_ths_member.side_effect = _ths_member

    # 基础接口 mock（涨停池等不测试）
    ts.get_limit_list.return_value = pd.DataFrame()

    return ts


# ════════════════════════════════════════════════════════
# _calc_period_pct_change 测试
# ════════════════════════════════════════════════════════

class TestCalcPeriodPctChange:
    def test_basic(self):
        df = pd.DataFrame({
            "ts_code": ["A", "A", "B", "B"],
            "trade_date": ["20260414", "20260418", "20260414", "20260418"],
            "close": [100.0, 110.0, 200.0, 190.0],
        })
        result = _calc_period_pct_change(df, {"A", "B"})
        assert abs(result["A"] - 10.0) < 0.01
        assert abs(result["B"] - (-5.0)) < 0.01

    def test_single_day_skipped(self):
        df = pd.DataFrame({
            "ts_code": ["A"],
            "trade_date": ["20260418"],
            "close": [100.0],
        })
        result = _calc_period_pct_change(df, {"A"})
        assert "A" not in result

    def test_empty(self):
        result = _calc_period_pct_change(pd.DataFrame(), set())
        assert result == {}

    def test_zero_close_skipped(self):
        df = pd.DataFrame({
            "ts_code": ["A", "A"],
            "trade_date": ["20260414", "20260418"],
            "close": [0.0, 100.0],
        })
        result = _calc_period_pct_change(df, {"A"})
        assert "A" not in result


# ════════════════════════════════════════════════════════
# EventLayerLoader — 行业热度完整版
# ════════════════════════════════════════════════════════

class TestEventLayerLoaderIndustryHeatFull:
    def _make_loader(self, ts_mock):
        return EventLayerLoader(
            tushare_client=ts_mock,
            cache=None,
            run_date=dt.date(2026, 4, 18),
            trade_dates=[dt.date(2026, 4, d) for d in range(1, 19)],
            enable_lhb_module=False,
            enable_concept_heat_module=False,
        )

    def test_full_mode_loaded(self):
        ts = _make_mock_ts()
        loader = self._make_loader(ts)
        ctx = loader.load()

        assert ctx.industry_heat_mode == "ths_daily_full"
        assert len(ctx.industry_rank_pctile) == 4
        assert "半导体" in ctx.industry_rank_pctile
        assert "银行" in ctx.industry_rank_pctile
        # 半导体涨幅最大 (1000→1100=10%)，百分位应最高
        assert ctx.industry_rank_pctile["半导体"] > ctx.industry_rank_pctile["银行"]
        # 银行跌 (500→480=-4%)，应最低
        assert ctx.industry_rank_pctile["银行"] < 0.25

    def test_industry_cons_map(self):
        ts = _make_mock_ts()
        loader = self._make_loader(ts)
        ctx = loader.load()

        assert ctx.industry_cons_map.get("300750") == "半导体"  # 从 mock index_member_all 第一行
        assert ctx.industry_cons_map.get("600519") == "白酒"
        assert ctx.industry_cons_map.get("002475") == "新能源"

    def test_hist_map_values(self):
        ts = _make_mock_ts()
        loader = self._make_loader(ts)
        ctx = loader.load()

        # 半导体: 1000→1100 = 10%
        assert abs(ctx.industry_hist_map["半导体"] - 10.0) < 0.1
        # 银行: 500→480 = -4%
        assert abs(ctx.industry_hist_map["银行"] - (-4.0)) < 0.1

    def test_fallback_to_degraded_when_no_permission(self):
        ts = _make_mock_ts(permission=False)
        loader = self._make_loader(ts)
        ctx = loader.load()

        assert ctx.industry_heat_mode == "tushare_200_degraded"
        assert len(ctx.industry_rank_pctile) == 0

    def test_fallback_when_ths_daily_empty(self):
        ts = _make_mock_ts()
        ts.get_ths_daily.side_effect = None
        ts.get_ths_daily.return_value = pd.DataFrame()
        loader = self._make_loader(ts)
        ctx = loader.load()

        assert ctx.industry_heat_mode == "tushare_200_degraded"


# ════════════════════════════════════════════════════════
# EventLayerLoader — 概念热度完整版
# ════════════════════════════════════════════════════════

class TestEventLayerLoaderConceptHeatFull:
    def _make_loader(self, ts_mock):
        return EventLayerLoader(
            tushare_client=ts_mock,
            cache=None,
            run_date=dt.date(2026, 4, 18),
            trade_dates=[dt.date(2026, 4, d) for d in range(1, 19)],
            enable_lhb_module=False,
            enable_concept_heat_module=True,
        )

    def test_full_mode_loaded(self):
        ts = _make_mock_ts()
        loader = self._make_loader(ts)
        ctx = loader.load()

        assert ctx.concept_heat_mode == "ths_daily_full"
        assert len(ctx.concept_rank_pctile) == 3
        assert "人工智能" in ctx.concept_rank_pctile
        # 人工智能涨幅最大 (1500→1650=10%)，百分位应最高
        assert ctx.concept_rank_pctile["人工智能"] > ctx.concept_rank_pctile["华为概念"]

    def test_concept_spot_df_stored(self):
        ts = _make_mock_ts()
        loader = self._make_loader(ts)
        ctx = loader.load()

        assert ctx.concept_spot_df is not None
        assert len(ctx.concept_spot_df) == 3

    def test_fallback_when_no_permission(self):
        ts = _make_mock_ts(concept_idx=False)
        loader = self._make_loader(ts)
        ctx = loader.load()

        assert ctx.concept_heat_mode == "tushare_200_unavailable"

    def test_concept_disabled(self):
        ts = _make_mock_ts()
        loader = EventLayerLoader(
            tushare_client=ts,
            cache=None,
            run_date=dt.date(2026, 4, 18),
            trade_dates=[dt.date(2026, 4, d) for d in range(1, 19)],
            enable_lhb_module=False,
            enable_concept_heat_module=False,  # 关闭
        )
        ctx = loader.load()
        assert ctx.concept_heat_mode == "none"


# ════════════════════════════════════════════════════════
# EventLayerProcessor — 行业热度完整版
# ════════════════════════════════════════════════════════

def _make_ctx_full(**kwargs) -> EventLayerContext:
    ctx = EventLayerContext()
    ctx.zt_pool_available = False
    ctx.strong_pool_available = False
    ctx.lhb_available = False
    for k, v in kwargs.items():
        if not k.startswith("_"):
            setattr(ctx, k, v)
    return ctx


class TestEventLayerProcessorIndustryHeatFull:
    def test_full_mode_fills_pctile(self):
        ctx = _make_ctx_full(
            industry_heat_mode="ths_daily_full",
            industry_rank_pctile={"白酒": 0.625, "半导体": 0.875, "银行": 0.125},
            industry_hist_map={"白酒": 2.0, "半导体": 10.0, "银行": -4.0},
            industry_cons_map={"600519": "白酒", "300750": "半导体"},
        )
        proc = EventLayerProcessor(ctx=ctx, tushare_client=MagicMock(), cache=None)
        result = proc.process(
            code="600519", industry="白酒", run_date=dt.date(2026, 4, 18),
            trade_dates_10d=[], trade_dates_3d=[], trade_dates_20d=[],
        )
        assert result.industry_heat_pctile_5d == 0.625
        assert result.industry_pct_5d == 2.0
        assert result.industry_heat_source == "ths_daily_full_direct"
        assert result.industry_name == "白酒"

    def test_full_mode_uses_cons_map_over_industry_param(self):
        """cons_map 映射优先于 industry 参数名。"""
        ctx = _make_ctx_full(
            industry_heat_mode="ths_daily_full",
            industry_rank_pctile={"电子": 0.75, "白酒": 0.25},
            industry_hist_map={"电子": 5.0, "白酒": 1.0},
            industry_cons_map={"600519": "电子"},  # 映射到"电子"而非"白酒"
        )
        proc = EventLayerProcessor(ctx=ctx, tushare_client=MagicMock(), cache=None)
        result = proc.process(
            code="600519", industry="白酒", run_date=dt.date(2026, 4, 18),
            trade_dates_10d=[], trade_dates_3d=[], trade_dates_20d=[],
        )
        assert result.industry_heat_pctile_5d == 0.75
        assert result.industry_name == "电子"

    def test_full_mode_fallback_to_industry_name(self):
        """cons_map 中无该股票时，fallback 到 industry 参数名。"""
        ctx = _make_ctx_full(
            industry_heat_mode="ths_daily_full",
            industry_rank_pctile={"白酒": 0.625},
            industry_hist_map={"白酒": 2.0},
            industry_cons_map={},  # 空映射
        )
        proc = EventLayerProcessor(ctx=ctx, tushare_client=MagicMock(), cache=None)
        result = proc.process(
            code="600519", industry="白酒", run_date=dt.date(2026, 4, 18),
            trade_dates_10d=[], trade_dates_3d=[], trade_dates_20d=[],
        )
        assert result.industry_heat_pctile_5d == 0.625
        assert result.industry_heat_source == "ths_daily_full_name_match"

    def test_full_mode_unmapped(self):
        """stock 的行业名在 rank_pctile 中找不到。"""
        ctx = _make_ctx_full(
            industry_heat_mode="ths_daily_full",
            industry_rank_pctile={"半导体": 0.875},
            industry_hist_map={"半导体": 10.0},
            industry_cons_map={},
        )
        proc = EventLayerProcessor(ctx=ctx, tushare_client=MagicMock(), cache=None)
        result = proc.process(
            code="600519", industry="未知行业", run_date=dt.date(2026, 4, 18),
            trade_dates_10d=[], trade_dates_3d=[], trade_dates_20d=[],
        )
        assert result.industry_heat_pctile_5d is None
        assert result.industry_heat_source == "ths_daily_full_unmapped"

    def test_full_mode_bridge_mapping(self):
        """桥接映射: stock 不在 cons_map 但 basic.industry 有桥接。"""
        ctx = _make_ctx_full(
            industry_heat_mode="ths_daily_full",
            industry_rank_pctile={"半导体": 0.875},
            industry_hist_map={"半导体": 10.0},
            industry_cons_map={},  # 该股票不在直接映射中
        )
        # 手动设置桥接映射: "元器件" → "半导体"
        ctx.basic_industry_to_ths["元器件"] = "半导体"
        proc = EventLayerProcessor(ctx=ctx, tushare_client=MagicMock(), cache=None)
        result = proc.process(
            code="002436", industry="元器件", run_date=dt.date(2026, 4, 18),
            trade_dates_10d=[], trade_dates_3d=[], trade_dates_20d=[],
        )
        assert result.industry_heat_pctile_5d == 0.875
        assert result.industry_heat_source == "ths_daily_full_bridge"
        assert result.industry_name == "半导体"

    def test_degraded_mode_unchanged(self):
        """降级模式行为不变。"""
        ctx = _make_ctx_full(industry_heat_mode="tushare_200_degraded")
        proc = EventLayerProcessor(ctx=ctx, tushare_client=MagicMock(), cache=None)
        result = proc.process(
            code="600519", industry="白酒", run_date=dt.date(2026, 4, 18),
            trade_dates_10d=[], trade_dates_3d=[], trade_dates_20d=[],
        )
        assert result.industry_heat_pctile_5d is None
        assert result.industry_heat_source == "tushare_200_degraded"


# ════════════════════════════════════════════════════════
# EventLayerProcessor — 概念热度完整版
# ════════════════════════════════════════════════════════

class TestEventLayerProcessorConceptHeatFull:
    def _make_concept_ctx(self, cons_map=None):
        return _make_ctx_full(
            concept_heat_mode="ths_daily_full",
            concept_rank_pctile={
                "人工智能": 0.833,
                "芯片": 0.500,
                "华为概念": 0.167,
            },
            concept_hist_map={
                "人工智能": 10.0,
                "芯片": 4.0,
                "华为概念": -2.0,
            },
            concept_cons_map=cons_map or {
                "002475": ["人工智能", "芯片"],
                "600519": ["华为概念"],
            },
            concept_spot_df=pd.DataFrame({
                "ts_code": ["885800.TI", "885801.TI", "885802.TI"],
                "name": ["华为概念", "人工智能", "芯片"],
            }),
        )

    def test_stock_with_multiple_concepts(self):
        """股票属于多个概念，取最强。"""
        ctx = self._make_concept_ctx()
        proc = EventLayerProcessor(
            ctx=ctx, tushare_client=MagicMock(), cache=None, enable_concept=True,
        )
        result = proc.process(
            code="002475", industry="电子", run_date=dt.date(2026, 4, 18),
            trade_dates_10d=[], trade_dates_3d=[], trade_dates_20d=[],
        )
        # 002475 属于 人工智能(0.833) 和 芯片(0.500)，取最强
        assert result.concept_heat_pctile_5d == 0.833
        assert result.advanced_concept_module_available is True
        assert result.concept_heat_source == "ths_daily_full"
        assert result.concept_names[0] == "人工智能"
        assert len(result.concept_names) == 2

    def test_stock_with_one_concept(self):
        ctx = self._make_concept_ctx()
        proc = EventLayerProcessor(
            ctx=ctx, tushare_client=MagicMock(), cache=None, enable_concept=True,
        )
        result = proc.process(
            code="600519", industry="白酒", run_date=dt.date(2026, 4, 18),
            trade_dates_10d=[], trade_dates_3d=[], trade_dates_20d=[],
        )
        assert result.concept_heat_pctile_5d == 0.167
        assert result.concept_names == ["华为概念"]

    def test_stock_not_in_any_concept_with_ths_member_lookup(self):
        """stock 不在 cons_map 中，通过 ths_member 实时查询。"""
        ctx = self._make_concept_ctx(cons_map={})

        ts_mock = MagicMock()
        # ths_member 返回 300750 属于 芯片板块
        ts_mock.get_ths_member.return_value = pd.DataFrame({
            "ts_code": ["885802.TI"],
            "con_code": ["300750.SZ"],
            "con_name": ["宁德时代"],
        })

        proc = EventLayerProcessor(
            ctx=ctx, tushare_client=ts_mock, cache=None, enable_concept=True,
        )
        result = proc.process(
            code="300750", industry="新能源", run_date=dt.date(2026, 4, 18),
            trade_dates_10d=[], trade_dates_3d=[], trade_dates_20d=[],
        )
        # 通过 ths_member 查到属于芯片(885802.TI → 芯片)
        assert result.concept_heat_pctile_5d == 0.500
        assert result.concept_names == ["芯片"]

    def test_stock_truly_no_concept(self):
        """stock 完全没有概念归属。"""
        ctx = self._make_concept_ctx(cons_map={})

        ts_mock = MagicMock()
        ts_mock.get_ths_member.return_value = pd.DataFrame()

        proc = EventLayerProcessor(
            ctx=ctx, tushare_client=ts_mock, cache=None, enable_concept=True,
        )
        result = proc.process(
            code="999999", industry="未知", run_date=dt.date(2026, 4, 18),
            trade_dates_10d=[], trade_dates_3d=[], trade_dates_20d=[],
        )
        assert result.concept_heat_pctile_5d is None
        assert result.advanced_concept_module_available is True
        assert result.concept_heat_source == "ths_daily_full_no_concept"

    def test_concept_unavailable_mode(self):
        """概念热度不可用（200元档）。"""
        ctx = _make_ctx_full(concept_heat_mode="tushare_200_unavailable")
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

    def test_concept_disabled(self):
        """概念热度模块关闭（enable_concept=False）。"""
        ctx = self._make_concept_ctx()
        proc = EventLayerProcessor(
            ctx=ctx, tushare_client=MagicMock(), cache=None, enable_concept=False,
        )
        result = proc.process(
            code="002475", industry="电子", run_date=dt.date(2026, 4, 18),
            trade_dates_10d=[], trade_dates_3d=[], trade_dates_20d=[],
        )
        # enable_concept=False 时不填充概念字段
        assert result.concept_heat_pctile_5d is None


# ════════════════════════════════════════════════════════
# 集成测试：Loader + Processor 完整流程
# ════════════════════════════════════════════════════════

class TestEventLayerIntegration:
    def test_full_pipeline(self):
        """从 loader 到 processor 的完整流程。"""
        ts = _make_mock_ts()
        trade_dates = [dt.date(2026, 4, d) for d in range(1, 19)]

        loader = EventLayerLoader(
            tushare_client=ts,
            cache=None,
            run_date=dt.date(2026, 4, 18),
            trade_dates=trade_dates,
            enable_lhb_module=False,
            enable_concept_heat_module=True,
        )
        ctx = loader.load()

        assert ctx.industry_heat_mode == "ths_daily_full"
        assert ctx.concept_heat_mode == "ths_daily_full"

        proc = EventLayerProcessor(
            ctx=ctx, tushare_client=ts, cache=None,
            enable_lhb=False, enable_concept=True,
        )

        # 测试茅台 — 白酒行业 + 华为概念
        result = proc.process(
            code="600519", industry="白酒", run_date=dt.date(2026, 4, 18),
            trade_dates_10d=trade_dates[-10:],
            trade_dates_3d=trade_dates[-3:],
            trade_dates_20d=trade_dates,
        )
        assert result.industry_heat_pctile_5d is not None
        assert result.industry_heat_source.startswith("ths_daily_full")
        # 茅台在华为概念里（通过 ths_member 查询）
        assert result.concept_heat_source in ("ths_daily_full", "ths_daily_full_no_concept")

    def test_graceful_degradation(self):
        """无权限时优雅降级。"""
        ts = _make_mock_ts(permission=False)
        trade_dates = [dt.date(2026, 4, d) for d in range(1, 19)]

        loader = EventLayerLoader(
            tushare_client=ts,
            cache=None,
            run_date=dt.date(2026, 4, 18),
            trade_dates=trade_dates,
            enable_lhb_module=False,
            enable_concept_heat_module=True,
        )
        ctx = loader.load()

        assert ctx.industry_heat_mode == "tushare_200_degraded"
        assert ctx.concept_heat_mode == "tushare_200_unavailable"
