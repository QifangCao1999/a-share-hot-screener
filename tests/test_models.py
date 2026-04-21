"""test_models.py – 测试数据结构基本行为."""

import pytest

from a_share_hot_screener.models import (
    HotStockDetail,
    HotStockSummary,
    RejectedRecord,
    RunMetadata,
)


class TestHotStockDetail:
    def test_defaults(self):
        d = HotStockDetail(code="600519")
        assert d.code == "600519"
        assert d.name == ""
        assert d.pass_stage1 is False
        assert d.latest_price is None
        assert d.warnings == []
        assert d.flags == {}

    def test_to_cache_dict_roundtrip(self):
        d = HotStockDetail(
            code="600519",
            name="贵州茅台",
            exchange="SH",
            latest_price=1800.0,
            pass_stage1=True,
            warnings=["test warning"],
        )
        cache_dict = d.to_cache_dict()
        assert isinstance(cache_dict, dict)
        assert cache_dict["code"] == "600519"
        assert cache_dict["latest_price"] == 1800.0

        restored = HotStockDetail.from_cache_dict(cache_dict)
        assert restored.code == d.code
        assert restored.name == d.name
        assert restored.latest_price == d.latest_price
        assert restored.pass_stage1 == d.pass_stage1
        assert restored.warnings == d.warnings

    def test_from_cache_dict_extra_keys_ignored(self):
        d = {"code": "000858", "future_field_xyz": "ignored"}
        restored = HotStockDetail.from_cache_dict(d)
        assert restored.code == "000858"

    def test_from_cache_dict_missing_keys_use_defaults(self):
        d = {"code": "300750"}
        restored = HotStockDetail.from_cache_dict(d)
        assert restored.pass_stage1 is False
        assert restored.warnings == []


class TestHotStockSummary:
    def test_from_detail(self):
        d = HotStockDetail(
            code="600519",
            name="贵州茅台",
            exchange="SH",
            input_order=0,
            pass_stage1=True,
            latest_price=1800.0,
            float_market_cap=2e12,
            pct_change_1d=5.0,
            amount_avg_5d=3e9,    # Session 3: 字段改名 amount_5d_avg → amount_avg_5d
            turnover_rate_1d=1.2,
            total_score=85.0,
            passed_hard_filter=True,
            warnings=["warn1", "warn2"],
        )
        s = HotStockSummary.from_detail(d)
        assert s.code == "600519"
        assert s.pass_stage1 is True
        assert s.total_score == 85.0
        assert s.warnings_count == 2
        assert s.amount_avg_5d == 3e9


class TestRejectedRecord:
    def test_defaults(self):
        r = RejectedRecord(code="ABCDEF")
        assert r.code == "ABCDEF"
        assert r.reject_stage == ""
        assert r.reject_reason == ""

    def test_full_fields(self):
        r = RejectedRecord(
            code="000001",
            name="平安银行",
            input_order=1,
            reject_stage="hard_filter",
            reject_reason="industry_finance",
            reject_detail="银行业，被硬筛排除",
        )
        assert r.reject_stage == "hard_filter"
        assert r.reject_reason == "industry_finance"


class TestRunMetadata:
    def test_defaults(self):
        m = RunMetadata()
        assert m.version == "0.1.0"
        assert m.pass_stage1_count == 0
        assert m.modules_enabled == {}
        assert m.output_files == {}
