"""test_s9.py — Session 9 P2 改进项测试.

P2-1: detail 输出改为长表（每指标一行）
P2-2: summary 补齐缺失字段
P2-4: metadata 字段名对齐 + rejected_before_scoring_count
"""

from __future__ import annotations

import csv
import dataclasses
import json
import os
from dataclasses import fields as dc_fields

import pytest

from a_share_hot_screener.limit_rules import infer_limit_pct
from a_share_hot_screener.models import (
    HotStockDetail,
    HotStockSummary,
    RejectedRecord,
    RunMetadata,
)
from a_share_hot_screener.output import (
    OutputWriter,
    _detail_to_long_rows,
    _fmt_float,
    _fmt_val,
    _DETAIL_LONG_COLUMNS,
)


# ════════════════════════════════════════════════════════
# 测试数据工厂
# ════════════════════════════════════════════════════════

def _make_detail_with_scores() -> HotStockDetail:
    """构造带有完整四轴评分 subscores 的 detail."""
    d = HotStockDetail(
        code="600519",
        name="贵州茅台",
        exchange="SH",
        ts_code="600519.SHG",
        input_order=0,
        industry="白酒",
        listing_days=8000,
        latest_price=1800.0,
        market_cap=2.2e12,
        float_market_cap=2.0e12,
        pct_change_1d=2.5,
        amount_avg_5d=5e9,
        amount_avg_20d=4e9,
        return_3d=3.5,
        return_5d=5.0,
        return_10d=8.0,
        volume_ratio_20d=1.5,
        close_position_20d=0.85,
        clv_latest=0.75,
        amount_ratio_5d_to_20d=1.25,
        abs_distance_to_ma10=0.02,
        ma5=1790.0,
        ma10=1770.0,
        ma20=1750.0,
        passed_hard_filter=True,
        pass_stage1=True,
        total_score=0.72,
        data_coverage=0.95,
        hot_theme_score=0.80,
        hot_theme_coverage=1.0,
        hot_theme_subscores={
            "axis": "hot_theme_score",
            "score": 0.80,
            "coverage": 1.0,
            "items": [
                {"name": "return_5d_pctile", "raw_value": 5.0, "derived_value": 0.85,
                 "subscore": 0.70, "weight": 8.0, "weighted_score": 5.6,
                 "is_applicable": True, "is_data_available": True,
                 "note": "近5日收益率百分位→三段型"},
                {"name": "return_10d_pctile", "raw_value": 8.0, "derived_value": 0.90,
                 "subscore": 0.85, "weight": 6.0, "weighted_score": 5.1,
                 "is_applicable": True, "is_data_available": True,
                 "note": "近10日收益率百分位→三段型"},
                {"name": "limit_up_count_10d", "raw_value": 2, "derived_value": None,
                 "subscore": 0.70, "weight": 8.0, "weighted_score": 5.6,
                 "is_applicable": True, "is_data_available": True,
                 "note": "离散映射"},
                {"name": "strong_pool_entry_3d", "raw_value": 1, "derived_value": None,
                 "subscore": 0.60, "weight": 6.0, "weighted_score": 3.6,
                 "is_applicable": True, "is_data_available": True,
                 "note": "离散映射"},
                {"name": "industry_heat_pctile_5d", "raw_value": 75.0, "derived_value": None,
                 "subscore": 0.70, "weight": 7.0, "weighted_score": 4.9,
                 "is_applicable": True, "is_data_available": True,
                 "note": "三段下限型"},
            ],
        },
        trend_flow_score=0.65,
        trend_flow_coverage=1.0,
        trend_flow_subscores={
            "axis": "trend_flow_score",
            "score": 0.65,
            "coverage": 1.0,
            "items": [
                {"name": "close_position_20d", "raw_value": 0.85, "derived_value": None,
                 "subscore": 0.80, "weight": 7.0, "weighted_score": 5.6,
                 "is_applicable": True, "is_data_available": True, "note": "三段下限型"},
                {"name": "ma_bullish_alignment", "raw_value": 3, "derived_value": 3,
                 "subscore": 1.0, "weight": 6.0, "weighted_score": 6.0,
                 "is_applicable": True, "is_data_available": True, "note": "离散型"},
                {"name": "volume_ratio_20d", "raw_value": 1.5, "derived_value": None,
                 "subscore": 0.40, "weight": 6.0, "weighted_score": 2.4,
                 "is_applicable": True, "is_data_available": True, "note": "三段下限型"},
                {"name": "clv_latest", "raw_value": 0.75, "derived_value": None,
                 "subscore": 0.52, "weight": 5.0, "weighted_score": 2.6,
                 "is_applicable": True, "is_data_available": True, "note": "三段下限型"},
                {"name": "amount_ratio_5d_to_20d", "raw_value": 1.25, "derived_value": None,
                 "subscore": 0.35, "weight": 6.0, "weighted_score": 2.1,
                 "is_applicable": True, "is_data_available": True, "note": "三段下限型"},
            ],
        },
        liquidity_execution_score=0.70,
        liquidity_execution_coverage=1.0,
        liquidity_execution_subscores={
            "axis": "liquidity_execution_score",
            "score": 0.70,
            "coverage": 1.0,
            "items": [
                {"name": "amount_avg_5d", "raw_value": 5e9, "derived_value": None,
                 "subscore": 0.75, "weight": 8.0, "weighted_score": 6.0,
                 "is_applicable": True, "is_data_available": True, "note": "三段下限型"},
                {"name": "turnover_avg_5d_approx", "raw_value": 10.0, "derived_value": None,
                 "subscore": 0.60, "weight": 5.0, "weighted_score": 3.0,
                 "is_applicable": True, "is_data_available": True, "note": "三段下限型"},
                {"name": "float_market_cap_bracket", "raw_value": 2e12, "derived_value": 20000.0,
                 "subscore": 0.20, "weight": 4.0, "weighted_score": 0.8,
                 "is_applicable": True, "is_data_available": True, "note": "离散分档"},
                {"name": "lhb_count_20d", "raw_value": 1, "derived_value": None,
                 "subscore": 0.60, "weight": 3.0, "weighted_score": 1.8,
                 "is_applicable": True, "is_data_available": True, "note": "离散映射"},
            ],
        },
        risk_control_score=0.75,
        risk_control_coverage=1.0,
        risk_control_subscores={
            "axis": "risk_control_score",
            "score": 0.75,
            "coverage": 1.0,
            "items": [
                {"name": "latest_limit_board", "raw_value": False, "derived_value": None,
                 "subscore": 1.0, "weight": 4.0, "weighted_score": 4.0,
                 "is_applicable": True, "is_data_available": True, "note": "非一字板"},
                {"name": "amp_norm_5d_ratio", "raw_value": 0.8, "derived_value": None,
                 "subscore": 0.70, "weight": 3.0, "weighted_score": 2.1,
                 "is_applicable": True, "is_data_available": True, "note": "三段上限型"},
                {"name": "abs_deviation_ma10", "raw_value": 2.0, "derived_value": None,
                 "subscore": 1.0, "weight": 4.0, "weighted_score": 4.0,
                 "is_applicable": True, "is_data_available": True, "note": "三段上限型"},
                {"name": "upper_shadow_count_5d", "raw_value": 0, "derived_value": None,
                 "subscore": 1.0, "weight": 2.0, "weighted_score": 2.0,
                 "is_applicable": True, "is_data_available": True, "note": "离散型"},
                {"name": "return_3d_vs_limit", "raw_value": 0.5, "derived_value": None,
                 "subscore": 1.0, "weight": 2.0, "weighted_score": 2.0,
                 "is_applicable": True, "is_data_available": True, "note": "三段上限型"},
            ],
        },
        flags={
            "shareholder_net_reduction_ratio_3m": None,
            "shareholder_reduction_flag_3m": None,
            "restricted_shares_unlock_ratio_20d": None,
            "unlock_risk_flag_20d": None,
            "pledge_ratio_latest": None,
            "pledge_ratio_flag": None,
        },
    )
    return d


def _make_detail_no_scores() -> HotStockDetail:
    """构造不含评分的 detail（未通过硬筛）."""
    return HotStockDetail(
        code="000001",
        name="平安银行",
        exchange="SZ",
        passed_hard_filter=False,
        hard_filter_reason="industry_finance",
        pass_stage1=False,
        flags={},
    )


# ════════════════════════════════════════════════════════
# P2-1 测试：detail 长表格式
# ════════════════════════════════════════════════════════

class TestP2_1_DetailLongTable:
    """P2-1: detail 输出改为长表."""

    def test_long_columns_complete(self):
        """长表列名定义完整."""
        expected = {"code", "name", "axis", "indicator_id", "indicator_name",
                    "raw_value", "derived_value", "subscore", "weight",
                    "weighted_score", "is_applicable", "is_data_available", "note"}
        assert expected == set(_DETAIL_LONG_COLUMNS)

    def test_detail_to_long_rows_with_scores(self):
        """有评分的 detail 应展开为多行：每指标+轴汇总+全局汇总."""
        d = _make_detail_with_scores()
        rows = _detail_to_long_rows(d)

        # 4轴 × (指标数 + 1轴汇总行) + 5全局汇总行
        # HT:5+1=6, TF:5+1=6, LE:4+1=5, RC:5+1=6 = 23轴行
        # + 5 summary行 = 28
        assert len(rows) == 29  # S10: blocked_by 加入 summary_items (+1)

        # 所有行都属于 code=600519
        for r in rows:
            assert r["code"] == "600519"

        # 检查轴行
        axis_names = {r["axis"] for r in rows}
        assert axis_names == {"hot_theme", "trend_flow", "liquidity_execution",
                              "risk_control", "summary"}

        # 检查 indicator_id 前缀
        ht_ids = [r["indicator_id"] for r in rows if r["axis"] == "hot_theme"]
        assert "HT1" in ht_ids
        assert "HT5" in ht_ids
        assert "HT_TOTAL" in ht_ids

        tf_ids = [r["indicator_id"] for r in rows if r["axis"] == "trend_flow"]
        assert "TF1" in tf_ids
        assert "TF_TOTAL" in tf_ids

        le_ids = [r["indicator_id"] for r in rows if r["axis"] == "liquidity_execution"]
        assert "LE1" in le_ids
        assert "LE_TOTAL" in le_ids

        rc_ids = [r["indicator_id"] for r in rows if r["axis"] == "risk_control"]
        assert "RC1" in rc_ids
        assert "RC_TOTAL" in rc_ids

    def test_detail_to_long_rows_no_scores(self):
        """未通过硬筛的 detail 只产生 summary 行."""
        d = _make_detail_no_scores()
        rows = _detail_to_long_rows(d)
        # 只有 5 行 summary（total_score/data_coverage/pass_stage1/passed_hard_filter/hard_filter_reason）
        assert len(rows) == 6  # S10: blocked_by 加入 summary_items (+1)
        for r in rows:
            assert r["axis"] == "summary"

    def test_detail_long_rows_summary_fields(self):
        """全局 summary 行应包含关键指标."""
        d = _make_detail_with_scores()
        rows = _detail_to_long_rows(d)
        summary_rows = [r for r in rows if r["axis"] == "summary"]
        summary_names = {r["indicator_name"] for r in summary_rows}
        assert "total_score" in summary_names
        assert "pass_stage1" in summary_names
        assert "data_coverage" in summary_names
        assert "passed_hard_filter" in summary_names
        assert "hard_filter_reason" in summary_names

    def test_detail_long_score_item_fields(self):
        """每个指标行应包含完整的评分溯源信息."""
        d = _make_detail_with_scores()
        rows = _detail_to_long_rows(d)
        ht1 = [r for r in rows if r["indicator_id"] == "HT1"][0]
        assert ht1["indicator_name"] == "return_5d_pctile"
        assert ht1["raw_value"] == "5.0000"
        assert ht1["subscore"] == "0.7000"
        assert ht1["weight"] == "8.0000"
        assert ht1["weighted_score"] == "5.6000"
        assert ht1["is_applicable"] is True
        assert ht1["is_data_available"] is True

    def test_write_detail_long_csv(self, tmp_path):
        """OutputWriter.write_detail 应写入长表 CSV."""
        writer = OutputWriter(str(tmp_path / "output"))
        details = [_make_detail_with_scores(), _make_detail_no_scores()]
        path = os.path.join(str(tmp_path / "output"), "test_detail.csv")
        writer.write_detail(details, path)

        with open(path, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        # 第一只股票 28 行 + 第二只 5 行 = 33 行
        assert len(rows) == 35  # S10: blocked_by 加入 summary_items (×2 stocks = +2)
        codes = {r["code"] for r in rows}
        assert codes == {"600519", "000001"}

    def test_write_detail_empty(self, tmp_path):
        """空 details 应写入只有表头的文件."""
        writer = OutputWriter(str(tmp_path / "output"))
        path = os.path.join(str(tmp_path / "output"), "test_detail_empty.csv")
        writer.write_detail([], path)
        with open(path, "r", encoding="utf-8-sig") as f:
            lines = f.readlines()
        assert len(lines) == 1  # 只有表头
        header = lines[0].strip()
        for col in _DETAIL_LONG_COLUMNS:
            assert col in header

    def test_axis_total_row_has_coverage(self):
        """轴汇总行的 note 应包含 coverage 信息."""
        d = _make_detail_with_scores()
        rows = _detail_to_long_rows(d)
        ht_total = [r for r in rows if r["indicator_id"] == "HT_TOTAL"][0]
        assert "coverage=" in ht_total["note"]
        assert ht_total["subscore"] == "0.8000"  # hot_theme_score


# ════════════════════════════════════════════════════════
# P2-2 测试：summary 补齐字段
# ════════════════════════════════════════════════════════

class TestP2_2_SummaryFields:
    """P2-2: summary 补齐缺失字段."""

    def test_summary_has_new_fields(self):
        """HotStockSummary 应包含 P2-2 新增的所有字段."""
        field_names = {f.name for f in dc_fields(HotStockSummary)}
        new_fields = {
            "validation_status", "ts_code", "industry",
            "market_cap", "listing_days", "board_limit_pct",
            "return_3d", "return_10d",
            "amount_avg_20d", "clv_latest",
            "amount_ratio_5d_to_20d", "abs_distance_to_ma10",
            "ma_bullish_alignment_count",
            # 减持/质押/解禁 6 个字段
            "shareholder_net_reduction_ratio_3m",
            "shareholder_reduction_flag_3m",
            "restricted_shares_unlock_ratio_20d",
            "unlock_risk_flag_20d",
            "pledge_ratio_latest",
            "pledge_ratio_flag",
        }
        for f in new_fields:
            assert f in field_names, f"HotStockSummary 缺少字段: {f}"

    def test_from_detail_populates_new_fields(self):
        """from_detail 应正确填充新增字段."""
        d = _make_detail_with_scores()
        s = HotStockSummary.from_detail(d)

        assert s.ts_code == "600519.SHG"
        assert s.validation_status == "valid"
        assert s.industry == "白酒"
        assert s.listing_days == 8000
        assert s.board_limit_pct == 10.0  # 600xxx → 主板 10%
        assert s.market_cap == 2.2e12
        assert s.return_3d == 3.5
        assert s.return_10d == 8.0
        assert s.amount_avg_20d == 4e9
        assert s.clv_latest == 0.75
        assert s.amount_ratio_5d_to_20d == 1.25
        assert s.abs_distance_to_ma10 == 0.02
        assert s.ma_bullish_alignment_count == 3  # close>ma5, ma5>ma10, ma10>ma20

    def test_from_detail_board_limit_pct_chinext(self):
        """创业板/科创板应返回 20%."""
        d = _make_detail_with_scores()
        d.code = "300750"
        s = HotStockSummary.from_detail(d)
        assert s.board_limit_pct == 20.0

        d.code = "688001"
        s = HotStockSummary.from_detail(d)
        assert s.board_limit_pct == 20.0

    def test_from_detail_ma_alignment_none_when_data_missing(self):
        """均线数据缺失时 ma_bullish_alignment_count 应为 None."""
        d = _make_detail_with_scores()
        d.ma5 = None
        s = HotStockSummary.from_detail(d)
        assert s.ma_bullish_alignment_count is None

    def test_from_detail_reduction_fields_from_flags(self):
        """减持/质押/解禁字段应从 flags dict 读取."""
        d = _make_detail_with_scores()
        d.flags["pledge_ratio_latest"] = 0.15
        d.flags["pledge_ratio_flag"] = True
        s = HotStockSummary.from_detail(d)
        assert s.pledge_ratio_latest == 0.15
        assert s.pledge_ratio_flag is True
        # 其余仍为 None
        assert s.shareholder_net_reduction_ratio_3m is None


# ════════════════════════════════════════════════════════
# P2-4 测试：metadata 字段名对齐
# ════════════════════════════════════════════════════════

class TestP2_4_MetadataAlignment:
    """P2-4: metadata 字段名对齐."""

    def test_input_pool_size_replaces_input_count(self):
        """input_pool_size 应为正式字段名，input_count 为向后兼容属性."""
        field_names = {f.name for f in dc_fields(RunMetadata)}
        assert "input_pool_size" in field_names
        # input_count 不再是 dataclass 字段，而是 @property
        assert "input_count" not in field_names

        m = RunMetadata(input_pool_size=10)
        assert m.input_pool_size == 10
        assert m.input_count == 10  # 向后兼容

    def test_rejected_before_scoring_count(self):
        """新增 rejected_before_scoring_count 字段."""
        field_names = {f.name for f in dc_fields(RunMetadata)}
        assert "rejected_before_scoring_count" in field_names

        m = RunMetadata(
            validation_rejected=3,
            hard_filter_rejected=5,
            rejected_before_scoring_count=8,
        )
        assert m.rejected_before_scoring_count == 8

    def test_axis_weights_field(self):
        """新增 axis_weights dict 字段."""
        field_names = {f.name for f in dc_fields(RunMetadata)}
        assert "axis_weights" in field_names

        m = RunMetadata(axis_weights={
            "hot_theme": 35.0,
            "trend_flow": 30.0,
            "liquidity_execution": 20.0,
            "risk_control": 15.0,
        })
        assert m.axis_weights["hot_theme"] == 35.0
        assert sum(m.axis_weights.values()) == 100.0

    def test_metadata_json_serialization(self, tmp_path):
        """metadata JSON 序列化应包含新字段."""
        m = RunMetadata(
            run_date="2026-04-18",
            input_pool_size=10,
            rejected_before_scoring_count=3,
            axis_weights={"hot_theme": 35.0, "trend_flow": 30.0,
                          "liquidity_execution": 20.0, "risk_control": 15.0},
        )
        writer = OutputWriter(str(tmp_path / "output"))
        path = os.path.join(str(tmp_path / "output"), "test_meta.json")
        writer.write_metadata(m, path)

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        assert data["input_pool_size"] == 10
        assert data["rejected_before_scoring_count"] == 3
        assert data["axis_weights"]["hot_theme"] == 35.0
        assert "input_count" not in data  # property 不在 dataclass.asdict 里


# ════════════════════════════════════════════════════════
# 格式化工具函数测试
# ════════════════════════════════════════════════════════

class TestFormatUtils:
    """格式化工具函数."""

    def test_fmt_float_none(self):
        assert _fmt_float(None) == ""

    def test_fmt_float_number(self):
        assert _fmt_float(0.123456) == "0.1235"

    def test_fmt_float_int(self):
        assert _fmt_float(3) == "3.0000"

    def test_fmt_val_none(self):
        assert _fmt_val(None) == ""

    def test_fmt_val_bool(self):
        assert _fmt_val(True) == "True"
        assert _fmt_val(False) == "False"

    def test_fmt_val_dict(self):
        result = _fmt_val({"a": 1})
        assert '"a"' in result

    def test_fmt_val_float(self):
        assert _fmt_val(1.5) == "1.5000"
