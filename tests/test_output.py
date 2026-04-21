"""test_output.py – 测试 OutputWriter 输出格式."""

import csv
import json
import os
import dataclasses

import pytest

from a_share_hot_screener.models import (
    HotStockDetail,
    RejectedRecord,
    RunMetadata,
)
from a_share_hot_screener.output import OutputWriter


@pytest.fixture
def writer(tmp_path):
    return OutputWriter(str(tmp_path / "output"))


def make_detail(code: str = "600519", pass_s1: bool = True) -> HotStockDetail:
    return HotStockDetail(
        code=code,
        name="测试股票",
        exchange="SH",
        input_order=0,
        pass_stage1=pass_s1,
        latest_price=100.0,
        float_market_cap=5e9,
        total_score=80.0,
        passed_hard_filter=True,
        warnings=["warn1"],
    )


def make_rejected() -> RejectedRecord:
    return RejectedRecord(
        code="ABCDEF",
        reject_stage="code_parse",
        reject_reason="invalid_code_format",
        reject_detail="无法解析",
    )


def make_metadata() -> RunMetadata:
    return RunMetadata(
        run_date="2026-04-18",
        generated_at="2026-04-18T00:00:00Z",
        input_pool_size=2,
        pass_stage1_count=1,
    )


class TestOutputWriter:
    def test_write_summary_creates_csv(self, writer, tmp_path):
        details = [make_detail("600519", True), make_detail("000858", False)]
        path = os.path.join(str(tmp_path / "output"), "2026-04-18_stage1_hot_summary.csv")
        writer.write_summary(details, path)
        assert os.path.exists(path)
        with open(path, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert len(rows) == 2
        assert rows[0]["code"] == "600519"

    def test_write_detail_creates_csv(self, writer, tmp_path):
        """P2-1: detail 现在是长表格式，每只股票产生多行（评分指标 + summary 行）."""
        details = [make_detail()]
        path = os.path.join(str(tmp_path / "output"), "2026-04-18_stage1_hot_detail.csv")
        writer.write_detail(details, path)
        assert os.path.exists(path)
        with open(path, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        # 长表格式：至少有 summary 行（total_score/data_coverage/pass_stage1/passed_hard_filter/hard_filter_reason）
        assert len(rows) >= 5, f"detail 长表应至少有5行summary，实际{len(rows)}行"
        # 所有行都属于同一只股票
        for row in rows:
            assert row["code"] == "600519"
        # 必须包含长表标准列
        expected_cols = {"code", "name", "axis", "indicator_id", "indicator_name",
                        "raw_value", "derived_value", "subscore", "weight",
                        "weighted_score", "is_applicable", "is_data_available", "note"}
        assert expected_cols.issubset(set(rows[0].keys()))

    def test_write_rejected_creates_csv(self, writer, tmp_path):
        rejected = [make_rejected()]
        path = os.path.join(str(tmp_path / "output"), "2026-04-18_stage1_hot_rejected.csv")
        writer.write_rejected(rejected, path)
        assert os.path.exists(path)
        with open(path, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert len(rows) == 1
        assert rows[0]["reject_stage"] == "code_parse"

    def test_write_metadata_creates_json(self, writer, tmp_path):
        metadata = make_metadata()
        path = os.path.join(str(tmp_path / "output"), "2026-04-18_stage1_hot_metadata.json")
        writer.write_metadata(metadata, path)
        assert os.path.exists(path)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert data["run_date"] == "2026-04-18"
        assert data["pass_stage1_count"] == 1

    def test_write_all_creates_all_files(self, writer, tmp_path):
        details = [make_detail()]
        rejected = [make_rejected()]
        metadata = make_metadata()
        writer.write_all(details=details, rejected=rejected, metadata=metadata)

        out_dir = str(tmp_path / "output")
        files = os.listdir(out_dir)
        assert any("summary" in f for f in files)
        assert any("detail" in f for f in files)
        assert any("rejected" in f for f in files)
        assert any("metadata" in f for f in files)

    def test_empty_details_writes_header_only(self, writer, tmp_path):
        path = os.path.join(str(tmp_path / "output"), "2026-04-18_stage1_hot_summary.csv")
        writer.write_summary([], path)
        with open(path, "r", encoding="utf-8-sig") as f:
            lines = f.readlines()
        assert len(lines) == 1  # 只有表头
