"""test_batch_runner.py – 测试批量运行编排器."""

import csv
import json
import os
from unittest.mock import patch, MagicMock

import pytest

from a_share_hot_screener.batch_runner import (
    _split,
    _merge_csv,
    _merge_metadata,
    _save_progress,
    _load_progress,
    run_batched,
)
from a_share_hot_screener.config import HotScreenerConfig
from a_share_hot_screener.models import RunMetadata

import datetime as dt


# ════════════════════════════════════════════════════════
# 工具函数测试
# ════════════════════════════════════════════════════════

class TestSplit:
    def test_even_split(self):
        result = _split(list(range(10)), 5)
        assert result == [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]

    def test_uneven_split(self):
        result = _split(list(range(7)), 3)
        assert len(result) == 3
        assert result[0] == [0, 1, 2]
        assert result[1] == [3, 4, 5]
        assert result[2] == [6]

    def test_single_batch(self):
        result = _split(["a", "b"], 10)
        assert result == [["a", "b"]]

    def test_empty(self):
        assert _split([], 5) == []


class TestProgress:
    def test_save_and_load(self, tmp_path):
        _save_progress(str(tmp_path), "2026-04-20", {0: "batch_1", 1: "batch_2"})
        progress = _load_progress(str(tmp_path))
        assert progress is not None
        assert progress["run_date"] == "2026-04-20"
        assert "0" in progress["completed"]

    def test_load_nonexistent(self, tmp_path):
        assert _load_progress(str(tmp_path / "nope")) is None


class TestMergeCsv:
    def _write_csv(self, path, fieldnames, rows):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(rows)

    def test_merge_summary(self, tmp_path):
        bd1 = str(tmp_path / "batch_1")
        bd2 = str(tmp_path / "batch_2")
        fields = ["code", "name", "input_order", "total_score"]

        self._write_csv(
            os.path.join(bd1, "2026-04-20_summary.csv"),
            fields,
            [{"code": "600519", "name": "贵州茅台", "input_order": "0", "total_score": "0.80"}],
        )
        self._write_csv(
            os.path.join(bd2, "2026-04-20_summary.csv"),
            fields,
            [{"code": "000858", "name": "五粮液", "input_order": "1", "total_score": "0.75"}],
        )

        out_path = str(tmp_path / "merged.csv")
        _merge_csv([bd1, bd2], "2026-04-20", "summary.csv", out_path, sort_key="input_order")

        with open(out_path, "r", encoding="utf-8-sig") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 2
        assert rows[0]["code"] == "600519"
        assert rows[1]["code"] == "000858"

    def test_merge_missing_batch(self, tmp_path):
        """某批目录不存在时跳过."""
        bd1 = str(tmp_path / "batch_1")
        bd2 = str(tmp_path / "batch_2_missing")

        os.makedirs(bd1, exist_ok=True)
        fields = ["code"]
        self._write_csv(
            os.path.join(bd1, "2026-04-20_data.csv"),
            fields,
            [{"code": "600519"}],
        )

        out_path = str(tmp_path / "merged.csv")
        _merge_csv([bd1, bd2], "2026-04-20", "data.csv", out_path)

        with open(out_path, "r", encoding="utf-8-sig") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 1


class TestMergeMetadata:
    def test_aggregate_counts(self):
        config = HotScreenerConfig(
            tushare_token="x",
            run_date=dt.date(2026, 4, 20),
            stock_codes=["600519", "000858", "300750"],
            output_dir="/tmp/test",
        )
        m1 = RunMetadata(
            run_date="2026-04-20",
            trade_date_used="2026-04-20",
            input_pool_size=2,
            valid_input_count=2,
            pass_stage1_count=1,
            hard_filter_passed=2,
            input_stock_codes=["600519", "000858"],
            average_data_coverage=0.85,
        )
        m2 = RunMetadata(
            run_date="2026-04-20",
            trade_date_used="2026-04-20",
            input_pool_size=1,
            valid_input_count=1,
            pass_stage1_count=0,
            hard_filter_passed=1,
            input_stock_codes=["300750"],
            average_data_coverage=0.90,
        )

        merged = _merge_metadata(config, [m1, m2], 100.0, 2)

        assert merged.input_pool_size == 3
        assert merged.pass_stage1_count == 1
        assert merged.hard_filter_passed == 3
        assert len(merged.input_stock_codes) == 3
        assert merged.elapsed_seconds == 100.0
        assert any("batch_mode" in w for w in merged.global_warnings)


# ════════════════════════════════════════════════════════
# 端到端集成测试（mock pipeline）
# ════════════════════════════════════════════════════════

class TestRunBatched:
    @patch("a_share_hot_screener.batch_runner.Stage1HotPipeline")
    def test_splits_and_merges(self, MockPipeline, tmp_path):
        """验证 run_batched 按 batch_size 分批并合并结果."""
        out_dir = str(tmp_path / "output")

        # Mock pipeline.run() → 写入 CSV + 返回 metadata
        call_count = [0]

        def mock_run_side_effect():
            call_count[0] += 1
            idx = call_count[0]
            # 获取传给 Pipeline 的 config
            mock_config = MockPipeline.call_args[0][0]
            batch_dir = mock_config.output_dir
            os.makedirs(batch_dir, exist_ok=True)
            run_date = mock_config.run_date_str

            # 写一个 summary CSV
            fields = ["code", "name", "input_order"]
            with open(
                os.path.join(batch_dir, f"{run_date}_stage1_hot_summary.csv"),
                "w", newline="", encoding="utf-8-sig",
            ) as f:
                w = csv.DictWriter(f, fieldnames=fields)
                w.writeheader()
                for c in mock_config.stock_codes:
                    w.writerow({"code": c, "name": f"stock_{c}", "input_order": "0"})

            # 写空的 detail 和 rejected
            for suffix in ["stage1_hot_detail.csv", "stage1_hot_rejected.csv"]:
                with open(
                    os.path.join(batch_dir, f"{run_date}_{suffix}"),
                    "w", newline="", encoding="utf-8-sig",
                ) as f:
                    f.write("code\n")

            return RunMetadata(
                run_date=run_date,
                trade_date_used=run_date,
                input_pool_size=len(mock_config.stock_codes),
                valid_input_count=len(mock_config.stock_codes),
                pass_stage1_count=1 if idx == 1 else 0,
                hard_filter_passed=len(mock_config.stock_codes),
                input_stock_codes=list(mock_config.stock_codes),
                elapsed_seconds=10.0,
            )

        mock_instance = MagicMock()
        mock_instance.run.side_effect = mock_run_side_effect
        MockPipeline.return_value = mock_instance

        config = HotScreenerConfig(
            tushare_token="test",
            run_date=dt.date(2026, 4, 20),
            stock_codes=["600519", "000858", "300750", "601318", "000001"],
            output_dir=out_dir,
            batch_size=2,
        )

        metadata = run_batched(config)

        # 应该分成 3 批（2+2+1）
        assert MockPipeline.call_count == 3
        assert metadata.input_pool_size == 5
        assert metadata.pass_stage1_count == 1

        # 合并 summary 应有 5 行
        summary_path = os.path.join(out_dir, "2026-04-20_stage1_hot_summary.csv")
        assert os.path.exists(summary_path)
        with open(summary_path, "r", encoding="utf-8-sig") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 5

    @patch("a_share_hot_screener.batch_runner.Stage1HotPipeline")
    def test_resume_skips_completed(self, MockPipeline, tmp_path):
        """--resume 跳过已完成的批次."""
        out_dir = str(tmp_path / "output")

        # 预写进度文件：batch 0 已完成
        os.makedirs(out_dir, exist_ok=True)
        batch_0_dir = os.path.join(out_dir, "batch_1")
        os.makedirs(batch_0_dir, exist_ok=True)

        # 写一个假 metadata
        run_date = "2026-04-20"
        meta = RunMetadata(
            run_date=run_date,
            trade_date_used=run_date,
            input_pool_size=2,
            valid_input_count=2,
            pass_stage1_count=1,
            input_stock_codes=["600519", "000858"],
        )
        meta_path = os.path.join(batch_0_dir, f"{run_date}_stage1_hot_metadata.json")
        import dataclasses
        with open(meta_path, "w") as f:
            json.dump(dataclasses.asdict(meta), f, default=str)

        # 写假 CSV
        for suffix in ["stage1_hot_summary.csv", "stage1_hot_detail.csv", "stage1_hot_rejected.csv"]:
            with open(os.path.join(batch_0_dir, f"{run_date}_{suffix}"), "w", encoding="utf-8-sig") as f:
                f.write("code\n600519\n")

        # 写进度
        progress = {"run_date": "2026-04-20", "completed": {"0": batch_0_dir}}
        with open(os.path.join(out_dir, ".batch_progress.json"), "w") as f:
            json.dump(progress, f)

        # batch 1 正常跑
        def mock_run():
            mock_config = MockPipeline.call_args[0][0]
            bd = mock_config.output_dir
            os.makedirs(bd, exist_ok=True)
            for suffix in ["stage1_hot_summary.csv", "stage1_hot_detail.csv", "stage1_hot_rejected.csv"]:
                with open(os.path.join(bd, f"{run_date}_{suffix}"), "w", encoding="utf-8-sig") as f:
                    f.write("code\n300750\n")
            return RunMetadata(
                run_date=run_date,
                trade_date_used=run_date,
                input_pool_size=1,
                valid_input_count=1,
                input_stock_codes=["300750"],
            )

        mock_instance = MagicMock()
        mock_instance.run.side_effect = mock_run
        MockPipeline.return_value = mock_instance

        config = HotScreenerConfig(
            tushare_token="test",
            run_date=dt.date(2026, 4, 20),
            stock_codes=["600519", "000858", "300750"],
            output_dir=out_dir,
            batch_size=2,
            resume=True,
        )

        metadata = run_batched(config)

        # 只应跑 1 批（batch 0 被跳过）
        assert MockPipeline.call_count == 1
        assert metadata.input_pool_size == 3
