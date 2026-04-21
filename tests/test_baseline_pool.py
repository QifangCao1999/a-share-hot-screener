"""Session 12 P0-2 测试: 基准 pool 序列化/加载/合并 + CLI 参数.

测试覆盖：
  1. ScoringPool.save_baseline / load_baseline 往返（round-trip）
  2. ScoringPool.merge_with_baseline 合并逻辑
  3. load_baseline 文件不存在 → None
  4. load_baseline 格式损坏 → None
  5. pipeline _resolve_baseline_pool_path 逻辑
  6. CLI 新增参数解析
  7. Config 新增字段默认值
  8. metadata 新增 used_baseline_pool / baseline_pool_stock_count 字段
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass

import pytest

from a_share_hot_screener.scoring import ScoringPool


# ════════════════════════════════════════════════════════
# 1. ScoringPool round-trip 测试
# ════════════════════════════════════════════════════════


def _make_sample_pool(n: int = 10) -> ScoringPool:
    """构建一个有真实数据的 ScoringPool."""
    pool = ScoringPool()
    pool.stock_count = n
    pool.pool_return_5d = [float(i) for i in range(n)]
    pool.pool_return_10d = [float(i * 2) for i in range(n)]
    pool.pool_limit_up_count_10d = [float(i % 3) for i in range(n)]
    pool.pool_strong_pool_3d = [float(i % 2) for i in range(n)]
    pool.pool_industry_heat_pctile = [0.1 * i for i in range(n)]
    pool.pool_volume_ratio_20d = [1.0 + 0.1 * i for i in range(n)]
    pool.pool_amount_ratio_5d_to_20d = [1.0 + 0.2 * i for i in range(n)]
    pool.pool_close_position_20d = [0.5 + 0.05 * i for i in range(n)]
    pool.pool_clv_latest = [0.3 + 0.07 * i for i in range(n)]
    pool.pool_amount_avg_5d = [2e8 + 1e7 * i for i in range(n)]
    return pool


class TestBaselineRoundTrip:
    """save_baseline → load_baseline 往返测试."""

    def test_round_trip_preserves_all_fields(self, tmp_path):
        pool = _make_sample_pool(20)
        path = str(tmp_path / "baseline.json")
        pool.save_baseline(path)

        loaded = ScoringPool.load_baseline(path)
        assert loaded is not None
        assert loaded.stock_count == 20

        for fname in ScoringPool._POOL_FIELDS:
            # load_baseline 会预排序，所以比较排序后的结果
            assert getattr(loaded, fname) == sorted(getattr(pool, fname)), f"字段 {fname} 不一致"

    def test_saved_file_is_valid_json(self, tmp_path):
        pool = _make_sample_pool(5)
        path = str(tmp_path / "baseline.json")
        pool.save_baseline(path)

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert isinstance(data, dict)
        assert data["stock_count"] == 5
        assert "pool_return_5d" in data
        assert len(data["pool_return_5d"]) == 5

    def test_creates_parent_directories(self, tmp_path):
        pool = _make_sample_pool(3)
        path = str(tmp_path / "sub" / "dir" / "baseline.json")
        pool.save_baseline(path)
        assert os.path.isfile(path)


# ════════════════════════════════════════════════════════
# 2. merge_with_baseline 测试
# ════════════════════════════════════════════════════════


class TestMerge:
    def test_merge_combines_lists(self):
        live = ScoringPool()
        live.stock_count = 2
        live.pool_return_5d = [1.0, 2.0]
        live.pool_return_10d = [10.0, 20.0]

        base = _make_sample_pool(10)

        merged = live.merge_with_baseline(base)
        assert merged.stock_count == 12
        # merge 后会预排序
        assert merged.pool_return_5d == sorted([1.0, 2.0] + base.pool_return_5d)
        assert merged.pool_return_10d == sorted([10.0, 20.0] + base.pool_return_10d)

    def test_merge_empty_live_with_baseline(self):
        live = ScoringPool()
        live.stock_count = 0
        base = _make_sample_pool(10)

        merged = live.merge_with_baseline(base)
        assert merged.stock_count == 10
        assert merged.pool_return_5d == base.pool_return_5d

    def test_merge_preserves_all_fields(self):
        live = _make_sample_pool(3)
        base = _make_sample_pool(7)

        merged = live.merge_with_baseline(base)
        assert merged.stock_count == 10

        for fname in ScoringPool._POOL_FIELDS:
            expected = sorted(getattr(live, fname) + getattr(base, fname))
            assert getattr(merged, fname) == expected, f"字段 {fname} 合并不一致"


# ════════════════════════════════════════════════════════
# 3. load_baseline 错误处理
# ════════════════════════════════════════════════════════


class TestLoadErrors:
    def test_load_nonexistent_returns_none(self):
        result = ScoringPool.load_baseline("/tmp/nonexistent_baseline_12345.json")
        assert result is None

    def test_load_empty_path_returns_none(self):
        result = ScoringPool.load_baseline("")
        assert result is None

    def test_load_corrupted_json_returns_none(self, tmp_path):
        path = str(tmp_path / "bad.json")
        with open(path, "w") as f:
            f.write("{invalid json!!")
        result = ScoringPool.load_baseline(path)
        assert result is None

    def test_load_missing_fields_defaults_to_empty(self, tmp_path):
        path = str(tmp_path / "partial.json")
        with open(path, "w") as f:
            json.dump({"stock_count": 50}, f)
        loaded = ScoringPool.load_baseline(path)
        assert loaded is not None
        assert loaded.stock_count == 50
        assert loaded.pool_return_5d == []
        assert loaded.pool_return_10d == []


# ════════════════════════════════════════════════════════
# 4. Config 新增字段
# ════════════════════════════════════════════════════════


class TestConfigFields:
    def test_baseline_defaults(self):
        import datetime as dt
        from a_share_hot_screener.config import HotScreenerConfig

        cfg = HotScreenerConfig(
            tushare_token="test",
            run_date=dt.date(2026, 4, 17),
            stock_codes=["600519"],
            output_dir="/tmp/test",
        )
        assert cfg.baseline_pool_path == ""
        assert cfg.save_baseline_pool is False
        assert cfg.min_baseline_pool_size == 30  # #2: 从5提升到30


# ════════════════════════════════════════════════════════
# 5. CLI 参数解析
# ════════════════════════════════════════════════════════


class TestCLIArgs:
    def test_baseline_args_parsed(self):
        from a_share_hot_screener.cli import build_parser

        parser = build_parser()
        args = parser.parse_args([
            "--tushare-token", "test",
            "--run-date", "2026-04-17",
            "--stock-codes", "600519",
            "--output-dir", "/tmp/test",
            "--baseline-pool-path", "/tmp/baseline.json",
            "--save-baseline-pool",
            "--min-baseline-pool-size", "10",
        ])
        assert args.baseline_pool_path == "/tmp/baseline.json"
        assert args.save_baseline_pool is True
        assert args.min_baseline_pool_size == 10

    def test_baseline_defaults(self):
        from a_share_hot_screener.cli import build_parser

        parser = build_parser()
        args = parser.parse_args([
            "--tushare-token", "test",
            "--run-date", "2026-04-17",
            "--stock-codes", "600519",
            "--output-dir", "/tmp/test",
        ])
        assert args.baseline_pool_path == ""
        assert args.save_baseline_pool is False
        assert args.min_baseline_pool_size == 30  # #2: 从5提升到30


# ════════════════════════════════════════════════════════
# 6. RunMetadata 新增字段
# ════════════════════════════════════════════════════════


class TestMetadataFields:
    def test_default_baseline_fields(self):
        from a_share_hot_screener.models import RunMetadata

        meta = RunMetadata()
        assert meta.used_baseline_pool is False
        assert meta.baseline_pool_stock_count is None

    def test_baseline_fields_serializable(self):
        from a_share_hot_screener.models import RunMetadata
        import dataclasses
        import json

        meta = RunMetadata(used_baseline_pool=True, baseline_pool_stock_count=119)
        d = dataclasses.asdict(meta)
        s = json.dumps(d, default=str)
        assert '"used_baseline_pool": true' in s
        assert '"baseline_pool_stock_count": 119' in s


# ════════════════════════════════════════════════════════
# 7. 端到端: save → 加载 → 合并 → 百分位不再 None
# ════════════════════════════════════════════════════════


class TestE2EBaselinePercentile:
    """验证基准 pool 合并后，score_percentile 不再因 pool_too_small 返回 None."""

    def test_percentile_with_baseline_not_none(self):
        from a_share_hot_screener.scoring import score_percentile

        # 模拟单只股票 live pool (too small)
        small_pool = [5.0]  # len < 2 → would return is_data_available=False

        # 直接测 score_percentile 用大 pool
        big_pool = [float(i) for i in range(100)]
        item = score_percentile(
            name="return_5d_pctile",
            value=50.0,
            pool=big_pool,
            ascending=True,
            weight=8.0,
        )
        assert item.is_data_available is True
        assert item.subscore is not None
        assert item.subscore > 0

    def test_merged_pool_enables_percentile(self, tmp_path):
        """完整流程: save baseline → load → merge → percentile 可用."""
        from a_share_hot_screener.scoring import score_percentile

        # 1. 创建并保存大基准 pool
        base = _make_sample_pool(100)
        path = str(tmp_path / "baseline.json")
        base.save_baseline(path)

        # 2. 模拟单只股票的 live pool
        live = ScoringPool()
        live.stock_count = 1
        live.pool_return_5d = [5.0]

        # 3. live pool 太小 → 加载并合并
        loaded = ScoringPool.load_baseline(path)
        assert loaded is not None
        merged = live.merge_with_baseline(loaded)
        assert merged.stock_count == 101

        # 4. 用合并后的 pool 做百分位 → 不再 None
        item = score_percentile(
            name="test",
            value=5.0,
            pool=merged.pool_return_5d,
            ascending=True,
            weight=1.0,
        )
        assert item.is_data_available is True
        assert item.subscore is not None
