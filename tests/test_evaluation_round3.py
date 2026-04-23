"""Round 3 评估框架扩展测试 — timing 相关标签 + 分组维度.

覆盖:
  - StockLabel timing 标签 (touched_support/broke_invalidation/touched_resistance)
  - LabelGenerator._fill_timing_labels()
  - ScoredStock timing/context 字段
  - EvaluationHarness timing 分组 (by_action/by_regime/by_confidence)
  - timing_score 十分位单调性
  - GroupMetrics timing 专项命中率
  - EvaluationReport timing 文本/CSV 输出
"""

import csv
import os
import tempfile
from dataclasses import asdict

import numpy as np
import pytest

from a_share_hot_screener.evaluation.label_generator import (
    LabelGenerator,
    StockLabel,
)
from a_share_hot_screener.evaluation.harness import (
    EvaluationHarness,
    EvaluationResult,
    GroupMetrics,
    ScoredStock,
)
from a_share_hot_screener.evaluation.report import EvaluationReport


# ════════════════════════════════════════════════════════
# StockLabel timing 标签
# ════════════════════════════════════════════════════════

class TestStockLabelTimingFields:
    """StockLabel 新增 timing 标签字段."""

    def test_default_none(self):
        label = StockLabel(code="600519", run_date="20260422")
        assert label.touched_support_zone_5d is None
        assert label.broke_invalidation_5d is None
        assert label.touched_resistance_5d is None

    def test_set_values(self):
        label = StockLabel(
            code="600519",
            run_date="20260422",
            touched_support_zone_5d=True,
            broke_invalidation_5d=False,
            touched_resistance_5d=True,
        )
        assert label.touched_support_zone_5d is True
        assert label.broke_invalidation_5d is False
        assert label.touched_resistance_5d is True

    def test_asdict_includes_timing(self):
        label = StockLabel(code="600519", run_date="20260422")
        d = asdict(label)
        assert "touched_support_zone_5d" in d
        assert "broke_invalidation_5d" in d
        assert "touched_resistance_5d" in d


# ════════════════════════════════════════════════════════
# _fill_timing_labels
# ════════════════════════════════════════════════════════

class TestFillTimingLabels:
    """LabelGenerator._fill_timing_labels 静态方法."""

    def test_touched_support_zone(self):
        label = StockLabel(code="600519", run_date="20260422", future_days_available=5)
        lows = np.array([45.0, 44.5, 44.0, 45.5, 46.0])
        highs = np.array([46.0, 45.5, 45.0, 47.0, 48.0])
        timing = {"support_zone_high": 44.5, "invalidation_level": 43.0, "resistance_1": 50.0}
        LabelGenerator._fill_timing_labels(label, lows, highs, timing)
        assert label.touched_support_zone_5d is True  # min_low=44.0 <= 44.5
        assert label.broke_invalidation_5d is False    # 44.0 >= 43.0
        assert label.touched_resistance_5d is False    # max_high=48.0 < 50.0

    def test_broke_invalidation(self):
        label = StockLabel(code="600519", run_date="20260422", future_days_available=5)
        lows = np.array([45.0, 44.5, 42.0, 43.0, 44.0])
        highs = np.array([46.0, 45.5, 43.0, 44.0, 45.0])
        timing = {"support_zone_high": 44.0, "invalidation_level": 43.0, "resistance_1": 50.0}
        LabelGenerator._fill_timing_labels(label, lows, highs, timing)
        assert label.broke_invalidation_5d is True  # min_low=42.0 < 43.0

    def test_touched_resistance(self):
        label = StockLabel(code="600519", run_date="20260422", future_days_available=5)
        lows = np.array([48.0, 49.0, 50.0, 51.0, 52.0])
        highs = np.array([49.0, 50.5, 51.0, 52.0, 53.0])
        timing = {"support_zone_high": 44.0, "invalidation_level": 43.0, "resistance_1": 50.0}
        LabelGenerator._fill_timing_labels(label, lows, highs, timing)
        assert label.touched_resistance_5d is True  # max_high=53.0 >= 50.0

    def test_no_touch_any(self):
        label = StockLabel(code="600519", run_date="20260422", future_days_available=5)
        lows = np.array([46.0, 46.5, 46.2, 46.8, 47.0])
        highs = np.array([47.0, 47.5, 47.2, 47.8, 48.0])
        timing = {"support_zone_high": 44.0, "invalidation_level": 43.0, "resistance_1": 50.0}
        LabelGenerator._fill_timing_labels(label, lows, highs, timing)
        assert label.touched_support_zone_5d is False
        assert label.broke_invalidation_5d is False
        assert label.touched_resistance_5d is False

    def test_partial_timing_signal(self):
        """timing_signal 缺少某些价位 → 对应标签 None."""
        label = StockLabel(code="600519", run_date="20260422", future_days_available=5)
        lows = np.array([46.0, 46.5, 46.2, 46.8, 47.0])
        highs = np.array([47.0, 47.5, 47.2, 47.8, 48.0])
        timing = {"support_zone_high": 47.0}  # 只有 support
        LabelGenerator._fill_timing_labels(label, lows, highs, timing)
        assert label.touched_support_zone_5d is True  # 46.0 <= 47.0
        assert label.broke_invalidation_5d is None    # 没有 invalidation_level
        assert label.touched_resistance_5d is None    # 没有 resistance_1

    def test_empty_lows_highs(self):
        """空数组不填充."""
        label = StockLabel(code="600519", run_date="20260422", future_days_available=0)
        lows = np.array([])
        highs = np.array([])
        timing = {"support_zone_high": 44.0, "invalidation_level": 43.0, "resistance_1": 50.0}
        LabelGenerator._fill_timing_labels(label, lows, highs, timing)
        assert label.touched_support_zone_5d is None
        assert label.broke_invalidation_5d is None
        assert label.touched_resistance_5d is None

    def test_fewer_than_5_days(self):
        """不足5天仍正常计算."""
        label = StockLabel(code="600519", run_date="20260422", future_days_available=3)
        lows = np.array([45.0, 44.0, 46.0])
        highs = np.array([46.0, 45.0, 47.0])
        timing = {"support_zone_high": 44.5, "invalidation_level": 43.0, "resistance_1": 48.0}
        LabelGenerator._fill_timing_labels(label, lows, highs, timing)
        assert label.touched_support_zone_5d is True  # min(45,44,46) = 44 <= 44.5
        assert label.broke_invalidation_5d is False
        assert label.touched_resistance_5d is False  # max(46,45,47)=47 < 48


# ════════════════════════════════════════════════════════
# ScoredStock timing/context 字段
# ════════════════════════════════════════════════════════

class TestScoredStockTimingFields:
    """ScoredStock 新增 timing/context 字段."""

    def test_default_values(self):
        s = ScoredStock(code="600519")
        assert s.timing_score is None
        assert s.timing_action == ""
        assert s.timing_level_confidence == ""
        assert s.timing_risk_score is None
        assert s.timing_market_regime == ""
        assert s.ht8_score is None
        assert s.ht9_score is None
        assert s.ht10_score is None
        assert s.touched_support_zone_5d is None
        assert s.broke_invalidation_5d is None
        assert s.touched_resistance_5d is None

    def test_set_timing(self):
        s = ScoredStock(
            code="600519",
            timing_score=78.5,
            timing_action="watch",
            timing_level_confidence="high",
            timing_risk_score=0.65,
            timing_market_regime="bull",
        )
        assert s.timing_score == 78.5
        assert s.timing_action == "watch"


# ════════════════════════════════════════════════════════
# EvaluationHarness: timing 分组
# ════════════════════════════════════════════════════════

def _make_scored_stocks_with_timing(n=50):
    """生成带 timing 数据的测试数据."""
    import random
    random.seed(42)

    actions = ["setup_ready", "watch", "wait", "avoid_chase"]
    regimes = ["bull", "neutral", "bear"]
    confs = ["high", "medium", "low"]

    stocks = []
    for i in range(n):
        s = ScoredStock(
            code=f"{600000 + i}",
            name=f"Stock{i}",
            total_score=random.uniform(0.5, 0.95),
            pass_stage1=i < 30,
            passed_hard_filter=True,
            candidate_pool_type="tradeable" if i < 20 else ("watch_only" if i < 30 else "failed_score"),
            hot_theme_score=random.uniform(0.5, 0.9),
            trend_flow_score=random.uniform(0.4, 0.8),
            liquidity_execution_score=random.uniform(0.3, 0.7),
            risk_control_score=random.uniform(0.3, 0.8),
            # timing
            timing_score=random.uniform(30, 95),
            timing_action=random.choice(actions),
            timing_level_confidence=random.choice(confs),
            timing_risk_score=random.uniform(0.1, 0.9),
            timing_market_regime=random.choice(regimes),
            # context
            ht8_score=random.uniform(0.3, 1.0),
            ht9_score=random.uniform(0.2, 0.9),
            ht10_score=random.uniform(0.2, 0.9),
            # labels
            return_t1=random.uniform(-3, 5),
            mfe_3d=random.uniform(0, 10),
            mfe_5d=random.uniform(0, 15),
            mae_5d=random.uniform(-8, 0),
            hit_limit_up_3d=random.random() > 0.8,
            beat_index_5d=random.random() > 0.5,
            touched_support_zone_5d=random.random() > 0.6,
            broke_invalidation_5d=random.random() > 0.85,
            touched_resistance_5d=random.random() > 0.5,
            label_quality="full",
        )
        stocks.append(s)
    return stocks


class TestTimingGroupEvaluation:
    """EvaluationHarness timing 分组评估."""

    def test_timing_by_action_groups(self):
        stocks = _make_scored_stocks_with_timing(50)
        harness = EvaluationHarness()
        result = harness.evaluate(stocks, run_date="20260422")

        assert len(result.timing_by_action) > 0
        # 每个 action 都应有对应分组
        for action in ["setup_ready", "watch", "wait", "avoid_chase"]:
            if action in result.timing_by_action:
                gm = result.timing_by_action[action]
                assert gm.count > 0
                assert gm.avg_mfe_5d is not None

    def test_timing_by_regime_groups(self):
        stocks = _make_scored_stocks_with_timing(50)
        harness = EvaluationHarness()
        result = harness.evaluate(stocks, run_date="20260422")

        assert len(result.timing_by_regime) > 0
        for regime in ["bull", "neutral", "bear"]:
            if regime in result.timing_by_regime:
                gm = result.timing_by_regime[regime]
                assert gm.count > 0

    def test_timing_by_confidence_groups(self):
        stocks = _make_scored_stocks_with_timing(50)
        harness = EvaluationHarness()
        result = harness.evaluate(stocks, run_date="20260422")

        assert len(result.timing_by_confidence) > 0
        for conf in ["high", "medium", "low"]:
            if conf in result.timing_by_confidence:
                gm = result.timing_by_confidence[conf]
                assert gm.count > 0

    def test_timing_monotonicity(self):
        stocks = _make_scored_stocks_with_timing(50)
        harness = EvaluationHarness()
        result = harness.evaluate(stocks, run_date="20260422")

        mono = result.timing_monotonicity
        assert mono is not None
        assert len(mono.decile_groups) > 0
        # 每个 decile 有 score_min/max
        for g in mono.decile_groups:
            assert "decile" in g
            assert "score_min" in g
            assert "avg_mfe_5d" in g

    def test_timing_monotonicity_too_few_stocks(self):
        """不足10只时 decile_groups 为空."""
        stocks = _make_scored_stocks_with_timing(5)
        harness = EvaluationHarness()
        result = harness.evaluate(stocks, run_date="20260422")

        mono = result.timing_monotonicity
        assert mono is not None
        assert len(mono.decile_groups) == 0

    def test_no_timing_data(self):
        """stocks 没有 timing → timing 分组为空."""
        stocks = [
            ScoredStock(
                code="600519",
                total_score=0.8,
                passed_hard_filter=True,
                candidate_pool_type="tradeable",
                mfe_5d=5.0,
                return_t1=1.0,
                label_quality="full",
            )
        ]
        harness = EvaluationHarness()
        result = harness.evaluate(stocks, run_date="20260422")
        assert len(result.timing_by_action) == 0
        assert len(result.timing_by_regime) == 0
        assert len(result.timing_by_confidence) == 0


# ════════════════════════════════════════════════════════
# GroupMetrics: timing 专项命中率
# ════════════════════════════════════════════════════════

class TestGroupMetricsTimingHitRates:
    """GroupMetrics 新增 timing 命中率字段."""

    def test_timing_hit_rates_computed(self):
        stocks = _make_scored_stocks_with_timing(50)
        harness = EvaluationHarness()
        result = harness.evaluate(stocks, run_date="20260422")

        # all_scored 分组应有 timing 命中率
        gm = result.group_metrics.get("all_scored")
        assert gm is not None
        assert gm.hit_rate_touched_support is not None
        assert gm.hit_rate_broke_invalidation is not None
        assert gm.hit_rate_touched_resistance is not None
        # 值在 0~1 范围
        assert 0 <= gm.hit_rate_touched_support <= 1
        assert 0 <= gm.hit_rate_broke_invalidation <= 1
        assert 0 <= gm.hit_rate_touched_resistance <= 1

    def test_timing_hit_rates_none_when_no_data(self):
        stocks = [
            ScoredStock(
                code="600519",
                total_score=0.8,
                passed_hard_filter=True,
                candidate_pool_type="tradeable",
                mfe_5d=5.0,
                label_quality="full",
            )
        ]
        harness = EvaluationHarness()
        result = harness.evaluate(stocks, run_date="20260422")
        gm = result.group_metrics.get("all_scored")
        assert gm is not None
        assert gm.hit_rate_touched_support is None
        assert gm.hit_rate_broke_invalidation is None
        assert gm.hit_rate_touched_resistance is None


# ════════════════════════════════════════════════════════
# EvaluationReport: timing 输出
# ════════════════════════════════════════════════════════

class TestEvaluationReportTiming:
    """EvaluationReport timing 文本/CSV 输出."""

    def _make_result(self):
        stocks = _make_scored_stocks_with_timing(50)
        harness = EvaluationHarness()
        return harness.evaluate(stocks, run_date="20260422")

    def test_text_report_includes_timing(self):
        result = self._make_result()
        report = EvaluationReport(result)
        text = report.to_text()

        assert "Setup Timing 分组" in text
        assert "Action" in text
        assert "Market Regime" in text
        assert "Level Confidence" in text
        assert "timing_score 十分位" in text

    def test_text_report_timing_action_labels(self):
        result = self._make_result()
        report = EvaluationReport(result)
        text = report.to_text()

        # 至少一个 action 出现在报告中
        found_action = False
        for action in ["setup_ready", "watch", "wait", "avoid_chase"]:
            if f"[{action}]" in text:
                found_action = True
                break
        assert found_action

    def test_text_report_timing_hit_rates(self):
        result = self._make_result()
        report = EvaluationReport(result)
        text = report.to_text()

        # timing 专项应在某处出现
        assert "触支撑" in text or "破失效" in text or "触压力" in text

    def test_group_csv_includes_timing_groups(self):
        result = self._make_result()
        report = EvaluationReport(result)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "groups.csv")
            report.save_group_csv(path)

            with open(path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                rows = list(reader)

            group_names = [r["group_name"] for r in rows]
            # 应包含 timing_ 前缀的分组
            timing_groups = [g for g in group_names if g.startswith("timing_")]
            assert len(timing_groups) > 0

            # 应有 timing 命中率列
            assert "hit_rate_touched_support" in reader.fieldnames
            assert "hit_rate_broke_invalidation" in reader.fieldnames
            assert "hit_rate_touched_resistance" in reader.fieldnames

    def test_timing_decile_csv(self):
        result = self._make_result()
        report = EvaluationReport(result)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "timing_deciles.csv")
            report.save_timing_decile_csv(path)

            with open(path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                rows = list(reader)

            assert len(rows) > 0
            assert "decile" in reader.fieldnames
            assert "avg_mfe_5d" in reader.fieldnames

    def test_save_all_includes_timing_decile(self):
        result = self._make_result()
        report = EvaluationReport(result)

        with tempfile.TemporaryDirectory() as tmpdir:
            paths = report.save_all(tmpdir, prefix="test")
            assert "timing_deciles_csv" in paths
            assert os.path.exists(paths["timing_deciles_csv"])

    def test_timing_decile_csv_empty_when_no_timing(self):
        """没有 timing 数据时 CSV 仍有表头."""
        result = EvaluationResult(run_date="20260422")
        report = EvaluationReport(result)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "timing_deciles.csv")
            report.save_timing_decile_csv(path)

            with open(path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                rows = list(reader)

            assert len(rows) == 0  # 空内容
            assert "decile" in reader.fieldnames  # 有表头


# ════════════════════════════════════════════════════════
# LabelGenerator CSV 序列化/反序列化
# ════════════════════════════════════════════════════════

class TestLabelCSVRoundTrip:
    """timing 标签的 CSV 序列化/反序列化."""

    def test_save_load_timing_labels(self):
        from a_share_hot_screener.evaluation.label_generator import LabelResult

        label = StockLabel(
            code="600519",
            run_date="20260422",
            return_t1=2.5,
            mfe_3d=5.0,
            mfe_5d=8.0,
            mae_5d=-3.0,
            hit_limit_up_3d=False,
            beat_index_5d=True,
            touched_support_zone_5d=True,
            broke_invalidation_5d=False,
            touched_resistance_5d=True,
            close_on_run_date=50.0,
            future_days_available=5,
            label_quality="full",
        )
        result = LabelResult(
            run_date="20260422",
            labels=[label],
            total_requested=1,
            total_labeled=1,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "labels.csv")
            LabelGenerator.save_csv(result, path)
            loaded = LabelGenerator.load_csv(path)

            assert len(loaded.labels) == 1
            ll = loaded.labels[0]
            assert ll.touched_support_zone_5d is True
            assert ll.broke_invalidation_5d is False
            assert ll.touched_resistance_5d is True

    def test_load_missing_timing_columns(self):
        """旧版 CSV 没有 timing 列 → None."""
        from a_share_hot_screener.evaluation.label_generator import LabelResult

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "labels.csv")
            # 写一个不含 timing 列的 CSV
            with open(path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=[
                    "code", "run_date", "return_t1", "mfe_3d", "mfe_5d",
                    "mae_5d", "hit_limit_up_3d", "beat_index_5d",
                    "close_on_run_date", "future_days_available", "label_quality",
                ])
                writer.writeheader()
                writer.writerow({
                    "code": "600519", "run_date": "20260422",
                    "return_t1": "2.5000", "mfe_3d": "5.0000", "mfe_5d": "8.0000",
                    "mae_5d": "-3.0000", "hit_limit_up_3d": "False",
                    "beat_index_5d": "True", "close_on_run_date": "50.0000",
                    "future_days_available": "5", "label_quality": "full",
                })

            loaded = LabelGenerator.load_csv(path)
            ll = loaded.labels[0]
            # 旧版 CSV 没有这些列 → None
            assert ll.touched_support_zone_5d is None
            assert ll.broke_invalidation_5d is None
            assert ll.touched_resistance_5d is None


# ════════════════════════════════════════════════════════
# load_merged_data 读取 timing/context 字段
# ════════════════════════════════════════════════════════

class TestLoadMergedDataTimingFields:
    """load_merged_data 正确读取 timing/context 字段."""

    def test_loads_timing_from_summary(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            summary_path = os.path.join(tmpdir, "summary.csv")
            label_path = os.path.join(tmpdir, "labels.csv")

            # 写 summary CSV
            with open(summary_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=[
                    "code", "name", "total_score", "candidate_pool_type",
                    "pass_stage1", "passed_hard_filter",
                    "hot_theme_score", "trend_flow_score",
                    "liquidity_execution_score", "risk_control_score",
                    "timing_score", "timing_action", "timing_level_confidence",
                    "ht8_score", "ht9_score", "ht10_score", "ht10_position_type",
                ])
                writer.writeheader()
                writer.writerow({
                    "code": "600519", "name": "茅台", "total_score": "0.82",
                    "candidate_pool_type": "tradeable", "pass_stage1": "True",
                    "passed_hard_filter": "True",
                    "hot_theme_score": "0.78", "trend_flow_score": "0.72",
                    "liquidity_execution_score": "0.65", "risk_control_score": "0.58",
                    "timing_score": "75.5", "timing_action": "watch",
                    "timing_level_confidence": "high",
                    "ht8_score": "0.80", "ht9_score": "0.60", "ht10_score": "0.70",
                    "ht10_position_type": "frontline_like",
                })

            # 写 label CSV
            with open(label_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=[
                    "code", "run_date", "return_t1", "mfe_3d", "mfe_5d",
                    "mae_5d", "hit_limit_up_3d", "beat_index_5d",
                    "touched_support_zone_5d", "broke_invalidation_5d",
                    "touched_resistance_5d",
                    "close_on_run_date", "future_days_available", "label_quality",
                ])
                writer.writeheader()
                writer.writerow({
                    "code": "600519", "run_date": "20260422",
                    "return_t1": "2.5", "mfe_3d": "5.0", "mfe_5d": "8.0",
                    "mae_5d": "-3.0", "hit_limit_up_3d": "False",
                    "beat_index_5d": "True",
                    "touched_support_zone_5d": "True",
                    "broke_invalidation_5d": "False",
                    "touched_resistance_5d": "True",
                    "close_on_run_date": "50.0", "future_days_available": "5",
                    "label_quality": "full",
                })

            harness = EvaluationHarness()
            stocks = harness.load_merged_data(summary_path, label_path)

            assert len(stocks) == 1
            s = stocks[0]
            # timing
            assert s.timing_score == 75.5
            assert s.timing_action == "watch"
            assert s.timing_level_confidence == "high"
            # context
            assert s.ht8_score == 0.80
            assert s.ht9_score == 0.60
            assert s.ht10_score == 0.70
            assert s.ht10_position_type == "frontline_like"
            # timing labels
            assert s.touched_support_zone_5d is True
            assert s.broke_invalidation_5d is False
            assert s.touched_resistance_5d is True
