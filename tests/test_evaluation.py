"""Phase 1 评估框架测试.

覆盖:
  - LabelGenerator: 标签生成逻辑 (mock Tushare)
  - EvaluationHarness: 分组 + 指标计算 + 单调性 + 分离度 + 消融
  - EvaluationReport: 文本 + CSV 输出
  - StockLabel / LabelResult: 数据结构
  - CSV 序列化 / 反序列化
"""

import csv
import os
import tempfile
from dataclasses import asdict
from typing import List
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from a_share_hot_screener.evaluation.label_generator import (
    LabelGenerator,
    LabelResult,
    StockLabel,
)
from a_share_hot_screener.evaluation.harness import (
    EvaluationHarness,
    EvaluationResult,
    GroupMetrics,
    ScoredStock,
    _mean,
    _median,
    _safe_float,
    _spearman_corr,
)
from a_share_hot_screener.evaluation.report import EvaluationReport


# ════════════════════════════════════════════════════════
# Fixtures
# ════════════════════════════════════════════════════════

def _make_mock_client():
    """创建 mock TushareClient."""
    client = MagicMock()
    return client


def _make_mock_calendar():
    """创建 mock TradeCalendar."""
    cal = MagicMock()
    # is_trade_date: 周一~周五为交易日
    import datetime as dt
    def is_trade_date(d):
        if isinstance(d, str):
            d = dt.datetime.strptime(d.replace("-", ""), "%Y%m%d").date()
        return d.weekday() < 5

    cal.is_trade_date = is_trade_date
    return cal


def _make_daily_df(base_close: float, days: int, ts_code: str, start_date: str):
    """生成模拟日线数据."""
    import datetime as dt
    start = dt.datetime.strptime(start_date, "%Y%m%d").date()
    rows = []
    close = base_close
    for i in range(days):
        d = start + dt.timedelta(days=i)
        # 跳过周末
        while d.weekday() >= 5:
            d = d + dt.timedelta(days=1)
        open_ = close * 0.99
        high = close * 1.02
        low = close * 0.98
        pct_chg = (close - base_close) / base_close * 100 if i > 0 else 0
        rows.append({
            "ts_code": ts_code,
            "trade_date": d.strftime("%Y%m%d"),
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "pre_close": close * 0.99,
            "change": close * 0.01,
            "pct_chg": pct_chg,
            "vol": 100000,
            "amount": 500000,
        })
        close = close * 1.01  # 每天涨 1%
    return pd.DataFrame(rows)


# ════════════════════════════════════════════════════════
# LabelGenerator Tests
# ════════════════════════════════════════════════════════

class TestStockLabel:
    """StockLabel 数据结构测试."""

    def test_default_values(self):
        label = StockLabel(code="600519", run_date="20260418")
        assert label.code == "600519"
        assert label.return_t1 is None
        assert label.label_quality == "full"

    def test_to_dict(self):
        label = StockLabel(code="600519", run_date="20260418", return_t1=2.5)
        d = asdict(label)
        assert d["code"] == "600519"
        assert d["return_t1"] == 2.5


class TestLabelGenerator:
    """LabelGenerator 标签生成测试."""

    def test_generate_single_stock_full(self):
        """完整标签生成 (5 个未来交易日)."""
        client = _make_mock_client()
        cal = _make_mock_calendar()

        # 模拟 run_date=20260420 (周一) 的行情
        # 返回 run_date + 5 个未来交易日
        df = pd.DataFrame([
            {"ts_code": "600519.SH", "trade_date": "20260420", "open": 100, "high": 102, "low": 98, "close": 100, "pre_close": 99, "change": 1, "pct_chg": 1.0, "vol": 10000, "amount": 50000},
            {"ts_code": "600519.SH", "trade_date": "20260421", "open": 100, "high": 105, "low": 99, "close": 103, "pre_close": 100, "change": 3, "pct_chg": 3.0, "vol": 12000, "amount": 60000},
            {"ts_code": "600519.SH", "trade_date": "20260422", "open": 103, "high": 108, "low": 101, "close": 106, "pre_close": 103, "change": 3, "pct_chg": 2.9, "vol": 15000, "amount": 70000},
            {"ts_code": "600519.SH", "trade_date": "20260423", "open": 106, "high": 107, "low": 102, "close": 104, "pre_close": 106, "change": -2, "pct_chg": -1.9, "vol": 11000, "amount": 55000},
            {"ts_code": "600519.SH", "trade_date": "20260424", "open": 104, "high": 110, "low": 103, "close": 109, "pre_close": 104, "change": 5, "pct_chg": 4.8, "vol": 18000, "amount": 90000},
            {"ts_code": "600519.SH", "trade_date": "20260427", "open": 109, "high": 112, "low": 107, "close": 111, "pre_close": 109, "change": 2, "pct_chg": 1.8, "vol": 14000, "amount": 70000},
        ])
        client.get_daily.return_value = df

        gen = LabelGenerator(client, cal)
        result = gen.generate(
            run_date="20260420",
            stock_codes=["600519"],
            ts_code_map={"600519": "600519.SH"},
        )

        assert result.total_requested == 1
        assert result.total_labeled == 1
        assert len(result.labels) == 1

        label = result.labels[0]
        assert label.code == "600519"
        assert label.close_on_run_date == 100
        assert label.future_days_available == 5
        assert label.label_quality == "full"

        # T+1: (103/100 - 1) * 100 = 3.0%
        assert label.return_t1 == pytest.approx(3.0, abs=0.01)

        # MFE 3d: max(105, 108, 107) / 100 - 1 = 8%
        assert label.mfe_3d == pytest.approx(8.0, abs=0.01)

        # MFE 5d: max(105, 108, 107, 110, 112) / 100 - 1 = 12%
        assert label.mfe_5d == pytest.approx(12.0, abs=0.01)

        # MAE 5d: min(99, 101, 102, 103, 107) / 100 - 1 = -1%
        assert label.mae_5d == pytest.approx(-1.0, abs=0.01)

    def test_generate_no_future_data(self):
        """无未来数据."""
        client = _make_mock_client()
        cal = _make_mock_calendar()

        # run_date 行有但无未来行情
        df = pd.DataFrame([
            {"ts_code": "600519.SH", "trade_date": "20260420", "open": 100, "high": 102, "low": 98, "close": 100, "pre_close": 99, "change": 1, "pct_chg": 1.0, "vol": 10000, "amount": 50000},
        ])
        client.get_daily.return_value = df

        gen = LabelGenerator(client, cal)
        result = gen.generate(
            run_date="20260420",
            stock_codes=["600519"],
            ts_code_map={"600519": "600519.SH"},
        )

        label = result.labels[0]
        assert label.label_quality == "missing"

    def test_generate_api_returns_none(self):
        """API 返回 None."""
        client = _make_mock_client()
        cal = _make_mock_calendar()
        client.get_daily.return_value = None

        gen = LabelGenerator(client, cal)
        result = gen.generate(
            run_date="20260420",
            stock_codes=["600519"],
            ts_code_map={"600519": "600519.SH"},
        )

        label = result.labels[0]
        assert label.label_quality == "missing"

    def test_generate_partial_data(self):
        """部分数据 (仅 2 个未来交易日)."""
        client = _make_mock_client()
        cal = _make_mock_calendar()

        df = pd.DataFrame([
            {"ts_code": "600519.SH", "trade_date": "20260420", "open": 100, "high": 102, "low": 98, "close": 100, "pre_close": 99, "change": 1, "pct_chg": 1.0, "vol": 10000, "amount": 50000},
            {"ts_code": "600519.SH", "trade_date": "20260421", "open": 100, "high": 105, "low": 99, "close": 103, "pre_close": 100, "change": 3, "pct_chg": 3.0, "vol": 12000, "amount": 60000},
            {"ts_code": "600519.SH", "trade_date": "20260422", "open": 103, "high": 108, "low": 101, "close": 106, "pre_close": 103, "change": 3, "pct_chg": 2.9, "vol": 15000, "amount": 70000},
        ])
        client.get_daily.return_value = df

        gen = LabelGenerator(client, cal)
        result = gen.generate(
            run_date="20260420",
            stock_codes=["600519"],
            ts_code_map={"600519": "600519.SH"},
        )

        label = result.labels[0]
        assert label.future_days_available == 2
        assert label.label_quality == "partial"
        assert label.return_t1 is not None
        assert label.mfe_3d is not None  # 用 2 天数据计算

    def test_generate_multiple_stocks(self):
        """多只股票."""
        client = _make_mock_client()
        cal = _make_mock_calendar()

        def mock_get_daily(ts_code, start_date, end_date, **kwargs):
            close = 100 if ts_code == "600519.SH" else 50
            return pd.DataFrame([
                {"ts_code": ts_code, "trade_date": "20260420", "open": close, "high": close*1.02, "low": close*0.98, "close": close, "pre_close": close*0.99, "change": 1, "pct_chg": 1.0, "vol": 10000, "amount": 50000},
                {"ts_code": ts_code, "trade_date": "20260421", "open": close, "high": close*1.05, "low": close*0.99, "close": close*1.03, "pre_close": close, "change": 3, "pct_chg": 3.0, "vol": 12000, "amount": 60000},
                {"ts_code": ts_code, "trade_date": "20260422", "open": close*1.03, "high": close*1.08, "low": close*1.01, "close": close*1.06, "pre_close": close*1.03, "change": 3, "pct_chg": 2.9, "vol": 15000, "amount": 70000},
                {"ts_code": ts_code, "trade_date": "20260423", "open": close*1.06, "high": close*1.07, "low": close*1.02, "close": close*1.04, "pre_close": close*1.06, "change": -2, "pct_chg": -1.9, "vol": 11000, "amount": 55000},
                {"ts_code": ts_code, "trade_date": "20260424", "open": close*1.04, "high": close*1.10, "low": close*1.03, "close": close*1.09, "pre_close": close*1.04, "change": 5, "pct_chg": 4.8, "vol": 18000, "amount": 90000},
                {"ts_code": ts_code, "trade_date": "20260427", "open": close*1.09, "high": close*1.12, "low": close*1.07, "close": close*1.11, "pre_close": close*1.09, "change": 2, "pct_chg": 1.8, "vol": 14000, "amount": 70000},
            ])

        client.get_daily.side_effect = mock_get_daily

        gen = LabelGenerator(client, cal)
        result = gen.generate(
            run_date="20260420",
            stock_codes=["600519", "000858"],
            ts_code_map={"600519": "600519.SH", "000858": "000858.SZ"},
        )

        assert result.total_requested == 2
        assert result.total_labeled == 2
        assert len(result.labels) == 2

    def test_hit_limit_up_3d(self):
        """涨停检测."""
        client = _make_mock_client()
        cal = _make_mock_calendar()

        df = pd.DataFrame([
            {"ts_code": "002436.SZ", "trade_date": "20260420", "open": 10, "high": 10.5, "low": 9.8, "close": 10, "pre_close": 9.9, "change": 0.1, "pct_chg": 1.0, "vol": 10000, "amount": 50000},
            {"ts_code": "002436.SZ", "trade_date": "20260421", "open": 10, "high": 11.1, "low": 10, "close": 11.0, "pre_close": 10, "change": 1, "pct_chg": 10.0, "vol": 20000, "amount": 100000},
            {"ts_code": "002436.SZ", "trade_date": "20260422", "open": 11, "high": 12.1, "low": 10.8, "close": 12.0, "pre_close": 11, "change": 1, "pct_chg": 9.09, "vol": 25000, "amount": 130000},
            {"ts_code": "002436.SZ", "trade_date": "20260423", "open": 12, "high": 12.5, "low": 11.5, "close": 12.2, "pre_close": 12, "change": 0.2, "pct_chg": 1.67, "vol": 15000, "amount": 80000},
            {"ts_code": "002436.SZ", "trade_date": "20260424", "open": 12.2, "high": 13, "low": 12, "close": 12.8, "pre_close": 12.2, "change": 0.6, "pct_chg": 4.9, "vol": 18000, "amount": 90000},
            {"ts_code": "002436.SZ", "trade_date": "20260427", "open": 12.8, "high": 13.5, "low": 12.5, "close": 13.2, "pre_close": 12.8, "change": 0.4, "pct_chg": 3.1, "vol": 16000, "amount": 85000},
        ])
        client.get_daily.return_value = df

        gen = LabelGenerator(client, cal)
        result = gen.generate(
            run_date="20260420",
            stock_codes=["002436"],
            ts_code_map={"002436": "002436.SZ"},
        )

        label = result.labels[0]
        # T+1 pct_chg = 10.0% >= 9.8% → 涨停
        assert label.hit_limit_up_3d is True

    def test_csv_save_and_load(self):
        """CSV 保存和加载."""
        label = StockLabel(
            code="600519",
            run_date="20260420",
            return_t1=3.0,
            mfe_3d=8.0,
            mfe_5d=12.0,
            mae_5d=-1.0,
            hit_limit_up_3d=False,
            beat_index_5d=True,
            close_on_run_date=100.0,
            future_days_available=5,
            label_quality="full",
        )
        result = LabelResult(
            run_date="20260420",
            labels=[label],
            total_requested=1,
            total_labeled=1,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = os.path.join(tmpdir, "labels.csv")
            LabelGenerator.save_csv(result, csv_path)

            # 验证文件存在
            assert os.path.exists(csv_path)

            # 加载回来
            loaded = LabelGenerator.load_csv(csv_path)
            assert len(loaded.labels) == 1
            loaded_label = loaded.labels[0]
            assert loaded_label.code == "600519"
            assert loaded_label.return_t1 == pytest.approx(3.0, abs=0.01)
            assert loaded_label.mfe_5d == pytest.approx(12.0, abs=0.01)
            assert loaded_label.beat_index_5d is True
            assert loaded_label.label_quality == "full"


# ════════════════════════════════════════════════════════
# EvaluationHarness Tests
# ════════════════════════════════════════════════════════

def _make_scored_stocks(n: int = 50) -> List[ScoredStock]:
    """生成 n 只模拟股票用于测试."""
    import random
    random.seed(42)
    stocks = []
    for i in range(n):
        score = 0.3 + random.random() * 0.5
        mfe = score * 15 + random.gauss(0, 3)  # 分数越高 MFE 越大
        mae = -(random.random() * 5 + 1)

        cpt = "tradeable" if score > 0.68 else ("watch_only" if score > 0.65 else "failed_score")

        stocks.append(ScoredStock(
            code=f"{600000 + i}",
            name=f"Stock{i}",
            total_score=score,
            candidate_pool_type=cpt,
            pass_stage1=(cpt == "tradeable"),
            passed_hard_filter=True,
            hot_theme_score=score * 0.9,
            trend_flow_score=score * 0.85,
            liquidity_execution_score=score * 0.95,
            risk_control_score=score * 1.1,
            return_t1=mfe * 0.3 + random.gauss(0, 1),
            mfe_3d=mfe * 0.6,
            mfe_5d=mfe,
            mae_5d=mae,
            hit_limit_up_3d=(mfe > 10),
            beat_index_5d=(mfe > 5),
            label_quality="full",
        ))
    return stocks


class TestHarnessUtils:
    """工具函数测试."""

    def test_safe_float_normal(self):
        assert _safe_float("3.14") == pytest.approx(3.14)

    def test_safe_float_none(self):
        assert _safe_float(None) is None

    def test_safe_float_empty(self):
        assert _safe_float("") is None

    def test_safe_float_invalid(self):
        assert _safe_float("abc") is None

    def test_mean_normal(self):
        assert _mean([1.0, 2.0, 3.0]) == pytest.approx(2.0)

    def test_mean_empty(self):
        assert _mean([]) is None

    def test_mean_with_none(self):
        assert _mean([1.0, None, 3.0]) == pytest.approx(2.0)

    def test_median_odd(self):
        assert _median([1.0, 2.0, 3.0]) == pytest.approx(2.0)

    def test_median_even(self):
        assert _median([1.0, 2.0, 3.0, 4.0]) == pytest.approx(2.5)

    def test_median_empty(self):
        assert _median([]) is None

    def test_spearman_perfect(self):
        """完美正相关."""
        assert _spearman_corr([1, 2, 3, 4, 5], [10, 20, 30, 40, 50]) == pytest.approx(1.0)

    def test_spearman_inverse(self):
        """完美负相关."""
        assert _spearman_corr([1, 2, 3, 4, 5], [50, 40, 30, 20, 10]) == pytest.approx(-1.0)

    def test_spearman_too_few(self):
        """样本不足."""
        assert _spearman_corr([1, 2], [3, 4]) is None


class TestEvaluationHarness:
    """评估引擎测试."""

    def test_evaluate_basic(self):
        """基本评估流程."""
        stocks = _make_scored_stocks(50)
        harness = EvaluationHarness(top_n=5)
        result = harness.evaluate(stocks, run_date="20260418")

        assert result.total_stocks == 50
        assert "pass_tradeable" in result.group_metrics
        assert "failed_score" in result.group_metrics
        assert "top_N" in result.group_metrics
        assert "bottom_N" in result.group_metrics

    def test_group_counts(self):
        """分组数量正确."""
        stocks = _make_scored_stocks(50)
        harness = EvaluationHarness(top_n=5)
        result = harness.evaluate(stocks, run_date="20260418")

        # top_N 应该有 5 只
        assert result.group_metrics["top_N"].count == 5
        assert result.group_metrics["bottom_N"].count == 5

        # tradeable + watch_only + failed_score 应该覆盖大部分
        total_grouped = (
            result.group_metrics.get("pass_tradeable", GroupMetrics("")).count +
            result.group_metrics.get("pass_watch_only", GroupMetrics("")).count +
            result.group_metrics.get("failed_score", GroupMetrics("")).count
        )
        assert total_grouped > 0

    def test_group_metrics_computed(self):
        """指标计算正确."""
        stocks = _make_scored_stocks(50)
        harness = EvaluationHarness(top_n=5)
        result = harness.evaluate(stocks, run_date="20260418")

        top_n = result.group_metrics["top_N"]
        assert top_n.avg_mfe_5d is not None
        assert top_n.avg_mae_5d is not None
        assert top_n.hit_rate_mfe3d_gt3 is not None

    def test_monotonicity(self):
        """单调性检验."""
        stocks = _make_scored_stocks(100)
        harness = EvaluationHarness(top_n=10)
        result = harness.evaluate(stocks, run_date="20260418")

        assert result.monotonicity is not None
        assert len(result.monotonicity.decile_groups) > 0
        # 由于数据有噪声，单调性不一定成立，但 Spearman 应该可以计算
        assert result.monotonicity.spearman_corr is not None

    def test_separation(self):
        """分离度计算."""
        stocks = _make_scored_stocks(50)
        harness = EvaluationHarness(top_n=5)
        result = harness.evaluate(stocks, run_date="20260418")

        sep = result.separation
        assert sep is not None
        # tradeable 组的 MFE 应该高于 failed 组 (因为我们的 mock 数据正相关)
        if sep.mfe_5d_diff is not None:
            assert sep.mfe_5d_diff > 0

    def test_load_merged_data(self):
        """从 CSV 加载合并数据."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # 创建 summary CSV
            summary_path = os.path.join(tmpdir, "summary.csv")
            with open(summary_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["code", "name", "total_score", "candidate_pool_type",
                                "pass_stage1", "passed_hard_filter",
                                "hot_theme_score", "trend_flow_score",
                                "liquidity_execution_score", "risk_control_score"])
                writer.writerow(["600519", "贵州茅台", "0.8200", "tradeable",
                                "True", "True", "0.78", "0.73", "0.65", "0.91"])
                writer.writerow(["000858", "五粮液", "0.6500", "failed_score",
                                "False", "True", "0.60", "0.55", "0.50", "0.70"])

            # 创建 label CSV
            label_path = os.path.join(tmpdir, "labels.csv")
            with open(label_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["code", "run_date", "return_t1", "mfe_3d", "mfe_5d",
                                "mae_5d", "hit_limit_up_3d", "beat_index_5d",
                                "close_on_run_date", "future_days_available", "label_quality"])
                writer.writerow(["600519", "20260418", "3.0", "8.0", "12.0",
                                "-1.0", "False", "True", "100.0", "5", "full"])
                writer.writerow(["000858", "20260418", "1.0", "3.0", "5.0",
                                "-2.0", "False", "False", "50.0", "5", "full"])

            harness = EvaluationHarness()
            stocks = harness.load_merged_data(summary_path, label_path)

            assert len(stocks) == 2
            assert stocks[0].code == "600519" or stocks[1].code == "600519"

            maotai = [s for s in stocks if s.code == "600519"][0]
            assert maotai.total_score == pytest.approx(0.82, abs=0.01)
            assert maotai.candidate_pool_type == "tradeable"
            assert maotai.mfe_5d == pytest.approx(12.0, abs=0.01)

    def test_ablation_compare(self):
        """消融实验对比."""
        baseline = _make_scored_stocks(50)
        # 变体: 所有分数加 0.05
        variant = []
        for s in baseline:
            v = ScoredStock(
                code=s.code,
                name=s.name,
                total_score=(s.total_score or 0) + 0.05,
                candidate_pool_type=s.candidate_pool_type,
                pass_stage1=s.pass_stage1,
                passed_hard_filter=s.passed_hard_filter,
                return_t1=s.return_t1,
                mfe_3d=s.mfe_3d,
                mfe_5d=s.mfe_5d,
                mae_5d=s.mae_5d,
                label_quality=s.label_quality,
            )
            variant.append(v)

        harness = EvaluationHarness(top_n=5)
        comparison = harness.ablation_compare(
            baseline, variant,
            run_date="20260418",
            variant_name="score_boost",
        )

        assert comparison["variant_name"] == "score_boost"
        assert "groups" in comparison
        assert "monotonicity" in comparison

    def test_empty_stocks(self):
        """空列表不崩溃."""
        harness = EvaluationHarness()
        result = harness.evaluate([], run_date="20260418")
        assert result.total_stocks == 0

    def test_all_missing_labels(self):
        """所有标签缺失."""
        stocks = [
            ScoredStock(code="600519", total_score=0.8, label_quality="missing"),
            ScoredStock(code="000858", total_score=0.6, label_quality="missing"),
        ]
        harness = EvaluationHarness()
        result = harness.evaluate(stocks, run_date="20260418")
        # 无 labeled 数据，分组为空
        assert result.total_stocks == 2


# ════════════════════════════════════════════════════════
# EvaluationReport Tests
# ════════════════════════════════════════════════════════

class TestEvaluationReport:
    """评估报告测试."""

    def test_to_text(self):
        """文本报告生成."""
        stocks = _make_scored_stocks(50)
        harness = EvaluationHarness(top_n=5)
        result = harness.evaluate(stocks, run_date="20260418")
        report = EvaluationReport(result)

        text = report.to_text()
        assert "评估报告" in text
        assert "20260418" in text
        assert "pass_tradeable" in text

    def test_save_group_csv(self):
        """分组 CSV 保存."""
        stocks = _make_scored_stocks(50)
        harness = EvaluationHarness(top_n=5)
        result = harness.evaluate(stocks, run_date="20260418")
        report = EvaluationReport(result)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "groups.csv")
            report.save_group_csv(path)
            assert os.path.exists(path)

            # 验证 CSV 内容
            with open(path) as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                assert len(rows) > 0
                assert "group_name" in rows[0]

    def test_save_decile_csv(self):
        """十分位 CSV 保存."""
        stocks = _make_scored_stocks(100)
        harness = EvaluationHarness(top_n=10)
        result = harness.evaluate(stocks, run_date="20260418")
        report = EvaluationReport(result)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "deciles.csv")
            report.save_decile_csv(path)
            assert os.path.exists(path)

    def test_save_all(self):
        """保存所有报告."""
        stocks = _make_scored_stocks(50)
        harness = EvaluationHarness(top_n=5)
        result = harness.evaluate(stocks, run_date="20260418")
        report = EvaluationReport(result)

        with tempfile.TemporaryDirectory() as tmpdir:
            paths = report.save_all(tmpdir, prefix="20260418")
            assert "text" in paths
            assert "groups_csv" in paths
            assert "deciles_csv" in paths
            assert all(os.path.exists(p) for p in paths.values())

    def test_empty_result_report(self):
        """空结果不崩溃."""
        result = EvaluationResult(run_date="20260418")
        report = EvaluationReport(result)
        text = report.to_text()
        assert "评估报告" in text
