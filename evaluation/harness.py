"""评估引擎 — 核心评估指标计算 + 分组 + 消融实验框架.

指标 (DESIGN_v2.md §1.3):
  - Top N 命中率 (MFE > X%)
  - Top N 平均 MFE / MAE
  - pass vs fail 分离度
  - tradeable vs watch_only 差异
  - 分数排序单调性 (十分位分组)

分组 (DESIGN_v2.md §1.2):
  - pass_tradeable
  - pass_watch_only
  - failed_score
  - top_N
  - bottom_N
"""

from __future__ import annotations

import csv
import logging
import os
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger("a_share_hot_screener.evaluation.harness")


# ════════════════════════════════════════════════════════
# 数据结构
# ════════════════════════════════════════════════════════

@dataclass
class ScoredStock:
    """评估引擎使用的股票数据 (从 summary CSV + label CSV 合并)."""

    code: str
    name: str = ""
    total_score: float = 0.0
    candidate_pool_type: str = ""   # tradeable / watch_only / failed_score / ...
    pass_stage1: bool = False
    passed_hard_filter: bool = False

    # 四轴分数
    hot_theme_score: Optional[float] = None
    trend_flow_score: Optional[float] = None
    liquidity_execution_score: Optional[float] = None
    risk_control_score: Optional[float] = None

    # Setup Timing (Phase 4)
    timing_score: Optional[float] = None         # 0~100
    timing_action: str = ""                       # setup_ready/watch/wait/avoid_chase
    timing_level_confidence: str = ""             # high/medium/low
    timing_risk_score: Optional[float] = None     # 0~1 风险维度分
    timing_market_regime: str = ""                # bull/neutral/bear

    # Context Scores (Phase 3: HT8/9/10)
    ht8_score: Optional[float] = None
    ht8_confirmation_level: str = ""
    ht9_score: Optional[float] = None
    ht10_score: Optional[float] = None
    ht10_position_type: str = ""

    # 标签
    return_t1: Optional[float] = None
    mfe_3d: Optional[float] = None
    mfe_5d: Optional[float] = None
    mae_5d: Optional[float] = None
    hit_limit_up_3d: Optional[bool] = None
    beat_index_5d: Optional[bool] = None
    touched_support_zone_5d: Optional[bool] = None
    broke_invalidation_5d: Optional[bool] = None
    touched_resistance_5d: Optional[bool] = None
    label_quality: str = "missing"


@dataclass
class GroupMetrics:
    """一个分组的评估指标."""

    group_name: str
    count: int = 0

    # MFE/MAE 统计
    avg_mfe_3d: Optional[float] = None
    avg_mfe_5d: Optional[float] = None
    avg_mae_5d: Optional[float] = None
    median_mfe_3d: Optional[float] = None
    median_mfe_5d: Optional[float] = None
    median_mae_5d: Optional[float] = None

    # 收益统计
    avg_return_t1: Optional[float] = None
    median_return_t1: Optional[float] = None

    # 命中率
    hit_rate_mfe3d_gt3: Optional[float] = None   # MFE 3d > 3%
    hit_rate_mfe3d_gt5: Optional[float] = None   # MFE 3d > 5%
    hit_rate_mfe5d_gt5: Optional[float] = None   # MFE 5d > 5%
    hit_rate_mfe5d_gt8: Optional[float] = None   # MFE 5d > 8%
    hit_rate_limit_up_3d: Optional[float] = None # 3日内涨停率
    hit_rate_beat_index: Optional[float] = None  # 跑赢沪深300率

    # 风险
    avg_reward_risk: Optional[float] = None      # 平均 MFE5d / |MAE5d|

    # timing 专项
    hit_rate_touched_support: Optional[float] = None   # 触及支撑区间率
    hit_rate_broke_invalidation: Optional[float] = None  # 跌破失效位率
    hit_rate_touched_resistance: Optional[float] = None  # 触及压力位率


@dataclass
class MonotonicityResult:
    """分数排序单调性检验结果."""

    decile_groups: List[Dict[str, Any]] = field(default_factory=list)
    is_monotonic: bool = False
    spearman_corr: Optional[float] = None
    kendall_tau: Optional[float] = None


@dataclass
class SeparationResult:
    """pass vs fail 分离度结果."""

    pass_group: Optional[GroupMetrics] = None
    fail_group: Optional[GroupMetrics] = None
    mfe_3d_diff: Optional[float] = None      # pass 中位数 - fail 中位数
    mfe_5d_diff: Optional[float] = None
    return_t1_diff: Optional[float] = None


@dataclass
class EvaluationResult:
    """完整评估结果."""

    run_date: str
    total_stocks: int = 0

    # 分组指标
    group_metrics: Dict[str, GroupMetrics] = field(default_factory=dict)

    # 排序单调性
    monotonicity: Optional[MonotonicityResult] = None

    # pass vs fail 分离度
    separation: Optional[SeparationResult] = None

    # tradeable vs watch_only
    tradeable_vs_watch: Optional[SeparationResult] = None

    # ── Round 3: timing 分组 ──
    timing_by_action: Dict[str, GroupMetrics] = field(default_factory=dict)
    timing_by_regime: Dict[str, GroupMetrics] = field(default_factory=dict)
    timing_by_confidence: Dict[str, GroupMetrics] = field(default_factory=dict)
    timing_monotonicity: Optional[MonotonicityResult] = None  # timing_score 十分位


# ════════════════════════════════════════════════════════
# 评估引擎
# ════════════════════════════════════════════════════════

class EvaluationHarness:
    """核心评估引擎.

    用法:
        harness = EvaluationHarness()
        stocks = harness.load_merged_data(summary_csv, label_csv)
        result = harness.evaluate(stocks, run_date="20260418")
    """

    def __init__(self, top_n: int = 10):
        self.top_n = top_n

    def load_merged_data(
        self,
        summary_csv: str,
        label_csv: str,
    ) -> List[ScoredStock]:
        """从 summary CSV + label CSV 合并加载数据.

        summary_csv: Stage 1 的 stage1_hot_summary.csv (全量)
        label_csv: label_generator 输出的标签 CSV
        """
        # 加载 summary
        summary_map: Dict[str, Dict[str, Any]] = {}
        with open(summary_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                code = row.get("code", "")
                if code:
                    summary_map[code] = row

        # 加载 labels
        label_map: Dict[str, Dict[str, Any]] = {}
        with open(label_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                code = row.get("code", "")
                if code:
                    label_map[code] = row

        # 合并
        stocks = []
        for code, srow in summary_map.items():
            stock = ScoredStock(code=code)
            stock.name = srow.get("name", "")
            stock.total_score = _safe_float(srow.get("total_score"))
            stock.candidate_pool_type = srow.get("candidate_pool_type", "")
            stock.pass_stage1 = srow.get("pass_stage1", "") == "True"
            stock.passed_hard_filter = srow.get("passed_hard_filter", "") == "True"

            stock.hot_theme_score = _safe_float(srow.get("hot_theme_score"))
            stock.trend_flow_score = _safe_float(srow.get("trend_flow_score"))
            stock.liquidity_execution_score = _safe_float(srow.get("liquidity_execution_score"))
            stock.risk_control_score = _safe_float(srow.get("risk_control_score"))

            # Setup Timing 字段
            stock.timing_score = _safe_float(srow.get("timing_score"))
            stock.timing_action = srow.get("timing_action", "")
            stock.timing_level_confidence = srow.get("timing_level_confidence", "")
            stock.timing_risk_score = _safe_float(srow.get("timing_risk_score"))  # 可能不在 summary
            stock.timing_market_regime = srow.get("timing_market_regime", srow.get("market_regime", ""))

            # Context Scores 字段
            stock.ht8_score = _safe_float(srow.get("ht8_score"))
            stock.ht8_confirmation_level = srow.get("ht8_confirmation_level", "")
            stock.ht9_score = _safe_float(srow.get("ht9_score"))
            stock.ht10_score = _safe_float(srow.get("ht10_score"))
            stock.ht10_position_type = srow.get("ht10_position_type", "")

            # 合并标签
            lrow = label_map.get(code, {})
            if lrow:
                stock.return_t1 = _safe_float(lrow.get("return_t1"))
                stock.mfe_3d = _safe_float(lrow.get("mfe_3d"))
                stock.mfe_5d = _safe_float(lrow.get("mfe_5d"))
                stock.mae_5d = _safe_float(lrow.get("mae_5d"))
                stock.hit_limit_up_3d = lrow.get("hit_limit_up_3d") == "True"
                stock.beat_index_5d = lrow.get("beat_index_5d") == "True"
                stock.touched_support_zone_5d = lrow.get("touched_support_zone_5d") == "True" if lrow.get("touched_support_zone_5d") else None
                stock.broke_invalidation_5d = lrow.get("broke_invalidation_5d") == "True" if lrow.get("broke_invalidation_5d") else None
                stock.touched_resistance_5d = lrow.get("touched_resistance_5d") == "True" if lrow.get("touched_resistance_5d") else None
                stock.label_quality = lrow.get("label_quality", "missing")

            stocks.append(stock)

        logger.info(
            "[harness] Loaded %d stocks (%d with labels)",
            len(stocks),
            sum(1 for s in stocks if s.label_quality != "missing"),
        )
        return stocks

    def evaluate(
        self,
        stocks: List[ScoredStock],
        run_date: str = "",
        top_n: Optional[int] = None,
    ) -> EvaluationResult:
        """运行完整评估.

        Args:
            stocks: 全量 scored stocks (含标签)
            run_date: 运行日期
            top_n: Top N 分组大小 (默认用构造函数的 self.top_n)
        """
        n = top_n or self.top_n
        result = EvaluationResult(run_date=run_date, total_stocks=len(stocks))

        # 只评估有标签的
        labeled = [s for s in stocks if s.label_quality in ("full", "partial")]

        # ── 分组 ──────────────────────────────────
        groups = self._build_groups(labeled, n)
        for gname, members in groups.items():
            result.group_metrics[gname] = self._compute_group_metrics(gname, members)

        # ── 排序单调性 ──────────────────────────────
        # 只用通过硬筛的
        hard_passed = [s for s in labeled if s.passed_hard_filter]
        result.monotonicity = self._compute_monotonicity(hard_passed)

        # ── pass vs fail 分离度 ──────────────────────
        result.separation = self._compute_separation(
            pass_group=groups.get("pass_tradeable", []),
            fail_group=groups.get("failed_score", []),
        )

        # ── tradeable vs watch_only ──────────────────
        result.tradeable_vs_watch = self._compute_separation(
            pass_group=groups.get("pass_tradeable", []),
            fail_group=groups.get("pass_watch_only", []),
        )

        # ── Round 3: timing 分组 ──────────────────────
        self._evaluate_timing_groups(result, labeled)

        return result

    def _build_groups(
        self,
        stocks: List[ScoredStock],
        top_n: int,
    ) -> Dict[str, List[ScoredStock]]:
        """按 DESIGN_v2.md 定义分组."""
        groups: Dict[str, List[ScoredStock]] = {
            "pass_tradeable": [],
            "pass_watch_only": [],
            "failed_score": [],
            "top_N": [],
            "bottom_N": [],
            "all_scored": [],
        }

        for s in stocks:
            groups["all_scored"].append(s)
            cpt = s.candidate_pool_type
            if cpt == "tradeable":
                groups["pass_tradeable"].append(s)
            elif cpt == "watch_only":
                groups["pass_watch_only"].append(s)
            elif cpt == "failed_score":
                groups["failed_score"].append(s)

        # Top N / Bottom N (按 total_score 排序，只看通过硬筛的)
        hard_passed = [s for s in stocks if s.passed_hard_filter and s.total_score is not None]
        sorted_by_score = sorted(hard_passed, key=lambda s: s.total_score or 0, reverse=True)
        groups["top_N"] = sorted_by_score[:top_n]
        groups["bottom_N"] = sorted_by_score[-top_n:] if len(sorted_by_score) >= top_n else sorted_by_score

        return groups

    def _compute_group_metrics(
        self,
        group_name: str,
        members: List[ScoredStock],
    ) -> GroupMetrics:
        """计算一个分组的所有指标."""
        m = GroupMetrics(group_name=group_name, count=len(members))
        if not members:
            return m

        # 收集有效值
        mfe_3d_vals = [s.mfe_3d for s in members if s.mfe_3d is not None]
        mfe_5d_vals = [s.mfe_5d for s in members if s.mfe_5d is not None]
        mae_5d_vals = [s.mae_5d for s in members if s.mae_5d is not None]
        return_t1_vals = [s.return_t1 for s in members if s.return_t1 is not None]

        # 均值
        m.avg_mfe_3d = _mean(mfe_3d_vals)
        m.avg_mfe_5d = _mean(mfe_5d_vals)
        m.avg_mae_5d = _mean(mae_5d_vals)
        m.avg_return_t1 = _mean(return_t1_vals)

        # 中位数
        m.median_mfe_3d = _median(mfe_3d_vals)
        m.median_mfe_5d = _median(mfe_5d_vals)
        m.median_mae_5d = _median(mae_5d_vals)
        m.median_return_t1 = _median(return_t1_vals)

        # 命中率
        if mfe_3d_vals:
            m.hit_rate_mfe3d_gt3 = sum(1 for v in mfe_3d_vals if v > 3) / len(mfe_3d_vals)
            m.hit_rate_mfe3d_gt5 = sum(1 for v in mfe_3d_vals if v > 5) / len(mfe_3d_vals)
        if mfe_5d_vals:
            m.hit_rate_mfe5d_gt5 = sum(1 for v in mfe_5d_vals if v > 5) / len(mfe_5d_vals)
            m.hit_rate_mfe5d_gt8 = sum(1 for v in mfe_5d_vals if v > 8) / len(mfe_5d_vals)

        limit_up_vals = [s.hit_limit_up_3d for s in members if s.hit_limit_up_3d is not None]
        if limit_up_vals:
            m.hit_rate_limit_up_3d = sum(1 for v in limit_up_vals if v) / len(limit_up_vals)

        beat_vals = [s.beat_index_5d for s in members if s.beat_index_5d is not None]
        if beat_vals:
            m.hit_rate_beat_index = sum(1 for v in beat_vals if v) / len(beat_vals)

        # 风险收益比
        if mfe_5d_vals and mae_5d_vals:
            avg_mfe = _mean(mfe_5d_vals)
            avg_mae = _mean(mae_5d_vals)
            if avg_mae is not None and avg_mfe is not None and avg_mae != 0:
                m.avg_reward_risk = abs(avg_mfe / avg_mae)

        # timing 专项命中率
        support_vals = [s.touched_support_zone_5d for s in members if s.touched_support_zone_5d is not None]
        if support_vals:
            m.hit_rate_touched_support = sum(1 for v in support_vals if v) / len(support_vals)

        inv_vals = [s.broke_invalidation_5d for s in members if s.broke_invalidation_5d is not None]
        if inv_vals:
            m.hit_rate_broke_invalidation = sum(1 for v in inv_vals if v) / len(inv_vals)

        res_vals = [s.touched_resistance_5d for s in members if s.touched_resistance_5d is not None]
        if res_vals:
            m.hit_rate_touched_resistance = sum(1 for v in res_vals if v) / len(res_vals)

        return m

    def _compute_monotonicity(
        self,
        stocks: List[ScoredStock],
    ) -> MonotonicityResult:
        """计算 total_score 十分位分组 MFE 单调性."""
        result = MonotonicityResult()

        valid = [s for s in stocks if s.total_score is not None and s.mfe_5d is not None]
        if len(valid) < 10:
            return result

        # 按 total_score 降序排列
        sorted_stocks = sorted(valid, key=lambda s: s.total_score or 0, reverse=True)
        n = len(sorted_stocks)
        decile_size = max(1, n // 10)

        decile_groups = []
        for i in range(10):
            start = i * decile_size
            end = start + decile_size if i < 9 else n
            group = sorted_stocks[start:end]
            if not group:
                continue

            scores = [s.total_score for s in group]
            mfe_vals = [s.mfe_5d for s in group if s.mfe_5d is not None]

            decile_groups.append({
                "decile": i + 1,
                "count": len(group),
                "score_min": min(scores),
                "score_max": max(scores),
                "avg_mfe_5d": _mean(mfe_vals),
                "median_mfe_5d": _median(mfe_vals),
                "avg_return_t1": _mean([s.return_t1 for s in group if s.return_t1 is not None]),
            })

        result.decile_groups = decile_groups

        # 检查单调性 (MFE 从 decile 1 到 10 递减)
        mfe_means = [g["avg_mfe_5d"] for g in decile_groups if g["avg_mfe_5d"] is not None]
        if len(mfe_means) >= 5:
            # 简单检查: decile 1 > decile 10
            result.is_monotonic = mfe_means[0] > mfe_means[-1]

            # Spearman 相关 (rank vs MFE)
            try:
                ranks = list(range(1, len(mfe_means) + 1))
                result.spearman_corr = _spearman_corr(ranks, mfe_means)
            except Exception:
                pass

        return result

    def _compute_separation(
        self,
        pass_group: List[ScoredStock],
        fail_group: List[ScoredStock],
    ) -> SeparationResult:
        """计算两组的分离度."""
        result = SeparationResult()

        if pass_group:
            result.pass_group = self._compute_group_metrics("pass", pass_group)
        if fail_group:
            result.fail_group = self._compute_group_metrics("fail", fail_group)

        if result.pass_group and result.fail_group:
            if result.pass_group.median_mfe_3d is not None and result.fail_group.median_mfe_3d is not None:
                result.mfe_3d_diff = result.pass_group.median_mfe_3d - result.fail_group.median_mfe_3d
            if result.pass_group.median_mfe_5d is not None and result.fail_group.median_mfe_5d is not None:
                result.mfe_5d_diff = result.pass_group.median_mfe_5d - result.fail_group.median_mfe_5d
            if result.pass_group.median_return_t1 is not None and result.fail_group.median_return_t1 is not None:
                result.return_t1_diff = result.pass_group.median_return_t1 - result.fail_group.median_return_t1

        return result

    # ════════════════════════════════════════════════════════
    # Round 3: Timing 分组评估
    # ════════════════════════════════════════════════════════

    def _evaluate_timing_groups(
        self,
        result: EvaluationResult,
        labeled: List[ScoredStock],
    ) -> None:
        """计算 timing 相关分组指标 (Round 3).

        分组维度:
          1. 按 timing_action: setup_ready / watch / wait / avoid_chase
          2. 按 timing_market_regime: bull / neutral / bear
          3. 按 timing_level_confidence: high / medium / low
          4. timing_score 十分位单调性
        """
        # 筛选有 timing 数据的股票
        timed = [s for s in labeled if s.timing_action]

        # 1. 按 action 分组
        action_groups: Dict[str, List[ScoredStock]] = {}
        for s in timed:
            action_groups.setdefault(s.timing_action, []).append(s)
        for action_name, members in action_groups.items():
            result.timing_by_action[action_name] = self._compute_group_metrics(
                f"timing_action_{action_name}", members
            )

        # 2. 按 market_regime 分组
        regime_groups: Dict[str, List[ScoredStock]] = {}
        for s in timed:
            regime = s.timing_market_regime or "unknown"
            regime_groups.setdefault(regime, []).append(s)
        for regime_name, members in regime_groups.items():
            result.timing_by_regime[regime_name] = self._compute_group_metrics(
                f"timing_regime_{regime_name}", members
            )

        # 3. 按 level_confidence 分组
        conf_groups: Dict[str, List[ScoredStock]] = {}
        for s in timed:
            conf = s.timing_level_confidence or "unknown"
            conf_groups.setdefault(conf, []).append(s)
        for conf_name, members in conf_groups.items():
            result.timing_by_confidence[conf_name] = self._compute_group_metrics(
                f"timing_confidence_{conf_name}", members
            )

        # 4. timing_score 十分位单调性
        valid_timing = [s for s in timed if s.timing_score is not None and s.mfe_5d is not None]
        result.timing_monotonicity = self._compute_timing_monotonicity(valid_timing)

    def _compute_timing_monotonicity(
        self,
        stocks: List[ScoredStock],
    ) -> MonotonicityResult:
        """计算 timing_score 十分位分组 MFE 单调性."""
        result = MonotonicityResult()

        if len(stocks) < 10:
            return result

        sorted_stocks = sorted(stocks, key=lambda s: s.timing_score or 0, reverse=True)
        n = len(sorted_stocks)
        decile_size = max(1, n // 10)

        decile_groups = []
        for i in range(10):
            start = i * decile_size
            end = start + decile_size if i < 9 else n
            group = sorted_stocks[start:end]
            if not group:
                continue

            timing_scores = [s.timing_score for s in group if s.timing_score is not None]
            mfe_vals = [s.mfe_5d for s in group if s.mfe_5d is not None]

            decile_groups.append({
                "decile": i + 1,
                "count": len(group),
                "score_min": min(timing_scores) if timing_scores else None,
                "score_max": max(timing_scores) if timing_scores else None,
                "avg_mfe_5d": _mean(mfe_vals),
                "median_mfe_5d": _median(mfe_vals),
                "avg_return_t1": _mean([s.return_t1 for s in group if s.return_t1 is not None]),
            })

        result.decile_groups = decile_groups

        mfe_means = [g["avg_mfe_5d"] for g in decile_groups if g["avg_mfe_5d"] is not None]
        if len(mfe_means) >= 5:
            result.is_monotonic = mfe_means[0] > mfe_means[-1]
            try:
                ranks = list(range(1, len(mfe_means) + 1))
                result.spearman_corr = _spearman_corr(ranks, mfe_means)
            except Exception:
                pass

        return result

    # ════════════════════════════════════════════════════════
    # 消融实验
    # ════════════════════════════════════════════════════════

    def ablation_compare(
        self,
        baseline_stocks: List[ScoredStock],
        variant_stocks: List[ScoredStock],
        run_date: str = "",
        variant_name: str = "variant",
    ) -> Dict[str, Any]:
        """对比两种配置的评估结果 (消融实验).

        Args:
            baseline_stocks: 基线配置的评分 + 标签
            variant_stocks: 变体配置的评分 + 标签
            variant_name: 变体名称

        Returns:
            对比报告 dict
        """
        baseline_result = self.evaluate(baseline_stocks, run_date=run_date)
        variant_result = self.evaluate(variant_stocks, run_date=run_date)

        comparison = {
            "run_date": run_date,
            "variant_name": variant_name,
            "baseline_total": baseline_result.total_stocks,
            "variant_total": variant_result.total_stocks,
            "groups": {},
        }

        # 对比每个分组的指标
        all_groups = set(baseline_result.group_metrics.keys()) | set(variant_result.group_metrics.keys())
        for gname in sorted(all_groups):
            bm = baseline_result.group_metrics.get(gname)
            vm = variant_result.group_metrics.get(gname)
            comparison["groups"][gname] = {
                "baseline": asdict(bm) if bm else None,
                "variant": asdict(vm) if vm else None,
            }

        # 单调性对比
        comparison["monotonicity"] = {
            "baseline": asdict(baseline_result.monotonicity) if baseline_result.monotonicity else None,
            "variant": asdict(variant_result.monotonicity) if variant_result.monotonicity else None,
        }

        # 分离度对比
        comparison["separation"] = {
            "baseline": {
                "mfe_3d_diff": baseline_result.separation.mfe_3d_diff if baseline_result.separation else None,
                "mfe_5d_diff": baseline_result.separation.mfe_5d_diff if baseline_result.separation else None,
            },
            "variant": {
                "mfe_3d_diff": variant_result.separation.mfe_3d_diff if variant_result.separation else None,
                "mfe_5d_diff": variant_result.separation.mfe_5d_diff if variant_result.separation else None,
            },
        }

        return comparison


# ════════════════════════════════════════════════════════
# 工具函数
# ════════════════════════════════════════════════════════

def _safe_float(val: Any) -> Optional[float]:
    """安全转换为 float, 空/None → None."""
    if val is None or val == "":
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


def _mean(vals: List[Optional[float]]) -> Optional[float]:
    """计算均值, 忽略 None."""
    clean = [v for v in vals if v is not None]
    if not clean:
        return None
    return sum(clean) / len(clean)


def _median(vals: List[Optional[float]]) -> Optional[float]:
    """计算中位数, 忽略 None."""
    clean = sorted(v for v in vals if v is not None)
    if not clean:
        return None
    n = len(clean)
    if n % 2 == 1:
        return clean[n // 2]
    return (clean[n // 2 - 1] + clean[n // 2]) / 2


def _spearman_corr(x: List[float], y: List[float]) -> Optional[float]:
    """简易 Spearman 秩相关 (不依赖 scipy)."""
    n = len(x)
    if n < 3:
        return None

    def _rank(vals):
        sorted_idx = sorted(range(len(vals)), key=lambda i: vals[i])
        ranks = [0.0] * len(vals)
        for rank, idx in enumerate(sorted_idx, 1):
            ranks[idx] = float(rank)
        return ranks

    rx = _rank(x)
    ry = _rank(y)

    d_sq = sum((a - b) ** 2 for a, b in zip(rx, ry))
    return 1 - (6 * d_sq) / (n * (n ** 2 - 1))
