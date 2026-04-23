"""评估报告生成器 — 文本 + CSV 格式输出.

输出:
  - 文本报告 (终端友好)
  - CSV 分组指标表
  - CSV 十分位分析表
"""

from __future__ import annotations

import csv
import logging
import os
from dataclasses import asdict
from typing import Any, Dict, List, Optional

from a_share_hot_screener.evaluation.harness import (
    EvaluationResult,
    GroupMetrics,
    MonotonicityResult,
    SeparationResult,
)

# timing action 输出顺序
_TIMING_ACTION_ORDER = ["setup_ready", "watch", "wait", "avoid_chase"]
_TIMING_REGIME_ORDER = ["bull", "neutral", "bear"]
_TIMING_CONFIDENCE_ORDER = ["high", "medium", "low"]

logger = logging.getLogger("a_share_hot_screener.evaluation.report")


class EvaluationReport:
    """评估报告生成器."""

    def __init__(self, result: EvaluationResult):
        self._result = result

    def to_text(self) -> str:
        """生成文本报告."""
        r = self._result
        lines = []
        lines.append("=" * 70)
        lines.append(f"  评估报告 — {r.run_date}")
        lines.append(f"  样本总数: {r.total_stocks}")
        lines.append("=" * 70)
        lines.append("")

        # ── 分组指标 ──────────────────────────────
        lines.append("─── 分组指标 ───")
        lines.append("")

        group_order = [
            "pass_tradeable", "pass_watch_only", "failed_score",
            "top_N", "bottom_N", "all_scored",
        ]

        for gname in group_order:
            gm = r.group_metrics.get(gname)
            if not gm or gm.count == 0:
                continue

            lines.append(f"  [{gname}] (n={gm.count})")
            lines.append(f"    MFE 3d:  avg={_fmt(gm.avg_mfe_3d)}%  med={_fmt(gm.median_mfe_3d)}%  hit>3%={_fmt_pct(gm.hit_rate_mfe3d_gt3)}  hit>5%={_fmt_pct(gm.hit_rate_mfe3d_gt5)}")
            lines.append(f"    MFE 5d:  avg={_fmt(gm.avg_mfe_5d)}%  med={_fmt(gm.median_mfe_5d)}%  hit>5%={_fmt_pct(gm.hit_rate_mfe5d_gt5)}  hit>8%={_fmt_pct(gm.hit_rate_mfe5d_gt8)}")
            lines.append(f"    MAE 5d:  avg={_fmt(gm.avg_mae_5d)}%  med={_fmt(gm.median_mae_5d)}%")
            lines.append(f"    T+1:     avg={_fmt(gm.avg_return_t1)}%  med={_fmt(gm.median_return_t1)}%")
            lines.append(f"    涨停3d:  {_fmt_pct(gm.hit_rate_limit_up_3d)}  超指数5d: {_fmt_pct(gm.hit_rate_beat_index)}  R/R: {_fmt(gm.avg_reward_risk)}")
            lines.append("")

        # ── 排序单调性 ──────────────────────────────
        if r.monotonicity:
            lines.append("─── 排序单调性 (total_score 十分位) ───")
            lines.append("")
            mono = r.monotonicity
            lines.append(f"  单调: {'✅ 是' if mono.is_monotonic else '❌ 否'}  Spearman={_fmt(mono.spearman_corr)}")
            lines.append("")

            if mono.decile_groups:
                lines.append(f"  {'分位':>4}  {'数量':>4}  {'分数区间':>16}  {'MFE5d均值':>10}  {'MFE5d中位':>10}  {'T+1均值':>10}")
                lines.append("  " + "-" * 62)
                for g in mono.decile_groups:
                    lines.append(
                        f"  D{g['decile']:>3}  {g['count']:>4}  "
                        f"{_fmt(g.get('score_min'))}~{_fmt(g.get('score_max'))}  "
                        f"{_fmt(g.get('avg_mfe_5d')):>10}%  "
                        f"{_fmt(g.get('median_mfe_5d')):>10}%  "
                        f"{_fmt(g.get('avg_return_t1')):>10}%"
                    )
                lines.append("")

        # ── 分离度 ──────────────────────────────────
        if r.separation:
            lines.append("─── 分离度: tradeable vs failed_score ───")
            lines.append(_fmt_separation(r.separation))
            lines.append("")

        if r.tradeable_vs_watch:
            lines.append("─── 分离度: tradeable vs watch_only ───")
            lines.append(_fmt_separation(r.tradeable_vs_watch))
            lines.append("")

        # ── Round 3: Timing 分组 ────────────────
        if r.timing_by_action:
            lines.append("─── Setup Timing 分组: 按 Action ───")
            lines.append("")
            for action in _TIMING_ACTION_ORDER:
                gm = r.timing_by_action.get(action)
                if gm and gm.count > 0:
                    lines.extend(_fmt_timing_group(action, gm))
            lines.append("")

        if r.timing_by_regime:
            lines.append("─── Setup Timing 分组: 按 Market Regime ───")
            lines.append("")
            for regime in _TIMING_REGIME_ORDER:
                gm = r.timing_by_regime.get(regime)
                if gm and gm.count > 0:
                    lines.extend(_fmt_timing_group(regime, gm))
            # 未知 regime
            for rname, gm in r.timing_by_regime.items():
                if rname not in _TIMING_REGIME_ORDER and gm.count > 0:
                    lines.extend(_fmt_timing_group(rname, gm))
            lines.append("")

        if r.timing_by_confidence:
            lines.append("─── Setup Timing 分组: 按 Level Confidence ───")
            lines.append("")
            for conf in _TIMING_CONFIDENCE_ORDER:
                gm = r.timing_by_confidence.get(conf)
                if gm and gm.count > 0:
                    lines.extend(_fmt_timing_group(conf, gm))
            lines.append("")

        if r.timing_monotonicity and r.timing_monotonicity.decile_groups:
            lines.append("─── 排序单调性 (timing_score 十分位) ───")
            lines.append("")
            mono = r.timing_monotonicity
            lines.append(f"  单调: {'\u2705 是' if mono.is_monotonic else '\u274c 否'}  Spearman={_fmt(mono.spearman_corr)}")
            lines.append("")
            lines.append(f"  {'分位':>4}  {'数量':>4}  {'分数区间':>16}  {'MFE5d均值':>10}  {'MFE5d中位':>10}  {'T+1均值':>10}")
            lines.append("  " + "-" * 62)
            for g in mono.decile_groups:
                lines.append(
                    f"  D{g['decile']:>3}  {g['count']:>4}  "
                    f"{_fmt(g.get('score_min'))}~{_fmt(g.get('score_max'))}  "
                    f"{_fmt(g.get('avg_mfe_5d')):>10}%  "
                    f"{_fmt(g.get('median_mfe_5d')):>10}%  "
                    f"{_fmt(g.get('avg_return_t1')):>10}%"
                )
            lines.append("")

        lines.append("=" * 70)
        return "\n".join(lines)

    def save_group_csv(self, output_path: str) -> str:
        """保存分组指标 CSV."""
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

        fieldnames = [
            "group_name", "count",
            "avg_mfe_3d", "avg_mfe_5d", "avg_mae_5d",
            "median_mfe_3d", "median_mfe_5d", "median_mae_5d",
            "avg_return_t1", "median_return_t1",
            "hit_rate_mfe3d_gt3", "hit_rate_mfe3d_gt5",
            "hit_rate_mfe5d_gt5", "hit_rate_mfe5d_gt8",
            "hit_rate_limit_up_3d", "hit_rate_beat_index",
            "avg_reward_risk",
            "hit_rate_touched_support", "hit_rate_broke_invalidation",
            "hit_rate_touched_resistance",
        ]

        # 收集所有分组（原有 + timing 分组）
        all_groups: Dict[str, GroupMetrics] = {}
        all_groups.update(self._result.group_metrics)
        for action, gm in self._result.timing_by_action.items():
            all_groups[f"timing_action_{action}"] = gm
        for regime, gm in self._result.timing_by_regime.items():
            all_groups[f"timing_regime_{regime}"] = gm
        for conf, gm in self._result.timing_by_confidence.items():
            all_groups[f"timing_confidence_{conf}"] = gm

        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for gname, gm in all_groups.items():
                row = asdict(gm)
                filtered = {k: row.get(k, "") for k in fieldnames}
                for k, v in filtered.items():
                    if v is None:
                        filtered[k] = ""
                    elif isinstance(v, float):
                        filtered[k] = f"{v:.4f}"
                writer.writerow(filtered)

        logger.info("[report] Group metrics CSV saved to %s", output_path)
        return output_path

    def save_decile_csv(self, output_path: str) -> str:
        """保存十分位分析 CSV."""
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

        if not self._result.monotonicity or not self._result.monotonicity.decile_groups:
            logger.warning("[report] No decile data to save")
            return output_path

        fieldnames = [
            "decile", "count", "score_min", "score_max",
            "avg_mfe_5d", "median_mfe_5d", "avg_return_t1",
        ]

        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for g in self._result.monotonicity.decile_groups:
                row = {k: g.get(k, "") for k in fieldnames}
                for k, v in row.items():
                    if v is None:
                        row[k] = ""
                    elif isinstance(v, float):
                        row[k] = f"{v:.4f}"
                writer.writerow(row)

        logger.info("[report] Decile CSV saved to %s", output_path)
        return output_path

    def save_timing_decile_csv(self, output_path: str) -> str:
        """保存 timing_score 十分位分析 CSV."""
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

        mono = self._result.timing_monotonicity
        if not mono or not mono.decile_groups:
            # 创建空文件（仅表头）
            with open(output_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=[
                    "decile", "count", "score_min", "score_max",
                    "avg_mfe_5d", "median_mfe_5d", "avg_return_t1",
                ])
                writer.writeheader()
            return output_path

        fieldnames = [
            "decile", "count", "score_min", "score_max",
            "avg_mfe_5d", "median_mfe_5d", "avg_return_t1",
        ]

        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for g in mono.decile_groups:
                row = {k: g.get(k, "") for k in fieldnames}
                for k, v in row.items():
                    if v is None:
                        row[k] = ""
                    elif isinstance(v, float):
                        row[k] = f"{v:.4f}"
                writer.writerow(row)

        logger.info("[report] Timing decile CSV saved to %s", output_path)
        return output_path

    def save_all(self, output_dir: str, prefix: str = "") -> Dict[str, str]:
        """保存所有报告文件.

        Returns:
            {report_type: file_path}
        """
        os.makedirs(output_dir, exist_ok=True)
        pfx = f"{prefix}_" if prefix else ""

        paths = {}

        # 文本报告
        text_path = os.path.join(output_dir, f"{pfx}evaluation_report.txt")
        with open(text_path, "w", encoding="utf-8") as f:
            f.write(self.to_text())
        paths["text"] = text_path

        # 分组 CSV
        group_path = os.path.join(output_dir, f"{pfx}evaluation_groups.csv")
        self.save_group_csv(group_path)
        paths["groups_csv"] = group_path

        # 十分位 CSV (total_score)
        decile_path = os.path.join(output_dir, f"{pfx}evaluation_deciles.csv")
        self.save_decile_csv(decile_path)
        paths["deciles_csv"] = decile_path

        # 十分位 CSV (timing_score) — Round 3
        timing_decile_path = os.path.join(output_dir, f"{pfx}evaluation_timing_deciles.csv")
        self.save_timing_decile_csv(timing_decile_path)
        paths["timing_deciles_csv"] = timing_decile_path

        logger.info("[report] All reports saved to %s", output_dir)
        return paths


# ════════════════════════════════════════════════════════
# 工具函数
# ════════════════════════════════════════════════════════

def _fmt(val: Optional[float], decimals: int = 2) -> str:
    """格式化浮点数."""
    if val is None:
        return "N/A"
    return f"{val:.{decimals}f}"


def _fmt_pct(val: Optional[float]) -> str:
    """格式化百分比."""
    if val is None:
        return "N/A"
    return f"{val:.1%}"


def _fmt_timing_group(label: str, gm: GroupMetrics) -> List[str]:
    """格式化 timing 分组指标."""
    lines = []
    lines.append(f"  [{label}] (n={gm.count})")
    lines.append(
        f"    MFE 3d: avg={_fmt(gm.avg_mfe_3d)}%  med={_fmt(gm.median_mfe_3d)}%  "
        f"hit>3%={_fmt_pct(gm.hit_rate_mfe3d_gt3)}  hit>5%={_fmt_pct(gm.hit_rate_mfe3d_gt5)}"
    )
    lines.append(
        f"    MFE 5d: avg={_fmt(gm.avg_mfe_5d)}%  med={_fmt(gm.median_mfe_5d)}%  "
        f"hit>5%={_fmt_pct(gm.hit_rate_mfe5d_gt5)}  hit>8%={_fmt_pct(gm.hit_rate_mfe5d_gt8)}"
    )
    lines.append(f"    T+1:    avg={_fmt(gm.avg_return_t1)}%  med={_fmt(gm.median_return_t1)}%  R/R: {_fmt(gm.avg_reward_risk)}")
    # timing 专项
    timing_parts = []
    if gm.hit_rate_touched_support is not None:
        timing_parts.append(f"触支撑={_fmt_pct(gm.hit_rate_touched_support)}")
    if gm.hit_rate_broke_invalidation is not None:
        timing_parts.append(f"破失效={_fmt_pct(gm.hit_rate_broke_invalidation)}")
    if gm.hit_rate_touched_resistance is not None:
        timing_parts.append(f"触压力={_fmt_pct(gm.hit_rate_touched_resistance)}")
    if timing_parts:
        lines.append(f"    Timing: {' '.join(timing_parts)}")
    lines.append("")
    return lines


def _fmt_separation(sep: SeparationResult) -> str:
    """格式化分离度结果."""
    lines = []
    if sep.pass_group:
        lines.append(f"  Pass组 (n={sep.pass_group.count}): MFE3d中位={_fmt(sep.pass_group.median_mfe_3d)}%  MFE5d中位={_fmt(sep.pass_group.median_mfe_5d)}%  T+1中位={_fmt(sep.pass_group.median_return_t1)}%")
    if sep.fail_group:
        lines.append(f"  Fail组 (n={sep.fail_group.count}): MFE3d中位={_fmt(sep.fail_group.median_mfe_3d)}%  MFE5d中位={_fmt(sep.fail_group.median_mfe_5d)}%  T+1中位={_fmt(sep.fail_group.median_return_t1)}%")
    lines.append(f"  差异: MFE3d={_fmt(sep.mfe_3d_diff)}pp  MFE5d={_fmt(sep.mfe_5d_diff)}pp  T+1={_fmt(sep.return_t1_diff)}pp")
    return "\n".join(lines)
