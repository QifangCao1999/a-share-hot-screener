"""输出文件写入器.

输出文件：
  stage1_hot_summary.csv    — 每只股票一行，宽表摘要（通过/未通过均输出）
  stage1_hot_detail.csv     — **长表**：每只股票×每个指标一行，便于审计溯源
  stage1_hot_rejected.csv   — 被淘汰股票，含 reject_stage / reject_reason
  stage1_hot_metadata.json  — 本次运行元数据

文件命名规范：包含 run_date 前缀，如：
  2026-04-18_stage1_hot_summary.csv

设计说明：
  - 所有字段写入 None 时以空字符串表示
  - float 保留 4 位小数
  - 不写 parquet（避免引入 pyarrow 依赖；后续 session 可按需切换）

Session 9 P2-1 变更：
  detail.csv 从宽表（每只股票一行）改为长表（每指标一行），列结构：
    code, name, axis, indicator_id, indicator_name, raw_value, derived_value,
    subscore, weight, weighted_score, is_applicable, is_data_available, note
  追加 summary 行（axis="summary"）：包含 total_score / pass_stage1 / data_coverage 等。
"""

from __future__ import annotations

import csv
import dataclasses
import datetime as dt
import json
import logging
import os
from typing import Any, Dict, List, Optional

from a_share_hot_screener.models import (
    HotStockDetail,
    HotStockSummary,
    RejectedRecord,
    RunMetadata,
)

logger = logging.getLogger("a_share_hot_screener.output")


class OutputWriter:
    """写入四类输出文件."""

    def __init__(self, output_dir: str) -> None:
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def write_all(
        self,
        details: List[HotStockDetail],
        rejected: List[RejectedRecord],
        metadata: RunMetadata,
    ) -> None:
        """写入全部输出文件，并将路径更新到 metadata.output_files."""
        # Bug fix: 使用 trade_date_used 而非 run_date 作为文件名前缀
        # 非交易日运行时 run_date 与实际数据日期不一致，用 trade_date_used 语义更准确
        date_prefix = metadata.trade_date_used or metadata.run_date  # YYYY-MM-DD

        summary_path = self._path(date_prefix, "stage1_hot_summary.csv")
        detail_path = self._path(date_prefix, "stage1_hot_detail.csv")
        rejected_path = self._path(date_prefix, "stage1_hot_rejected.csv")
        metadata_path = self._path(date_prefix, "stage1_hot_metadata.json")

        self.write_summary(details, summary_path)
        self.write_detail(details, detail_path)
        self.write_rejected(rejected, rejected_path)

        metadata.output_files = {
            "summary": summary_path,
            "detail": detail_path,
            "rejected": rejected_path,
            "metadata": metadata_path,
        }
        self.write_metadata(metadata, metadata_path)

        logger.info(
            "输出完成: summary=%s, detail=%s, rejected=%s, metadata=%s",
            summary_path,
            detail_path,
            rejected_path,
            metadata_path,
        )

    # ── summary.csv ───────────────────────────────────────

    def write_summary(
        self, details: List[HotStockDetail], path: str
    ) -> None:
        """写入 stage1_hot_summary.csv."""
        summaries = [HotStockSummary.from_detail(d) for d in details]
        if not summaries:
            logger.info("summary: 0 行，跳过写入")
            _write_empty_csv(path, list(dataclasses.fields(HotStockSummary)))
            return

        fieldnames = [f.name for f in dataclasses.fields(HotStockSummary)]
        rows = [dataclasses.asdict(s) for s in summaries]
        _write_csv(path, fieldnames, rows)
        logger.info("summary: %d 行 → %s", len(rows), path)

    # ── detail.csv（长表：每指标一行）──────────────────────

    def write_detail(
        self, details: List[HotStockDetail], path: str
    ) -> None:
        """写入 stage1_hot_detail.csv（长表格式）.

        Session 9 P2-1：从宽表改为长表，每只股票×每个指标一行。
        列结构：
          code, name, axis, indicator_id, indicator_name, raw_value,
          derived_value, subscore, weight, weighted_score,
          is_applicable, is_data_available, note
        追加 summary 行（axis="summary"）：total_score / axis_scores / pass_stage1 等。
        """
        fieldnames = _DETAIL_LONG_COLUMNS
        if not details:
            logger.info("detail: 0 行，跳过写入")
            _write_empty_csv(path, fieldnames)
            return

        rows = []
        for d in details:
            rows.extend(_detail_to_long_rows(d))
        _write_csv(path, fieldnames, rows)
        logger.info("detail(long): %d 行（%d只股票）→ %s", len(rows), len(details), path)

    # ── rejected.csv ──────────────────────────────────────

    def write_rejected(
        self, rejected: List[RejectedRecord], path: str
    ) -> None:
        """写入 stage1_hot_rejected.csv."""
        if not rejected:
            logger.info("rejected: 0 行，跳过写入")
            _write_empty_csv(path, list(dataclasses.fields(RejectedRecord)))
            return

        fieldnames = [f.name for f in dataclasses.fields(RejectedRecord)]
        rows = [dataclasses.asdict(r) for r in rejected]
        _write_csv(path, fieldnames, rows)
        logger.info("rejected: %d 行 → %s", len(rows), path)

    # ── metadata.json ─────────────────────────────────────

    def write_metadata(self, metadata: RunMetadata, path: str) -> None:
        """写入 stage1_hot_metadata.json."""
        data = dataclasses.asdict(metadata)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2, default=str)
        logger.info("metadata → %s", path)

    # ── 内部 ─────────────────────────────────────────────

    def _path(self, run_date: str, filename: str) -> str:
        """组合输出路径：output_dir / {run_date}_{filename}."""
        return os.path.join(self.output_dir, f"{run_date}_{filename}")


# ── CSV 工具函数 ──────────────────────────────────────────

def _write_csv(path: str, fieldnames: List[str], rows: List[Dict[str, Any]]) -> None:
    """写入 CSV 文件（UTF-8 BOM，Excel 兼容）."""
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _write_empty_csv(path: str, fields) -> None:
    """写入只有表头的空 CSV（fields 接受 dataclass Field 列表或字段名列表）."""
    if fields and hasattr(fields[0], "name"):
        fieldnames = [f.name for f in fields]
    else:
        fieldnames = list(fields)
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()


# ── detail 长表工具函数（Session 9 P2-1）──────────────────

_DETAIL_LONG_COLUMNS = [
    "code", "name", "axis", "indicator_id", "indicator_name",
    "raw_value", "derived_value", "subscore", "weight", "weighted_score",
    "is_applicable", "is_data_available", "note",
]

# 轴名 → subscores dict 字段名
_AXIS_SUBSCORE_FIELDS = [
    ("hot_theme",            "hot_theme_subscores"),
    ("trend_flow",           "trend_flow_subscores"),
    ("liquidity_execution",  "liquidity_execution_subscores"),
    ("risk_control",         "risk_control_subscores"),
]


def _detail_to_long_rows(d: HotStockDetail) -> List[Dict[str, Any]]:
    """将一只股票的 HotStockDetail 展开为长表行列表.

    结构：
      1. 每个轴的每个 ScoreItem → 一行
      2. 每个轴的轴级汇总行（axis_score / axis_coverage）→ 一行
      3. 全局汇总行（total_score / data_coverage / pass_stage1）→ 若干行
    """
    rows: List[Dict[str, Any]] = []
    code = d.code
    name = d.name

    # ── 1. 四轴指标行 ────────────────────────────────────
    for axis_name, subscores_attr in _AXIS_SUBSCORE_FIELDS:
        subscores_dict = getattr(d, subscores_attr, {})
        if not subscores_dict:
            continue
        items = subscores_dict.get("items", [])
        for idx, item in enumerate(items, 1):
            indicator_id = f"{axis_name.upper()[:2]}{idx}"  # HO1,HO2... / TR1... / LI1... / RI1...
            # 使用更规范的 ID 前缀
            prefix_map = {
                "hot_theme": "HT",
                "trend_flow": "TF",
                "liquidity_execution": "LE",
                "risk_control": "RC",
            }
            indicator_id = f"{prefix_map.get(axis_name, axis_name[:2].upper())}{idx}"
            rows.append({
                "code": code,
                "name": name,
                "axis": axis_name,
                "indicator_id": indicator_id,
                "indicator_name": item.get("name", ""),
                "raw_value": _fmt_val(item.get("raw_value")),
                "derived_value": _fmt_val(item.get("derived_value")),
                "subscore": _fmt_float(item.get("subscore")),
                "weight": _fmt_float(item.get("weight")),
                "weighted_score": _fmt_float(item.get("weighted_score")),
                "is_applicable": item.get("is_applicable", ""),
                "is_data_available": item.get("is_data_available", ""),
                "note": item.get("note", ""),
            })

        # 轴级汇总行
        axis_score = subscores_dict.get("score")
        axis_coverage = subscores_dict.get("coverage")
        rows.append({
            "code": code,
            "name": name,
            "axis": axis_name,
            "indicator_id": f"{prefix_map.get(axis_name, axis_name[:2].upper())}_TOTAL",
            "indicator_name": f"{axis_name}_score",
            "raw_value": "",
            "derived_value": "",
            "subscore": _fmt_float(axis_score),
            "weight": "",
            "weighted_score": "",
            "is_applicable": "",
            "is_data_available": "",
            "note": f"coverage={_fmt_float(axis_coverage)}",
        })

    # ── 2. 全局汇总行 ────────────────────────────────────
    summary_items = [
        ("total_score", d.total_score),
        ("data_coverage", d.data_coverage),
        ("pass_stage1", d.pass_stage1),
        ("blocked_by", ",".join(d.blocked_by) if d.blocked_by else ""),
        ("passed_hard_filter", d.passed_hard_filter),
        ("hard_filter_reason", d.hard_filter_reason or ""),
    ]
    # Session 14 P2-6: 时序 delta 汇总行
    td = d.trend_delta
    if td:
        summary_items.extend([
            ("prev_run_date", td.get("prev_run_date", "")),
            ("total_score_delta", td.get("total_score_delta")),
            ("pass_stage1_change", td.get("pass_stage1_change", "")),
            ("score_accelerating", td.get("score_accelerating")),
            ("score_decelerating", td.get("score_decelerating")),
        ])
    for sname, sval in summary_items:
        rows.append({
            "code": code,
            "name": name,
            "axis": "summary",
            "indicator_id": "",
            "indicator_name": sname,
            "raw_value": _fmt_val(sval),
            "derived_value": "",
            "subscore": "",
            "weight": "",
            "weighted_score": "",
            "is_applicable": "",
            "is_data_available": "",
            "note": "",
        })

    return rows


def _fmt_float(val: Any) -> str:
    """格式化 float 为 4 位小数字符串，None → 空串."""
    if val is None:
        return ""
    try:
        return f"{float(val):.4f}"
    except (ValueError, TypeError):
        return str(val)


def _fmt_val(val: Any) -> str:
    """格式化任意值为字符串，None → 空串."""
    if val is None:
        return ""
    if isinstance(val, float):
        return f"{val:.4f}"
    if isinstance(val, bool):
        return str(val)
    if isinstance(val, (dict, list)):
        return json.dumps(val, ensure_ascii=False, default=str)
    return str(val)
