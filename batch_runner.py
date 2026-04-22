"""批量运行编排器 — 自动分批 + 合并 + 断点续跑.

用法（由 CLI 自动路由）：
    python3 -m a_share_hot_screener \
        --batch-size 100 --resume \
        --stock-codes "600519,000858,..." \
        --run-date today --output-dir ./output

流程（默认全局池模式，P2）：
  1. 将 stock_codes 按 batch_size 分块
  2. 每批运行数据收集阶段（Steps 0-6.12），不评分
  3. 构建跨批次全局 ScoringPool
  4. 对所有股票统一评分 + flags + 判定 + 输出

旧模式（--no-global-pool）：
  每批在 output_dir/batch_N/ 下运行独立 pipeline，最后合并 CSV。
"""

from __future__ import annotations

import copy
import csv
import dataclasses
import json
import logging
import os
import time
from typing import Any, Dict, List, Optional

from a_share_hot_screener.config import HotScreenerConfig
from a_share_hot_screener.models import HotStockDetail, RejectedRecord, RunMetadata
from a_share_hot_screener.pipeline import Stage1HotPipeline

logger = logging.getLogger("a_share_hot_screener.batch_runner")

_PROGRESS_FILE = ".batch_progress.json"


# ════════════════════════════════════════════════════════
# 公共入口
# ════════════════════════════════════════════════════════

def run_batched(config: HotScreenerConfig) -> RunMetadata:
    """分批运行 pipeline 并合并结果.

    P2: 根据 config.global_pool 选择模式。
    - global_pool=True（默认）: 全局池模式，跨批次统一评分
    - global_pool=False: 旧模式，每批独立评分后合并 CSV
    """
    if config.global_pool:
        return _run_batched_global(config)
    else:
        return _run_batched_local(config)


# ════════════════════════════════════════════════════════
# P2: 全局池模式
# ════════════════════════════════════════════════════════

def _run_batched_global(config: HotScreenerConfig) -> RunMetadata:
    """全局池模式: 收集所有批次数据后统一评分.

    Phase 1: 逐批运行 pipeline.run_data_only() 收集 details
    Phase 2: 构建全局 ScoringPool
    Phase 3: 统一评分 + flags + 判定
    Phase 4: 输出
    """
    from a_share_hot_screener.date_utils import now_utc_iso
    from a_share_hot_screener.flags import compute_flags
    from a_share_hot_screener.logger import WarningsCollector
    from a_share_hot_screener.output import OutputWriter
    from a_share_hot_screener.scoring import ScoringPool
    from a_share_hot_screener.scoring_aggregator import apply_four_axis_scores
    from a_share_hot_screener.stage1_judge import judge_pass_stage1

    batch_size = config.batch_size
    all_codes = list(config.stock_codes)
    batches = _split(all_codes, batch_size)
    n_batches = len(batches)

    logger.info(
        "=== 批量运行(全局池) | %d 只股票 | batch_size=%d | %d 批 ===",
        len(all_codes), batch_size, n_batches,
    )

    start_ts = time.time()
    all_details: List[HotStockDetail] = []
    all_rejected: List[RejectedRecord] = []
    trade_date_used: str = ""

    # Phase 1: 逐批数据收集
    for idx, batch_codes in enumerate(batches):
        batch_dir = os.path.join(config.output_dir, f"batch_{idx + 1}")
        logger.info(
            "[batch %d/%d] 数据收集 — %d 只股票",
            idx + 1, n_batches, len(batch_codes),
        )
        batch_config = _make_batch_config(config, batch_codes, batch_dir)
        try:
            pipeline = Stage1HotPipeline(batch_config)
            details, rejected, tds = pipeline.run_data_only()
            all_details.extend(details)
            all_rejected.extend(rejected)
            if not trade_date_used:
                trade_date_used = tds
            hf_pass = sum(1 for d in details if d.passed_hard_filter)
            logger.info(
                "[batch %d/%d] 数据收集完成 — %d 只通过硬筛",
                idx + 1, n_batches, hf_pass,
            )
        except Exception as e:
            logger.error(
                "[batch %d/%d] 数据收集失败: %s", idx + 1, n_batches, e, exc_info=True,
            )
            continue

    # Phase 2: 构建全局 ScoringPool
    scoring_pool = ScoringPool.build(all_details)
    logger.info(
        "[global_pool] 构建完成: %d 只股票参与横截面",
        scoring_pool.stock_count,
    )

    # 基准 pool 合并
    used_baseline = False
    baseline_path = _resolve_baseline_path(config)
    if scoring_pool.stock_count < config.min_baseline_pool_size and baseline_path:
        baseline = ScoringPool.load_baseline(baseline_path)
        if baseline is not None and baseline.stock_count >= config.min_baseline_pool_size:
            scoring_pool = scoring_pool.merge_with_baseline(baseline)
            used_baseline = True

    # Phase 3: 统一评分
    warnings = WarningsCollector()
    warnings.add_global(
        f"[global_pool_mode] {n_batches} 批数据收集 → 全局统一评分"
        f"（{scoring_pool.stock_count}只参与横截面）"
    )

    for detail in all_details:
        if detail.passed_hard_filter:
            apply_four_axis_scores(detail, scoring_pool, config, warnings)

    # Step 7.5: flags
    for detail in all_details:
        try:
            flags = compute_flags(
                detail,
                enable_lhb_module=config.enable_lhb_module,
                enable_unlock_risk_module=config.enable_unlock_risk_module,
                enable_concept_heat_module=config.enable_concept_heat_module,
            )
            detail.flags = flags
        except Exception as e:
            logger.error("compute_flags(%s) 异常: %s", detail.code, e, exc_info=True)
            warnings.add(detail.code, f"[flags] compute_flags 异常: {e}")

    # Step 8: 判定
    for detail in all_details:
        rej = judge_pass_stage1(detail, config)
        if rej is not None:
            all_rejected.append(rej)

    # Step 9: 时序连续性（可选）
    trend_compare_enabled = False
    prev_run_date = ""
    prev_dir = config.prev_run_dir or config.output_dir
    if prev_dir:
        try:
            from a_share_hot_screener.trend_compare import compute_all_deltas, load_prev_run
            snapshot = load_prev_run(prev_dir, prev_run_date=config.prev_run_date)
            if snapshot is not None and snapshot.trade_date_used != trade_date_used:
                deltas = compute_all_deltas(all_details, snapshot)
                for detail in all_details:
                    delta = deltas.get(detail.code)
                    if delta is not None:
                        detail.trend_delta = delta.to_dict()
                trend_compare_enabled = True
                prev_run_date = snapshot.trade_date_used
                logger.info(
                    "[trend_compare] 时序对比完成: %d 只 vs %s",
                    len(deltas), prev_run_date,
                )
        except Exception as e:
            logger.debug("[trend_compare] 加载失败（全局池模式）: %s", e)

    # Phase 4: 输出
    total_elapsed = time.time() - start_ts
    os.makedirs(config.output_dir, exist_ok=True)
    writer = OutputWriter(output_dir=config.output_dir)

    metadata = _build_global_metadata(
        config=config,
        all_details=all_details,
        all_rejected=all_rejected,
        trade_date_used=trade_date_used,
        scoring_pool=scoring_pool,
        used_baseline=used_baseline,
        trend_compare_enabled=trend_compare_enabled,
        prev_run_date=prev_run_date,
        warnings=warnings,
        elapsed=total_elapsed,
        n_batches=n_batches,
    )
    writer.write_all(
        details=all_details,
        rejected=all_rejected,
        metadata=metadata,
    )

    # Step 11: 保存基准 pool
    if config.save_baseline_pool and scoring_pool.stock_count >= config.min_baseline_pool_size and not used_baseline:
        cache_dir = config.cache_dir or os.path.join(
            os.path.expanduser("~"), ".a_share_hot_screener", "cache"
        )
        save_path = os.path.join(cache_dir, "baseline_pool.json")
        scoring_pool.save_baseline(save_path)

    logger.info(
        "=== 批量运行(全局池)完成 | %d 批 | pass_stage1=%d | 总耗时=%.1fs ===",
        n_batches, metadata.pass_stage1_count, total_elapsed,
    )
    return metadata


def _build_global_metadata(
    *,
    config: HotScreenerConfig,
    all_details: List[HotStockDetail],
    all_rejected: List[RejectedRecord],
    trade_date_used: str,
    scoring_pool,
    used_baseline: bool,
    trend_compare_enabled: bool,
    prev_run_date: str,
    warnings,
    elapsed: float,
    n_batches: int,
) -> RunMetadata:
    """为全局池模式构建 RunMetadata."""
    from a_share_hot_screener.date_utils import now_utc_iso

    pass_count = sum(1 for d in all_details if d.pass_stage1)
    hf_pass = sum(1 for d in all_details if d.passed_hard_filter)
    hf_rej = sum(1 for r in all_rejected if r.reject_stage == "hard_filter")
    val_rej = sum(1 for r in all_rejected if r.reject_stage == "validation")
    dc_rej = sum(1 for r in all_rejected if r.reject_stage == "data_coverage")
    dc_pass = hf_pass - dc_rej

    hf_passed_details = [d for d in all_details if d.passed_hard_filter]
    if hf_passed_details:
        coverages = [d.data_coverage for d in hf_passed_details if d.data_coverage is not None]
        avg_coverage = round(sum(coverages) / len(coverages), 4) if coverages else None
    else:
        avg_coverage = None

    return RunMetadata(
        run_date=config.run_date_str,
        trade_date_used=trade_date_used,
        generated_at=now_utc_iso(),
        version="0.1.0",
        input_pool_size=len(config.stock_codes),
        input_stock_codes=list(config.stock_codes),
        valid_input_count=len(all_details) + val_rej + hf_rej,
        invalid_input_count=0,
        validation_passed=len(all_details) + hf_rej,
        validation_rejected=val_rej,
        hard_filter_passed=hf_pass,
        hard_filter_rejected=hf_rej,
        rejected_before_scoring_count=val_rej + hf_rej,
        data_coverage_passed=dc_pass,
        data_coverage_rejected=dc_rej,
        pass_stage1_count=pass_count,
        fail_stage1_count=max(0, len(all_details) - pass_count),
        scoring_pool_size=scoring_pool.stock_count,
        average_data_coverage=avg_coverage,
        min_data_coverage=config.min_data_coverage,
        min_price=config.min_price,
        min_amount_avg_5d=config.min_amount_avg_5d,
        min_float_market_cap=config.min_float_market_cap,
        min_trading_days=config.min_trading_days,
        include_beijing=config.include_beijing,
        enable_concept_heat_module=config.enable_concept_heat_module,
        enable_lhb_module=config.enable_lhb_module,
        enable_unlock_risk_module=config.enable_unlock_risk_module,
        max_workers=config.max_workers,
        pass_stage1_thresholds={
            "total_score": config.min_total_score,
            "hot_theme_score": config.min_hot_theme_score,
            "trend_flow_score": config.min_trend_flow_score,
            "liquidity_execution_score": config.min_liquidity_execution_score,
            "risk_control_score": config.min_risk_control_score,
            "data_coverage": config.min_data_coverage,
        },
        axis_weights={
            "hot_theme": config.axis_weight_hot_theme,
            "trend_flow": config.axis_weight_trend_flow,
            "liquidity_execution": config.axis_weight_liquidity_execution,
            "risk_control": config.axis_weight_risk_control,
        },
        elapsed_seconds=round(elapsed, 2),
        modules_enabled={
            "concept_heat": config.enable_concept_heat_module,
            "lhb": config.enable_lhb_module,
            "unlock_risk": config.enable_unlock_risk_module,
        },
        used_baseline_pool=used_baseline,
        baseline_pool_stock_count=(
            scoring_pool.stock_count if used_baseline else None
        ),
        global_warnings=warnings.global_warnings(),
        trend_compare_enabled=trend_compare_enabled,
        trend_compare_prev_run_date=prev_run_date,
    )


def _resolve_baseline_path(config: HotScreenerConfig) -> str:
    """解析 baseline pool 文件路径."""
    if config.baseline_pool_path:
        return config.baseline_pool_path if os.path.isfile(config.baseline_pool_path) else ""
    cache_dir = config.cache_dir or os.path.join(
        os.path.expanduser("~"), ".a_share_hot_screener", "cache"
    )
    auto_path = os.path.join(cache_dir, "baseline_pool.json")
    return auto_path if os.path.isfile(auto_path) else ""


# ════════════════════════════════════════════════════════
# 旧模式：每批独立评分 + CSV 合并
# ════════════════════════════════════════════════════════

def _run_batched_local(config: HotScreenerConfig) -> RunMetadata:
    """旧批量模式: 每批独立运行完整 pipeline，最后合并 CSV.

    使用 --no-global-pool 激活此模式。
    """
    batch_size = config.batch_size
    all_codes = list(config.stock_codes)
    batches = _split(all_codes, batch_size)
    n_batches = len(batches)

    logger.info(
        "=== 批量运行(本地池) | %d 只股票 | batch_size=%d | %d 批 ===",
        len(all_codes), batch_size, n_batches,
    )

    progress = _load_progress(config.output_dir) if config.resume else None
    completed: Dict[int, str] = {}  # batch_idx → batch_dir
    if progress and progress.get("run_date") == config.run_date_str:
        completed = {int(k): v for k, v in progress.get("completed", {}).items()}
        logger.info("[resume] 已完成 %d/%d 批", len(completed), n_batches)

    start_ts = time.time()
    batch_metadata: List[RunMetadata] = []
    batch_dirs: List[str] = []

    for idx, batch_codes in enumerate(batches):
        batch_dir = os.path.join(config.output_dir, f"batch_{idx + 1}")
        batch_dirs.append(batch_dir)

        if idx in completed:
            logger.info("[batch %d/%d] 已完成，跳过 (%d 只)", idx + 1, n_batches, len(batch_codes))
            # 读取已有 metadata
            meta_path = _find_metadata(batch_dir, config.run_date_str)
            if meta_path:
                batch_metadata.append(_load_metadata(meta_path))
            continue

        logger.info(
            "[batch %d/%d] 开始 — %d 只股票",
            idx + 1, n_batches, len(batch_codes),
        )

        batch_config = _make_batch_config(config, batch_codes, batch_dir)
        try:
            pipeline = Stage1HotPipeline(batch_config)
            meta = pipeline.run()
            batch_metadata.append(meta)
            completed[idx] = batch_dir
            _save_progress(config.output_dir, config.run_date_str, completed)
            logger.info(
                "[batch %d/%d] 完成 — pass_stage1=%d, elapsed=%.1fs",
                idx + 1, n_batches, meta.pass_stage1_count, meta.elapsed_seconds,
            )
        except Exception as e:
            logger.error(
                "[batch %d/%d] 失败: %s", idx + 1, n_batches, e, exc_info=True,
            )
            # 继续下一批（resume 可重跑失败批次）
            continue

    # 合并输出
    total_elapsed = time.time() - start_ts
    merged_meta = _merge_outputs(config, batch_dirs, batch_metadata, total_elapsed)

    # 清理进度文件（全部完成时）
    if len(completed) == n_batches:
        prog_path = os.path.join(config.output_dir, _PROGRESS_FILE)
        if os.path.exists(prog_path):
            os.remove(prog_path)

    logger.info(
        "=== 批量运行(本地池)完成 | %d 批 | pass_stage1=%d | 总耗时=%.1fs ===",
        n_batches, merged_meta.pass_stage1_count, total_elapsed,
    )
    return merged_meta


# ════════════════════════════════════════════════════════
# 内部工具
# ════════════════════════════════════════════════════════

def _split(codes: List[str], size: int) -> List[List[str]]:
    """将列表分成指定大小的块."""
    return [codes[i:i + size] for i in range(0, len(codes), size)]


def _make_batch_config(
    base: HotScreenerConfig,
    codes: List[str],
    output_dir: str,
) -> HotScreenerConfig:
    """为单批创建独立 config（共享 token/cache/阈值，替换 codes 和 output_dir）."""
    cfg = copy.copy(base)
    cfg.stock_codes = codes
    cfg.output_dir = output_dir
    # 批量模式下禁用内部分批（防递归）
    cfg.batch_size = 0
    cfg.resume = False
    return cfg


# ── 进度文件 ─────────────────────────────────────────────

def _save_progress(output_dir: str, run_date: str, completed: Dict[int, str]) -> None:
    path = os.path.join(output_dir, _PROGRESS_FILE)
    os.makedirs(output_dir, exist_ok=True)
    data = {
        "run_date": run_date,
        "completed": {str(k): v for k, v in completed.items()},
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _load_progress(output_dir: str) -> Optional[Dict]:
    path = os.path.join(output_dir, _PROGRESS_FILE)
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


# ── Metadata IO ──────────────────────────────────────────

def _find_metadata(batch_dir: str, run_date: str) -> Optional[str]:
    """在 batch_dir 中查找 metadata JSON."""
    candidate = os.path.join(batch_dir, f"{run_date}_stage1_hot_metadata.json")
    if os.path.isfile(candidate):
        return candidate
    # fallback：找任何 metadata 文件
    for f in os.listdir(batch_dir) if os.path.isdir(batch_dir) else []:
        if f.endswith("_metadata.json"):
            return os.path.join(batch_dir, f)
    return None


def _load_metadata(path: str) -> RunMetadata:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    valid_fields = {f.name for f in dataclasses.fields(RunMetadata)}
    filtered = {k: v for k, v in data.items() if k in valid_fields}
    return RunMetadata(**filtered)


# ── 合并 ─────────────────────────────────────────────────

def _merge_outputs(
    config: HotScreenerConfig,
    batch_dirs: List[str],
    batch_metadata: List[RunMetadata],
    total_elapsed: float,
) -> RunMetadata:
    """将多批输出合并为最终文件."""
    run_date = config.run_date_str
    out_dir = config.output_dir
    os.makedirs(out_dir, exist_ok=True)

    # 合并 CSV
    summary_path = os.path.join(out_dir, f"{run_date}_stage1_hot_summary.csv")
    detail_path = os.path.join(out_dir, f"{run_date}_stage1_hot_detail.csv")
    rejected_path = os.path.join(out_dir, f"{run_date}_stage1_hot_rejected.csv")

    _merge_csv(batch_dirs, run_date, "stage1_hot_summary.csv", summary_path, sort_key="input_order")
    _merge_csv(batch_dirs, run_date, "stage1_hot_detail.csv", detail_path)
    _merge_csv(batch_dirs, run_date, "stage1_hot_rejected.csv", rejected_path)

    # 合并 metadata
    merged_meta = _merge_metadata(config, batch_metadata, total_elapsed, len(batch_dirs))
    merged_meta.output_files = {
        "summary": summary_path,
        "detail": detail_path,
        "rejected": rejected_path,
        "metadata": os.path.join(out_dir, f"{run_date}_stage1_hot_metadata.json"),
    }

    meta_path = os.path.join(out_dir, f"{run_date}_stage1_hot_metadata.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(dataclasses.asdict(merged_meta), f, ensure_ascii=False, indent=2, default=str)

    logger.info(
        "合并完成: summary=%s, detail=%s, rejected=%s",
        summary_path, detail_path, rejected_path,
    )
    return merged_meta


def _merge_csv(
    batch_dirs: List[str],
    run_date: str,
    filename: str,
    output_path: str,
    sort_key: str = "",
) -> None:
    """合并多批 CSV 文件."""
    all_rows: List[Dict[str, Any]] = []
    fieldnames: Optional[List[str]] = None

    for bd in batch_dirs:
        csv_path = os.path.join(bd, f"{run_date}_{filename}")
        if not os.path.isfile(csv_path):
            continue
        with open(csv_path, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            if fieldnames is None:
                fieldnames = reader.fieldnames or []
            all_rows.extend(reader)

    if fieldnames is None:
        fieldnames = []

    if sort_key and all_rows and sort_key in all_rows[0]:
        try:
            all_rows.sort(key=lambda r: int(r.get(sort_key, 0)))
        except (ValueError, TypeError):
            pass

    with open(output_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_rows)


def _merge_metadata(
    config: HotScreenerConfig,
    metas: List[RunMetadata],
    total_elapsed: float,
    n_batches: int,
) -> RunMetadata:
    """聚合多批 metadata 为一个合并记录."""
    if not metas:
        return RunMetadata(run_date=config.run_date_str, elapsed_seconds=round(total_elapsed, 2))

    # 取第一批的基础字段
    base = metas[0]

    # 累加计数字段
    sum_fields = [
        "input_pool_size", "valid_input_count", "invalid_input_count",
        "validation_passed", "validation_rejected",
        "hard_filter_passed", "hard_filter_rejected",
        "rejected_before_scoring_count",
        "data_coverage_passed", "data_coverage_rejected",
        "pass_stage1_count", "fail_stage1_count",
        "scoring_pool_size",
    ]

    merged = RunMetadata(
        run_date=config.run_date_str,
        trade_date_used=base.trade_date_used,
        generated_at=base.generated_at,
        version=base.version,
        min_data_coverage=base.min_data_coverage,
        min_price=base.min_price,
        min_amount_avg_5d=base.min_amount_avg_5d,
        min_float_market_cap=base.min_float_market_cap,
        min_trading_days=base.min_trading_days,
        include_beijing=base.include_beijing,
        enable_concept_heat_module=base.enable_concept_heat_module,
        enable_lhb_module=base.enable_lhb_module,
        enable_unlock_risk_module=base.enable_unlock_risk_module,
        max_workers=base.max_workers,
        pass_stage1_thresholds=base.pass_stage1_thresholds,
        axis_weights=base.axis_weights,
        modules_enabled=base.modules_enabled,
        elapsed_seconds=round(total_elapsed, 2),
    )

    # 聚合所有输入代码
    all_codes: List[str] = []
    for m in metas:
        all_codes.extend(m.input_stock_codes)
    merged.input_stock_codes = all_codes

    # 累加计数
    for field_name in sum_fields:
        total = sum(getattr(m, field_name, 0) for m in metas)
        setattr(merged, field_name, total)

    # 加权平均 data_coverage
    coverages = [m.average_data_coverage for m in metas if m.average_data_coverage is not None]
    if coverages:
        merged.average_data_coverage = round(sum(coverages) / len(coverages), 4)

    # 全局 warnings 合并
    all_warnings: List[str] = []
    for m in metas:
        all_warnings.extend(getattr(m, "global_warnings", []))
    all_warnings.insert(0, f"[batch_mode] {n_batches} 批合并，每批独立评分（横截面仅含本批）")
    merged.global_warnings = all_warnings

    return merged
