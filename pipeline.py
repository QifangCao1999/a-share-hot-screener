"""Stage1HotPipeline – 短线热点第一阶段筛选主流程（薄编排层）.

流程：
  Step 0:   stock_codes 解析
  Step 0.5: 交易日历加载 + trade_date_used 确定
  Step 1:   Tushare daily_basic 全市场表加载（SpotUniverse）
  Step 2:   股票池校验（StockValidator） → 同步收集 spot 字段
  Step 2.5: 事件层批量加载（EventLayerLoader） → 产出 EventLayerContext
  Step 3~6: 每只股票并发处理 → stock_processor.process_single_stock()
  Step 7:   四轴评分 → scoring_aggregator.apply_four_axis_scores()
  Step 7.5: structured flags
  Step 8:   pass_stage1 判定 → stage1_judge.judge_pass_stage1()
  Step 9:   时序连续性对比
  Step 10:  输出四类文件
  Step 11:  保存基准 pool

P2 重构: 将 run() 拆分为 _run_data_collection() 和评分/输出阶段，
         新增 run_data_only() 供 batch_runner 全局池模式使用。
"""

from __future__ import annotations

import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional

from a_share_hot_screener.cache import LocalCache
from a_share_hot_screener.clients.tushare_client import TushareClient
from a_share_hot_screener.config import HotScreenerConfig
from a_share_hot_screener.date_utils import now_utc_iso
from a_share_hot_screener.event_layer import EventLayerContext, EventLayerLoader, EventLayerProcessor
from a_share_hot_screener.flags import compute_flags
from a_share_hot_screener.logger import WarningsCollector, setup_logger
from a_share_hot_screener.models import (
    HotStockDetail,
    RejectedRecord,
    RunMetadata,
    ValidatedHotStock,
)
from a_share_hot_screener.output import OutputWriter
from a_share_hot_screener.scoring import ScoringPool
from a_share_hot_screener.scoring_aggregator import apply_four_axis_scores
from a_share_hot_screener.stage1_judge import judge_pass_stage1
from a_share_hot_screener.stock_processor import process_single_stock
from a_share_hot_screener.stock_codes import parse_stock_codes
from a_share_hot_screener.trend_compare import compute_all_deltas, load_prev_run, PrevRunSnapshot
from a_share_hot_screener.trade_calendar import TradeCalendar
from a_share_hot_screener.validation import SpotUniverse, StockValidator

logger = logging.getLogger("a_share_hot_screener.pipeline")


class Stage1HotPipeline:
    """A 股短线第一阶段热点筛选 Pipeline."""

    def __init__(self, config: HotScreenerConfig) -> None:
        self.config = config
        self.warnings = WarningsCollector()

        self._cache = LocalCache(config.cache_dir)
        self._tushare = TushareClient(token=config.tushare_token, cache=self._cache)

        self._spot_universe = SpotUniverse(
            tushare=self._tushare,
            cache=self._cache,
            warnings=self.warnings,
        )
        self._trade_cal = TradeCalendar(config.run_date, cache=self._cache)
        self._writer = OutputWriter(output_dir=config.output_dir)

        self._details: List[HotStockDetail] = []
        self._rejected: List[RejectedRecord] = []
        self._event_ctx: Optional[EventLayerContext] = None

        # P2: 实例属性，由 _run_data_collection() 填充
        self._trade_date_str: str = ""
        self._valid_codes: List[str] = []
        self._invalid_codes: List[str] = []
        self._validated: List[ValidatedHotStock] = []

        setup_logger("a_share_hot_screener", level=config.log_level)

    # ── 数据收集阶段 (Steps 0-6.12) ──────────────────────

    def _run_data_collection(self) -> None:
        """Steps 0-6.12: 数据收集、校验、硬筛、特征提取、事件层填充.

        P2 重构: 从 run() 提取，供 run() 和 run_data_only() 复用。
        结果存储在 self._details, self._rejected 等实例属性中。
        """
        cfg = self.config

        logger.info(
            "=== A股短线热点筛选 开始 | run_date=%s | codes=%d只 ===",
            cfg.run_date_str, len(cfg.stock_codes),
        )

        # Step 0
        self._valid_codes, self._invalid_codes = parse_stock_codes(cfg.stock_codes)

        # Step 0.5: 交易日历加载 + trade_date_used
        logger.info("加载交易日历...")
        self._trade_cal.load(tushare_client=self._tushare)
        self._trade_cal.resolve()
        for w in self._trade_cal.get_warnings():
            self.warnings.add_global(w)
        trade_date_used = self._trade_cal.get_trade_date_used()
        self._trade_date_str = trade_date_used.isoformat()
        logger.info("trade_date_used=%s (fallback=%s)", self._trade_date_str, self._trade_cal.is_fallback)

        # Step 1
        logger.info("加载 Tushare daily_basic 全市场表...")
        self._spot_universe.load(cfg.run_date_str)

        # Step 2
        validator = StockValidator(
            spot_universe=self._spot_universe,
            warnings=self.warnings,
            include_beijing=cfg.include_beijing,
        )
        self._validated, pre_rejected = validator.validate(self._valid_codes, self._invalid_codes)
        self._rejected.extend(pre_rejected)
        logger.info("校验: 通过=%d, 淘汰=%d", len(self._validated), len(pre_rejected))

        # Step 2.5: 事件层批量加载
        logger.info("开始事件层批量加载...")
        event_loader = EventLayerLoader(
            tushare_client=self._tushare,
            cache=self._cache,
            run_date=trade_date_used,
            trade_dates=self._trade_cal._trade_dates,
            enable_lhb_module=cfg.enable_lhb_module,
            enable_concept_heat_module=cfg.enable_concept_heat_module,
        )
        self._event_ctx = event_loader.load()
        for w in self._event_ctx.global_warnings:
            self.warnings.add_global(w)

        # Step 2.7: 预加载风控数据（pledge/float/holdnum）到缓存
        # 串行执行，避免并发工作线程中的 API 限流竞争
        ts_codes_for_prefetch = [v.ts_code for v in self._validated if v.ts_code]
        if ts_codes_for_prefetch:
            self._tushare.prefetch_risk_data(ts_codes_for_prefetch, self._trade_date_str)

        # Step 2.8: 预加载资金流向数据（moneyflow/holdertrade/margin）— Session 22
        if ts_codes_for_prefetch:
            import datetime as _dt
            _td = _dt.date.fromisoformat(self._trade_date_str)
            _flow_start = (_td - _dt.timedelta(days=60)).strftime("%Y%m%d")
            _flow_end = self._trade_date_str.replace("-", "")
            self._tushare.prefetch_flow_data(
                ts_codes_for_prefetch,
                start_date=_flow_start,
                end_date=_flow_end,
            )

        # Step 2.9: 板块轮动分析（Session 22，需 6000积分，无权限自动跳过）
        self._sector_momentum: Dict[str, str] = {}  # industry_name → momentum_switch
        if cfg.enable_sector_rotation:
            self._run_sector_rotation(self._trade_date_str)

        # Step 3~6: 并发处理（price_features + hard_filters + event_layer）
        processed = self._run_concurrent(self._validated, self._trade_date_str)
        self._details.extend(processed)

        # Step 6.12: 填充板块轮动信号到 detail.flags（Session 22）
        if self._sector_momentum and self._event_ctx is not None:
            for detail in self._details:
                # 通过 industry_cons_map 查找股票所属行业，fallback 到 detail.industry
                ind_name = self._event_ctx.industry_cons_map.get(detail.code, detail.industry)
                momentum = self._sector_momentum.get(ind_name, "neutral")
                detail.flags["sector_momentum_signal"] = momentum

    # ── 公共接口 ─────────────────────────────────────────

    def run_data_only(self):
        """Run data collection phase only (Steps 0-6.12).

        P2: 用于 batch_runner 全局池模式，先收集各批数据再统一评分。

        Returns:
            (details, rejected, trade_date_str)
        """
        self._run_data_collection()
        return self._details, self._rejected, self._trade_date_str

    # ── 主入口 ────────────────────────────────────────────

    def run(self) -> RunMetadata:
        start_ts = time.time()

        # Phase 1: 数据收集 (Steps 0-6.12)
        self._run_data_collection()

        cfg = self.config

        # Step 7: 四轴评分
        scoring_pool = ScoringPool.build(self._details)
        logger.info(
            "[scoring] scoring_pool 构建完成: %d只股票参与横截面",
            scoring_pool.stock_count,
        )

        # Step 7a: 基准 pool 合并
        used_baseline = False
        baseline_path = self._resolve_baseline_pool_path()
        if scoring_pool.stock_count < cfg.min_baseline_pool_size and baseline_path:
            baseline = ScoringPool.load_baseline(baseline_path)
            if baseline is not None and baseline.stock_count >= cfg.min_baseline_pool_size:
                scoring_pool = scoring_pool.merge_with_baseline(baseline)
                used_baseline = True
                self.warnings.add_global(
                    f"[baseline_pool] scoring_pool 不足({scoring_pool.stock_count - baseline.stock_count}只)"
                    f"，已合并基准 pool({baseline.stock_count}只) → 合并后 {scoring_pool.stock_count}只"
                )
            else:
                logger.warning(
                    "[baseline] scoring_pool 不足(%d) 且无可用基准 pool (path=%s)",
                    scoring_pool.stock_count, baseline_path or "none",
                )

        for detail in self._details:
            if detail.passed_hard_filter:
                apply_four_axis_scores(detail, scoring_pool, self.config, self.warnings)

        # Step 7.5: structured flags
        for detail in self._details:
            try:
                flags = compute_flags(
                    detail,
                    enable_lhb_module=cfg.enable_lhb_module,
                    enable_unlock_risk_module=cfg.enable_unlock_risk_module,
                    enable_concept_heat_module=cfg.enable_concept_heat_module,
                )
                detail.flags = flags
            except Exception as e:
                logger.error("compute_flags(%s) 异常: %s", detail.code, e, exc_info=True)
                self.warnings.add(detail.code, f"[flags] compute_flags 异常: {e}")

        # Step 8: pass_stage1 判定
        for detail in self._details:
            rejected = judge_pass_stage1(detail, cfg)
            if rejected is not None:
                self._rejected.append(rejected)

        # Step 9: 时序连续性 / 趋势加速信号
        prev_snapshot = self._load_prev_run()
        trend_compare_enabled = prev_snapshot is not None
        if prev_snapshot is not None:
            deltas = compute_all_deltas(self._details, prev_snapshot)
            for detail in self._details:
                delta = deltas.get(detail.code)
                if delta is not None:
                    detail.trend_delta = delta.to_dict()
            logger.info(
                "[trend_compare] 时序对比完成: %d 只股票 vs 上次运行(%s)",
                len(deltas), prev_snapshot.trade_date_used,
            )

        # Step 10: 输出
        metadata = self._build_metadata(
            raw_input=cfg.stock_codes,
            valid_codes=self._valid_codes,
            invalid_codes=self._invalid_codes,
            validated=self._validated,
            trade_date_used=self._trade_date_str,
            elapsed=time.time() - start_ts,
            scoring_pool=scoring_pool,
            used_baseline=used_baseline,
            trend_compare_enabled=trend_compare_enabled,
            prev_run_date=prev_snapshot.trade_date_used if prev_snapshot else "",
        )
        self._writer.write_all(
            details=self._details,
            rejected=self._rejected,
            metadata=metadata,
        )

        # Step 11: 保存基准 pool
        if cfg.save_baseline_pool and scoring_pool.stock_count >= cfg.min_baseline_pool_size and not used_baseline:
            save_path = self._resolve_baseline_pool_path(for_save=True)
            if save_path:
                scoring_pool.save_baseline(save_path)

        logger.info(
            "=== 完成 | 耗时=%.1fs | 通过硬筛=%d | rejected合计=%d ===",
            metadata.elapsed_seconds,
            metadata.hard_filter_passed,
            len(self._rejected),
        )
        return metadata

    # ── 时序连续性加载 ─────────────────────────────────────

    def _load_prev_run(self) -> "Optional[PrevRunSnapshot]":
        cfg = self.config
        prev_dir = cfg.prev_run_dir
        if not prev_dir:
            prev_dir = cfg.output_dir
        if not prev_dir:
            return None
        try:
            snapshot = load_prev_run(prev_dir, prev_run_date=cfg.prev_run_date)
            if snapshot is None:
                return None
            trade_date_str = self._trade_cal.get_trade_date_used().isoformat()
            if snapshot.trade_date_used == trade_date_str:
                logger.info(
                    "[trend_compare] 上次运行日期与当前相同(%s)，跳过时序对比",
                    trade_date_str,
                )
                return None
            return snapshot
        except Exception as e:
            logger.warning("[trend_compare] 加载上次运行失败: %s", e)
            self.warnings.add_global(f"[trend_compare] 加载失败: {e}")
            return None

    # ── 基准 pool 路径解析 ─────────────────────────────────

    def _resolve_baseline_pool_path(self, for_save: bool = False) -> str:
        cfg = self.config
        if cfg.baseline_pool_path:
            if for_save:
                return cfg.baseline_pool_path
            return cfg.baseline_pool_path if os.path.isfile(cfg.baseline_pool_path) else ""
        auto_path = os.path.join(cfg.cache_dir, "baseline_pool.json")
        if for_save:
            return auto_path
        return auto_path if os.path.isfile(auto_path) else ""

    # ── 板块轮动 (Session 22) ───────────────────────────────

    def _run_sector_rotation(self, trade_date_str: str) -> None:
        """Step 2.9: 板块轮动分析 + 输出 sector_heat.csv + 构建 momentum 查找表."""
        import datetime as _dt
        from a_share_hot_screener.sector_rotation import SectorRotationAnalyzer

        try:
            run_date = _dt.date.fromisoformat(trade_date_str)
            analyzer = SectorRotationAnalyzer(
                tushare_client=self._tushare,
                cache=self._cache,
                run_date=run_date,
                trade_dates=self._trade_cal._trade_dates,
            )
            rows = analyzer.analyze()
            if not rows:
                logger.warning("[sector_rotation] 无数据（可能无权限），跳过")
                return

            # 输出 sector_heat.csv
            csv_path = os.path.join(
                self.config.output_dir,
                f"{trade_date_str}_sector_heat.csv",
            )
            analyzer.to_csv(rows, csv_path)

            # 构建行业名称 → momentum_switch 查找表（仅取行业类型 I）
            for row in rows:
                if row.type == "I" and row.name:
                    self._sector_momentum[row.name] = row.momentum_switch

            logger.info(
                "[sector_rotation] 完成: %d 个板块, 行业动量查找表 %d 项",
                len(rows), len(self._sector_momentum),
            )
        except Exception as e:
            logger.warning("[sector_rotation] 分析失败: %s", e)
            self.warnings.add_global(f"[sector_rotation] 分析失败: {e}")

    # ── 并发调度 ─────────────────────────────────────────

    def _make_event_processor(self) -> Optional[EventLayerProcessor]:
        if self._event_ctx is None:
            return None
        cfg = self.config
        return EventLayerProcessor(
            ctx=self._event_ctx,
            tushare_client=self._tushare,
            cache=self._cache,
            enable_lhb=cfg.enable_lhb_module,
            enable_concept=cfg.enable_concept_heat_module,
        )

    def _run_concurrent(
        self,
        validated: List[ValidatedHotStock],
        trade_date_str: str,
    ) -> List[HotStockDetail]:
        if not validated:
            return []
        max_w = min(self.config.max_workers, len(validated))
        results: List[HotStockDetail] = []
        event_proc = self._make_event_processor()
        with ThreadPoolExecutor(max_workers=max_w) as ex:
            fut_map = {
                ex.submit(
                    process_single_stock,
                    vs, trade_date_str, self.config,
                    self._tushare, self._spot_universe, self._trade_cal,
                    event_proc,
                ): vs.code
                for vs in validated
            }
            for fut in as_completed(fut_map):
                code = fut_map[fut]
                try:
                    d = fut.result()
                    if d is not None:
                        # 合并 warnings 到全局收集器
                        for w in d.warnings:
                            self.warnings.add(code, w)
                        d.warnings = self.warnings.get(code)
                        results.append(d)
                except Exception as e:
                    logger.error("处理 %s 异常: %s", code, e, exc_info=True)
                    self._rejected.append(RejectedRecord(
                        code=code,
                        reject_stage="pipeline_error",
                        reject_reason="unhandled_exception",
                        reject_detail=str(e)[:200],
                    ))
        results.sort(key=lambda d: d.input_order)
        return results

    # ── metadata ─────────────────────────────────────────

    def _build_metadata(
        self,
        raw_input, valid_codes, invalid_codes, validated,
        trade_date_used, elapsed,
        scoring_pool: Optional["ScoringPool"] = None,
        used_baseline: bool = False,
        trend_compare_enabled: bool = False,
        prev_run_date: str = "",
    ) -> RunMetadata:
        cfg = self.config
        pass_count = sum(1 for d in self._details if d.pass_stage1)
        hf_pass = sum(1 for d in self._details if d.passed_hard_filter)
        hf_rej = sum(1 for r in self._rejected if r.reject_stage == "hard_filter")
        val_rej = sum(1 for r in self._rejected if r.reject_stage == "validation")
        dc_rej = sum(1 for r in self._rejected if r.reject_stage == "data_coverage")
        dc_pass = hf_pass - dc_rej

        hf_passed_details = [d for d in self._details if d.passed_hard_filter]
        if hf_passed_details:
            coverages = [d.data_coverage for d in hf_passed_details if d.data_coverage is not None]
            avg_coverage = round(sum(coverages) / len(coverages), 4) if coverages else None
        else:
            avg_coverage = None

        return RunMetadata(
            run_date=cfg.run_date_str,
            trade_date_used=trade_date_used,
            generated_at=now_utc_iso(),
            version="0.1.0",
            input_pool_size=len(raw_input),
            input_stock_codes=list(raw_input),
            valid_input_count=len(valid_codes),
            invalid_input_count=len(invalid_codes),
            validation_passed=len(validated),
            validation_rejected=val_rej,
            hard_filter_passed=hf_pass,
            hard_filter_rejected=hf_rej,
            rejected_before_scoring_count=val_rej + hf_rej,
            data_coverage_passed=dc_pass,
            data_coverage_rejected=dc_rej,
            pass_stage1_count=pass_count,
            fail_stage1_count=max(0, len(self._details) - pass_count),
            scoring_pool_size=scoring_pool.stock_count if scoring_pool else hf_pass,
            average_data_coverage=avg_coverage,
            cache_hit_rate=self._cache.get_hit_rate(),
            min_data_coverage=cfg.min_data_coverage,
            min_price=cfg.min_price,
            min_amount_avg_5d=cfg.min_amount_avg_5d,
            min_float_market_cap=cfg.min_float_market_cap,
            min_trading_days=cfg.min_trading_days,
            include_beijing=cfg.include_beijing,
            enable_concept_heat_module=cfg.enable_concept_heat_module,
            enable_lhb_module=cfg.enable_lhb_module,
            enable_unlock_risk_module=cfg.enable_unlock_risk_module,
            max_workers=cfg.max_workers,
            pass_stage1_thresholds={
                "total_score": cfg.min_total_score,
                "hot_theme_score": cfg.min_hot_theme_score,
                "trend_flow_score": cfg.min_trend_flow_score,
                "liquidity_execution_score": cfg.min_liquidity_execution_score,
                "risk_control_score": cfg.min_risk_control_score,
                "data_coverage": cfg.min_data_coverage,
            },
            axis_weights={
                "hot_theme": cfg.axis_weight_hot_theme,
                "trend_flow": cfg.axis_weight_trend_flow,
                "liquidity_execution": cfg.axis_weight_liquidity_execution,
                "risk_control": cfg.axis_weight_risk_control,
            },
            elapsed_seconds=round(elapsed, 2),
            modules_enabled={
                "concept_heat": cfg.enable_concept_heat_module,
                "lhb": cfg.enable_lhb_module,
                "unlock_risk": cfg.enable_unlock_risk_module,
            },
            used_baseline_pool=used_baseline,
            baseline_pool_stock_count=(
                scoring_pool.stock_count if used_baseline and scoring_pool else None
            ),
            global_warnings=self.warnings.global_warnings(),
            trend_compare_enabled=trend_compare_enabled,
            trend_compare_prev_run_date=prev_run_date,
        )
