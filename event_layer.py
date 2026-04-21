"""事件层模块 — Tushare 数据源版.

涨停池、强势股池、龙虎榜、行业/概念热度。

设计原则:
  1. 批量拉取优先：所有事件表按日期区间一次性拉取，再按股票过滤。
  2. 降级策略明确：接口不可得时返回 None + warnings。
  3. 不做分数整合（交给 scorers）。
  4. strict as-of：只使用 run_date 及之前已知数据。

数据源：
  涨停池      → limit_list_d(trade_date, limit_type='U')       [5000pt]
  强势股池    → limit_list_d where limit_times > 1（衍生）       [5000pt]
  龙虎榜      → top_list(trade_date)                             [2000pt]
  行业热度    → ths_daily(行业指数行情) + index_member_all       [6000pt]
  概念热度    → ths_index(概念列表) + ths_member + ths_daily      [6000pt]

降级策略（6000pt 不可用时）:
  行业热度 → industry_heat_mode="tushare_200_degraded"，使用 stock_basic 中的行业名，无分位排名
  概念热度 → concept_heat_mode="tushare_200_unavailable"，跳过整模块
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import os
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd

logger = logging.getLogger("a_share_hot_screener.event_layer")


# ════════════════════════════════════════════════════════
# 数据结构
# ════════════════════════════════════════════════════════

@dataclass
class EventLayerResult:
    """单只股票的事件层中间结果."""

    limit_up_count_5d: Optional[int] = None
    limit_up_count_10d: Optional[int] = None
    max_consecutive_limit_up_10d: Optional[int] = None
    limit_up_source: str = ""

    strong_pool_entry_count_3d: Optional[int] = None
    strong_pool_source: str = ""

    lhb_count_20d: Optional[int] = None
    lhb_on_board: Optional[bool] = None
    lhb_source: str = ""

    industry_heat_pctile_5d: Optional[float] = None
    industry_pct_5d: Optional[float] = None
    industry_name: str = ""
    industry_heat_source: str = ""

    concept_heat_pctile_5d: Optional[float] = None
    concept_names: List[str] = field(default_factory=list)
    advanced_concept_module_available: bool = False
    concept_heat_source: str = ""

    warnings: List[str] = field(default_factory=list)


@dataclass
class EventLayerContext:
    """批量拉取的事件层共享上下文."""

    zt_pool_map: Dict[str, Set[str]] = field(default_factory=dict)
    zt_pool_available: bool = False

    strong_pool_map: Dict[str, Set[str]] = field(default_factory=dict)
    strong_pool_available: bool = False

    lhb_df: Optional[pd.DataFrame] = None
    lhb_available: bool = False

    industry_hist_map: Dict[str, float] = field(default_factory=dict)
    industry_rank_pctile: Dict[str, float] = field(default_factory=dict)
    industry_spot_df: Optional[pd.DataFrame] = None
    industry_heat_mode: str = "none"
    industry_snapshot_days_used: int = 0

    concept_hist_map: Dict[str, float] = field(default_factory=dict)
    concept_rank_pctile: Dict[str, float] = field(default_factory=dict)
    concept_spot_df: Optional[pd.DataFrame] = None
    concept_heat_mode: str = "none"

    concept_cons_map: Dict[str, List[str]] = field(default_factory=dict)
    industry_cons_map: Dict[str, str] = field(default_factory=dict)

    global_warnings: List[str] = field(default_factory=list)

    _loaded: bool = False
    _lock: threading.Lock = field(default_factory=threading.Lock)


# ════════════════════════════════════════════════════════
# EventLayerLoader
# ════════════════════════════════════════════════════════

class EventLayerLoader:
    """批量加载事件层原始数据（Tushare 数据源）."""

    def __init__(
        self,
        tushare_client,
        cache,
        run_date: dt.date,
        trade_dates: List[dt.date],
        enable_lhb_module: bool = True,
        enable_concept_heat_module: bool = False,
    ) -> None:
        self._ts = tushare_client
        self._cache = cache
        self._run_date = run_date
        self._trade_dates = trade_dates
        self._enable_lhb = enable_lhb_module
        self._enable_concept = enable_concept_heat_module

    def load(self) -> EventLayerContext:
        ctx = EventLayerContext()
        logger.info("[event_layer] 开始批量加载事件层数据（Tushare）...")

        dates_10d = self._prev_n_trade_dates(self._run_date, n=10)
        dates_5d = dates_10d[-5:] if len(dates_10d) >= 5 else dates_10d
        dates_3d = dates_10d[-3:] if len(dates_10d) >= 3 else dates_10d
        dates_20d = self._prev_n_trade_dates(self._run_date, n=20)

        self._load_zt_pool(ctx, dates_10d)
        self._load_strong_pool(ctx, dates_3d)

        if self._enable_lhb:
            self._load_lhb(ctx, dates_20d)

        # 行业热度：尝试完整版（ths_daily 6000pt），失败则降级
        if not self._try_load_industry_heat_full(ctx, dates_5d):
            self._load_industry_heat_degraded(ctx)

        # 概念热度：尝试完整版，失败则跳过
        if self._enable_concept:
            if not self._try_load_concept_heat_full(ctx, dates_5d):
                msg = ("[event_layer] GLOBAL: concept_heat_unavailable: "
                       "同花顺概念数据需6000积分，概念热度模块跳过。升级到600元/年可解锁。")
                ctx.global_warnings.append(msg)
                logger.warning(msg)
                ctx.concept_heat_mode = "tushare_200_unavailable"

        ctx._loaded = True
        logger.info(
            "[event_layer] 加载完成 | zt_pool=%s | strong_pool=%s | lhb=%s | "
            "industry=%s | concept=%s | warnings=%d",
            ctx.zt_pool_available, ctx.strong_pool_available,
            ctx.lhb_available, ctx.industry_heat_mode,
            ctx.concept_heat_mode, len(ctx.global_warnings),
        )
        return ctx

    # ── 涨停池（Tushare limit_list_d）────────────────────

    def _load_zt_pool(self, ctx: EventLayerContext, trade_dates: List[dt.date]) -> None:
        from concurrent.futures import ThreadPoolExecutor, as_completed

        date_strs = [d.strftime("%Y%m%d") for d in trade_dates]
        any_success = False

        # 并发拉取各天涨停池（各天无依赖，可并行）
        with ThreadPoolExecutor(max_workers=min(5, len(date_strs))) as ex:
            fut_map = {
                ex.submit(self._fetch_zt_pool_one_day, ds): ds
                for ds in date_strs
            }
            for fut in as_completed(fut_map):
                ds = fut_map[fut]
                try:
                    codes = fut.result()
                    if codes is not None:
                        ctx.zt_pool_map[ds] = codes
                        any_success = True
                    else:
                        msg = f"[event_layer] GLOBAL: limit_up_pool_unavailable: limit_list_d({ds}) 失败"
                        ctx.global_warnings.append(msg)
                        logger.warning(msg)
                except Exception as e:
                    msg = f"[event_layer] GLOBAL: limit_up_pool_error: limit_list_d({ds}) 异常: {e}"
                    ctx.global_warnings.append(msg)
                    logger.warning(msg)

        ctx.zt_pool_available = any_success
        if not any_success:
            msg = "[event_layer] GLOBAL: limit_up_pool_all_failed: 近10个交易日涨停池全部加载失败"
            ctx.global_warnings.append(msg)

    def _fetch_zt_pool_one_day(self, date_str: str) -> Optional[Set[str]]:
        """拉取单日涨停池（Tushare limit_list_d），返回6位代码集合."""
        cache_key = f"zt_pool_{date_str}"
        if self._cache:
            cached = self._cache.get("event_zt_pool", cache_key, ttl=86400 * 3)
            if cached is not None:
                return set(cached)

        df = self._ts.get_limit_list(date_str, limit_type="U")
        if df is None or df.empty:
            return None

        codes: Set[str] = set()
        for ts_code in df["ts_code"].dropna():
            code = str(ts_code).split(".")[0]
            if len(code) == 6 and code.isdigit():
                codes.add(code)

        if self._cache and codes:
            self._cache.put("event_zt_pool", cache_key, list(codes), ttl=86400 * 3)

        logger.info("[event_layer] zt_pool(%s): %d 只涨停股", date_str, len(codes))
        return codes

    # ── 强势股池（从 limit_list_d 衍生，limit_times > 1）─

    def _load_strong_pool(self, ctx: EventLayerContext, trade_dates: List[dt.date]) -> None:
        """从涨停池数据衍生强势股池：limit_times > 1 = 连板股 = 强势股."""
        any_success = False
        for d in trade_dates:
            date_str = d.strftime("%Y%m%d")
            codes = self._fetch_strong_pool_one_day(date_str)
            if codes is not None:
                ctx.strong_pool_map[date_str] = codes
                any_success = True

        ctx.strong_pool_available = any_success

    def _fetch_strong_pool_one_day(self, date_str: str) -> Optional[Set[str]]:
        """从 limit_list_d 衍生强势股（limit_times > 1）."""
        cache_key = f"strong_pool_{date_str}"
        if self._cache:
            cached = self._cache.get("event_strong_pool", cache_key, ttl=86400 * 3)
            if cached is not None:
                return set(cached)

        df = self._ts.get_limit_list(date_str, limit_type="U")
        if df is None or df.empty:
            return None

        if "limit_times" not in df.columns:
            return None

        strong = df[df["limit_times"] > 1]
        codes: Set[str] = set()
        for ts_code in strong["ts_code"].dropna():
            code = str(ts_code).split(".")[0]
            if len(code) == 6 and code.isdigit():
                codes.add(code)

        if self._cache:
            self._cache.put("event_strong_pool", cache_key, list(codes), ttl=86400 * 3)

        logger.info("[event_layer] strong_pool(%s): %d 只连板股", date_str, len(codes))
        return codes

    # ── 龙虎榜（Tushare top_list）────────────────────────

    def _load_lhb(self, ctx: EventLayerContext, trade_dates: List[dt.date]) -> None:
        """逐日拉取龙虎榜并合并."""
        if not trade_dates:
            return

        start_date = trade_dates[0].strftime("%Y%m%d")
        end_date = trade_dates[-1].strftime("%Y%m%d")
        cache_key = f"lhb_{start_date}_{end_date}"

        if self._cache:
            cached = self._cache.get("event_lhb", cache_key, ttl=86400 * 2)
            if cached is not None:
                try:
                    ctx.lhb_df = pd.DataFrame(cached)
                    ctx.lhb_available = not ctx.lhb_df.empty
                    logger.info("[event_layer] lhb 从缓存加载: %d 行", len(ctx.lhb_df))
                    return
                except Exception:
                    pass

        frames = []
        for d in trade_dates:
            date_str = d.strftime("%Y%m%d")
            df = self._ts.get_top_list(date_str)
            if df is not None and not df.empty:
                frames.append(df)

        if not frames:
            msg = f"[event_layer] GLOBAL: lhb_empty: top_list({start_date}~{end_date}) 全部返回空"
            ctx.global_warnings.append(msg)
            logger.warning(msg)
            return

        combined = pd.concat(frames, ignore_index=True)
        ctx.lhb_df = combined
        ctx.lhb_available = True

        if self._cache:
            self._cache.put("event_lhb", cache_key, combined.to_dict("records"), ttl=86400 * 2)
        logger.info("[event_layer] lhb 加载完成: %d 行", len(combined))

    # ── 行业热度（完整版 ths_daily 6000pt）────────────────

    def _try_load_industry_heat_full(
        self, ctx: EventLayerContext, dates_5d: List[dt.date],
    ) -> bool:
        """尝试用 ths_daily 加载完整行业热度。成功返回 True，无权限/失败返回 False。"""
        if not dates_5d:
            return False

        try:
            # 1. 拉取同花顺行业指数列表
            idx_df = self._ts.get_ths_index(exchange="A", type_="I")
            if idx_df is None or idx_df.empty:
                logger.info("[event_layer] ths_index(I) 返回空，行业热度降级")
                return False

            # 2. 拉取 dates_5d 每天的行业指数行情
            daily_frames = []
            for d in dates_5d:
                date_str = d.strftime("%Y%m%d")
                df = self._ts.get_ths_daily(trade_date=date_str)
                if df is not None and not df.empty:
                    daily_frames.append(df)

            if len(daily_frames) < 2:
                logger.warning(
                    "[event_layer] ths_daily 只拿到 %d 天数据，行业热度降级",
                    len(daily_frames),
                )
                return False

            all_daily = pd.concat(daily_frames, ignore_index=True)

            # 3. 筛选行业指数（只保留 idx_df 中 type='I' 的 ts_code）
            industry_codes = set(idx_df["ts_code"].dropna())
            industry_daily = all_daily[all_daily["ts_code"].isin(industry_codes)].copy()
            if industry_daily.empty:
                return False

            # 4. 计算每个行业指数的 5 日累计涨幅
            code_to_name = dict(zip(idx_df["ts_code"], idx_df["name"]))
            pct_map = _calc_period_pct_change(industry_daily, industry_codes)

            if not pct_map:
                return False

            # 5. 排名 → 百分位
            sorted_items = sorted(pct_map.items(), key=lambda x: x[1])
            n = len(sorted_items)
            for rank_0, (ts_code, pct) in enumerate(sorted_items):
                name = code_to_name.get(ts_code, ts_code)
                pctile = (rank_0 + 0.5) / n  # 中点百分位
                ctx.industry_rank_pctile[name] = pctile
                ctx.industry_hist_map[name] = pct

            # 6. 建立股票→行业映射（使用 index_member_all）
            member_df = self._ts.get_index_member_all()
            if member_df is not None and not member_df.empty:
                current = member_df[
                    member_df["is_new"].astype(str).str.upper() == "Y"
                ] if "is_new" in member_df.columns else member_df
                for _, row in current.iterrows():
                    stock_ts = str(row.get("ts_code", ""))
                    code6 = stock_ts.split(".")[0] if "." in stock_ts else stock_ts
                    if len(code6) == 6 and code6.isdigit():
                        l1_name = str(row.get("l1_name", ""))
                        if l1_name:
                            ctx.industry_cons_map[code6] = l1_name

            ctx.industry_heat_mode = "ths_daily_full"
            ctx.industry_snapshot_days_used = len(daily_frames)
            logger.info(
                "[event_layer] 行业热度完整版加载成功: %d 个行业, %d 只股票映射, %d 天数据",
                len(ctx.industry_rank_pctile), len(ctx.industry_cons_map),
                len(daily_frames),
            )
            return True

        except Exception as e:
            logger.warning("[event_layer] 行业热度完整版加载失败，降级: %s", e)
            return False

    # ── 概念热度（完整版 ths_index/ths_member/ths_daily 6000pt）──

    def _try_load_concept_heat_full(
        self, ctx: EventLayerContext, dates_5d: List[dt.date],
    ) -> bool:
        """尝试用 ths_index + ths_member + ths_daily 加载完整概念热度。成功返回 True。"""
        if not dates_5d:
            return False

        try:
            # 1. 拉取同花顺概念指数列表
            idx_df = self._ts.get_ths_index(exchange="A", type_="N")
            if idx_df is None or idx_df.empty:
                logger.info("[event_layer] ths_index(N) 返回空，概念热度不可用")
                return False

            # 2. 拉取 dates_5d 每天行情（复用行业热度已拉的 ths_daily—如果同天拉过会命中缓存）
            daily_frames = []
            for d in dates_5d:
                date_str = d.strftime("%Y%m%d")
                df = self._ts.get_ths_daily(trade_date=date_str)
                if df is not None and not df.empty:
                    daily_frames.append(df)

            if len(daily_frames) < 2:
                logger.warning(
                    "[event_layer] ths_daily 只拿到 %d 天数据，概念热度不可用",
                    len(daily_frames),
                )
                return False

            all_daily = pd.concat(daily_frames, ignore_index=True)

            # 3. 筛选概念指数
            concept_codes = set(idx_df["ts_code"].dropna())
            concept_daily = all_daily[all_daily["ts_code"].isin(concept_codes)].copy()
            if concept_daily.empty:
                return False

            code_to_name = dict(zip(idx_df["ts_code"], idx_df["name"]))

            # 4. 计算每个概念 5 日累计涨幅
            pct_map = _calc_period_pct_change(concept_daily, concept_codes)
            if not pct_map:
                return False

            # 5. 排名 → 百分位
            sorted_items = sorted(pct_map.items(), key=lambda x: x[1])
            n = len(sorted_items)
            for rank_0, (ts_code, pct) in enumerate(sorted_items):
                name = code_to_name.get(ts_code, ts_code)
                pctile = (rank_0 + 0.5) / n
                ctx.concept_rank_pctile[name] = pctile
                ctx.concept_hist_map[name] = pct

            # 6. 存储 ts_code→name 映射到 concept_spot_df 供 processor 反查
            ctx.concept_spot_df = idx_df[["ts_code", "name"]].copy()
            ctx.concept_heat_mode = "ths_daily_full"
            logger.info(
                "[event_layer] 概念热度完整版加载成功: %d 个概念",
                len(ctx.concept_rank_pctile),
            )
            return True

        except Exception as e:
            logger.warning("[event_layer] 概念热度完整版加载失败: %s", e)
            return False

    # ── 行业热度（200元档降级）──────────────────────────

    def _load_industry_heat_degraded(self, ctx: EventLayerContext) -> None:
        """行业热度降级模式.

        200元档（5000积分）无 ths_daily（同花顺行业指数行情），
        无法计算行业间的横向排名。

        降级方案：使用 stock_basic 中的申万行业（industry 字段）做行业归属，
        但不计算行业分位排名，设 industry_heat_mode = "tushare_200_degraded"。
        """
        msg = ("[event_layer] GLOBAL: industry_heat_degraded: "
               "行业指数行情需6000积分（当前5000），行业热度分位排名不可用。"
               "升级到600元/年可解锁 ths_daily。")
        ctx.global_warnings.append(msg)
        logger.info(msg)
        ctx.industry_heat_mode = "tushare_200_degraded"

    # ── 辅助 ──────────────────────────────────────────────

    def _prev_n_trade_dates(self, ref: dt.date, n: int) -> List[dt.date]:
        if not self._trade_dates:
            result = []
            d = ref
            while len(result) < n:
                if d.weekday() < 5:
                    result.append(d)
                d -= dt.timedelta(days=1)
            return list(reversed(result))

        idx = _bisect_right(self._trade_dates, ref)
        start = max(0, idx - n)
        return self._trade_dates[start:idx]


# ════════════════════════════════════════════════════════
# EventLayerProcessor
# ════════════════════════════════════════════════════════

class EventLayerProcessor:
    """从 EventLayerContext 提取单只股票的事件层字段."""

    def __init__(
        self,
        ctx: EventLayerContext,
        tushare_client,
        cache,
        enable_lhb: bool = True,
        enable_concept: bool = False,
    ) -> None:
        self._ctx = ctx
        self._ts = tushare_client
        self._cache = cache
        self._enable_lhb = enable_lhb
        self._enable_concept = enable_concept

    def process(
        self,
        code: str,
        industry: str,
        run_date: dt.date,
        trade_dates_10d: List[dt.date],
        trade_dates_3d: List[dt.date],
        trade_dates_20d: List[dt.date],
        limit_board_count_5d_proxy: Optional[int] = None,
    ) -> EventLayerResult:
        result = EventLayerResult()
        ctx = self._ctx

        self._fill_limit_up(result, code, trade_dates_10d, limit_board_count_5d_proxy)
        self._fill_strong_pool(result, code, trade_dates_3d)

        if self._enable_lhb:
            self._fill_lhb(result, code, trade_dates_20d)

        self._fill_industry_heat(result, code, industry)

        if self._enable_concept:
            self._fill_concept_heat(result, code)

        return result

    def _fill_limit_up(self, result, code, trade_dates_10d, proxy):
        ctx = self._ctx
        if not ctx.zt_pool_available:
            result.limit_up_count_5d = None
            result.limit_up_count_10d = None
            result.max_consecutive_limit_up_10d = None
            result.limit_up_source = "none"
            result.warnings.append(
                f"[event_layer] {code}: limit_up_pool_unavailable"
            )
            return

        dates_5d = trade_dates_10d[-5:] if len(trade_dates_10d) >= 5 else trade_dates_10d
        count_5d = 0
        count_10d = 0
        hit_flags: List[bool] = []

        for d in trade_dates_10d:
            date_str = d.strftime("%Y%m%d")
            pool = ctx.zt_pool_map.get(date_str)
            if pool is None:
                hit_flags.append(False)
                continue
            hit = code in pool
            hit_flags.append(hit)
            if hit:
                count_10d += 1
                if d in dates_5d:
                    count_5d += 1

        result.limit_up_count_5d = count_5d
        result.limit_up_count_10d = count_10d
        result.max_consecutive_limit_up_10d = _max_consecutive(hit_flags)
        result.limit_up_source = "tushare_limit_list_d"

    def _fill_strong_pool(self, result, code, trade_dates_3d):
        ctx = self._ctx
        if not ctx.strong_pool_available:
            if result.limit_up_count_5d is not None:
                fallback = min(result.limit_up_count_5d, 3)
                result.strong_pool_entry_count_3d = fallback
                result.strong_pool_source = "fallback_zt_count"
            else:
                result.strong_pool_entry_count_3d = None
                result.strong_pool_source = "none"
            return

        count = 0
        for d in trade_dates_3d:
            date_str = d.strftime("%Y%m%d")
            pool = ctx.strong_pool_map.get(date_str)
            if pool is not None and code in pool:
                count += 1

        result.strong_pool_entry_count_3d = count
        result.strong_pool_source = "tushare_limit_list_d_derived"

    def _fill_lhb(self, result, code, trade_dates_20d):
        ctx = self._ctx
        if not ctx.lhb_available or ctx.lhb_df is None:
            result.lhb_count_20d = None
            result.lhb_on_board = None
            result.lhb_source = "none"
            result.warnings.append(
                f"[event_layer] {code}: lhb_unavailable"
            )
            return

        df = ctx.lhb_df
        # Tushare top_list 字段: ts_code, trade_date
        if "ts_code" not in df.columns:
            result.lhb_count_20d = None
            result.lhb_on_board = None
            result.lhb_source = "none"
            return

        try:
            # 匹配代码：ts_code 格式 000001.SZ，提取前6位
            mask = df["ts_code"].astype(str).str[:6] == code
            sub = df[mask]

            if trade_dates_20d:
                start_str = trade_dates_20d[0].strftime("%Y%m%d")
                end_str = trade_dates_20d[-1].strftime("%Y%m%d")
                if "trade_date" in sub.columns:
                    td = sub["trade_date"].astype(str)
                    sub = sub[(td >= start_str) & (td <= end_str)]

            if not sub.empty and "trade_date" in sub.columns:
                result.lhb_count_20d = int(sub["trade_date"].nunique())
            else:
                result.lhb_count_20d = len(sub)

            result.lhb_on_board = (result.lhb_count_20d > 0)
            result.lhb_source = "tushare_top_list"

        except Exception as e:
            result.lhb_count_20d = None
            result.lhb_on_board = None
            result.lhb_source = "none"
            result.warnings.append(
                f"[event_layer] {code}: lhb_filter_error: {e}"
            )

    def _fill_industry_heat(self, result, code, industry):
        ctx = self._ctx
        result.industry_name = industry

        if ctx.industry_heat_mode == "none":
            result.industry_heat_pctile_5d = None
            result.industry_pct_5d = None
            result.industry_heat_source = "none"
            return

        if ctx.industry_heat_mode == "ths_daily_full":
            # 完整版：先尝试 industry_cons_map（精确映射），再 fallback 到 industry 名称
            mapped_name = ctx.industry_cons_map.get(code, "")
            lookup_name = mapped_name or industry
            if lookup_name and lookup_name in ctx.industry_rank_pctile:
                result.industry_heat_pctile_5d = ctx.industry_rank_pctile[lookup_name]
                result.industry_pct_5d = ctx.industry_hist_map.get(lookup_name)
                result.industry_heat_source = "ths_daily_full"
                result.industry_name = lookup_name
            else:
                # 股票不在映射里或行业名不匹配
                result.industry_heat_pctile_5d = None
                result.industry_pct_5d = None
                result.industry_heat_source = "ths_daily_full_unmapped"
                result.warnings.append(
                    f"[event_layer] {code}: industry_heat_unmapped: "
                    f"industry='{industry}', mapped='{mapped_name}'"
                )
            return

        # 200元档降级：有行业名但无分位排名
        result.industry_heat_pctile_5d = None
        result.industry_pct_5d = None
        result.industry_heat_source = ctx.industry_heat_mode


    def _fill_concept_heat(self, result, code):
        """填充概念热度字段。"""
        ctx = self._ctx

        if ctx.concept_heat_mode == "ths_daily_full":
            # 完整版：查股票所属概念，取最热概念的百分位
            ts_code = f"{code}.SH" if code.startswith("6") else f"{code}.SZ"
            concepts = ctx.concept_cons_map.get(code, [])

            if not concepts:
                # 实时查询该股票属于哪些概念板块
                concepts = self._lookup_stock_concepts(code, ts_code)
                if concepts:
                    ctx.concept_cons_map[code] = concepts

            # 从 concept_rank_pctile 中查找该股票所属概念的最高百分位
            if concepts:
                scored = []
                for cname in concepts:
                    p = ctx.concept_rank_pctile.get(cname)
                    if p is not None:
                        scored.append((cname, p))

                if scored:
                    scored.sort(key=lambda x: x[1], reverse=True)
                    result.concept_heat_pctile_5d = scored[0][1]  # 最强概念
                    result.advanced_concept_module_available = True
                    result.concept_heat_source = "ths_daily_full"
                    result.concept_names = [n for n, _ in scored[:5]]
                    return

            # 模块可用但该股票未被任何概念收录
            result.concept_heat_pctile_5d = None
            result.advanced_concept_module_available = True
            result.concept_heat_source = "ths_daily_full_no_concept"
            return

        # 降级：不可用
        result.concept_heat_pctile_5d = None
        result.advanced_concept_module_available = False
        result.concept_heat_source = ctx.concept_heat_mode or "tushare_200_unavailable"

    def _lookup_stock_concepts(self, code: str, ts_code: str) -> List[str]:
        """用 ths_member 查询单只股票所属的概念板块名称列表。"""
        ctx = self._ctx
        try:
            member_df = self._ts.get_ths_member(con_code=ts_code)
            if member_df is None or member_df.empty:
                return []

            # ths_member 返回 ts_code(板块code), con_code, con_name
            # 需要从板块 ts_code 映射到概念名称
            # concept_spot_df 存了 ts_code→name 映射
            idx_code_to_name = {}
            if ctx.concept_spot_df is not None and not ctx.concept_spot_df.empty:
                idx_code_to_name = dict(
                    zip(ctx.concept_spot_df["ts_code"], ctx.concept_spot_df["name"])
                )

            names = []
            for _, row in member_df.iterrows():
                idx_code = str(row.get("ts_code", ""))
                name = idx_code_to_name.get(idx_code, "")
                if name and name in ctx.concept_rank_pctile:
                    names.append(name)

            return names

        except Exception as e:
            logger.debug("[event_layer] %s: ths_member 查询失败: %s", code, e)
            return []


# ════════════════════════════════════════════════════════
# 工具函数
# ════════════════════════════════════════════════════════

def _calc_period_pct_change(
    daily_df: pd.DataFrame,
    index_codes: set,
) -> Dict[str, float]:
    """计算一组指数在给定区间内的累计涨跌幅（%）.

    使用区间首日和末日的 close 计算：(close_last / close_first - 1) * 100。
    如果某指数数据不足 2 天，跳过。

    Args:
        daily_df: 合并后的 ths_daily DataFrame（含 ts_code, trade_date, close）
        index_codes: 需要计算的指数 ts_code 集合

    Returns:
        {ts_code: pct_change_5d} 字典
    """
    if daily_df.empty or "ts_code" not in daily_df.columns or "close" not in daily_df.columns:
        return {}

    result: Dict[str, float] = {}
    # 确保 trade_date 可排序
    daily_df = daily_df.copy()
    daily_df["trade_date"] = daily_df["trade_date"].astype(str)

    for code in index_codes:
        sub = daily_df[daily_df["ts_code"] == code].sort_values("trade_date")
        if len(sub) < 2:
            continue
        close_first = sub.iloc[0]["close"]
        close_last = sub.iloc[-1]["close"]
        if close_first is None or close_last is None or close_first == 0:
            continue
        pct = (float(close_last) / float(close_first) - 1.0) * 100.0
        result[code] = round(pct, 4)

    return result


def _max_consecutive(flags: List[bool]) -> int:
    max_c = cur = 0
    for f in flags:
        if f:
            cur += 1
            max_c = max(max_c, cur)
        else:
            cur = 0
    return max_c


def _bisect_right(sorted_list: List[dt.date], target: dt.date) -> int:
    lo, hi = 0, len(sorted_list)
    while lo < hi:
        mid = (lo + hi) // 2
        if sorted_list[mid] <= target:
            lo = mid + 1
        else:
            hi = mid
    return lo
