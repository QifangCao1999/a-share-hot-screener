"""Tushare Pro API 客户端 — 统一数据源（替代 EODHD + AkShare）.

订阅等级：200元/年（≈5000积分）基础接口 + 600元/年（≈6000积分）同花顺板块
可用接口（≥5000pt）：
  daily / trade_cal / stock_basic / adj_factor / daily_basic /
  top_list / top_inst / dividend / pledge_stat / share_float /
  stk_holdernumber / index_classify / index_member_all / limit_list_d
需6000积分（600元/年）：ths_index / ths_daily / ths_member

单位说明：
  - daily.vol：手（1手=100股）
  - daily.amount：千元
  - daily_basic.total_mv / circ_mv：万元
"""

from __future__ import annotations

import logging
import random
import time
import threading
from typing import Any, Dict, List, Optional

import pandas as pd

from a_share_hot_screener.cache import LocalCache

logger = logging.getLogger("a_share_hot_screener.tushare_client")

_MAX_RETRIES = 3
_RETRY_BACKOFF = 2.0
_RETRY_JITTER = 0.5  # 随机抖动系数（0~0.5倍 backoff）


# ═══════════════════════════════════════════════════════
# TokenBucket 限流器
# ═══════════════════════════════════════════════════════

class TokenBucket:
    """Token bucket 限流器，允许突发请求，比固定 sleep 更高效.

    Tushare 200元档限流约 3 次/秒（根据实测）。
    capacity=3, refill_rate=3.0 即每秒补充 3 个 token，最多突发 3 个。
    """

    def __init__(self, capacity: int = 3, refill_rate: float = 3.0) -> None:
        self._capacity = capacity
        self._tokens = float(capacity)
        self._refill_rate = refill_rate  # tokens per second
        self._last_refill = time.monotonic()
        self._lock = threading.Lock()

    def acquire(self, timeout: float = 30.0) -> bool:
        """获取一个 token，必要时等待。返回 True 表示成功。"""
        deadline = time.monotonic() + timeout
        while True:
            with self._lock:
                self._refill()
                if self._tokens >= 1.0:
                    self._tokens -= 1.0
                    return True
                # 计算需要等待多久才能有 1 个 token
                wait_time = (1.0 - self._tokens) / self._refill_rate
            if time.monotonic() + wait_time > deadline:
                return False
            time.sleep(min(wait_time, 0.5))  # 最多睡 0.5s 再重试

    def _refill(self) -> None:
        now = time.monotonic()
        elapsed = now - self._last_refill
        self._tokens = min(self._capacity, self._tokens + elapsed * self._refill_rate)
        self._last_refill = now


# 全局 token bucket 实例
_token_bucket = TokenBucket(capacity=3, refill_rate=3.0)


class TushareClient:
    """Tushare Pro API 客户端.

    封装限流、重试、缓存，统一替代 EODHDClient + AkShareClient。
    """

    def __init__(
        self,
        token: str,
        cache: Optional[LocalCache] = None,
    ) -> None:
        import tushare as ts
        self.pro = ts.pro_api(token)
        self.cache = cache

    # ── 内部：带重试的调用 ─────────────────────────────────

    def _call(
        self,
        api_name: str,
        label: str = "",
        max_retries: int = _MAX_RETRIES,
        **kwargs,
    ) -> Optional[pd.DataFrame]:
        """带 token-bucket 限流 + jitter 重试的 Tushare API 调用."""
        delay = _RETRY_BACKOFF
        tag = label or api_name
        for attempt in range(1, max_retries + 1):
            try:
                _token_bucket.acquire()
                df = getattr(self.pro, api_name)(**kwargs)
                if df is None:
                    raise ValueError(f"{tag} 返回 None")
                return df
            except Exception as e:
                err_str = str(e)
                if "没有接口访问权限" in err_str:
                    logger.warning("[tushare] %s 无权限（积分不足）: %s", tag, e)
                    return None
                logger.warning(
                    "[tushare] %s attempt=%d/%d 失败: %s",
                    tag, attempt, max_retries, e,
                )
                if attempt < max_retries:
                    jitter = delay * random.uniform(0, _RETRY_JITTER)
                    time.sleep(delay + jitter)
                    delay *= 2
        logger.error("[tushare] %s 最终失败", tag)
        return None

    # ═══════════════════════════════════════════════════════
    # 交易日历 (120pt)
    # ═══════════════════════════════════════════════════════

    def get_trade_dates(
        self,
        start_date: str = "19900101",
        end_date: str = "20271231",
        use_cache: bool = True,
        cache_ttl: int = 7 * 86400,
    ) -> Optional[List[str]]:
        """获取交易日列表，返回 YYYY-MM-DD 格式的日期字符串列表（升序）."""
        cache_key = f"trade_dates_{start_date}_{end_date}"
        if use_cache and self.cache:
            cached = self.cache.get("tushare_cal", cache_key, ttl=cache_ttl)
            if cached is not None:
                return cached

        df = self._call(
            "trade_cal",
            label="trade_cal",
            exchange="SSE",
            start_date=start_date,
            end_date=end_date,
        )
        if df is None or df.empty:
            return None

        open_days = df[df["is_open"] == 1]["cal_date"].tolist()
        # 转为 YYYY-MM-DD 格式
        dates = sorted([
            f"{d[:4]}-{d[4:6]}-{d[6:8]}" for d in open_days
            if len(str(d)) == 8
        ])

        if use_cache and self.cache and dates:
            self.cache.put("tushare_cal", cache_key, dates, ttl=cache_ttl)

        logger.info("[tushare] trade_cal: %d 个交易日", len(dates))
        return dates

    # ═══════════════════════════════════════════════════════
    # 股票列表 (120pt)
    # ═══════════════════════════════════════════════════════

    def get_stock_basic(
        self,
        use_cache: bool = True,
        cache_ttl: int = 86400,
    ) -> Optional[pd.DataFrame]:
        """获取全A股在市股票列表.

        返回字段：ts_code, symbol, name, area, industry, market, list_date, is_hs
        """
        cache_key = "stock_basic_listed"
        if use_cache and self.cache:
            cached = self.cache.get("tushare_basic", cache_key, ttl=cache_ttl)
            if cached is not None:
                try:
                    return pd.DataFrame(cached)
                except Exception:
                    pass

        df = self._call(
            "stock_basic",
            label="stock_basic",
            exchange="",
            list_status="L",
            fields="ts_code,symbol,name,area,industry,market,list_date,is_hs",
        )
        if df is not None and not df.empty and use_cache and self.cache:
            self.cache.put("tushare_basic", cache_key, df.to_dict("records"), ttl=cache_ttl)

        return df

    # ═══════════════════════════════════════════════════════
    # 日线行情 (120pt) — 替代 EODHD EOD
    # ═══════════════════════════════════════════════════════

    def get_daily(
        self,
        ts_code: str,
        start_date: str,
        end_date: str,
        use_cache: bool = True,
        cache_ttl: int = 86400,
    ) -> Optional[pd.DataFrame]:
        """获取个股日线行情.

        Args:
            ts_code: Tushare 格式，如 '000001.SZ'
            start_date: YYYYMMDD
            end_date: YYYYMMDD

        Returns:
            DataFrame: ts_code, trade_date, open, high, low, close, pre_close,
                       change, pct_chg, vol(手), amount(千元)
        """
        cache_key = f"daily_{ts_code}_{start_date}_{end_date}"
        if use_cache and self.cache:
            cached = self.cache.get("tushare_daily", cache_key, ttl=cache_ttl)
            if cached is not None:
                try:
                    return pd.DataFrame(cached)
                except Exception:
                    pass

        df = self._call(
            "daily",
            label=f"daily({ts_code})",
            ts_code=ts_code,
            start_date=start_date,
            end_date=end_date,
        )
        if df is not None and not df.empty and use_cache and self.cache:
            self.cache.put("tushare_daily", cache_key, df.to_dict("records"), ttl=cache_ttl)

        return df

    # ═══════════════════════════════════════════════════════
    # 每日指标 (2000pt) — 替代 AkShare spot
    # ═══════════════════════════════════════════════════════

    def get_daily_basic_by_date(
        self,
        trade_date: str,
        use_cache: bool = True,
        cache_ttl: int = 4 * 3600,
    ) -> Optional[pd.DataFrame]:
        """获取全市场单日指标（PE/PB/换手率/市值/量比）.

        Args:
            trade_date: YYYYMMDD

        Returns:
            DataFrame: ts_code, trade_date, close, turnover_rate, volume_ratio,
                       pe, pe_ttm, pb, total_mv(万元), circ_mv(万元), ...
        """
        cache_key = f"daily_basic_{trade_date}"
        if use_cache and self.cache:
            cached = self.cache.get("tushare_daily_basic", cache_key, ttl=cache_ttl)
            if cached is not None:
                try:
                    return pd.DataFrame(cached)
                except Exception:
                    pass

        df = self._call(
            "daily_basic",
            label=f"daily_basic(date={trade_date})",
            trade_date=trade_date,
        )
        if df is not None and not df.empty and use_cache and self.cache:
            self.cache.put("tushare_daily_basic", cache_key, df.to_dict("records"), ttl=cache_ttl)

        return df

    def get_daily_basic_by_code(
        self,
        ts_code: str,
        start_date: str,
        end_date: str,
    ) -> Optional[pd.DataFrame]:
        """获取个股区间每日指标."""
        return self._call(
            "daily_basic",
            label=f"daily_basic({ts_code})",
            ts_code=ts_code,
            start_date=start_date,
            end_date=end_date,
        )

    # ═══════════════════════════════════════════════════════
    # 涨停池 (5000pt) — 替代 AkShare stock_zt_pool_em
    # ═══════════════════════════════════════════════════════

    def get_limit_list(
        self,
        trade_date: str,
        limit_type: str = "U",
        use_cache: bool = True,
        cache_ttl: int = 86400 * 3,
    ) -> Optional[pd.DataFrame]:
        """获取单日涨停/跌停池.

        Args:
            trade_date: YYYYMMDD
            limit_type: 'U'=涨停, 'D'=跌停

        Returns:
            DataFrame: ts_code, name, close, pct_chg, fd_amount, limit_times, ...
        """
        cache_key = f"limit_{trade_date}_{limit_type}"
        if use_cache and self.cache:
            cached = self.cache.get("tushare_limit", cache_key, ttl=cache_ttl)
            if cached is not None:
                try:
                    return pd.DataFrame(cached)
                except Exception:
                    pass

        df = self._call(
            "limit_list_d",
            label=f"limit_list_d({trade_date},{limit_type})",
            trade_date=trade_date,
            limit_type=limit_type,
        )
        if df is not None and not df.empty and use_cache and self.cache:
            self.cache.put("tushare_limit", cache_key, df.to_dict("records"), ttl=cache_ttl)

        return df

    # ═══════════════════════════════════════════════════════
    # 龙虎榜 (2000pt) — 替代 AkShare stock_lhb_detail_em
    # ═══════════════════════════════════════════════════════

    def get_top_list(
        self,
        trade_date: str,
        use_cache: bool = True,
        cache_ttl: int = 86400 * 2,
    ) -> Optional[pd.DataFrame]:
        """获取单日龙虎榜明细.

        Returns:
            DataFrame: trade_date, ts_code, name, l_buy, l_sell, net_amount, reason, ...
        """
        cache_key = f"top_list_{trade_date}"
        if use_cache and self.cache:
            cached = self.cache.get("tushare_lhb", cache_key, ttl=cache_ttl)
            if cached is not None:
                try:
                    return pd.DataFrame(cached)
                except Exception:
                    pass

        df = self._call(
            "top_list",
            label=f"top_list({trade_date})",
            trade_date=trade_date,
        )
        if df is not None and not df.empty and use_cache and self.cache:
            self.cache.put("tushare_lhb", cache_key, df.to_dict("records"), ttl=cache_ttl)

        return df

    # ═══════════════════════════════════════════════════════
    # 质押统计 (2000pt) — 替代 AkShare stock_gpzy_pledge_ratio_em
    # ═══════════════════════════════════════════════════════

    def get_pledge_stat(
        self,
        ts_code: str,
        use_cache: bool = True,
        cache_ttl: int = 86400,
    ) -> Optional[pd.DataFrame]:
        """获取个股质押统计.

        Returns:
            DataFrame: ts_code, end_date, pledge_count, unrest_pledge,
                       rest_pledge, total_share, pledge_ratio
        """
        cache_key = f"pledge_{ts_code}"
        if use_cache and self.cache:
            cached = self.cache.get("tushare_pledge", cache_key, ttl=cache_ttl)
            if cached is not None:
                try:
                    return pd.DataFrame(cached)
                except Exception:
                    pass

        df = self._call(
            "pledge_stat",
            label=f"pledge_stat({ts_code})",
            ts_code=ts_code,
        )
        if df is not None and not df.empty and use_cache and self.cache:
            self.cache.put("tushare_pledge", cache_key, df.to_dict("records"), ttl=cache_ttl)

        return df

    # ═══════════════════════════════════════════════════════
    # 限售解禁 (3000pt) — 替代 AkShare stock_restricted_release_queue_em
    # ═══════════════════════════════════════════════════════

    def get_share_float(
        self,
        ts_code: str,
        use_cache: bool = True,
        cache_ttl: int = 86400,
    ) -> Optional[pd.DataFrame]:
        """获取个股限售解禁计划.

        Returns:
            DataFrame: ts_code, ann_date, float_date, float_share, float_ratio,
                       holder_name, share_type
        """
        cache_key = f"share_float_{ts_code}"
        if use_cache and self.cache:
            cached = self.cache.get("tushare_float", cache_key, ttl=cache_ttl)
            if cached is not None:
                try:
                    return pd.DataFrame(cached)
                except Exception:
                    pass

        df = self._call(
            "share_float",
            label=f"share_float({ts_code})",
            ts_code=ts_code,
        )
        if df is not None and not df.empty and use_cache and self.cache:
            self.cache.put("tushare_float", cache_key, df.to_dict("records"), ttl=cache_ttl)

        return df

    # ═══════════════════════════════════════════════════════
    # 股东人数 (2000pt) — 替代 AkShare stock_hold_num_cninfo
    # ═══════════════════════════════════════════════════════

    def get_stk_holdernumber(
        self,
        ts_code: str,
        start_date: str = "",
        use_cache: bool = True,
        cache_ttl: int = 86400,
    ) -> Optional[pd.DataFrame]:
        """获取个股股东人数变动.

        Returns:
            DataFrame: ts_code, ann_date, end_date, holder_num
        """
        cache_key = f"holdnum_{ts_code}_{start_date}"
        if use_cache and self.cache:
            cached = self.cache.get("tushare_holdnum", cache_key, ttl=cache_ttl)
            if cached is not None:
                try:
                    return pd.DataFrame(cached)
                except Exception:
                    pass

        kwargs = {"ts_code": ts_code}
        if start_date:
            kwargs["start_date"] = start_date

        df = self._call(
            "stk_holdernumber",
            label=f"stk_holdernumber({ts_code})",
            **kwargs,
        )
        if df is not None and not df.empty and use_cache and self.cache:
            self.cache.put("tushare_holdnum", cache_key, df.to_dict("records"), ttl=cache_ttl)

        return df

    # ═══════════════════════════════════════════════════════
    # 申万行业分类 (2000pt)
    # ═══════════════════════════════════════════════════════

    def get_index_classify(
        self,
        level: str = "L1",
        src: str = "SW2021",
        use_cache: bool = True,
        cache_ttl: int = 7 * 86400,
    ) -> Optional[pd.DataFrame]:
        """获取申万行业分类列表."""
        cache_key = f"index_classify_{level}_{src}"
        if use_cache and self.cache:
            cached = self.cache.get("tushare_industry", cache_key, ttl=cache_ttl)
            if cached is not None:
                try:
                    return pd.DataFrame(cached)
                except Exception:
                    pass

        df = self._call(
            "index_classify",
            label=f"index_classify({level})",
            level=level,
            src=src,
        )
        if df is not None and not df.empty and use_cache and self.cache:
            self.cache.put("tushare_industry", cache_key, df.to_dict("records"), ttl=cache_ttl)

        return df

    def get_index_member_all(
        self,
        use_cache: bool = True,
        cache_ttl: int = 7 * 86400,
    ) -> Optional[pd.DataFrame]:
        """获取申万行业全部成分股映射（全量，不按单个行业拉取）.

        Returns:
            DataFrame: l1_code, l1_name, l2_code, l2_name, l3_code, l3_name,
                       ts_code, name, in_date, out_date, is_new
        """
        cache_key = "index_member_all"
        if use_cache and self.cache:
            cached = self.cache.get("tushare_industry", cache_key, ttl=cache_ttl)
            if cached is not None:
                try:
                    return pd.DataFrame(cached)
                except Exception:
                    pass

        df = self._call(
            "index_member_all",
            label="index_member_all",
        )
        if df is not None and not df.empty and use_cache and self.cache:
            self.cache.put("tushare_industry", cache_key, df.to_dict("records"), ttl=cache_ttl)

        return df

    # ═══════════════════════════════════════════════════════
    # 分红记录 (2000pt)
    # ═══════════════════════════════════════════════════════

    def get_dividend(
        self,
        ts_code: str,
        use_cache: bool = True,
        cache_ttl: int = 86400,
    ) -> Optional[pd.DataFrame]:
        """获取个股分红记录."""
        cache_key = f"dividend_{ts_code}"
        if use_cache and self.cache:
            cached = self.cache.get("tushare_div", cache_key, ttl=cache_ttl)
            if cached is not None:
                try:
                    return pd.DataFrame(cached)
                except Exception:
                    pass

        df = self._call(
            "dividend",
            label=f"dividend({ts_code})",
            ts_code=ts_code,
        )
        if df is not None and not df.empty and use_cache and self.cache:
            self.cache.put("tushare_div", cache_key, df.to_dict("records"), ttl=cache_ttl)

        return df

    # ═══════════════════════════════════════════════════════
    # 同花顺板块指数 (6000pt / 600元年)
    # ═══════════════════════════════════════════════════════

    def get_ths_index(
        self,
        exchange: str = "A",
        type_: Optional[str] = None,
        use_cache: bool = True,
        cache_ttl: int = 7 * 86400,
    ) -> Optional[pd.DataFrame]:
        """获取同花顺概念/行业指数列表（需 6000 积分）.

        Args:
            exchange: 'A'=A股, 'HK'=港股, 'US'=美股
            type_: 'N'=概念, 'I'=行业, 'S'=特色, None=全部

        Returns:
            DataFrame: ts_code, name, count, exchange, list_date, type
            无权限时返回 None。
        """
        suffix = type_ or "all"
        cache_key = f"ths_index_{exchange}_{suffix}"
        if use_cache and self.cache:
            cached = self.cache.get("tushare_ths", cache_key, ttl=cache_ttl)
            if cached is not None:
                try:
                    return pd.DataFrame(cached)
                except Exception:
                    pass

        kwargs: Dict[str, Any] = {"exchange": exchange}
        if type_ is not None:
            kwargs["type"] = type_

        df = self._call("ths_index", label=f"ths_index({exchange},{suffix})", **kwargs)
        if df is not None and not df.empty and use_cache and self.cache:
            self.cache.put("tushare_ths", cache_key, df.to_dict("records"), ttl=cache_ttl)

        return df

    def get_ths_daily(
        self,
        ts_code: str = "",
        trade_date: str = "",
        start_date: str = "",
        end_date: str = "",
        use_cache: bool = True,
        cache_ttl: int = 86400,
    ) -> Optional[pd.DataFrame]:
        """获取同花顺板块指数行情（需 6000 积分）.

        可按 ts_code+日期区间 或 trade_date 查询。

        Returns:
            DataFrame: ts_code, trade_date, close, open, high, low, pct_change, vol, ...
            无权限时返回 None。
        """
        if trade_date:
            cache_key = f"ths_daily_date_{trade_date}"
        elif ts_code:
            cache_key = f"ths_daily_{ts_code}_{start_date}_{end_date}"
        else:
            cache_key = f"ths_daily_{start_date}_{end_date}"

        if use_cache and self.cache:
            cached = self.cache.get("tushare_ths_daily", cache_key, ttl=cache_ttl)
            if cached is not None:
                try:
                    return pd.DataFrame(cached)
                except Exception:
                    pass

        kwargs: Dict[str, Any] = {}
        if ts_code:
            kwargs["ts_code"] = ts_code
        if trade_date:
            kwargs["trade_date"] = trade_date
        if start_date:
            kwargs["start_date"] = start_date
        if end_date:
            kwargs["end_date"] = end_date

        df = self._call(
            "ths_daily",
            label=f"ths_daily({ts_code or trade_date or start_date})",
            **kwargs,
        )
        if df is not None and not df.empty and use_cache and self.cache:
            self.cache.put("tushare_ths_daily", cache_key, df.to_dict("records"), ttl=cache_ttl)

        return df

    def get_ths_member(
        self,
        ts_code: str = "",
        con_code: str = "",
        use_cache: bool = True,
        cache_ttl: int = 7 * 86400,
    ) -> Optional[pd.DataFrame]:
        """获取同花顺概念板块成分（需 6000 积分）.

        Args:
            ts_code: 板块指数代码（查成分）
            con_code: 股票代码 Tushare 格式（查所属板块）

        Returns:
            DataFrame: ts_code, con_code, con_name
            无权限时返回 None。
        """
        if con_code:
            cache_key = f"ths_member_con_{con_code}"
        elif ts_code:
            cache_key = f"ths_member_{ts_code}"
        else:
            return None

        if use_cache and self.cache:
            cached = self.cache.get("tushare_ths_member", cache_key, ttl=cache_ttl)
            if cached is not None:
                try:
                    return pd.DataFrame(cached)
                except Exception:
                    pass

        kwargs: Dict[str, Any] = {}
        if ts_code:
            kwargs["ts_code"] = ts_code
        if con_code:
            kwargs["con_code"] = con_code

        df = self._call(
            "ths_member",
            label=f"ths_member({ts_code or con_code})",
            **kwargs,
        )
        if df is not None and not df.empty and use_cache and self.cache:
            self.cache.put("tushare_ths_member", cache_key, df.to_dict("records"), ttl=cache_ttl)

        return df

    # ═══════════════════════════════════════════════════════
    # 批量预加载（用于并发处理前的串行预热缓存）
    # ═══════════════════════════════════════════════════════

    def prefetch_risk_data(
        self,
        ts_codes: List[str],
        run_date_str: str = "",
        max_workers: int = 5,
    ) -> Dict[str, int]:
        """并发预加载质押/解禁/股东人数数据到缓存。

        在 pipeline Step 2.5 之后、Step 3 之前调用。
        TokenBucket 控制整体速率，并发线程提升吞吐量。
        已缓存的代码会被跳过（开销约 0）。

        Args:
            ts_codes: Tushare 格式代码列表
            run_date_str: YYYY-MM-DD
            max_workers: 预加载并发线程数

        Returns:
            {'pledge': N, 'float': N, 'holdnum': N} 实际 API 调用数
        """
        import datetime as dt
        from concurrent.futures import ThreadPoolExecutor, as_completed

        stats = {'pledge': 0, 'float': 0, 'holdnum': 0}
        if not ts_codes:
            return stats

        # holdernumber 的 start_date
        start_date = ""
        if run_date_str:
            try:
                rd = dt.date.fromisoformat(run_date_str)
                start_date = (rd - dt.timedelta(days=365)).strftime("%Y%m%d")
            except Exception:
                pass

        # 过滤已缓存的，收集需要拉取的任务
        tasks: List[tuple] = []  # (ts_code, api_type)
        for ts_code in ts_codes:
            if not (self.cache and self.cache.get("tushare_pledge", f"pledge_{ts_code}", ttl=86400) is not None):
                tasks.append((ts_code, 'pledge'))
            if not (self.cache and self.cache.get("tushare_float", f"share_float_{ts_code}", ttl=86400) is not None):
                tasks.append((ts_code, 'float'))
            holdnum_key = f"holdnum_{ts_code}_{start_date}"
            if not (self.cache and self.cache.get("tushare_holdnum", holdnum_key, ttl=86400) is not None):
                tasks.append((ts_code, 'holdnum'))

        if not tasks:
            logger.info("[prefetch] 全部已缓存，跳过预加载 (%d 只股票)", len(ts_codes))
            return stats

        logger.info(
            "[prefetch] 开始预加载风控数据: %d 只股票, %d 个待拉取任务 (%d 并发)",
            len(ts_codes), len(tasks), max_workers,
        )
        t0 = time.time()
        _stats_lock = threading.Lock()

        def _fetch_one(ts_code: str, api_type: str) -> str:
            if api_type == 'pledge':
                self.get_pledge_stat(ts_code)
            elif api_type == 'float':
                self.get_share_float(ts_code)
            elif api_type == 'holdnum':
                self.get_stk_holdernumber(ts_code, start_date=start_date)
            return api_type

        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futs = {ex.submit(_fetch_one, tc, at): (tc, at) for tc, at in tasks}
            for fut in as_completed(futs):
                try:
                    api_type = fut.result()
                    with _stats_lock:
                        stats[api_type] = stats.get(api_type, 0) + 1
                except Exception as e:
                    tc, at = futs[fut]
                    logger.warning("[prefetch] %s(%s) 失败: %s", at, tc, e)

        elapsed = time.time() - t0
        total_calls = sum(stats.values())
        logger.info(
            "[prefetch] 预加载完成: %d 次 API 调用 (pledge=%d, float=%d, holdnum=%d) | %.1fs",
            total_calls, stats['pledge'], stats['float'], stats['holdnum'], elapsed,
        )
        return stats
