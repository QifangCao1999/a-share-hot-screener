"""Universe 动态构建器 — Phase 2.

构建当日候选池:
  1. 静态底仓 (CSI300 + CSI500 + CSI1000, 月更)
  2. 近5日涨停池 (limit_list_d, Level 2)
  3. 近5日龙虎榜 (top_list, Level 1)
  4. 当日成交额 Top 200 (daily, Level 1)
  5. 热门板块成分 (ths_daily + ths_member, Level 3)
  6. 去重 + ST/停牌/上市天数过滤
  7. 行业过滤 (可选)

输出:
  - universe/daily_YYYYMMDD.txt
  - UniverseResult (代码列表 + 来源标记 + 可用性状态)

DESIGN_v2.md §5
"""

from __future__ import annotations

import datetime as dt
import logging
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

import pandas as pd

logger = logging.getLogger("a_share_hot_screener.universe_builder")

# 指数代码
_CSI300 = "000300.SH"
_CSI500 = "000905.SH"
_CSI1000 = "000852.SH"

# 静态底仓文件名
_STATIC_FILES = {
    "csi300": "static_csi300.txt",
    "csi500": "static_csi500.txt",
    "csi1000": "static_csi1000.txt",
}

# 默认配置
_DEFAULT_AMOUNT_TOP_N = 200
_DEFAULT_HOT_SECTOR_TOP_N = 10
_DEFAULT_MIN_LISTING_DAYS = 20
_STATIC_STALE_DAYS = 30  # 超过 30 天认为过期


@dataclass
class UniverseResult:
    """Universe 构建结果."""

    run_date: str                                    # YYYYMMDD
    codes: List[str] = field(default_factory=list)   # 6位纯数字代码列表
    total_count: int = 0

    # 来源标记: code -> set of source tags
    source_tags: Dict[str, Set[str]] = field(default_factory=dict)

    # 各来源计数
    static_count: int = 0
    zt_pool_count: int = 0
    lhb_count: int = 0
    amount_top_count: int = 0
    ths_hot_count: int = 0

    # 可用性状态
    stale_static_universe: bool = False
    zt_pool_available: bool = True
    top_list_available: bool = True
    ths_hot_available: bool = True
    daily_available: bool = True

    # 过滤统计
    filtered_st: int = 0
    filtered_listing_days: int = 0
    filtered_industry: int = 0
    pre_filter_count: int = 0


class UniverseBuilder:
    """动态候选池构建器.

    用法:
        builder = UniverseBuilder(tushare_client, trade_calendar, universe_dir="universe/")
        result = builder.build(run_date="20260422")
    """

    def __init__(
        self,
        tushare_client,
        trade_calendar,
        universe_dir: str = "universe",
        amount_top_n: int = _DEFAULT_AMOUNT_TOP_N,
        hot_sector_top_n: int = _DEFAULT_HOT_SECTOR_TOP_N,
        min_listing_days: int = _DEFAULT_MIN_LISTING_DAYS,
        excluded_industries: Optional[List[str]] = None,
    ):
        self._client = tushare_client
        self._cal = trade_calendar
        self._universe_dir = universe_dir
        self._amount_top_n = amount_top_n
        self._hot_sector_top_n = hot_sector_top_n
        self._min_listing_days = min_listing_days
        self._excluded_industries = excluded_industries or []

        os.makedirs(universe_dir, exist_ok=True)

    def build(self, run_date: str) -> UniverseResult:
        """构建当日候选池.

        Args:
            run_date: YYYYMMDD

        Returns:
            UniverseResult
        """
        run_date = run_date.replace("-", "")
        result = UniverseResult(run_date=run_date)

        # 加载 stock_basic 用于过滤
        stock_basic_df = self._client.get_stock_basic()
        if stock_basic_df is None or stock_basic_df.empty:
            logger.error("[universe] stock_basic 不可用，终止构建")
            raise RuntimeError("stock_basic 不可用，无法构建 Universe")

        basic_map = self._build_basic_map(stock_basic_df)

        # 收集所有候选代码 + 来源标记
        all_codes: Dict[str, Set[str]] = {}  # code -> source tags

        # Step 1: 静态底仓
        static_codes = self._load_static_base(run_date, result)
        for code in static_codes:
            all_codes.setdefault(code, set()).update(static_codes[code])
        result.static_count = len(static_codes)

        # Step 2: 涨停池扩展 (Level 2)
        zt_codes = self._expand_zt_pool(run_date, result)
        for code, tags in zt_codes.items():
            all_codes.setdefault(code, set()).update(tags)
        result.zt_pool_count = len(zt_codes)

        # Step 3: 龙虎榜扩展 (Level 1)
        lhb_codes = self._expand_lhb(run_date, result)
        for code, tags in lhb_codes.items():
            all_codes.setdefault(code, set()).update(tags)
        result.lhb_count = len(lhb_codes)

        # Step 4: 成交额 Top N (Level 1)
        amount_codes = self._expand_amount_top(run_date, result)
        for code, tags in amount_codes.items():
            all_codes.setdefault(code, set()).update(tags)
        result.amount_top_count = len(amount_codes)

        # Step 5: 热门板块成分 (Level 3)
        ths_codes = self._expand_ths_hot(run_date, result)
        for code, tags in ths_codes.items():
            all_codes.setdefault(code, set()).update(tags)
        result.ths_hot_count = len(ths_codes)

        result.pre_filter_count = len(all_codes)

        # Step 6: 过滤
        filtered_codes = self._apply_filters(all_codes, basic_map, run_date, result)

        # 最终结果
        result.codes = sorted(filtered_codes.keys())
        result.source_tags = {code: tags for code, tags in filtered_codes.items()}
        result.total_count = len(result.codes)

        # 保存到文件
        self._save_daily_file(result)

        logger.info(
            "[universe] run_date=%s: %d codes (static=%d zt=%d lhb=%d amount=%d ths=%d | "
            "filtered: ST=%d listing=%d industry=%d)",
            run_date,
            result.total_count,
            result.static_count,
            result.zt_pool_count,
            result.lhb_count,
            result.amount_top_count,
            result.ths_hot_count,
            result.filtered_st,
            result.filtered_listing_days,
            result.filtered_industry,
        )

        return result

    # ════════════════════════════════════════════════════════
    # Step 1: 静态底仓
    # ════════════════════════════════════════════════════════

    def _load_static_base(
        self,
        run_date: str,
        result: UniverseResult,
    ) -> Dict[str, Set[str]]:
        """加载 CSI300 + CSI500 + CSI1000 底仓."""
        codes: Dict[str, Set[str]] = {}

        index_map = {
            "csi300": _CSI300,
            "csi500": _CSI500,
            "csi1000": _CSI1000,
        }

        for label, index_code in index_map.items():
            file_path = os.path.join(self._universe_dir, _STATIC_FILES[label])

            # 检查文件是否过期
            needs_refresh = self._is_static_stale(file_path)

            if needs_refresh:
                refreshed = self._refresh_static_file(index_code, label, file_path)
                if not refreshed and not os.path.exists(file_path):
                    logger.warning("[universe] %s 文件不存在且无法刷新", label)
                    continue
                elif not refreshed:
                    result.stale_static_universe = True
                    logger.warning("[universe] %s 过期但无法刷新, 使用旧文件", label)

            # 读取文件
            file_codes = self._read_static_file(file_path)
            for code in file_codes:
                codes.setdefault(code, set()).add(label)

        return codes

    def _is_static_stale(self, file_path: str) -> bool:
        """检查静态文件是否过期 (> 30 天)."""
        if not os.path.exists(file_path):
            return True
        mtime = os.path.getmtime(file_path)
        age_days = (dt.datetime.now().timestamp() - mtime) / 86400
        return age_days > _STATIC_STALE_DAYS

    def _refresh_static_file(
        self,
        index_code: str,
        label: str,
        file_path: str,
    ) -> bool:
        """通过 index_weight API 刷新静态底仓文件."""
        try:
            df = self._client.get_index_weight(index_code=index_code)
            if df is None or df.empty:
                logger.warning("[universe] index_weight(%s) 返回空", index_code)
                return False

            # 提取 con_code → 6位代码
            codes = set()
            for ts_code in df["con_code"].unique():
                code = ts_code.split(".")[0]
                codes.add(code)

            # 写入文件
            self._write_static_file(file_path, sorted(codes))
            logger.info("[universe] 刷新 %s: %d 只", label, len(codes))
            return True

        except Exception as e:
            logger.warning("[universe] 刷新 %s 失败: %s", label, e)
            return False

    def _read_static_file(self, file_path: str) -> List[str]:
        """读取静态底仓文件 (每行一个6位代码)."""
        if not os.path.exists(file_path):
            return []
        with open(file_path, "r") as f:
            return [line.strip() for line in f if line.strip() and not line.startswith("#")]

    def _write_static_file(self, file_path: str, codes: List[str]) -> None:
        """写入静态底仓文件."""
        os.makedirs(os.path.dirname(file_path) or ".", exist_ok=True)
        with open(file_path, "w") as f:
            f.write(f"# Updated: {dt.datetime.now().isoformat()}\n")
            f.write(f"# Count: {len(codes)}\n")
            for code in codes:
                f.write(code + "\n")

    # ════════════════════════════════════════════════════════
    # Step 2: 涨停池扩展
    # ════════════════════════════════════════════════════════

    def _expand_zt_pool(
        self,
        run_date: str,
        result: UniverseResult,
    ) -> Dict[str, Set[str]]:
        """近5日涨停池扩展 (Level 2)."""
        codes: Dict[str, Set[str]] = {}

        trade_dates = self._get_recent_trade_dates(run_date, n=5)
        any_success = False

        for td in trade_dates:
            try:
                df = self._client.get_limit_list(trade_date=td, limit_type="U")
                if df is not None and not df.empty:
                    any_success = True
                    for ts_code in df["ts_code"].unique():
                        code = ts_code.split(".")[0]
                        codes.setdefault(code, set()).add("zt_pool")
            except Exception as e:
                logger.warning("[universe] limit_list_d(%s) 失败: %s", td, e)

        if not any_success:
            result.zt_pool_available = False
            logger.warning("[universe] 涨停池不可用 (Level 2)")

        return codes

    # ════════════════════════════════════════════════════════
    # Step 3: 龙虎榜扩展
    # ════════════════════════════════════════════════════════

    def _expand_lhb(
        self,
        run_date: str,
        result: UniverseResult,
    ) -> Dict[str, Set[str]]:
        """近5日龙虎榜扩展 (Level 1)."""
        codes: Dict[str, Set[str]] = {}

        trade_dates = self._get_recent_trade_dates(run_date, n=5)
        any_success = False

        for td in trade_dates:
            try:
                df = self._client.get_top_list(trade_date=td)
                if df is not None and not df.empty:
                    any_success = True
                    for ts_code in df["ts_code"].unique():
                        code = ts_code.split(".")[0]
                        codes.setdefault(code, set()).add("lhb")
            except Exception as e:
                logger.warning("[universe] top_list(%s) 失败: %s", td, e)

        if not any_success:
            result.top_list_available = False
            logger.warning("[universe] 龙虎榜不可用 (Level 1)")

        return codes

    # ════════════════════════════════════════════════════════
    # Step 4: 成交额 Top N
    # ════════════════════════════════════════════════════════

    def _expand_amount_top(
        self,
        run_date: str,
        result: UniverseResult,
    ) -> Dict[str, Set[str]]:
        """当日成交额 Top N 扩展 (Level 1)."""
        codes: Dict[str, Set[str]] = {}

        try:
            df = self._client.get_daily_by_date(trade_date=run_date)
            if df is None or df.empty:
                result.daily_available = False
                logger.error("[universe] daily(market,%s) 不可用，终止", run_date)
                raise RuntimeError(f"daily 全市场行情({run_date})不可用，无法构建 Universe")

            # 按成交额降序，取 Top N
            df = df.sort_values("amount", ascending=False)
            top_n = df.head(self._amount_top_n)

            for ts_code in top_n["ts_code"].values:
                code = ts_code.split(".")[0]
                codes.setdefault(code, set()).add("amount_top")

        except RuntimeError:
            raise
        except Exception as e:
            logger.error("[universe] 成交额 Top N 加载失败: %s", e)
            result.daily_available = False

        return codes

    # ════════════════════════════════════════════════════════
    # Step 5: 热门板块成分
    # ════════════════════════════════════════════════════════

    def _expand_ths_hot(
        self,
        run_date: str,
        result: UniverseResult,
    ) -> Dict[str, Set[str]]:
        """热门板块成分扩展 (Level 3)."""
        codes: Dict[str, Set[str]] = {}

        try:
            # 获取近 5 日涨幅 Top N 板块
            trade_dates = self._get_recent_trade_dates(run_date, n=5)
            if not trade_dates:
                result.ths_hot_available = False
                return codes

            start_date = trade_dates[0]
            end_date = trade_dates[-1]

            # 获取所有概念指数列表
            ths_index_df = self._client.get_ths_index(exchange="", type_="N")
            if ths_index_df is None or ths_index_df.empty:
                result.ths_hot_available = False
                logger.warning("[universe] ths_index 不可用 (Level 3)")
                return codes

            # 获取每个板块近5日行情，计算涨幅
            sector_returns = []
            for _, row in ths_index_df.iterrows():
                ts_code = row["ts_code"]
                try:
                    daily_df = self._client.get_ths_daily(
                        ts_code=ts_code,
                        start_date=start_date,
                        end_date=end_date,
                    )
                    if daily_df is not None and len(daily_df) >= 2:
                        daily_df = daily_df.sort_values("trade_date")
                        first_close = float(daily_df.iloc[0]["close"])
                        last_close = float(daily_df.iloc[-1]["close"])
                        if first_close > 0:
                            ret = (last_close / first_close - 1) * 100
                            sector_returns.append((ts_code, row.get("name", ""), ret))
                except Exception:
                    pass

            if not sector_returns:
                result.ths_hot_available = False
                return codes

            # 取涨幅 Top N 板块
            sector_returns.sort(key=lambda x: x[2], reverse=True)
            top_sectors = sector_returns[:self._hot_sector_top_n]

            # 获取成分股
            for ts_code, name, ret in top_sectors:
                try:
                    member_df = self._client.get_ths_member(ts_code=ts_code)
                    if member_df is not None and not member_df.empty:
                        col = "con_code" if "con_code" in member_df.columns else "code"
                        for con_code in member_df[col].unique():
                            code = con_code.split(".")[0]
                            codes.setdefault(code, set()).add("ths_concept_hot")
                except Exception as e:
                    logger.warning("[universe] ths_member(%s) 失败: %s", ts_code, e)

        except Exception as e:
            logger.warning("[universe] 热门板块扩展失败: %s", e)
            result.ths_hot_available = False

        return codes

    # ════════════════════════════════════════════════════════
    # Step 6: 过滤
    # ════════════════════════════════════════════════════════

    def _apply_filters(
        self,
        all_codes: Dict[str, Set[str]],
        basic_map: Dict[str, Dict],
        run_date: str,
        result: UniverseResult,
    ) -> Dict[str, Set[str]]:
        """去重 + ST/停牌/上市天数/行业过滤."""
        filtered: Dict[str, Set[str]] = {}
        run_dt = dt.datetime.strptime(run_date, "%Y%m%d").date()

        for code, tags in all_codes.items():
            info = basic_map.get(code, {})
            name = info.get("name", "")

            # ST / *ST / 退市
            if any(marker in name for marker in ["ST", "*ST", "退"]):
                result.filtered_st += 1
                continue

            # 上市天数
            if self._min_listing_days > 0:
                list_date_str = info.get("list_date", "")
                if list_date_str:
                    try:
                        list_date = dt.datetime.strptime(list_date_str, "%Y%m%d").date()
                        listing_days = (run_dt - list_date).days
                        if listing_days < self._min_listing_days:
                            result.filtered_listing_days += 1
                            continue
                    except (ValueError, TypeError):
                        pass

            # 行业过滤
            if self._excluded_industries:
                industry = info.get("industry", "")
                if industry in self._excluded_industries:
                    result.filtered_industry += 1
                    continue

            filtered[code] = tags

        return filtered

    def _build_basic_map(self, df: pd.DataFrame) -> Dict[str, Dict]:
        """从 stock_basic DataFrame 构建 code -> info 映射."""
        result = {}
        for _, row in df.iterrows():
            ts_code = row.get("ts_code", "")
            code = ts_code.split(".")[0] if ts_code else row.get("symbol", "")
            if code:
                result[code] = {
                    "name": row.get("name", ""),
                    "industry": row.get("industry", ""),
                    "list_date": row.get("list_date", ""),
                    "market": row.get("market", ""),
                    "ts_code": ts_code,
                }
        return result

    # ════════════════════════════════════════════════════════
    # 输出
    # ════════════════════════════════════════════════════════

    def _save_daily_file(self, result: UniverseResult) -> str:
        """保存每日 Universe 文件."""
        file_path = os.path.join(
            self._universe_dir,
            f"daily_{result.run_date}.txt",
        )

        with open(file_path, "w") as f:
            f.write(f"# Universe for {result.run_date}\n")
            f.write(f"# Count: {result.total_count}\n")
            f.write(f"# Static: {result.static_count} | ZT: {result.zt_pool_count} | "
                    f"LHB: {result.lhb_count} | Amount: {result.amount_top_count} | "
                    f"THS: {result.ths_hot_count}\n")
            for code in result.codes:
                tags = result.source_tags.get(code, set())
                f.write(f"{code}\t{','.join(sorted(tags))}\n")

        logger.info("[universe] 保存 %s (%d codes)", file_path, result.total_count)
        return file_path

    @staticmethod
    def load_daily_file(file_path: str) -> List[str]:
        """加载每日 Universe 文件, 返回代码列表."""
        codes = []
        with open(file_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                code = line.split("\t")[0].strip()
                if code:
                    codes.append(code)
        return codes

    # ════════════════════════════════════════════════════════
    # 工具方法
    # ════════════════════════════════════════════════════════

    def _get_recent_trade_dates(self, run_date: str, n: int = 5) -> List[str]:
        """获取 run_date 当日及前 n-1 个交易日 (YYYYMMDD, 降序→升序)."""
        run_dt = dt.datetime.strptime(run_date, "%Y%m%d").date()
        dates = []
        current = run_dt
        attempts = 0
        max_attempts = n * 3 + 10

        while len(dates) < n and attempts < max_attempts:
            if self._cal.is_trade_date(current):
                dates.append(current.strftime("%Y%m%d"))
            current = current - dt.timedelta(days=1)
            attempts += 1

        return sorted(dates)  # 升序
