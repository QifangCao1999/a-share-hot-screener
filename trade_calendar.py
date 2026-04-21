"""交易日基础工具层（Session 3 完整实现）.

数据源：Tushare trade_cal (120积分)
  - 返回完整 A 肢交易日历
  - 字段 cal_date(YYYYMMDD) + is_open(0/1)
  - 带缓存（TTL 7 天）

核心接口：
  - TradeCalendar.resolve()             确定 trade_date_used
  - TradeCalendar.get_trade_date_used() 返回实际使用的交易日
  - TradeCalendar.is_trade_date(d)      判断是否是交易日
  - TradeCalendar.prev_trade_date(d)    前一个交易日
  - TradeCalendar.n_trade_dates_before(d, n) 向前 n 个交易日
  - TradeCalendar.count_trade_days_between(start, end) 区间交易日数量

as-of 口径：
  - run_date 当日数据是否“完整”依赖于运行时刻（盘中 vs 收盘后），
    当前版本不做盘中判断：若 run_date 是交易日，一律使用 run_date。
  - 若 run_date 是非交易日，自动回退到最近的前一个交易日。
"""

from __future__ import annotations

import datetime as dt
import logging
from functools import lru_cache
from typing import List, Optional, Set

from a_share_hot_screener.cache import LocalCache

logger = logging.getLogger("a_share_hot_screener.trade_calendar")

_CALENDAR_CACHE_TTL = 7 * 86400  # 7 天


class TradeCalendar:
    """A 股交易日历工具类.

    使用前必须调用 load()(或 resolve(),它会自动调用 load())。
    若 load() 失败,降级为"仅排除周末"的简单逻辑,并产生 warning。
    """

    def __init__(
        self,
        run_date: dt.date,
        cache: Optional[LocalCache] = None,
    ) -> None:
        self._run_date = run_date
        self._cache = cache
        self._trade_dates: List[dt.date] = []          # 排好序的交易日列表
        self._trade_date_set: Set[dt.date] = set()     # 集合,用于 O(1) 查找
        self._loaded: bool = False
        self._fallback: bool = False                   # True = 降级为周末判断
        self._trade_date_used: Optional[dt.date] = None
        self._warnings: List[str] = []
        self._resolved: bool = False

    # ── 加载交易日历 ──────────────────────────────────────

    def load(self, tushare_client=None) -> bool:
        """加载完整交易日历（优先缓存，否则从 Tushare 拉取）.

        Args:
            tushare_client: TushareClient 实例（可选，缓存命中时不需要）

        Returns:
            True 表示成功加载（完整或降级模式）
        """
        if self._loaded:
            return True

        # 先尝试缓存
        if self._cache:
            cached = self._cache.get(
                "trade_calendar", "tushare_trade_dates", ttl=_CALENDAR_CACHE_TTL
            )
            if cached:
                self._trade_dates = [dt.date.fromisoformat(s) for s in cached]
                self._trade_date_set = set(self._trade_dates)
                self._loaded = True
                logger.info("从缓存加载交易日历: %d 个交易日", len(self._trade_dates))
                return True

        # 从 Tushare 拉取
        try:
            if tushare_client is not None:
                date_strs = tushare_client.get_trade_dates(
                    start_date="19900101", end_date="20271231",
                    use_cache=False,
                )
            else:
                # 直接调用 tushare
                import tushare as ts
                pro = ts.pro_api()
                df = pro.trade_cal(exchange="SSE", start_date="19900101", end_date="20271231")
                if df is None or df.empty:
                    raise ValueError("trade_cal 返回空")
                open_days = df[df["is_open"] == 1]["cal_date"].tolist()
                date_strs = sorted([
                    f"{d[:4]}-{d[4:6]}-{d[6:8]}" for d in open_days if len(str(d)) == 8
                ])

            if not date_strs:
                raise ValueError("Tushare trade_cal 返回空交易日列表")

            dates = sorted([dt.date.fromisoformat(s) for s in date_strs])
            self._trade_dates = dates
            self._trade_date_set = set(dates)
            self._loaded = True
            self._fallback = False

            # 写缓存
            if self._cache:
                self._cache.put(
                    "trade_calendar", "tushare_trade_dates",
                    [d.isoformat() for d in dates],
                    ttl=_CALENDAR_CACHE_TTL,
                )
            logger.info("Tushare 交易日历加载完成: %d 个交易日", len(dates))
            return True

        except Exception as e:
            msg = (
                f"Tushare 交易日历加载失败（{e}），"
                f"降级为仅排除周末的简单逻辑，节假日可能被误判为交易日"
            )
            logger.warning(msg)
            self._warnings.append(msg)
            self._loaded = True
            self._fallback = True
            return True   # 返回 True 以便流程继续（降级运行）

    # ── resolve:确定 trade_date_used ────────────────────

    def resolve(self) -> "TradeCalendar":
        """确定本次实际使用的交易日(trade_date_used).

        逻辑:
          1. 加载交易日历(若未加载)
          2. 若 run_date 是交易日 → 使用 run_date
          3. 若 run_date 不是交易日(周末/节假日)→ 向前找最近交易日,产生 warning

        Returns:
            self(链式调用)
        """
        if self._resolved:
            return self
        self.load()

        if self.is_trade_date(self._run_date):
            self._trade_date_used = self._run_date
        else:
            prev = self.prev_trade_date(self._run_date)
            msg = (
                f"run_date={self._run_date} 不是交易日,"
                f"自动回退到最近交易日 {prev}"
            )
            logger.warning(msg)
            self._warnings.append(msg)
            self._trade_date_used = prev

        self._resolved = True
        return self

    # ── 主查询接口 ────────────────────────────────────────

    def get_trade_date_used(self) -> dt.date:
        """返回本次实际使用的交易日(若未 resolve 则自动触发)."""
        if not self._resolved:
            self.resolve()
        return self._trade_date_used  # type: ignore[return-value]

    def get_warnings(self) -> List[str]:
        return list(self._warnings)

    def is_trade_date(self, date: dt.date) -> bool:
        """判断是否是交易日(精确)."""
        if not self._loaded:
            self.load()
        if self._fallback:
            return date.weekday() < 5
        return date in self._trade_date_set

    def prev_trade_date(self, date: Optional[dt.date] = None) -> dt.date:
        """返回指定日期的前一个交易日.

        Args:
            date: 基准日期,默认为 run_date
        """
        if not self._loaded:
            self.load()
        target = date or self._run_date

        if self._fallback:
            d = target - dt.timedelta(days=1)
            while d.weekday() >= 5:
                d -= dt.timedelta(days=1)
            return d

        # 在有序列表中二分查找
        idx = _bisect_left(self._trade_dates, target)
        # target 本身可能在列表中(idx 是其位置),取 idx-1
        if idx > 0 and self._trade_dates[idx - 1] < target:
            return self._trade_dates[idx - 1]
        # target 不在列表中:idx 是第一个 >= target 的位置,取 idx-1
        if idx > 0:
            return self._trade_dates[idx - 1]
        raise ValueError(f"找不到 {target} 之前的交易日(日历数据不足?)")

    def n_trade_dates_before(self, date: dt.date, n: int) -> dt.date:
        """返回 date 向前 n 个交易日的日期(不含 date 本身).

        例:n=5 表示找"5个交易日前"。
        """
        if not self._loaded:
            self.load()

        if self._fallback:
            d = date
            count = 0
            while count < n:
                d -= dt.timedelta(days=1)
                if d.weekday() < 5:
                    count += 1
            return d

        idx = _bisect_left(self._trade_dates, date)
        # 若 date 在列表中,idx 是其位置;否则 idx 是第一个 > date 的位置
        # 向前 n 个:idx - n(date 本身不计入)
        if idx >= n:
            return self._trade_dates[idx - n]
        raise ValueError(
            f"交易日历向前 {n} 个不足(date={date}, 可用历史={idx})"
        )

    def eod_start_date(self, date: dt.date, window: int, extra: int = 10) -> dt.date:
        """计算拉取 EOD 历史的起始日期,覆盖 window 个交易日 + extra 缓冲.

        用于确保 window 个交易日的数据一定能拉到(calendar_days 转换)。

        Args:
            date:   基准日期(trade_date_used)
            window: 需要的交易日窗口(如 60)
            extra:  额外缓冲交易日数(默认 10,用于应对节假日)
        """
        return self.n_trade_dates_before(date, window + extra)

    def count_trade_days_between(
        self, start: dt.date, end: dt.date, inclusive: str = "both"
    ) -> int:
        """计算 [start, end] 区间内的交易日数量.

        Args:
            inclusive: 'both' / 'left' / 'right' / 'neither'
        """
        if not self._loaded:
            self.load()

        if self._fallback:
            count = 0
            d = start
            while d <= end:
                if d.weekday() < 5:
                    count += 1
                d += dt.timedelta(days=1)
            if inclusive == "left":
                count -= (1 if self.is_trade_date(end) else 0)
            elif inclusive == "right":
                count -= (1 if self.is_trade_date(start) else 0)
            elif inclusive == "neither":
                count -= (1 if self.is_trade_date(start) else 0)
                count -= (1 if self.is_trade_date(end) else 0)
            return max(count, 0)

        lo = _bisect_left(self._trade_dates, start)
        hi = _bisect_right(self._trade_dates, end)
        count = hi - lo
        if inclusive == "left" and end in self._trade_date_set:
            count -= 1
        elif inclusive == "right" and start in self._trade_date_set:
            count -= 1
        elif inclusive == "neither":
            count -= (1 if start in self._trade_date_set else 0)
            count -= (1 if end in self._trade_date_set else 0)
        return max(count, 0)

    @property
    def run_date(self) -> dt.date:
        return self._run_date

    @property
    def is_fallback(self) -> bool:
        return self._fallback

    # ── 向后兼容静态方法 ─────────────────────────────────

    @staticmethod
    def is_weekend(date: dt.date) -> bool:
        return date.weekday() >= 5

    @staticmethod
    def naive_is_trade_day(date: dt.date) -> bool:
        return date.weekday() < 5

    @staticmethod
    def naive_prev_workday(date: dt.date) -> dt.date:
        d = date - dt.timedelta(days=1)
        while d.weekday() >= 5:
            d -= dt.timedelta(days=1)
        return d

    def load_holiday_calendar(self) -> bool:
        """向后兼容接口,等价于 load()."""
        return self.load()


# ── 二分查找辅助(避免引入 bisect 模块歧义)──────────────

def _bisect_left(sorted_list: List[dt.date], target: dt.date) -> int:
    """在有序列表中找 target 的插入位置(左侧)."""
    lo, hi = 0, len(sorted_list)
    while lo < hi:
        mid = (lo + hi) // 2
        if sorted_list[mid] < target:
            lo = mid + 1
        else:
            hi = mid
    return lo


def _bisect_right(sorted_list: List[dt.date], target: dt.date) -> int:
    """在有序列表中找 target 的插入位置(右侧)."""
    lo, hi = 0, len(sorted_list)
    while lo < hi:
        mid = (lo + hi) // 2
        if sorted_list[mid] <= target:
            lo = mid + 1
        else:
            hi = mid
    return lo


def last_n_trade_dates(
    trade_dates: List[dt.date],
    ref: dt.date,
    n: int,
) -> List[dt.date]:
    """返回 ref（含）向前最近 n 个交易日的列表（升序）.

    若 trade_dates 为空（降级），回退为工作日逻辑。
    """
    if not trade_dates:
        result = []
        d = ref
        while len(result) < n:
            if d.weekday() < 5:
                result.append(d)
            d -= dt.timedelta(days=1)
        return list(reversed(result))

    # 二分找 ref 的右边界（第一个 > ref 的位置）
    idx = _bisect_right(trade_dates, ref)
    start = max(0, idx - n)
    return trade_dates[start:idx]
