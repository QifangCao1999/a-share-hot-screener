"""日期与 run_date 工具函数.

核心原则：as-of / point-in-time —— 只能使用 run_date 当天或之前已公开可见的数据。
短线版额外关注：交易日判断（后续 session 可扩展）。
"""

from __future__ import annotations

import datetime as dt
from typing import Optional


def parse_run_date(raw: str) -> dt.date:
    """解析 run_date 字符串为 date 对象.

    支持格式: YYYY-MM-DD / YYYYMMDD / today
    """
    raw = raw.strip().lower()
    if raw == "today":
        return dt.date.today()
    for fmt in ("%Y-%m-%d", "%Y%m%d"):
        try:
            return dt.datetime.strptime(raw, fmt).date()
        except ValueError:
            continue
    raise ValueError(
        f"无法解析 run_date: {raw!r}，请使用 YYYY-MM-DD 或 YYYYMMDD 格式"
    )


def is_on_or_before(date: dt.date, run_date: dt.date) -> bool:
    """判断 date 是否在 run_date 当天或之前（as-of 检查）."""
    return date <= run_date


def ensure_not_future(label: str, data_date: dt.date, run_date: dt.date) -> Optional[str]:
    """若 data_date > run_date，返回 warning 消息；否则返回 None.

    用于 as-of violation 检测。
    """
    if data_date > run_date:
        return (
            f"[as-of violation] {label}: "
            f"data_date={data_date} > run_date={run_date}"
        )
    return None


def days_since(past: dt.date, run_date: dt.date) -> int:
    """返回 past 到 run_date 之间的日历天数（正整数）."""
    return max(0, (run_date - past).days)


def eod_lookback_start(run_date: dt.date, calendar_days: int = 60) -> dt.date:
    """计算 EOD 历史数据的拉取起始日期.

    Args:
        run_date:      当前运行日期
        calendar_days: 向前推多少日历天（默认 60，覆盖约 40 个交易日）

    Returns:
        起始日期（含该日）
    """
    return run_date - dt.timedelta(days=calendar_days)


def date_to_str(d: dt.date) -> str:
    """date → 'YYYY-MM-DD' 字符串."""
    return d.isoformat()


def str_to_date(s: str) -> dt.date:
    """'YYYY-MM-DD' 字符串 → date."""
    return dt.date.fromisoformat(s)


def now_utc_iso() -> str:
    """返回当前 UTC 时间的 ISO8601 字符串，供 metadata 使用."""
    return dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")


# ── 短线专用：近期窗口辅助 ───────────────────────────

def lookback_dates(run_date: dt.date, n_days: int) -> list[dt.date]:
    """返回从 run_date 向前 n_days 个日历天的日期列表（含 run_date）.

    注意：这是日历天，不是交易日。
    TODO: 后续 session 如需精确交易日，接入交易日历。
    """
    return [run_date - dt.timedelta(days=i) for i in range(n_days)]
