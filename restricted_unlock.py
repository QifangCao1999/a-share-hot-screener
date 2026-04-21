"""限售解禁检测模块 — Tushare share_float 版.

功能：查询个股未来20天内限售解禁计划。
数据源：Tushare share_float (3000积分)

指标输出：
  restricted_shares_unlock_ratio_20d: Optional[float] — 未来20日解禁占比（%）
  unlock_risk_flag_20d: Optional[bool] — True = 比例 > 5%
"""

from __future__ import annotations

import datetime as dt
import logging
from dataclasses import dataclass, field
from typing import Any, List, Optional

import pandas as pd

logger = logging.getLogger("a_share_hot_screener.restricted_unlock")


@dataclass
class RestrictedUnlockResult:
    restricted_shares_unlock_ratio_20d: Optional[float] = None
    unlock_risk_flag_20d: Optional[bool] = None
    source: str = "none"
    upcoming_batch_count: int = 0
    nearest_unlock_date: Optional[str] = None
    warnings: List[str] = field(default_factory=list)


_UNLOCK_THRESHOLD_PCT = 5.0
_FORWARD_WINDOW_DAYS = 20


def compute_restricted_unlock(
    code: str,
    tushare_client: Any,
    run_date: dt.date,
    ts_code: str = "",
    warnings: Optional[List[str]] = None,
) -> RestrictedUnlockResult:
    """检测个股未来20天限售解禁（Tushare share_float）."""
    from a_share_hot_screener.ticker_mapping import code_to_tushare

    result = RestrictedUnlockResult()

    if not ts_code:
        ts_code = code_to_tushare(code) or ""
    if not ts_code:
        return result

    try:
        df = tushare_client.get_share_float(ts_code)
    except Exception as e:
        msg = f"[restricted_unlock] {code}: share_float 异常: {e}"
        result.warnings.append(msg)
        if warnings is not None:
            warnings.append(msg)
        return result

    if df is None or df.empty:
        result.restricted_shares_unlock_ratio_20d = 0.0
        result.unlock_risk_flag_20d = False
        result.source = "tushare_share_float"
        return result

    try:
        ratio, count, nearest = _calc_upcoming_unlock(df, run_date)
    except Exception as e:
        msg = f"[restricted_unlock] {code}: 解析异常: {e}"
        result.warnings.append(msg)
        if warnings is not None:
            warnings.append(msg)
        return result

    result.restricted_shares_unlock_ratio_20d = round(ratio, 4)
    result.unlock_risk_flag_20d = ratio > _UNLOCK_THRESHOLD_PCT
    result.source = "tushare_share_float"
    result.upcoming_batch_count = count
    result.nearest_unlock_date = nearest

    return result


def _calc_upcoming_unlock(df: pd.DataFrame, run_date: dt.date) -> tuple:
    """计算未来20天解禁比例.

    Tushare share_float 字段：float_date(YYYYMMDD), float_share(万股), float_ratio(%)
    """
    if "float_date" not in df.columns:
        return 0.0, 0, None

    window_end = run_date + dt.timedelta(days=_FORWARD_WINDOW_DAYS)
    run_str = run_date.strftime("%Y%m%d")
    end_str = window_end.strftime("%Y%m%d")

    total_ratio = 0.0
    count = 0
    nearest = None

    for _, row in df.iterrows():
        try:
            fd = str(row.get("float_date", ""))
            if len(fd) != 8:
                continue
            # 严格区间 (run_date, run_date + 20]
            if fd <= run_str:
                continue
            if fd > end_str:
                continue

            ratio = float(row.get("float_ratio", 0) or 0)
            total_ratio += ratio
            count += 1

            if nearest is None or fd < nearest:
                nearest = fd
        except (ValueError, TypeError):
            continue

    # 格式化 nearest
    if nearest and len(nearest) == 8:
        nearest = f"{nearest[:4]}-{nearest[4:6]}-{nearest[6:8]}"

    return total_ratio, count, nearest
