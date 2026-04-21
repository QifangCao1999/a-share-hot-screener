"""股东减持检测模块 — Tushare stk_holdernumber 版.

功能：检测近3个月内股东人数变化趋势，推断减持行为。
数据源：Tushare stk_holdernumber (2000积分)

指标输出：
  shareholder_net_reduction_ratio_3m: Optional[float] — 人数变化率（%）
  shareholder_reduction_flag_3m: Optional[bool] — True = 增幅 > 10%
"""

from __future__ import annotations

import datetime as dt
import logging
from dataclasses import dataclass, field
from typing import Any, List, Optional

logger = logging.getLogger("a_share_hot_screener.shareholder_reduction")


@dataclass
class ShareholderReductionResult:
    shareholder_net_reduction_ratio_3m: Optional[float] = None
    shareholder_reduction_flag_3m: Optional[bool] = None
    source: str = "none"
    data_points: int = 0
    latest_report_date: Optional[str] = None
    warnings: List[str] = field(default_factory=list)


_REDUCTION_THRESHOLD_PCT = 10.0


def compute_shareholder_reduction(
    code: str,
    tushare_client: Any,
    run_date: dt.date,
    ts_code: str = "",
    warnings: Optional[List[str]] = None,
) -> ShareholderReductionResult:
    """检测个股股东减持（Tushare stk_holdernumber）."""
    from a_share_hot_screener.ticker_mapping import code_to_tushare

    result = ShareholderReductionResult()

    if not ts_code:
        ts_code = code_to_tushare(code) or ""
    if not ts_code:
        return result

    # 查最近1年数据
    start_date = (run_date - dt.timedelta(days=365)).strftime("%Y%m%d")

    try:
        df = tushare_client.get_stk_holdernumber(ts_code, start_date=start_date)
    except Exception as e:
        msg = f"[shareholder_reduction] {code}: stk_holdernumber 异常: {e}"
        result.warnings.append(msg)
        if warnings is not None:
            warnings.append(msg)
        return result

    if df is None or df.empty:
        msg = f"[shareholder_reduction] {code}: 股东人数数据不可用"
        result.warnings.append(msg)
        if warnings is not None:
            warnings.append(msg)
        return result

    try:
        points = _parse_hold_num_points(df, run_date)
    except Exception as e:
        msg = f"[shareholder_reduction] {code}: 解析异常: {e}"
        result.warnings.append(msg)
        if warnings is not None:
            warnings.append(msg)
        return result

    result.data_points = len(points)
    if not points:
        return result

    result.latest_report_date = points[-1][0]
    result.source = "tushare_stk_holdernumber"

    if len(points) < 2:
        msg = f"[shareholder_reduction] {code}: 只有1个数据点"
        result.warnings.append(msg)
        if warnings is not None:
            warnings.append(msg)
        return result

    # 近3个月
    cutoff = (run_date - dt.timedelta(days=92)).isoformat()
    recent = [(d, n) for d, n in points if d >= cutoff]
    if len(recent) < 2:
        recent = points[-2:]

    if len(recent) < 2:
        return result

    earliest_num = recent[0][1]
    latest_num = recent[-1][1]

    if earliest_num <= 0:
        return result

    ratio = (latest_num - earliest_num) / earliest_num * 100.0
    result.shareholder_net_reduction_ratio_3m = round(ratio, 2)
    result.shareholder_reduction_flag_3m = ratio > _REDUCTION_THRESHOLD_PCT

    return result


def _parse_hold_num_points(df, run_date: dt.date) -> List[tuple]:
    """解析 Tushare stk_holdernumber 为 [(date_str, holder_count)] 列表.

    Tushare 字段: ts_code, ann_date(YYYYMMDD), end_date(YYYYMMDD), holder_num
    """
    import pandas as pd

    run_date_str = run_date.isoformat()
    points = []

    for _, row in df.iterrows():
        try:
            # 使用 end_date 作为报告期日期
            raw_date = str(row.get("end_date") or row.get("ann_date") or "")
            if len(raw_date) == 8 and raw_date.isdigit():
                d_str = f"{raw_date[:4]}-{raw_date[4:6]}-{raw_date[6:8]}"
            elif "-" in raw_date:
                d_str = raw_date[:10]
            else:
                continue

            if d_str > run_date_str:
                continue

            num = int(float(row.get("holder_num", 0) or 0))
            if num > 0:
                points.append((d_str, num))
        except (ValueError, TypeError):
            continue

    points.sort(key=lambda x: x[0])
    return points
