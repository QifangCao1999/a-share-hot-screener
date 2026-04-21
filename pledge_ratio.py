"""质押比例检测模块 — Tushare pledge_stat 版.

功能：查询个股当前质押比例。
数据源：Tushare pledge_stat (2000积分)

指标输出：
  pledge_ratio_latest: Optional[float] — 质押占总股本比例（%）
  pledge_ratio_flag: Optional[bool] — True = 质押比例 > 20%
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, List, Optional

logger = logging.getLogger("a_share_hot_screener.pledge_ratio")


@dataclass
class PledgeRatioResult:
    pledge_ratio_latest: Optional[float] = None
    pledge_ratio_flag: Optional[bool] = None
    source: str = "none"
    active_pledge_count: int = 0
    warnings: List[str] = field(default_factory=list)


_PLEDGE_THRESHOLD_PCT = 20.0


def compute_pledge_ratio(
    code: str,
    tushare_client: Any,
    ts_code: str = "",
    run_date: Any = None,
    warnings: Optional[List[str]] = None,
) -> PledgeRatioResult:
    """检测个股质押比例（Tushare pledge_stat）."""
    from a_share_hot_screener.ticker_mapping import code_to_tushare

    result = PledgeRatioResult()

    if not ts_code:
        ts_code = code_to_tushare(code) or ""
    if not ts_code:
        return result

    try:
        df = tushare_client.get_pledge_stat(ts_code)
    except Exception as e:
        msg = f"[pledge_ratio] {code}: pledge_stat 异常: {e}"
        result.warnings.append(msg)
        if warnings is not None:
            warnings.append(msg)
        return result

    if df is None or df.empty:
        # 无质押记录
        result.pledge_ratio_latest = 0.0
        result.pledge_ratio_flag = False
        result.source = "tushare_pledge_stat"
        return result

    try:
        # 取最新一条记录
        row = df.iloc[0]
        ratio = float(row.get("pledge_ratio", 0) or 0)
        count = int(row.get("pledge_count", 0) or 0)

        result.pledge_ratio_latest = round(ratio, 4)
        result.pledge_ratio_flag = ratio > _PLEDGE_THRESHOLD_PCT
        result.source = "tushare_pledge_stat"
        result.active_pledge_count = count
    except Exception as e:
        msg = f"[pledge_ratio] {code}: 数据解析异常: {e}"
        result.warnings.append(msg)
        if warnings is not None:
            warnings.append(msg)

    return result


def reset_market_cache() -> None:
    """兼容旧接口（Tushare 版按个股查询，无全市场缓存）."""
    pass
