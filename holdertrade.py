"""股东增减持分析模块 — Session 22 新增.

数据源：Tushare stk_holdertrade (2000积分)

指标输出：
  net_holder_reduction_ratio_30d: Optional[float] — 近30日股东净减持占流通比(%)
    正值=净减持，负值=净增持
"""

from __future__ import annotations

import datetime as dt
import logging
from dataclasses import dataclass, field
from typing import Any, List, Optional

logger = logging.getLogger("a_share_hot_screener.holdertrade")


@dataclass
class HolderTradeResult:
    net_holder_reduction_ratio_30d: Optional[float] = None
    source: str = "none"
    increase_count: int = 0
    decrease_count: int = 0
    warnings: List[str] = field(default_factory=list)


def compute_holder_trade_reduction(
    code: str,
    tushare_client: Any,
    run_date: dt.date,
    ts_code: str,
    warnings: Optional[List[str]] = None,
) -> HolderTradeResult:
    """计算近30日股东净减持占流通比.

    逻辑：
      1. 拉取近60天公告（buffer）
      2. 筛选 ann_date 在 [run_date-30, run_date] 范围内
      3. in_de='DE' 的 change_ratio 求和（减持占流通%）
      4. in_de='IN' 的 change_ratio 求和（增持占流通%）
      5. 净减持 = 减持 - 增持（正值=净减持，负值=净增持）
    """
    result = HolderTradeResult()

    start_date = (run_date - dt.timedelta(days=60)).strftime("%Y%m%d")
    end_date = run_date.strftime("%Y%m%d")

    try:
        df = tushare_client.get_stk_holdertrade(
            ts_code=ts_code, start_date=start_date, end_date=end_date
        )
    except Exception as e:
        msg = f"[holdertrade] {code}: 异常: {e}"
        result.warnings.append(msg)
        if warnings is not None:
            warnings.append(msg)
        return result

    if df is None or df.empty:
        # 无增减持记录 = 安全
        result.net_holder_reduction_ratio_30d = 0.0
        result.source = "tushare_stk_holdertrade"
        return result

    cutoff_str = (run_date - dt.timedelta(days=30)).strftime("%Y%m%d")

    try:
        total_reduction = 0.0
        total_increase = 0.0

        for _, row in df.iterrows():
            ann_date = str(row.get("ann_date", ""))
            if len(ann_date) != 8:
                continue
            if ann_date < cutoff_str or ann_date > end_date:
                continue

            in_de = str(row.get("in_de", "")).upper()
            change_ratio = float(row.get("change_ratio", 0) or 0)

            if in_de == "DE":
                total_reduction += abs(change_ratio)
                result.decrease_count += 1
            elif in_de == "IN":
                total_increase += abs(change_ratio)
                result.increase_count += 1

        net_reduction = total_reduction - total_increase
        result.net_holder_reduction_ratio_30d = round(net_reduction, 4)
        result.source = "tushare_stk_holdertrade"
    except Exception as e:
        msg = f"[holdertrade] {code}: 计算异常: {e}"
        result.warnings.append(msg)
        if warnings is not None:
            warnings.append(msg)

    return result
