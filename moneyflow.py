"""个股资金流向分析模块 — Session 22 新增.

数据源：Tushare moneyflow (2000积分)

指标输出：
  net_main_inflow_ratio_5d: Optional[float] — 近5日主力(大单+特大单)净流入占总成交额比(%)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, List, Optional

import pandas as pd

logger = logging.getLogger("a_share_hot_screener.moneyflow")


@dataclass
class MoneyFlowResult:
    net_main_inflow_ratio_5d: Optional[float] = None
    source: str = "none"
    data_days: int = 0
    warnings: List[str] = field(default_factory=list)


def compute_moneyflow_ratio(
    code: str,
    tushare_client: Any,
    ts_code: str,
    start_date: str,
    end_date: str,
    warnings: Optional[List[str]] = None,
) -> MoneyFlowResult:
    """计算近5日主力资金净流入占比.

    公式：
      net_main_5d = Σ(buy_lg_amount + buy_elg_amount - sell_lg_amount - sell_elg_amount)
      total_buy_5d = Σ(buy_sm_amount + buy_md_amount + buy_lg_amount + buy_elg_amount)
      ratio = net_main_5d / total_buy_5d * 100

    Tushare moneyflow amounts 单位：万元。比率计算单位抵消，无需转换。
    """
    result = MoneyFlowResult()

    try:
        df = tushare_client.get_moneyflow(ts_code, start_date=start_date, end_date=end_date)
    except Exception as e:
        msg = f"[moneyflow] {code}: 异常: {e}"
        result.warnings.append(msg)
        if warnings is not None:
            warnings.append(msg)
        return result

    if df is None or df.empty:
        return result

    # Sort by date descending, take last 5 days
    if "trade_date" in df.columns:
        df = df.sort_values("trade_date", ascending=False).head(5)
    else:
        df = df.head(5)

    result.data_days = len(df)
    if result.data_days < 3:
        msg = f"[moneyflow] {code}: 仅 {result.data_days} 天数据，跳过"
        result.warnings.append(msg)
        if warnings is not None:
            warnings.append(msg)
        return result

    try:
        def _col_sum(col_name: str) -> float:
            if col_name not in df.columns:
                return 0.0
            return pd.to_numeric(df[col_name], errors="coerce").fillna(0).sum()

        buy_lg = _col_sum("buy_lg_amount")
        buy_elg = _col_sum("buy_elg_amount")
        sell_lg = _col_sum("sell_lg_amount")
        sell_elg = _col_sum("sell_elg_amount")
        net_main = (buy_lg + buy_elg) - (sell_lg + sell_elg)

        buy_sm = _col_sum("buy_sm_amount")
        buy_md = _col_sum("buy_md_amount")
        total_buy = buy_sm + buy_md + buy_lg + buy_elg

        if total_buy <= 0:
            return result

        ratio = net_main / total_buy * 100.0
        result.net_main_inflow_ratio_5d = round(ratio, 4)
        result.source = "tushare_moneyflow"
    except Exception as e:
        msg = f"[moneyflow] {code}: 计算异常: {e}"
        result.warnings.append(msg)
        if warnings is not None:
            warnings.append(msg)

    return result
