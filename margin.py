"""融资融券分析模块 — Session 22 新增.

数据源：Tushare margin_detail (2000积分)

指标输出：
  margin_buy_net_ratio_5d: Optional[float] — 近5日融资净买入占成交额比(%)
  short_sell_ratio_change_5d: Optional[float] — 近5日融券余额变化率(%)
  is_margin_eligible: bool — 是否为两融标的

仅覆盖约1800只两融标的，非两融标的 is_margin_eligible=False。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, List, Optional

import pandas as pd

logger = logging.getLogger("a_share_hot_screener.margin")


@dataclass
class MarginResult:
    margin_buy_net_ratio_5d: Optional[float] = None
    short_sell_ratio_change_5d: Optional[float] = None
    is_margin_eligible: bool = False
    source: str = "none"
    data_days: int = 0
    warnings: List[str] = field(default_factory=list)


def compute_margin_metrics(
    code: str,
    tushare_client: Any,
    ts_code: str,
    start_date: str,
    end_date: str,
    amount_avg_5d: Optional[float] = None,
    warnings: Optional[List[str]] = None,
) -> MarginResult:
    """计算融资融券相关指标.

    TF7: 近5日融资净买入占成交额比
      = Σ(rzmre - rzche) / (amount_avg_5d × data_days) × 100

    RC10: 近5日融券余额变化率
      = (latest_rqye - earliest_rqye) / earliest_rqye × 100

    单位：margin_detail 金额均为元。amount_avg_5d 已为元。
    """
    result = MarginResult()

    try:
        df = tushare_client.get_margin_detail(
            ts_code=ts_code, start_date=start_date, end_date=end_date
        )
    except Exception as e:
        msg = f"[margin] {code}: 异常: {e}"
        result.warnings.append(msg)
        if warnings is not None:
            warnings.append(msg)
        return result

    if df is None or df.empty:
        result.is_margin_eligible = False
        return result

    result.is_margin_eligible = True

    if "trade_date" in df.columns:
        df = df.sort_values("trade_date", ascending=False).head(5)
    else:
        df = df.head(5)

    result.data_days = len(df)
    if result.data_days < 3:
        msg = f"[margin] {code}: 仅 {result.data_days} 天数据"
        result.warnings.append(msg)
        if warnings is not None:
            warnings.append(msg)
        return result

    try:
        # ── TF7: 融资净买入占比 ──
        rzmre = pd.to_numeric(df.get("rzmre", 0), errors="coerce").fillna(0).sum()
        rzche = pd.to_numeric(df.get("rzche", 0), errors="coerce").fillna(0).sum()
        margin_net_buy = rzmre - rzche

        if amount_avg_5d is not None and amount_avg_5d > 0:
            total_amount_5d = amount_avg_5d * result.data_days
            result.margin_buy_net_ratio_5d = round(
                margin_net_buy / total_amount_5d * 100, 4
            )

        # ── RC10: 融券余额变化率 ──
        df_asc = df.sort_values("trade_date", ascending=True)
        earliest_rqye = pd.to_numeric(
            df_asc.iloc[0].get("rqye", 0) if "rqye" in df_asc.columns else 0,
            errors="coerce",
        )
        latest_rqye = pd.to_numeric(
            df_asc.iloc[-1].get("rqye", 0) if "rqye" in df_asc.columns else 0,
            errors="coerce",
        )

        if pd.notna(earliest_rqye) and earliest_rqye > 0:
            change_pct = (latest_rqye - earliest_rqye) / earliest_rqye * 100
            result.short_sell_ratio_change_5d = round(change_pct, 4)
        elif pd.notna(latest_rqye) and latest_rqye > 0:
            result.short_sell_ratio_change_5d = 100.0
        else:
            result.short_sell_ratio_change_5d = 0.0

        result.source = "tushare_margin_detail"
    except Exception as e:
        msg = f"[margin] {code}: 计算异常: {e}"
        result.warnings.append(msg)
        if warnings is not None:
            warnings.append(msg)

    return result
