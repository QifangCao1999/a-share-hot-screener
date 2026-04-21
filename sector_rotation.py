"""板块轮动信号模块 (A4).

需要 6000 积分（600元/年）解锁同花顺板块指数数据。

输出：sector_heat.csv
  - 行业/概念指数近 5/10/20 日涨幅
  - 横截面排名百分位
  - 动量切换信号（前 N 日弱→近 5 日强 = 轮入）

设计原则：
  - 完全独立于 pipeline，可单独运行
  - 不修改已有模型
  - 输出标准 CSV，供后续消费
"""

from __future__ import annotations

import csv
import datetime as dt
import logging
import os
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Set

import pandas as pd

logger = logging.getLogger("a_share_hot_screener.sector_rotation")


# ════════════════════════════════════════════════════════
# 数据结构
# ════════════════════════════════════════════════════════

@dataclass
class SectorHeatRow:
    """单个板块指数的热度分析行。"""

    ts_code: str           # 行业/概念指数代码
    name: str
    type: str              # 'I'=行业, 'N'=概念
    pct_5d: Optional[float] = None
    pct_10d: Optional[float] = None
    pct_20d: Optional[float] = None
    rank_pctile_5d: Optional[float] = None   # 0~1
    rank_pctile_10d: Optional[float] = None
    rank_pctile_20d: Optional[float] = None
    momentum_switch: str = "neutral"
    # rotate_in / rotate_out / steady_strong / steady_weak / neutral
    member_count: int = 0


# ════════════════════════════════════════════════════════
# 动量信号判定
# ════════════════════════════════════════════════════════

def classify_momentum(
    rank_5d: Optional[float],
    rank_20d: Optional[float],
) -> str:
    """判定动量切换信号.

    Args:
        rank_5d: 近5日涨幅排名百分位 (0~1)
        rank_20d: 近20日涨幅排名百分位 (0~1)

    Returns:
        'rotate_in' / 'rotate_out' / 'steady_strong' / 'steady_weak' / 'neutral'
    """
    if rank_5d is None or rank_20d is None:
        return "neutral"

    # 轮入：前期弱势（20d<50%）+ 近期走强（5d>70%）
    if rank_20d < 0.50 and rank_5d > 0.70:
        return "rotate_in"

    # 轮出：前期强势（20d>70%）+ 近期走弱（5d<40%）
    if rank_20d > 0.70 and rank_5d < 0.40:
        return "rotate_out"

    # 持续强势
    if rank_5d > 0.70 and rank_20d > 0.60:
        return "steady_strong"

    # 持续弱势
    if rank_5d < 0.30 and rank_20d < 0.40:
        return "steady_weak"

    return "neutral"


# ════════════════════════════════════════════════════════
# SectorRotationAnalyzer
# ════════════════════════════════════════════════════════

class SectorRotationAnalyzer:
    """板块轮动分析器。

    独立运行，不依赖 pipeline 内部状态。
    需要 TushareClient 且积分 >= 6000。
    """

    def __init__(
        self,
        tushare_client,
        cache,
        run_date: dt.date,
        trade_dates: List[dt.date],
    ) -> None:
        self._ts = tushare_client
        self._cache = cache
        self._run_date = run_date
        self._trade_dates = trade_dates

    def analyze(self) -> List[SectorHeatRow]:
        """主分析流程。返回所有板块的热度行列表。"""
        logger.info("[sector_rotation] 开始板块轮动分析 run_date=%s", self._run_date)

        # 1. 获取板块列表
        industry_df = self._ts.get_ths_index(exchange="A", type_="I")
        concept_df = self._ts.get_ths_index(exchange="A", type_="N")

        if industry_df is None and concept_df is None:
            logger.warning("[sector_rotation] ths_index 无权限，无法分析")
            return []

        # 2. 确定日期窗口
        dates_20d = self._prev_n_trade_dates(20)
        dates_10d = dates_20d[-10:] if len(dates_20d) >= 10 else dates_20d
        dates_5d = dates_20d[-5:] if len(dates_20d) >= 5 else dates_20d

        if len(dates_5d) < 2:
            logger.warning("[sector_rotation] 交易日不足，跳过")
            return []

        # 3. 拉取全部行情数据
        daily_frames = []
        for d in dates_20d:
            date_str = d.strftime("%Y%m%d")
            df = self._ts.get_ths_daily(trade_date=date_str)
            if df is not None and not df.empty:
                daily_frames.append(df)

        if len(daily_frames) < 2:
            logger.warning("[sector_rotation] ths_daily 数据不足")
            return []

        all_daily = pd.concat(daily_frames, ignore_index=True)

        # 4. 合并处理
        results: List[SectorHeatRow] = []

        for idx_df, type_label in [(industry_df, "I"), (concept_df, "N")]:
            if idx_df is None or idx_df.empty:
                continue

            codes = set(idx_df["ts_code"].dropna())
            code_to_name = dict(zip(idx_df["ts_code"], idx_df["name"]))
            code_to_count = dict(zip(idx_df["ts_code"], idx_df.get("count", pd.Series(dtype=int))))

            # 提取该类型的行情
            type_daily = all_daily[all_daily["ts_code"].isin(codes)].copy()
            if type_daily.empty:
                continue

            # 计算三个窗口的涨幅
            pct_5d = _calc_period_pct(type_daily, codes, dates_5d)
            pct_10d = _calc_period_pct(type_daily, codes, dates_10d)
            pct_20d = _calc_period_pct(type_daily, codes, dates_20d)

            # 排名百分位
            rank_5d = _compute_rank_pctile(pct_5d)
            rank_10d = _compute_rank_pctile(pct_10d)
            rank_20d = _compute_rank_pctile(pct_20d)

            for code in codes:
                name = code_to_name.get(code, code)
                count = code_to_count.get(code, 0)
                try:
                    count = int(count) if count and not pd.isna(count) else 0
                except (ValueError, TypeError):
                    count = 0

                r5 = rank_5d.get(code)
                r20 = rank_20d.get(code)
                momentum = classify_momentum(r5, r20)

                results.append(SectorHeatRow(
                    ts_code=code,
                    name=name,
                    type=type_label,
                    pct_5d=pct_5d.get(code),
                    pct_10d=pct_10d.get(code),
                    pct_20d=pct_20d.get(code),
                    rank_pctile_5d=r5,
                    rank_pctile_10d=rank_10d.get(code),
                    rank_pctile_20d=r20,
                    momentum_switch=momentum,
                    member_count=count,
                ))

        # 排序：按 momentum 类型优先级 + pct_5d 降序
        _momentum_order = {
            "rotate_in": 0, "steady_strong": 1, "neutral": 2,
            "rotate_out": 3, "steady_weak": 4,
        }
        results.sort(key=lambda r: (
            _momentum_order.get(r.momentum_switch, 9),
            -(r.pct_5d or -999),
        ))

        logger.info(
            "[sector_rotation] 分析完成: %d 个板块 (rotate_in=%d, rotate_out=%d, "
            "steady_strong=%d, steady_weak=%d)",
            len(results),
            sum(1 for r in results if r.momentum_switch == "rotate_in"),
            sum(1 for r in results if r.momentum_switch == "rotate_out"),
            sum(1 for r in results if r.momentum_switch == "steady_strong"),
            sum(1 for r in results if r.momentum_switch == "steady_weak"),
        )
        return results

    def to_csv(self, rows: List[SectorHeatRow], output_path: str) -> str:
        """写出 sector_heat.csv，返回文件路径。"""
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

        fieldnames = [
            "ts_code", "name", "type",
            "pct_5d", "pct_10d", "pct_20d",
            "rank_pctile_5d", "rank_pctile_10d", "rank_pctile_20d",
            "momentum_switch", "member_count",
        ]

        with open(output_path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                d = asdict(row)
                # 格式化浮点数
                for k in ["pct_5d", "pct_10d", "pct_20d",
                           "rank_pctile_5d", "rank_pctile_10d", "rank_pctile_20d"]:
                    if d[k] is not None:
                        d[k] = round(d[k], 4)
                writer.writerow(d)

        logger.info("[sector_rotation] 已写出 %s (%d 行)", output_path, len(rows))
        return output_path

    # ── 辅助 ──────────────────────────────────────────────

    def _prev_n_trade_dates(self, n: int) -> List[dt.date]:
        """获取 run_date 之前（含）的 n 个交易日。"""
        if not self._trade_dates:
            result = []
            d = self._run_date
            while len(result) < n:
                if d.weekday() < 5:
                    result.append(d)
                d -= dt.timedelta(days=1)
            return list(reversed(result))

        from a_share_hot_screener.event_layer import _bisect_right
        idx = _bisect_right(self._trade_dates, self._run_date)
        start = max(0, idx - n)
        return self._trade_dates[start:idx]


# ════════════════════════════════════════════════════════
# 工具函数
# ════════════════════════════════════════════════════════

def _calc_period_pct(
    daily_df: pd.DataFrame,
    codes: Set[str],
    dates: List[dt.date],
) -> Dict[str, float]:
    """计算一组指数在指定交易日窗口内的累计涨跌幅。"""
    if daily_df.empty or len(dates) < 2:
        return {}

    date_strs = {d.strftime("%Y%m%d") for d in dates}
    sub = daily_df[daily_df["trade_date"].astype(str).isin(date_strs)].copy()
    if sub.empty:
        return {}

    from a_share_hot_screener.event_layer import _calc_period_pct_change
    return _calc_period_pct_change(sub, codes)


def _compute_rank_pctile(pct_map: Dict[str, float]) -> Dict[str, float]:
    """对涨幅字典计算横截面排名百分位 (0~1)。"""
    if not pct_map:
        return {}

    sorted_items = sorted(pct_map.items(), key=lambda x: x[1])
    n = len(sorted_items)
    return {code: (rank + 0.5) / n for rank, (code, _) in enumerate(sorted_items)}
