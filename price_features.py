"""日线派生特征计算层.

数据来源：Tushare daily API
  字段：trade_date / open / high / low / close / pre_close / vol(手) / amount(千元)
  Tushare 提供精确成交额，无需近似计算。

窗口规范：
  所有窗口均以"交易日数"计，不是"日历天数"。
  输入 eod_rows 已经过 as-of 截断（最晚日期 ≤ trade_date_used）。

as-of 口径：
  所有特征只使用 trade_date_used 当日及之前的数据。

字段覆盖率：
  若数据行数不足某窗口，对应字段返回 None + 在 coverage_notes 中记录。
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger("a_share_hot_screener.price_features")


# ════════════════════════════════════════════════════════
# 内部工具：构建归一化后的 EOD 行列表
# ════════════════════════════════════════════════════════

def _parse_eod_rows(raw_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """解析并清洗 Tushare daily 行，按日期升序排列.

    Tushare daily 字段:
      trade_date(YYYYMMDD) / open / high / low / close / vol(手) / amount(千元)
    也兼容旧格式 (date / volume / adjusted_close)。

    Returns:
        list of dict：{date_str(YYYY-MM-DD), open, high, low, close, volume(股), amount(元)}
    """
    parsed = []
    for r in raw_rows:
        try:
            # 日期：支持 trade_date(YYYYMMDD) 和 date(YYYY-MM-DD) 两种
            raw_date = str(r.get("trade_date") or r.get("date") or "")
            if len(raw_date) == 8 and raw_date.isdigit():
                date_str = f"{raw_date[:4]}-{raw_date[4:6]}-{raw_date[6:8]}"
            elif len(raw_date) >= 10 and "-" in raw_date:
                date_str = raw_date[:10]
            else:
                continue

            close = float(r.get("close") or 0)
            open_ = float(r.get("open") or 0)
            high = float(r.get("high") or 0)
            low = float(r.get("low") or 0)

            # volume: tushare vol(手) → 股, 或旧格式 volume(股)
            if "vol" in r:
                volume = float(r.get("vol") or 0) * 100  # 手→股
            else:
                volume = float(r.get("volume") or 0)

            # amount: tushare amount(千元) → 元, 或旧格式 amount(元)
            raw_amount = r.get("amount")
            if raw_amount is not None and "vol" in r:
                # tushare 格式：千元→元
                amount = float(raw_amount) * 1000
            elif raw_amount is not None:
                amount = float(raw_amount)
            else:
                amount = close * volume  # fallback

            if close <= 0 or volume < 0:
                continue

            parsed.append({
                "date": date_str,
                "open": open_,
                "high": high,
                "low": low,
                "close": close,
                "volume": volume,
                "amount": amount,
            })
        except (KeyError, ValueError, TypeError):
            continue
    return sorted(parsed, key=lambda x: x["date"])


def _safe_mean(vals: List[float]) -> Optional[float]:
    if not vals:
        return None
    return sum(vals) / len(vals)


def _safe_div(a: Optional[float], b: Optional[float]) -> Optional[float]:
    if a is None or b is None or b == 0:
        return None
    return a / b


# ════════════════════════════════════════════════════════
# 主函数：compute_price_features
# ════════════════════════════════════════════════════════

@dataclass
class PriceFeatures:
    """所有日线派生特征的容器."""

    # ── 最新行情（trade_date_used 当日或最近一个有数据的交易日）
    latest_date: Optional[str] = None          # 最新 EOD 日期（YYYY-MM-DD）
    latest_close: Optional[float] = None       # 最新收盘价
    latest_volume: Optional[float] = None      # 最新成交量（股）
    latest_amount: Optional[float] = None      # 最新成交额近似（元）

    # ── 涨跌幅（相对 trade_date_used 收盘价，向前 N 个交易日开盘前计算）
    return_3d: Optional[float] = None          # 近3个交易日收益率（%）
    return_5d: Optional[float] = None          # 近5个交易日收益率（%）
    return_10d: Optional[float] = None         # 近10个交易日收益率（%）

    # ── 成交额均值（）
    amount_avg_5d: Optional[float] = None      # 近5日均成交额（元）
    amount_avg_20d: Optional[float] = None     # 近20日均成交额（元）

    # ── 量比（最新成交量 / 近20日均量）
    volume_ratio_20d: Optional[float] = None   # 量比（20日均量基准）

    # ── 收盘位置（latest_close 在近20日高低点区间的位置，0~1）
    close_position_20d: Optional[float] = None  # 0=区间最低，1=区间最高

    # ── CLV（收盘价位置指标，类似 Williams %R 简化版）
    # CLV = (close - low) / (high - low)，当日值
    clv_latest: Optional[float] = None         # 当日 CLV（0~1）

    # ── 成交额比率（近5日均额 / 近20日均额，衡量量能放大程度）
    amount_ratio_5d_to_20d: Optional[float] = None

    # ── 价格与 MA10 的距离（归一化，正=在上方）
    abs_distance_to_ma10: Optional[float] = None  # (close - MA10) / MA10

    # ── 振幅均值（近5日振幅均值，用于评估波动强度）
    # NOTE: 振幅 = (high - low) / prev_close
    amp_norm_avg_5d: Optional[float] = None    # 近5日归一化振幅均值（%）

    # ── 上影线计数（近5日，proxy：upper_shadow > close*0.02）
    # upper_shadow = high - max(open, close)
    # NOTE: 此为 proxy，不能精确区分"真上影"与盘中波动
    upper_shadow_count_5d: Optional[int] = None

    # ── 一字板 proxy 计数（近5日）
    # 一字板 proxy：open == high == low == close（精确）或
    # abs(open-close)/close < 0.005 且 (high-low)/close < 0.01（近似）
    limit_board_count_5d: Optional[int] = None  # 近5日疑似一字板/T字板数量

    # ── 大涨天数（Session 11 新增）
    # 近10日内单日涨幅 > 阈值（默认 5%）的天数
    big_up_count_10d: Optional[int] = None      # 近10日大涨天数

    # ── 连续上涨天数（Session 11 新增计算）
    # 从最新交易日向前，连续收盘价上涨的天数
    consec_up_days: Optional[int] = None        # 连续收涨天数

    # ── 均线字段（Session 8 P1 新增）
    ma5: Optional[float] = None                # MA5（近5日收盘价均值）
    ma10: Optional[float] = None               # MA10（近10日收盘价均值）
    ma20: Optional[float] = None               # MA20（近20日收盘价均值）

    # ── 最新日一字板判定（Session 8 P1 新增）
    # 基于 open == high == low == close（浮点容差） + 涨跌停制度
    latest_is_limit_board: Optional[bool] = None        # 最新日是否一字板（proxy）
    latest_pct_change: Optional[float] = None           # 最新日涨跌幅（%）

    # ── 数据状态
    data_rows: int = 0                          # 日线实际可用行数
    coverage_notes: List[str] = field(default_factory=list)  # 覆盖率不足提示


def compute_price_features(
    raw_eod: List[Dict[str, Any]],
    trade_date_used_str: str,
    warnings: Optional[List[str]] = None,
) -> PriceFeatures:
    """从 Tushare daily 原始数据计算所有日线派生特征.

    Args:
        raw_eod:           Tushare get_eod_prices() 返回的原始 list
        trade_date_used_str: 'YYYY-MM-DD'，as-of 截断基准日
        warnings:          可选的 warning 列表，用于追加 coverage 不足的提示

    Returns:
        PriceFeatures
    """
    feat = PriceFeatures()
    if not raw_eod:
        feat.coverage_notes.append("日线数据为空")
        return feat

    rows = _parse_eod_rows(raw_eod)
    # as-of 截断：只保留 ≤ trade_date_used 的数据
    rows = [r for r in rows if r["date"] <= trade_date_used_str]

    feat.data_rows = len(rows)
    if not rows:
        feat.coverage_notes.append("as-of 截断后 日线数据为空")
        return feat

    # ── 最新行情 ────────────────────────────────────────
    latest = rows[-1]
    feat.latest_date = latest["date"]
    feat.latest_close = latest["close"]
    feat.latest_volume = latest["volume"]
    feat.latest_amount = latest["amount"]

    # ── CLV（当日）──────────────────────────────────────
    h, l, c = latest["high"], latest["low"], latest["close"]
    if h > l:
        feat.clv_latest = (c - l) / (h - l)
    elif h == l:
        feat.clv_latest = 0.5  # 一字板：取中间值

    # ── 最新日一字板判定 + 涨跌幅（Session 8 P1）────────
    o_latest = latest["open"]
    h_latest, l_latest = latest["high"], latest["low"]
    # 一字板 proxy: open == high == low == close（浮点容差 0.5%）
    if c > 0:
        price_range_pct = (h_latest - l_latest) / c
        body_pct = abs(o_latest - c) / c
        feat.latest_is_limit_board = (price_range_pct < 0.005 and body_pct < 0.003)
    # 最新日涨跌幅
    if len(rows) >= 2:
        prev_close = rows[-2]["close"]
        if prev_close > 0:
            feat.latest_pct_change = (c - prev_close) / prev_close * 100.0

    # ── 涨跌幅 ──────────────────────────────────────────
    feat.return_3d = _return_nd(rows, n=3)
    feat.return_5d = _return_nd(rows, n=5)
    feat.return_10d = _return_nd(rows, n=10)

    if feat.return_3d is None:
        feat.coverage_notes.append("return_3d: 行数不足 4（≥4 才能算）")
    if feat.return_5d is None:
        feat.coverage_notes.append("return_5d: 行数不足 6")
    if feat.return_10d is None:
        feat.coverage_notes.append("return_10d: 行数不足 11")

    # ── 成交额均值（）──────────
    amounts = [r["amount"] for r in rows]
    feat.amount_avg_5d = _safe_mean(amounts[-5:]) if len(amounts) >= 5 else None
    feat.amount_avg_20d = _safe_mean(amounts[-20:]) if len(amounts) >= 20 else None

    if feat.amount_avg_5d is None:
        feat.coverage_notes.append("amount_avg_5d: 行数不足 5（）")
    if feat.amount_avg_20d is None:
        feat.coverage_notes.append("amount_avg_20d: 行数不足 20（）")

    # ── 量比（20日均量基准）─────────────────────────────
    volumes = [r["volume"] for r in rows]
    if len(volumes) >= 20:
        avg_vol_20d = _safe_mean(volumes[-20:])
        feat.volume_ratio_20d = _safe_div(feat.latest_volume, avg_vol_20d)
    else:
        feat.coverage_notes.append("volume_ratio_20d: 行数不足 20")

    # ── 收盘位置（近20日高低区间）──────────────────────
    if len(rows) >= 20:
        recent_20 = rows[-20:]
        high_20 = max(r["high"] for r in recent_20)
        low_20 = min(r["low"] for r in recent_20)
        if high_20 > low_20:
            feat.close_position_20d = (c - low_20) / (high_20 - low_20)
        else:
            feat.close_position_20d = 0.5
    else:
        feat.coverage_notes.append("close_position_20d: 行数不足 20")

    # ── 成交额比率 ──────────────────────────────────────
    feat.amount_ratio_5d_to_20d = _safe_div(feat.amount_avg_5d, feat.amount_avg_20d)

    # ── 均线计算（MA5/MA10/MA20）+ MA10 距离 ──────────────
    closes = [r["close"] for r in rows]
    if len(closes) >= 5:
        feat.ma5 = _safe_mean(closes[-5:])
    else:
        feat.coverage_notes.append("ma5: 行数不足 5")
    if len(closes) >= 10:
        feat.ma10 = _safe_mean(closes[-10:])
        feat.abs_distance_to_ma10 = _safe_div(c - (feat.ma10 or 0), feat.ma10) if feat.ma10 else None
    else:
        feat.coverage_notes.append("abs_distance_to_ma10/ma10: 行数不足 10")
    if len(closes) >= 20:
        feat.ma20 = _safe_mean(closes[-20:])
    else:
        feat.coverage_notes.append("ma20: 行数不足 20")

    # ── 近5日振幅均值 ────────────────────────────────────
    if len(rows) >= 5:
        feat.amp_norm_avg_5d = _amp_avg(rows[-5:])
    else:
        feat.coverage_notes.append("amp_norm_avg_5d: 行数不足 5")

    # ── 近5日上影线计数（proxy）──────────────────────────
    if len(rows) >= 5:
        feat.upper_shadow_count_5d = _count_upper_shadow(rows[-5:], threshold_pct=0.02)
    else:
        feat.coverage_notes.append("upper_shadow_count_5d: 行数不足 5")

    # ── 近5日一字板 proxy 计数 ────────────────────────────
    if len(rows) >= 5:
        feat.limit_board_count_5d = _count_limit_board_proxy(rows[-5:])
    else:
        feat.coverage_notes.append("limit_board_count_5d: 行数不足 5")

    # ── 近10日大涨天数（Session 11 新增）───────────────────
    if len(rows) >= 2:
        feat.big_up_count_10d = _count_big_up_days(rows, n=10, threshold_pct=5.0)
    else:
        feat.coverage_notes.append("big_up_count_10d: 行数不足 2")

    # ── 连续上涨天数（Session 11 新增）──────────────────
    if len(rows) >= 2:
        feat.consec_up_days = _count_consec_up_days(rows)
    else:
        feat.coverage_notes.append("consec_up_days: 行数不足 2")

    # 追加到外部 warnings
    if warnings is not None:
        for note in feat.coverage_notes:
            warnings.append(f"[price_features] {note}")

    return feat


# ── 内部计算函数 ─────────────────────────────────────────

def _return_nd(rows: List[Dict], n: int) -> Optional[float]:
    """近 n 个交易日收益率（%）.

    定义：(close_latest - close_{n个交易日前}) / close_{n个交易日前} * 100
    需要至少 n+1 行数据。
    """
    if len(rows) < n + 1:
        return None
    c_now = rows[-1]["close"]
    c_prev = rows[-(n + 1)]["close"]
    if c_prev <= 0:
        return None
    return (c_now - c_prev) / c_prev * 100.0


def _amp_avg(rows: List[Dict]) -> Optional[float]:
    """计算近 N 行的平均振幅（%）.

    振幅 = (high - low) / prev_close * 100
    第一行无 prev_close 时用 open 代替。
    """
    if not rows:
        return None
    amps = []
    for i, r in enumerate(rows):
        prev_close = rows[i - 1]["close"] if i > 0 else r["open"]
        if prev_close and prev_close > 0:
            amp = (r["high"] - r["low"]) / prev_close * 100.0
            amps.append(amp)
    return _safe_mean(amps)


def _count_upper_shadow(
    rows: List[Dict], threshold_pct: float = 0.02
) -> int:
    """统计近 N 行中上影线超过 threshold_pct 的天数（proxy）.

    upper_shadow = high - max(open, close)
    若 upper_shadow / close > threshold_pct，视为有显著上影线。
    NOTE: proxy，不能完全区分真实龙头上影与盘中波动。
    """
    count = 0
    for r in rows:
        c = r["close"]
        o = r["open"]
        h = r["high"]
        if c <= 0:
            continue
        upper = h - max(o, c)
        if upper / c > threshold_pct:
            count += 1
    return count


def _count_limit_board_proxy(rows: List[Dict]) -> int:
    """统计近 N 行中疑似一字板/T字板的天数（proxy）.

    精确条件：open == high == low == close（严格一字板）
    近似条件：|open - close| / close < 0.005 且 (high - low) / close < 0.015
    NOTE: proxy，不能区分停牌与真实一字板。
    """
    count = 0
    for r in rows:
        c = r["close"]
        if c <= 0:
            continue
        o, h, l = r["open"], r["high"], r["low"]
        price_range = (h - l) / c
        body_size = abs(o - c) / c
        if price_range < 0.015 and body_size < 0.005:
            count += 1
    return count


def _count_big_up_days(rows: List[Dict], n: int = 10, threshold_pct: float = 5.0) -> int:
    """统计近 n 个交易日中单日涨幅超过 threshold_pct% 的天数（Session 11）.

    单日涨幅 = (close - prev_close) / prev_close * 100
    只计涨幅 > threshold_pct 的天数（不含等于）。
    若可用行数 < n+1，则在可用范围内计算。

    Args:
        rows:          升序排列的 EOD 行列表
        n:             窗口天数（默认 10）
        threshold_pct: 涨幅阈值（%，默认 5.0）

    Returns:
        大涨天数
    """
    # 取最后 n+1 行（需要 prev_close）
    window = rows[-(n + 1):] if len(rows) >= n + 1 else rows
    count = 0
    for i in range(1, len(window)):
        prev_close = window[i - 1]["close"]
        cur_close = window[i]["close"]
        if prev_close <= 0:
            continue
        pct_change = (cur_close - prev_close) / prev_close * 100.0
        if pct_change > threshold_pct:
            count += 1
    return count


def _count_consec_up_days(rows: List[Dict]) -> int:
    """计算从最新交易日向前连续收涨天数（Session 11）.

    收涨定义：当日 close > 前日 close（严格大于）。
    从最后一行向前遍历，遇到第一个非上涨日停止。

    Args:
        rows: 升序排列的 EOD 行列表（≥2 行）

    Returns:
        连续收涨天数（0 = 最新日未上涨）
    """
    count = 0
    for i in range(len(rows) - 1, 0, -1):
        prev_close = rows[i - 1]["close"]
        cur_close = rows[i]["close"]
        if prev_close <= 0:
            break
        if cur_close > prev_close:
            count += 1
        else:
            break
    return count
