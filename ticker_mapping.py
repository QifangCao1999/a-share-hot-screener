"""Ticker 映射工具 – 在不同数据源间转换 A 股代码格式.

支持格式：
  Tushare: 600519.SH / 000858.SZ
  内部:    600519 (纯 6 位)
  前缀格式: SH600519 / SZ000858

A股代码前缀分布：
  上交所 (SH):  600xxx / 601xxx / 603xxx / 605xxx / 688xxx
  深交所 (SZ):  000xxx / 001xxx / 002xxx / 003xxx / 300xxx / 301xxx
  北交所 (BJ):  8xxxxx / 4xxxxx
"""

from __future__ import annotations

from typing import Optional, Tuple

# ── Tushare 交易所后缀映射 ──────────────────────────────
_EXCHANGE_TO_TUSHARE_SUFFIX = {
    "SH": "SH",
    "SZ": "SZ",
    "BJ": "BJ",
}

# ── 代码前缀 → 交易所（3位精确 > 2位 > 1位）─────────
_PREFIX_3_TO_EXCHANGE = {
    "600": "SH", "601": "SH", "603": "SH", "605": "SH",
    "688": "SH",
    "000": "SZ", "001": "SZ", "002": "SZ", "003": "SZ",
    "300": "SZ", "301": "SZ",
}
_PREFIX_2_TO_EXCHANGE = {
    "60": "SH", "68": "SH",
    "00": "SZ", "30": "SZ",
}
_PREFIX_1_TO_EXCHANGE = {
    "8": "BJ", "4": "BJ",
}


def infer_exchange(code: str) -> str:
    """根据代码前缀推断交易所: SH / SZ / BJ / UNKNOWN."""
    if len(code) >= 3:
        ex = _PREFIX_3_TO_EXCHANGE.get(code[:3])
        if ex:
            return ex
    if len(code) >= 2:
        ex = _PREFIX_2_TO_EXCHANGE.get(code[:2])
        if ex:
            return ex
    if len(code) >= 1:
        ex = _PREFIX_1_TO_EXCHANGE.get(code[:1])
        if ex:
            return ex
    return "UNKNOWN"


def code_to_tushare(code: str, exchange: Optional[str] = None) -> Optional[str]:
    """将 6 位代码转为 Tushare 格式: '600519' -> '600519.SH'.

    Returns:
        Tushare ts_code 字符串，或 None（未知交易所）
    """
    if exchange is None:
        exchange = infer_exchange(code)
    suffix = _EXCHANGE_TO_TUSHARE_SUFFIX.get(exchange)
    if suffix is None:
        return None
    return f"{code}.{suffix}"


def tushare_to_code(ts_code: str) -> Optional[str]:
    """将 Tushare ts_code 转为 6 位纯数字代码: '600519.SH' -> '600519'."""
    parts = ts_code.split(".")
    if len(parts) == 2 and len(parts[0]) == 6 and parts[0].isdigit():
        return parts[0]
    return None


def code_to_prefix_format(code: str, exchange: Optional[str] = None) -> str:
    """转为 SH600519 / SZ000858 格式."""
    if exchange is None:
        exchange = infer_exchange(code)
    return f"{exchange}{code}"


def code_and_exchange(code: str) -> Tuple[str, str]:
    """返回 (code, exchange) 二元组，方便解构."""
    return code, infer_exchange(code)
