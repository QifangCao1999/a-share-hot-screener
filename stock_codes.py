"""stock_codes 解析与校验.

输入支持：
  - 逗号/分号/空格分隔字符串: "600519,000858,300750"
  - Python list: ["600519", "000858"]
  - 带交易所前缀: "SH600519" / "SZ000858"
  - 带 Tushare 后缀: "600519.SHG" / "000858.SHE"

输出：valid_codes 为 6 位纯数字列表（去重保序），invalid_codes 为无法解析的原始 token。
"""

from __future__ import annotations

import re
from typing import List, Tuple

from a_share_hot_screener.ticker_mapping import infer_exchange


def parse_stock_codes(raw: str | List[str]) -> Tuple[List[str], List[str]]:
    """解析输入的 stock_codes，返回 (valid_codes, invalid_codes).

    Returns:
        valid_codes:   6 位纯数字字符串列表，去重、保序
        invalid_codes: 无法解析的原始 token 列表
    """
    if isinstance(raw, str):
        tokens = re.split(r"[,;\s]+", raw.strip())
    else:
        tokens = [str(t).strip() for t in raw]

    valid: List[str] = []
    invalid: List[str] = []
    seen: set = set()

    for token in tokens:
        if not token:
            continue
        code = _extract_code(token)
        if code is not None and code not in seen:
            valid.append(code)
            seen.add(code)
        elif code is None:
            invalid.append(token)
        # 重复代码直接跳过（保序去重）

    return valid, invalid


def _extract_code(token: str) -> str | None:
    """从各种格式中提取 6 位纯数字代码，失败返回 None."""
    token = token.strip().upper()

    # 纯 6 位数字（最常见）
    if re.fullmatch(r"\d{6}", token):
        return token

    # SH600519 / SZ000858 / BJ830799
    m = re.fullmatch(r"(?:SH|SZ|BJ)(\d{6})", token)
    if m:
        return m.group(1)

    # sh.600519 / sz-000858
    m = re.fullmatch(r"(?:SH|SZ|BJ)[.\-_](\d{6})", token)
    if m:
        return m.group(1)

    # 600519.SHG / 000858.SHE / 600519.XSHG
    m = re.fullmatch(r"(\d{6})[.\-_](?:SHG|SHE|SH|SZ|BJ|XSHG|XSHE)", token)
    if m:
        return m.group(1)

    return None


def classify_exchange(code: str) -> str:
    """根据代码前缀判断交易所: SH / SZ / BJ / UNKNOWN."""
    return infer_exchange(code)


def filter_beijing(
    codes: List[str], include: bool
) -> Tuple[List[str], List[str]]:
    """若 include=False，过滤掉北交所代码，返回 (kept, removed)."""
    if include:
        return codes, []
    kept, removed = [], []
    for c in codes:
        (kept if classify_exchange(c) != "BJ" else removed).append(c)
    return kept, removed
