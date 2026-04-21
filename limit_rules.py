"""涨跌停幅度推断规则（可追踪的共享模块）.

Session 7 新增：将 risk_control.py / flags.py 中写死的 _infer_limit_pct 逻辑
抽取为共享函数，确保全项目口径一致，且规则来源可追踪。

规则说明：
  - 688xxx（科创板）→ 20%
  - 300xxx（创业板）→ 20%
  - 其余（沪深主板）→ 10%

北交所说明：
  北交所代码通常以 43/83/87 开头，实际涨跌停幅度为 30%。
  但 HotScreenerConfig.include_beijing=False 时北交所已被硬筛过滤，
  本函数暂按 10% 兜底处理，如需支持北交所请传入 exchange 参数并扩展规则。

NOTE（QA 追踪点）：
  本文件是全项目涨跌停幅度的唯一来源，修改规则只需改这里。
  调用方：scorers/risk_control.py、flags.py、tests/test_s7.py
"""

from __future__ import annotations

__all__ = ["infer_limit_pct", "LIMIT_PCT_STAR_BOARD", "LIMIT_PCT_MAIN_BOARD"]

LIMIT_PCT_STAR_BOARD: float = 20.0   # 科创板（688）/ 创业板（300）
LIMIT_PCT_MAIN_BOARD: float = 10.0   # 沪深主板


def infer_limit_pct(code: str) -> float:
    """根据股票代码推断涨跌停幅度（%）.

    Args:
        code: 6位纯数字 A 股代码（如 "600519" / "300750" / "688001"）

    Returns:
        涨跌停幅度百分比（float），如 10.0 或 20.0

    Examples:
        >>> infer_limit_pct("688001")
        20.0
        >>> infer_limit_pct("300750")
        20.0
        >>> infer_limit_pct("600519")
        10.0
        >>> infer_limit_pct("000858")
        10.0
        >>> infer_limit_pct("")
        10.0
    """
    if not code:
        return LIMIT_PCT_MAIN_BOARD
    if code.startswith("688") or code.startswith("300"):
        return LIMIT_PCT_STAR_BOARD
    return LIMIT_PCT_MAIN_BOARD
