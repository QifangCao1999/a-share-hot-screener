"""共用技术指标计算函数.

将分散在 price_features / risk_control / stage1_judge 中的重复计算
统一到此模块，消除口径不一致问题。

v2.1 P0-A 新增：
  upper_wick_ratio     — K 线形态描述（上影线比例）
  upper_reversal_ratio — 交易风险信号（冲高回落幅度）
"""

from __future__ import annotations


def upper_wick_ratio(open_: float, high: float, low: float, close: float) -> float:
    """经典 K 线上影线比率（形态描述）.

    定义: (high - max(open, close)) / (high - low)
    用途: price_features 形态描述、CSV 输出

    Returns:
        0.0 ~ 1.0，high == low 时返回 0.0
    """
    if high <= low:
        return 0.0
    return (high - max(open_, close)) / (high - low)


def upper_reversal_ratio(high: float, low: float, close: float) -> float:
    """冲高回落幅度（交易风险信号）.

    定义: (high - close) / (high - low)
    用途: RC4 评分、crowding cap 判定、Setup Timing 风险度量
    close 越远离 high，冲高回落越严重，风险越高。

    Returns:
        0.0 ~ 1.0，high == low 时返回 0.0
    """
    if high <= low:
        return 0.0
    return (high - close) / (high - low)
