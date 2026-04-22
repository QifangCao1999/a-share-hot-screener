"""观察时机评估模块 (Phase 4: Setup Timing).

输入: pass_stage1 tradeable 的股票 + 近 120 日 OHLCV 日线
输出: SetupSignal（评分 + 动作 + 关键价位 + 理由）

设计定位:
  - 观察周期 3~5 天，风格偏向低吸/回踩
  - 日频数据，不依赖盘中分时
  - 初始 experimental — 回测验证后再正式推送

v2.1 关键变更:
  - 默认只对 pass_stage1 (= tradeable) 运行
  - watch_only 若配置允许计算，action 最高只能到 watch
  - 权重归一: 0.22 + 0.28 + 0.22 + 0.16 + 0.12 = 1.00
  - 大盘环境仅乘数调节，不作为独立评分维度
  - level_confidence 评估参考价位可靠性
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("a_share_hot_screener.setup_timing")

# ════════════════════════════════════════════════════════
# 常量
# ════════════════════════════════════════════════════════

# 五维权重 (v2.1 归一: 和 = 1.00)
W_TREND = 0.22
W_PULLBACK = 0.28
W_VOLUME = 0.22
W_REPAIR = 0.16
W_RISK = 0.12

# 动作映射阈值
THRESHOLD_SETUP_READY = 80.0
THRESHOLD_WATCH = 65.0
THRESHOLD_WAIT = 45.0

# 大盘环境乘数
MARKET_MULT_BULL = 1.10
MARKET_MULT_NEUTRAL = 1.00
MARKET_MULT_BEAR = 0.75


# ════════════════════════════════════════════════════════
# 数据模型
# ════════════════════════════════════════════════════════

@dataclass
class SetupSignal:
    """单只股票的观察时机评估结果."""

    code: str
    name: str

    # 综合评分
    timing_score: float = 0.0           # 0~100
    action: str = "wait"                # wait / watch / setup_ready / avoid_chase

    # 关键参考价位
    support_zone_low: Optional[float] = None
    support_zone_high: Optional[float] = None
    invalidation_level: Optional[float] = None
    resistance_1: Optional[float] = None
    ref_reward_risk: Optional[float] = None
    level_confidence: str = "low"        # high / medium / low

    # 分项评分 (0~1)
    trend_score: float = 0.0
    pullback_score: float = 0.0
    volume_score: float = 0.0
    repair_score: float = 0.0
    risk_score: float = 0.0

    # 大盘
    market_regime: str = "neutral"
    market_multiplier: float = 1.0

    # 辅助
    support_basis: str = ""              # ma10 / ma20 / swing_low / box_low
    reason: str = ""
    warnings: List[str] = field(default_factory=list)
    confidence_reasons: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """序列化为 JSON 兼容 dict."""
        return {
            "code": self.code,
            "name": self.name,
            "timing_score": round(self.timing_score, 2),
            "action": self.action,
            "support_zone_low": _round_price(self.support_zone_low),
            "support_zone_high": _round_price(self.support_zone_high),
            "invalidation_level": _round_price(self.invalidation_level),
            "resistance_1": _round_price(self.resistance_1),
            "ref_reward_risk": round(self.ref_reward_risk, 2) if self.ref_reward_risk is not None else None,
            "level_confidence": self.level_confidence,
            "trend_score": round(self.trend_score, 4),
            "pullback_score": round(self.pullback_score, 4),
            "volume_score": round(self.volume_score, 4),
            "repair_score": round(self.repair_score, 4),
            "risk_score": round(self.risk_score, 4),
            "market_regime": self.market_regime,
            "market_multiplier": self.market_multiplier,
            "support_basis": self.support_basis,
            "reason": self.reason,
            "warnings": self.warnings,
            "confidence_reasons": self.confidence_reasons,
        }


# ════════════════════════════════════════════════════════
# 工具函数
# ════════════════════════════════════════════════════════

def _round_price(v: Optional[float], ndigits: int = 2) -> Optional[float]:
    return round(v, ndigits) if v is not None else None


def _safe_div(a: float, b: float, default: float = 0.0) -> float:
    """安全除法，b == 0 返回 default."""
    if b == 0:
        return default
    return a / b


def _clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, v))


def _bell_curve(x: float, center: float, width: float) -> float:
    """高斯钟形曲线评分, 峰值 center=1.0, 衰减由 width 控制.

    score = exp(-0.5 * ((x - center) / width)^2)
    """
    if width <= 0:
        return 1.0 if x == center else 0.0
    z = (x - center) / width
    return math.exp(-0.5 * z * z)


def _safe_mean(vals: List[float]) -> Optional[float]:
    if not vals:
        return None
    return sum(vals) / len(vals)


def _compute_ma(closes: List[float], n: int) -> Optional[float]:
    """计算最近 n 期均线."""
    if len(closes) < n:
        return None
    return sum(closes[-n:]) / n


def _compute_atr(rows: List[Dict[str, Any]], period: int = 14) -> Optional[float]:
    """计算 ATR (Average True Range).

    使用 Wilder 的定义:
      TR = max(high-low, |high-prev_close|, |low-prev_close|)
      ATR = SMA(TR, period)
    """
    if len(rows) < period + 1:
        return None
    trs = []
    for i in range(1, len(rows)):
        h = rows[i].get("high", 0)
        l = rows[i].get("low", 0)
        pc = rows[i - 1].get("close", 0)
        if h == 0 or l == 0 or pc == 0:
            continue
        tr = max(h - l, abs(h - pc), abs(l - pc))
        trs.append(tr)
    if len(trs) < period:
        return None
    return sum(trs[-period:]) / period


# ════════════════════════════════════════════════════════
# 1. 趋势状态评分 (权重 22%)
# ════════════════════════════════════════════════════════

def score_trend(
    close: float,
    ma5: Optional[float],
    ma10: Optional[float],
    ma20: Optional[float],
    ma60: Optional[float],
    closes_3d: List[float],
    ma20_prev: Optional[float],
) -> float:
    """评估趋势状态.

    组成:
      - MA 排列 (4线多头=1.0, 3线=0.7, 2线=0.4, 空头=0.0)  权重 0.35
      - 价格 vs MA20 (站上=加分, 跌破=减分)                    权重 0.25
      - 近3日方向 (上升/下降)                                   权重 0.20
      - MA20 斜率 (上行/走平/下行)                              权重 0.20

    低吸适配: 中等偏强(0.5~0.8)最适合低吸——趋势向上但正在回踩。
    """
    parts = []

    # 1) MA 排列
    ma_list = [v for v in [ma5, ma10, ma20, ma60] if v is not None]
    if len(ma_list) >= 2:
        bullish_count = sum(
            1 for i in range(len(ma_list) - 1) if ma_list[i] > ma_list[i + 1]
        )
        total_pairs = len(ma_list) - 1
        if total_pairs >= 3 and bullish_count == total_pairs:
            ma_score = 1.0   # 全多头
        elif bullish_count >= total_pairs - 1:
            ma_score = 0.7   # 差一条
        elif bullish_count >= 1:
            ma_score = 0.4   # 部分多头
        else:
            ma_score = 0.0   # 空头
    else:
        ma_score = 0.5  # 数据不足，中性
    parts.append((ma_score, 0.35))

    # 2) 价格 vs MA20
    if ma20 is not None and ma20 > 0:
        dist_ma20 = (close - ma20) / ma20
        if dist_ma20 > 0.02:
            price_vs_ma = 0.8
        elif dist_ma20 > -0.02:
            price_vs_ma = 0.6  # 在 MA20 附近
        elif dist_ma20 > -0.05:
            price_vs_ma = 0.3  # 略低于 MA20
        else:
            price_vs_ma = 0.1  # 大幅跌破
    else:
        price_vs_ma = 0.5
    parts.append((price_vs_ma, 0.25))

    # 3) 近3日方向
    if len(closes_3d) >= 2:
        ups = sum(1 for i in range(1, len(closes_3d)) if closes_3d[i] > closes_3d[i - 1])
        downs = sum(1 for i in range(1, len(closes_3d)) if closes_3d[i] < closes_3d[i - 1])
        if ups > downs:
            dir_score = 0.8
        elif ups == downs:
            dir_score = 0.5
        else:
            dir_score = 0.2  # 回调中——对低吸不一定是坏事
    else:
        dir_score = 0.5
    parts.append((dir_score, 0.20))

    # 4) MA20 斜率
    if ma20 is not None and ma20_prev is not None and ma20_prev > 0:
        slope = (ma20 - ma20_prev) / ma20_prev
        if slope > 0.005:
            slope_score = 0.9  # 上行
        elif slope > -0.002:
            slope_score = 0.5  # 走平
        else:
            slope_score = 0.1  # 下行
    else:
        slope_score = 0.5
    parts.append((slope_score, 0.20))

    return sum(s * w for s, w in parts) / sum(w for _, w in parts) if parts else 0.5


# ════════════════════════════════════════════════════════
# 2. 回踩位置评分 (权重 28%) ⭐ 核心
# ════════════════════════════════════════════════════════

def score_pullback(
    close: float,
    ma10: Optional[float],
    ma20: Optional[float],
    high_20d: float,
    low_20d: float,
) -> float:
    """评估回踩位置——低吸的核心维度.

    四个子指标，钟形曲线评分:
      - dist_ma10:      理想 -0.5%~+1%
      - dist_ma20:      理想 0%~+5%
      - range_pos:      理想 40%~65%
      - pullback_depth: 理想 3%~8%
    """
    scores = []
    weights = []

    # 1) 距离 MA10
    if ma10 is not None and ma10 > 0:
        dist_ma10 = (close - ma10) / ma10
        s = _bell_curve(dist_ma10, center=-0.005, width=0.03)
        scores.append(s)
        weights.append(0.30)

    # 2) 距离 MA20
    if ma20 is not None and ma20 > 0:
        dist_ma20 = (close - ma20) / ma20
        s = _bell_curve(dist_ma20, center=0.02, width=0.05)
        scores.append(s)
        weights.append(0.25)

    # 3) 20日区间位置
    range_span = high_20d - low_20d
    if range_span > 0:
        range_pos = (close - low_20d) / range_span
        s = _bell_curve(range_pos, center=0.50, width=0.25)
        scores.append(s)
        weights.append(0.20)

    # 4) 回踩深度
    if high_20d > 0:
        pullback_depth = (high_20d - close) / high_20d
        s = _bell_curve(pullback_depth, center=0.05, width=0.06)
        scores.append(s)
        weights.append(0.25)

    if not scores:
        return 0.5  # 数据不足
    total_w = sum(weights)
    return sum(s * w for s, w in zip(scores, weights)) / total_w


# ════════════════════════════════════════════════════════
# 3. 量价确认评分 (权重 22%)
# ════════════════════════════════════════════════════════

def score_volume(rows: List[Dict[str, Any]], current_volume_ratio: Optional[float]) -> float:
    """评估量价确认——回调缩量是低吸的关键信号.

    子指标:
      - 回调缩量: 近3日量递减 + 量比<0.8        权重 0.30
      - 量比区间: 0.5~1.5 区间内高分              权重 0.25
      - 下跌缩量比: 下跌日均量/上涨日均量 < 0.7   权重 0.25
      - 底部放量: 回踩低点量比>1.5                 权重 0.20
    """
    if len(rows) < 5:
        return 0.5

    recent_10 = rows[-10:] if len(rows) >= 10 else rows
    recent_3 = rows[-3:]

    scores = []
    weights = []

    # 1) 回调缩量: 近3日量递减
    vols_3 = [r.get("volume", 0) for r in recent_3 if r.get("volume")]
    if len(vols_3) >= 3:
        declining = all(vols_3[i] >= vols_3[i + 1] for i in range(len(vols_3) - 1))
        # 还要检查量比是否偏低
        avg_vol_20 = _safe_mean([r.get("volume", 0) for r in rows[-20:] if r.get("volume")])
        if avg_vol_20 and avg_vol_20 > 0:
            recent_ratio = vols_3[-1] / avg_vol_20
            if declining and recent_ratio < 0.8:
                s = 1.0
            elif declining:
                s = 0.7
            elif recent_ratio < 0.8:
                s = 0.6
            else:
                s = 0.3
        else:
            s = 0.5
    else:
        s = 0.5
    scores.append(s)
    weights.append(0.30)

    # 2) 量比区间 (0.5~1.5 最佳)
    if current_volume_ratio is not None and current_volume_ratio > 0:
        vr = current_volume_ratio
        s = _bell_curve(vr, center=1.0, width=0.6)
        scores.append(s)
        weights.append(0.25)

    # 3) 下跌缩量比
    up_vols = []
    down_vols = []
    for i in range(1, len(recent_10)):
        vol = recent_10[i].get("volume", 0)
        c = recent_10[i].get("close", 0)
        pc = recent_10[i - 1].get("close", 0)
        if vol > 0 and c > 0 and pc > 0:
            if c > pc:
                up_vols.append(vol)
            elif c < pc:
                down_vols.append(vol)
    avg_up = _safe_mean(up_vols)
    avg_down = _safe_mean(down_vols)
    if avg_up and avg_up > 0 and avg_down is not None:
        ratio = avg_down / avg_up
        if ratio < 0.5:
            s = 1.0
        elif ratio < 0.7:
            s = 0.8
        elif ratio < 1.0:
            s = 0.5
        else:
            s = 0.2
    else:
        s = 0.5
    scores.append(s)
    weights.append(0.25)

    # 4) 底部放量信号
    # 回踩最低点附近是否有放量(相对前几日均量)
    if len(recent_10) >= 5:
        lows = [r.get("low", float("inf")) for r in recent_10]
        min_low_idx = lows.index(min(lows))
        # 检查低点当日或次日是否放量
        check_indices = [min_low_idx]
        if min_low_idx + 1 < len(recent_10):
            check_indices.append(min_low_idx + 1)
        pre_avg_vol = _safe_mean(
            [r.get("volume", 0) for r in recent_10[:max(min_low_idx, 1)] if r.get("volume")]
        )
        if pre_avg_vol and pre_avg_vol > 0:
            max_ratio = max(
                recent_10[i].get("volume", 0) / pre_avg_vol
                for i in check_indices
                if recent_10[i].get("volume", 0) > 0
            ) if check_indices else 0
            if max_ratio > 1.5:
                s = 0.9  # 明显底部放量
            elif max_ratio > 1.2:
                s = 0.6
            else:
                s = 0.3
        else:
            s = 0.5
    else:
        s = 0.5
    scores.append(s)
    weights.append(0.20)

    total_w = sum(weights)
    return sum(s * w for s, w in zip(scores, weights)) / total_w if total_w > 0 else 0.5


# ════════════════════════════════════════════════════════
# 4. 分歧后修复结构评分 (权重 16%)
# ════════════════════════════════════════════════════════

def score_repair(rows_10d: List[Dict[str, Any]], ma20: Optional[float]) -> float:
    """评估分歧后修复结构.

    日频限制：无法判断盘中弱转强/封板质量/竞价承接，只能识别跨日模式。

    四个信号:
      - had_divergence:   近10日有大振幅日(>6%) 或 长上影日          权重 0.25
      - held_support:     分歧后未跌破 MA20 (容差3%)                 权重 0.30
      - volume_contracted: 后3日均量 < 分歧日量的60%                  权重 0.25
      - repair_candle:    最近2日有阳线 + 量温和放大                  权重 0.20
    """
    if len(rows_10d) < 5:
        return 0.5

    # 找分歧日: 振幅>6% 或 upper_reversal_ratio>0.45
    divergence_idx = None
    for i in range(len(rows_10d)):
        r = rows_10d[i]
        h = r.get("high", 0)
        l = r.get("low", 0)
        c = r.get("close", 0)
        o = r.get("open", 0)
        if h > 0 and l > 0:
            amp = (h - l) / l * 100
            # 冲高回落信号
            reversal = (h - c) / (h - l) if h != l else 0
            if amp > 6.0 or (reversal > 0.45 and amp > 3.0):
                divergence_idx = i

    if divergence_idx is None:
        # 没有分歧——不扣分也不加分
        return 0.5

    div_row = rows_10d[divergence_idx]
    after_div = rows_10d[divergence_idx + 1:] if divergence_idx + 1 < len(rows_10d) else []

    scores_w = []

    # (a) had_divergence = True (已确认)
    scores_w.append((1.0, 0.25))

    # (b) held_support: 分歧后未跌破 MA20
    if ma20 is not None and ma20 > 0 and after_div:
        min_close_after = min(r.get("close", float("inf")) for r in after_div)
        tolerance = ma20 * 0.97  # 3% 容差
        if min_close_after >= tolerance:
            scores_w.append((1.0, 0.30))
        elif min_close_after >= ma20 * 0.93:
            scores_w.append((0.4, 0.30))
        else:
            scores_w.append((0.0, 0.30))
    else:
        scores_w.append((0.5, 0.30))

    # (c) volume_contracted: 分歧后缩量
    div_vol = div_row.get("volume", 0)
    if div_vol > 0 and len(after_div) >= 2:
        after_vols = [r.get("volume", 0) for r in after_div[:3] if r.get("volume", 0) > 0]
        avg_after = _safe_mean(after_vols) if after_vols else 0
        if avg_after and avg_after > 0:
            ratio = avg_after / div_vol
            if ratio < 0.6:
                scores_w.append((1.0, 0.25))
            elif ratio < 0.8:
                scores_w.append((0.6, 0.25))
            else:
                scores_w.append((0.2, 0.25))
        else:
            scores_w.append((0.5, 0.25))
    else:
        scores_w.append((0.5, 0.25))

    # (d) repair_candle: 最近2日有阳线 + 量温和放大
    recent_2 = rows_10d[-2:]
    has_repair = False
    for r in recent_2:
        c = r.get("close", 0)
        o = r.get("open", 0)
        if c > o and c > 0:
            # 阳线 — 检查量是否温和放大
            has_repair = True
            break
    if has_repair:
        # 检查最后一天量是否比前一天大(温和放大)
        if len(recent_2) >= 2:
            v_last = recent_2[-1].get("volume", 0)
            v_prev = recent_2[-2].get("volume", 0)
            if v_last > v_prev * 1.1:
                scores_w.append((0.9, 0.20))
            else:
                scores_w.append((0.6, 0.20))
        else:
            scores_w.append((0.6, 0.20))
    else:
        scores_w.append((0.1, 0.20))

    total_w = sum(w for _, w in scores_w)
    return sum(s * w for s, w in scores_w) / total_w if total_w > 0 else 0.5


# ════════════════════════════════════════════════════════
# 5. 风险度量评分 (权重 12%)
# ════════════════════════════════════════════════════════

def score_risk(
    atr_pct: Optional[float],
    max_drawdown_5d: Optional[float],
    upper_reversal_count_5d: Optional[int],
    latest_is_limit_board: Optional[bool],
) -> float:
    """评估风险度量.

    子指标:
      - ATR% (14日): <3%=1.0, 3-5%=0.7, >8%=0.1   权重 0.30
      - 近5日最大回撤: <3%=1.0, >10%=0.2            权重 0.30
      - 上影线密度 (upper_reversal): 0天=1.0, ≥3天=0.1  权重 0.20
      - 一字板次日风险: 非一字=1.0, 一字涨停=0.2     权重 0.20
    """
    scores_w = []

    # ATR%
    if atr_pct is not None:
        if atr_pct < 3.0:
            s = 1.0
        elif atr_pct < 5.0:
            s = 0.7
        elif atr_pct < 8.0:
            s = 0.4
        else:
            s = 0.1
    else:
        s = 0.5
    scores_w.append((s, 0.30))

    # 近5日最大回撤
    if max_drawdown_5d is not None:
        dd = abs(max_drawdown_5d)
        if dd < 3.0:
            s = 1.0
        elif dd < 5.0:
            s = 0.7
        elif dd < 10.0:
            s = 0.4
        else:
            s = 0.2
    else:
        s = 0.5
    scores_w.append((s, 0.30))

    # 上影线密度
    if upper_reversal_count_5d is not None:
        cnt = upper_reversal_count_5d
        if cnt == 0:
            s = 1.0
        elif cnt == 1:
            s = 0.6
        elif cnt == 2:
            s = 0.3
        else:
            s = 0.1
    else:
        s = 0.5
    scores_w.append((s, 0.20))

    # 一字板次日风险
    if latest_is_limit_board is True:
        s = 0.2
    elif latest_is_limit_board is False:
        s = 1.0
    else:
        s = 0.5
    scores_w.append((s, 0.20))

    total_w = sum(w for _, w in scores_w)
    return sum(s * w for s, w in scores_w) / total_w if total_w > 0 else 0.5


# ════════════════════════════════════════════════════════
# 大盘环境 (仅乘数)
# ════════════════════════════════════════════════════════

def compute_market_regime(
    index_closes_20d: Optional[List[float]],
) -> Tuple[str, float]:
    """判断大盘环境 — 仅用作乘数调节.

    使用上证指数（或沪深300）近20日收盘价序列:
      - 牛市: 近20日涨幅>5% 且 最新价>MA20
      - 熊市: 近20日涨幅<-5% 且 最新价<MA20
      - 其余: 中性

    Args:
        index_closes_20d: 指数近20日收盘价序列（升序）

    Returns:
        (regime, multiplier)
    """
    if not index_closes_20d or len(index_closes_20d) < 10:
        return "neutral", MARKET_MULT_NEUTRAL

    first = index_closes_20d[0]
    last = index_closes_20d[-1]
    if first <= 0:
        return "neutral", MARKET_MULT_NEUTRAL

    ret_20d = (last - first) / first
    ma20 = sum(index_closes_20d) / len(index_closes_20d)

    if ret_20d > 0.05 and last > ma20:
        return "bull", MARKET_MULT_BULL
    elif ret_20d < -0.05 and last < ma20:
        return "bear", MARKET_MULT_BEAR
    else:
        return "neutral", MARKET_MULT_NEUTRAL


# ════════════════════════════════════════════════════════
# 参考价位计算
# ════════════════════════════════════════════════════════

def compute_reference_levels(
    close: float,
    ma10: Optional[float],
    ma20: Optional[float],
    atr: Optional[float],
    high_20d: float,
    low_20d: float,
) -> Dict[str, Any]:
    """计算关键参考价位.

    Returns:
        dict with support_zone_low, support_zone_high, invalidation_level,
              resistance_1, ref_reward_risk, support_basis
    """
    result: Dict[str, Any] = {}

    # 支撑区间: MA10~MA20 附近 (容差1%)
    if ma10 is not None and ma20 is not None:
        result["support_zone_low"] = round(min(ma10, ma20) * 0.99, 2)
        result["support_zone_high"] = round(max(ma10, ma20) * 1.01, 2)
        result["support_basis"] = "ma10" if abs(close - ma10) < abs(close - ma20) else "ma20"
    elif ma20 is not None:
        result["support_zone_low"] = round(ma20 * 0.98, 2)
        result["support_zone_high"] = round(ma20 * 1.02, 2)
        result["support_basis"] = "ma20"
    elif ma10 is not None:
        result["support_zone_low"] = round(ma10 * 0.98, 2)
        result["support_zone_high"] = round(ma10 * 1.02, 2)
        result["support_basis"] = "ma10"
    else:
        # 使用箱体低点
        result["support_zone_low"] = round(low_20d, 2)
        result["support_zone_high"] = round(low_20d * 1.03, 2)
        result["support_basis"] = "box_low"

    # 失效位: MA20 下方 1.5 倍 ATR
    if ma20 is not None and atr is not None and atr > 0:
        result["invalidation_level"] = round(ma20 - 1.5 * atr, 2)
    elif ma20 is not None:
        result["invalidation_level"] = round(ma20 * 0.95, 2)
    else:
        result["invalidation_level"] = round(low_20d * 0.97, 2)

    # 压力位: 近20日高点
    result["resistance_1"] = round(high_20d, 2)

    # 参考盈亏比
    inv = result.get("invalidation_level")
    if inv is not None and close > inv:
        risk = close - inv
        reward = high_20d - close
        result["ref_reward_risk"] = round(reward / risk, 2) if risk > 0 else 0.0
    else:
        result["ref_reward_risk"] = 0.0

    return result


# ════════════════════════════════════════════════════════
# 价位置信度
# ════════════════════════════════════════════════════════

def compute_level_confidence(
    close: float,
    ma10: Optional[float],
    ma20: Optional[float],
    atr: Optional[float],
    high_20d: float,
    low_20d: float,
    latest_is_limit_board: Optional[bool],
    limit_board_count: Optional[int],
) -> Tuple[str, List[str]]:
    """评估参考价位的可靠性.

    Returns:
        (confidence, reasons)  confidence = "high" / "medium" / "low"
    """
    reasons = []

    # 价格远离均线 → 均线支撑参考价值降低
    if ma20 is not None and ma20 > 0:
        if abs(close - ma20) / ma20 > 0.12:
            reasons.append("price_far_from_ma20")

    # 高波动 → ATR 计算的失效位波动大
    if atr is not None and close > 0:
        if atr / close > 0.06:
            reasons.append("high_volatility")

    # 连板结构 → 均线尚未跟上，支撑位意义不大
    if limit_board_count is not None and limit_board_count >= 2:
        reasons.append("consecutive_limit_up")

    # 箱体不清晰（20日高低差太小）
    if high_20d > 0 and (high_20d - low_20d) / high_20d < 0.05:
        reasons.append("narrow_range")

    # 近20日高点太近 → 压力位=当前位置，盈亏比失真
    if high_20d > 0 and (high_20d - close) / high_20d < 0.02:
        reasons.append("near_resistance")

    if len(reasons) >= 2:
        return "low", reasons
    elif len(reasons) == 1:
        return "medium", reasons
    else:
        return "high", []


# ════════════════════════════════════════════════════════
# 动作映射 + 硬规则
# ════════════════════════════════════════════════════════

def map_action(
    timing_score: float,
    latest_is_limit_board: Optional[bool],
    is_watch_only: bool = False,
) -> str:
    """将 timing_score 映射到动作字符串.

    硬规则:
      - 最新日一字涨停 → 强制 avoid_chase
      - watch_only → action 最高 watch
    """
    # 硬规则 1: 一字涨停
    if latest_is_limit_board is True:
        return "avoid_chase"

    # 基于分数的映射
    if timing_score >= THRESHOLD_SETUP_READY:
        action = "setup_ready"
    elif timing_score >= THRESHOLD_WATCH:
        action = "watch"
    elif timing_score >= THRESHOLD_WAIT:
        action = "wait"
    else:
        action = "avoid_chase"

    # 硬规则 2: watch_only cap
    if is_watch_only and action == "setup_ready":
        action = "watch"

    return action


# ════════════════════════════════════════════════════════
# 理由生成
# ════════════════════════════════════════════════════════

def generate_reason(signal: SetupSignal) -> str:
    """生成一句话理由."""
    parts = []

    # 趋势描述
    if signal.trend_score >= 0.7:
        parts.append("趋势多头")
    elif signal.trend_score >= 0.4:
        parts.append("趋势偏多")
    else:
        parts.append("趋势偏弱")

    # 回踩描述
    if signal.pullback_score >= 0.7:
        parts.append("回踩到位")
    elif signal.pullback_score >= 0.4:
        parts.append("回踩接近")
    else:
        parts.append("位置偏高")

    # 量价描述
    if signal.volume_score >= 0.7:
        parts.append("缩量确认")
    elif signal.volume_score < 0.3:
        parts.append("量能异常")

    # 风险描述
    if signal.risk_score < 0.4:
        parts.append("风险偏高")

    return "，".join(parts)


# ════════════════════════════════════════════════════════
# 生成警告
# ════════════════════════════════════════════════════════

def generate_warnings(signal: SetupSignal) -> List[str]:
    """生成风险提示列表."""
    warns = []
    if signal.risk_score < 0.3:
        warns.append("风险评分偏低，注意控制仓位")
    if signal.level_confidence == "low":
        warns.append("参考价位置信度低，仅供参考")
    if signal.market_regime == "bear":
        warns.append("大盘环境偏弱，谨慎操作")
    if signal.ref_reward_risk is not None and signal.ref_reward_risk < 1.0:
        warns.append("盈亏比不足1:1")
    return warns


# ════════════════════════════════════════════════════════
# 辅助: 从原始 EOD 行提取需要的中间值
# ════════════════════════════════════════════════════════

def _extract_eod_metrics(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """从 EOD 行列表中提取 setup_timing 需要的中间指标.

    Args:
        rows: 按日期升序的 EOD 行 (已标准化: date_str, open, high, low, close, volume, amount)

    Returns:
        dict 包含所有需要的中间值
    """
    closes = [r["close"] for r in rows if r.get("close")]

    ma5 = _compute_ma(closes, 5)
    ma10 = _compute_ma(closes, 10)
    ma20 = _compute_ma(closes, 20)
    ma60 = _compute_ma(closes, 60)

    close = closes[-1] if closes else 0.0

    # MA20 前一日值 (用于斜率计算)
    ma20_prev = _compute_ma(closes[:-1], 20) if len(closes) > 20 else None

    # 近20日高低点
    recent_20 = rows[-20:] if len(rows) >= 20 else rows
    highs_20d = [r.get("high", 0) for r in recent_20]
    lows_20d = [r.get("low", float("inf")) for r in recent_20]
    high_20d = max(highs_20d) if highs_20d else close
    low_20d = min(lows_20d) if lows_20d else close

    # ATR
    atr = _compute_atr(rows, period=14)
    atr_pct = (atr / close * 100) if atr and close > 0 else None

    # 近5日最大回撤
    max_drawdown_5d = None
    recent_5 = rows[-5:] if len(rows) >= 5 else rows
    if recent_5:
        peak = recent_5[0].get("close", 0)
        max_dd = 0.0
        for r in recent_5:
            c = r.get("close", 0)
            if c > peak:
                peak = c
            dd = (peak - c) / peak * 100 if peak > 0 else 0
            if dd > max_dd:
                max_dd = dd
        max_drawdown_5d = max_dd

    # 近3日收盘价
    closes_3d = closes[-3:] if len(closes) >= 3 else closes

    # 近10日 rows (用于 repair 评分)
    rows_10d = rows[-10:] if len(rows) >= 10 else rows

    return {
        "close": close,
        "ma5": ma5,
        "ma10": ma10,
        "ma20": ma20,
        "ma60": ma60,
        "ma20_prev": ma20_prev,
        "high_20d": high_20d,
        "low_20d": low_20d,
        "atr": atr,
        "atr_pct": atr_pct,
        "max_drawdown_5d": max_drawdown_5d,
        "closes_3d": closes_3d,
        "rows_10d": rows_10d,
    }


# ════════════════════════════════════════════════════════
# 标准化 EOD 行 (复用 price_features 的格式)
# ════════════════════════════════════════════════════════

def _normalize_eod_rows(raw_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """将 Tushare daily 行标准化为统一格式，按日期升序排列.

    兼容 tushare daily API 和已标准化的格式。
    """
    parsed = []
    for r in raw_rows:
        try:
            raw_date = str(r.get("trade_date") or r.get("date_str") or r.get("date") or "")
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

            # volume: tushare vol(手) → 股
            if "vol" in r and r.get("vol") is not None:
                volume = float(r["vol"]) * 100
            elif "volume" in r and r.get("volume") is not None:
                volume = float(r["volume"])
            else:
                volume = 0

            # amount: tushare amount(千元) → 元
            if "amount" in r and r.get("amount") is not None:
                raw_amount = float(r["amount"])
                # 判断是否已是元单位(>1亿通常已转换)
                amount = raw_amount * 1000 if raw_amount < 1_000_000_000 else raw_amount
            else:
                amount = 0

            if close <= 0:
                continue

            parsed.append({
                "date_str": date_str,
                "open": open_,
                "high": high,
                "low": low,
                "close": close,
                "volume": volume,
                "amount": amount,
            })
        except (ValueError, TypeError):
            continue

    parsed.sort(key=lambda x: x["date_str"])
    return parsed


# ════════════════════════════════════════════════════════
# 主入口: 单只股票评估
# ════════════════════════════════════════════════════════

def evaluate_setup_timing(
    code: str,
    name: str,
    eod_rows: List[Dict[str, Any]],
    upper_reversal_count_5d: Optional[int] = None,
    latest_is_limit_board: Optional[bool] = None,
    limit_board_count_5d: Optional[int] = None,
    volume_ratio: Optional[float] = None,
    is_watch_only: bool = False,
    market_regime: str = "neutral",
    market_multiplier: float = 1.0,
) -> SetupSignal:
    """评估单只股票的观察时机.

    Args:
        code: 6位股票代码
        name: 股票名称
        eod_rows: 标准化后的 EOD 行 (date_str, open, high, low, close, volume, amount)
        upper_reversal_count_5d: 近5日冲高回落天数 (来自 price_features)
        latest_is_limit_board: 最新日是否一字板 (来自 price_features)
        limit_board_count_5d: 近5日一字板天数 (来自 price_features)
        volume_ratio: 量比 (来自 spot)
        is_watch_only: 是否为 watch_only 候选
        market_regime: 大盘环境 (来自 compute_market_regime)
        market_multiplier: 大盘乘数

    Returns:
        SetupSignal
    """
    signal = SetupSignal(code=code, name=name)
    signal.market_regime = market_regime
    signal.market_multiplier = market_multiplier

    if not eod_rows or len(eod_rows) < 10:
        signal.action = "wait"
        signal.reason = "日线数据不足，无法评估"
        signal.warnings.append("EOD 数据不足10行")
        return signal

    # 提取中间指标
    metrics = _extract_eod_metrics(eod_rows)

    close = metrics["close"]
    if close <= 0:
        signal.action = "wait"
        signal.reason = "收盘价异常"
        return signal

    # ── 五维评分 ────────────────────────────────────────

    signal.trend_score = score_trend(
        close=close,
        ma5=metrics["ma5"],
        ma10=metrics["ma10"],
        ma20=metrics["ma20"],
        ma60=metrics["ma60"],
        closes_3d=metrics["closes_3d"],
        ma20_prev=metrics["ma20_prev"],
    )

    signal.pullback_score = score_pullback(
        close=close,
        ma10=metrics["ma10"],
        ma20=metrics["ma20"],
        high_20d=metrics["high_20d"],
        low_20d=metrics["low_20d"],
    )

    signal.volume_score = score_volume(
        rows=eod_rows,
        current_volume_ratio=volume_ratio,
    )

    signal.repair_score = score_repair(
        rows_10d=metrics["rows_10d"],
        ma20=metrics["ma20"],
    )

    signal.risk_score = score_risk(
        atr_pct=metrics["atr_pct"],
        max_drawdown_5d=metrics["max_drawdown_5d"],
        upper_reversal_count_5d=upper_reversal_count_5d,
        latest_is_limit_board=latest_is_limit_board,
    )

    # ── 综合评分 ────────────────────────────────────────

    raw_score = (
        signal.trend_score * W_TREND +
        signal.pullback_score * W_PULLBACK +
        signal.volume_score * W_VOLUME +
        signal.repair_score * W_REPAIR +
        signal.risk_score * W_RISK
    ) * 100

    signal.timing_score = _clamp(raw_score * market_multiplier, 0.0, 100.0)

    # ── 动作映射 ────────────────────────────────────────

    signal.action = map_action(
        timing_score=signal.timing_score,
        latest_is_limit_board=latest_is_limit_board,
        is_watch_only=is_watch_only,
    )

    # ── 参考价位 ────────────────────────────────────────

    ref_levels = compute_reference_levels(
        close=close,
        ma10=metrics["ma10"],
        ma20=metrics["ma20"],
        atr=metrics["atr"],
        high_20d=metrics["high_20d"],
        low_20d=metrics["low_20d"],
    )
    signal.support_zone_low = ref_levels.get("support_zone_low")
    signal.support_zone_high = ref_levels.get("support_zone_high")
    signal.invalidation_level = ref_levels.get("invalidation_level")
    signal.resistance_1 = ref_levels.get("resistance_1")
    signal.ref_reward_risk = ref_levels.get("ref_reward_risk")
    signal.support_basis = ref_levels.get("support_basis", "")

    # ── 置信度 ──────────────────────────────────────────

    confidence, conf_reasons = compute_level_confidence(
        close=close,
        ma10=metrics["ma10"],
        ma20=metrics["ma20"],
        atr=metrics["atr"],
        high_20d=metrics["high_20d"],
        low_20d=metrics["low_20d"],
        latest_is_limit_board=latest_is_limit_board,
        limit_board_count=limit_board_count_5d,
    )
    signal.level_confidence = confidence
    signal.confidence_reasons = conf_reasons

    # ── 理由 + 警告 ────────────────────────────────────

    signal.reason = generate_reason(signal)
    signal.warnings = generate_warnings(signal)

    return signal


# ════════════════════════════════════════════════════════
# 批量运行入口 — 供 pipeline 调用
# ════════════════════════════════════════════════════════

def run_setup_timing(
    details: List[Any],
    tushare_client: Any,
    trade_cal: Any,
    trade_date_str: str,
    index_closes_20d: Optional[List[float]] = None,
) -> List[SetupSignal]:
    """对 tradeable 的 details 批量运行观察时机评估.

    Args:
        details: 通过 pass_stage1 的 HotStockDetail 列表
        tushare_client: TushareClient 实例 (用于获取日线数据)
        trade_cal: TradeCalendar 实例
        trade_date_str: 运行日期 (YYYY-MM-DD)
        index_closes_20d: 指数近20日收盘价序列

    Returns:
        List[SetupSignal]
    """
    import datetime as _dt

    # 大盘环境
    regime, multiplier = compute_market_regime(index_closes_20d)
    logger.info("[setup_timing] 大盘环境: %s (乘数=%.2f)", regime, multiplier)

    signals: List[SetupSignal] = []
    trade_date = _dt.date.fromisoformat(trade_date_str)

    for detail in details:
        try:
            # 获取 120 日数据 (从缓存中读取，通常很快)
            ts_code = getattr(detail, "ts_code", "")
            if not ts_code:
                continue

            try:
                eod_start = trade_cal.eod_start_date(trade_date, window=120, extra=10)
            except Exception:
                eod_start = trade_date - _dt.timedelta(days=200)

            daily_df = tushare_client.get_daily(
                ts_code=ts_code,
                start_date=eod_start.strftime("%Y%m%d"),
                end_date=trade_date_str.replace("-", ""),
            )

            if daily_df is None or daily_df.empty:
                logger.warning("[setup_timing] %s 无法获取日线数据", detail.code)
                continue

            raw_eod = daily_df.to_dict("records")
            eod_rows = _normalize_eod_rows(raw_eod)

            if not eod_rows:
                continue

            is_watch_only = getattr(detail, "pass_stage1_watch", False) and not getattr(detail, "pass_stage1", False)

            signal = evaluate_setup_timing(
                code=detail.code,
                name=getattr(detail, "name", ""),
                eod_rows=eod_rows,
                upper_reversal_count_5d=getattr(detail, "upper_reversal_count_5d", None),
                latest_is_limit_board=getattr(detail, "latest_is_limit_board", None),
                limit_board_count_5d=getattr(detail, "limit_board_count_5d", None),
                volume_ratio=getattr(detail, "volume_ratio", None),
                is_watch_only=is_watch_only,
                market_regime=regime,
                market_multiplier=multiplier,
            )
            signals.append(signal)

        except Exception as e:
            logger.error("[setup_timing] %s 评估异常: %s", getattr(detail, "code", "?"), e, exc_info=True)

    logger.info(
        "[setup_timing] 完成: %d只评估 | setup_ready=%d watch=%d wait=%d avoid=%d",
        len(signals),
        sum(1 for s in signals if s.action == "setup_ready"),
        sum(1 for s in signals if s.action == "watch"),
        sum(1 for s in signals if s.action == "wait"),
        sum(1 for s in signals if s.action == "avoid_chase"),
    )
    return signals
