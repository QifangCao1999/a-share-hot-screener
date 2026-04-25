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

v3.0 关键变更 (Round 1):
  - 新增 SetupTimingConfig 独立配置类
  - 新增 Action Cap 硬规则体系 (14条规则，限制高风险结构的action上限)
  - 回踩参数波动率自适应 (ATR% 驱动钟形曲线参数)
  - 接入 Stage 1 四轴分数 (stage1_context 联动)
  - 增强 reason/warnings 解释性
  - 预留盘中执行接口 (requires_intraday_confirmation)
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

# 大盘环境乘数 (v2.1 原始)
MARKET_MULT_BULL = 1.10
MARKET_MULT_NEUTRAL = 1.00
MARKET_MULT_BEAR = 0.75

# v3.0 增强版乘数
MARKET_MULT_RISK_ON = 1.10
MARKET_MULT_RISK_OFF = 0.85

# Action 优先级 (越小越严格)
_ACTION_RANK = {
    "avoid_chase": 0,
    "wait": 1,
    "watch": 2,
    "setup_ready": 3,
}


# ════════════════════════════════════════════════════════
# 配置类
# ════════════════════════════════════════════════════════

@dataclass
class SetupTimingConfig:
    """Setup timing 内部参数配置.

    所有参数都有保守默认值。旧行为可通过关闭各 enable_ 开关恢复。
    """

    # ── 模块开关 ────────────────────────────────────────
    enable_action_caps: bool = True                 # 启用 action cap 硬规则
    enable_volatility_adaptive_pullback: bool = True  # 启用波动率自适应回踩参数
    enable_stage1_context: bool = True              # 启用 Stage 1 四轴联动

    # ── Action cap 阈值 ────────────────────────────────
    cap_min_risk_score_for_ready: float = 0.25      # risk_score < 此值 → 最高 wait
    cap_min_risk_score_critical: float = 0.15       # risk_score < 此值 → 强制 avoid_chase
    cap_min_reward_risk_for_ready: float = 1.0      # ref_reward_risk < 此值 → 最高 watch
    cap_min_reward_risk_critical: float = 0.6       # ref_reward_risk < 此值 → 最高 wait
    cap_max_dist_ma20_for_ready: float = 0.12       # 距 MA20 > 12% → 最高 watch
    cap_max_dist_ma20_critical: float = 0.18        # 距 MA20 > 18% → 最高 wait
    cap_max_atr_pct_for_ready: float = 10.0         # ATR% > 10% → 最高 watch
    cap_atr_drawdown_atr_threshold: float = 8.0     # ATR% > 8% 且 drawdown > 10% → wait
    cap_atr_drawdown_dd_threshold: float = 10.0     # 配合上条
    cap_max_upper_shadow_days: int = 2              # 上影线 > 此值 → 最高 wait
    cap_max_limit_board_days: int = 1               # 近5日一字板 > 此值 → 最高 watch
    cap_bear_ready_threshold: float = 85.0          # bear + timing < 此值 → 不允许 setup_ready
    cap_bear_min_risk_score: float = 0.4            # bear + risk_score < 此值 → 最高 wait

    # ── 波动率自适应回踩参数 ────────────────────────────
    pullback_min_center: float = 0.03               # 回踩深度 center 下限
    pullback_max_center: float = 0.10               # 回踩深度 center 上限
    pullback_atr_multiplier: float = 1.0            # center = atr_ratio * multiplier

    # ── Stage 1 联动阈值 ────────────────────────────────
    min_hot_theme_for_ready: float = 0.55           # hot_theme < 此值 → setup_ready 降级
    min_liquidity_for_ready: float = 0.50           # liquidity < 此值 → setup_ready 降级
    min_liquidity_critical: float = 0.35            # liquidity < 此值 → 最高 wait
    min_stage1_risk_for_ready: float = 0.40         # stage1 risk_control < 此值 → 降级
    min_stage1_risk_critical: float = 0.25          # stage1 risk_control < 此值 → 最高 wait
    high_priority_hot_theme: float = 0.75           # hot_theme >= 此值 且 timing >= 75 → high_priority
    high_priority_timing: float = 75.0              # 配合上条
    stage1_bottom_pctile_timing: float = 85.0       # total_score 后50% 需 timing >= 此值才能 setup_ready
    min_stage1_rank_sample_size: int = 20           # I10: 候选池 < 此值时不使用 rank_pctile 降级

    # ── Market Regime 增强 (v3.0) ──────────────────────
    enable_enhanced_market_regime: bool = False      # 启用增强版大盘环境判断
    risk_on_multiplier: float = 1.10                # risk_on 乘数
    risk_off_multiplier: float = 0.85               # risk_off 乘数

    # ── 参考价位增强 (v3.0) ──────────────────────
    enable_enhanced_reference_levels: bool = True    # 启用多来源支撑体系


# 默认配置 (模块级单例)
_DEFAULT_CONFIG = SetupTimingConfig()

# I5: 弱市 regime 集合 — basic regime 用 "bear"，enhanced regime 用 risk_off / ice_point
# action_cap 和 warnings 对这些 regime 一视同仁
WEAK_REGIMES: frozenset = frozenset({"bear", "risk_off", "ice_point"})


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
    resistance_2: Optional[float] = None     # v3.0: 前高/平台上沿
    ref_reward_risk: Optional[float] = None
    level_confidence: str = "low"        # high / medium / low
    candidate_support_levels: List[Dict[str, Any]] = field(default_factory=list)  # v3.0

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

    # v3.0 新增: action cap
    action_cap_reasons: List[str] = field(default_factory=list)
    score_components_commentary: Dict[str, str] = field(default_factory=dict)

    # v3.0 新增: Stage 1 context
    stage1_context_used: bool = False
    stage1_adjustment_reason: Optional[str] = None
    final_action_before_stage1_cap: str = ""
    final_action_after_stage1_cap: str = ""
    high_priority_watch: bool = False

    # v3.0 新增: 盘中执行预留
    requires_intraday_confirmation: bool = False
    intraday_check_hint: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """序列化为 JSON 兼容 dict."""
        d = {
            "code": self.code,
            "name": self.name,
            "timing_score": round(self.timing_score, 2),
            "action": self.action,
            "support_zone_low": _round_price(self.support_zone_low),
            "support_zone_high": _round_price(self.support_zone_high),
            "invalidation_level": _round_price(self.invalidation_level),
            "resistance_1": _round_price(self.resistance_1),
            "resistance_2": _round_price(self.resistance_2),
            "ref_reward_risk": round(self.ref_reward_risk, 2) if self.ref_reward_risk is not None else None,
            "candidate_support_levels": self.candidate_support_levels,
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
            # v3.0
            "action_cap_reasons": self.action_cap_reasons,
            "score_components_commentary": self.score_components_commentary,
            "stage1_context_used": self.stage1_context_used,
            "stage1_adjustment_reason": self.stage1_adjustment_reason,
            "high_priority_watch": self.high_priority_watch,
            "requires_intraday_confirmation": self.requires_intraday_confirmation,
            "intraday_check_hint": self.intraday_check_hint,
        }
        return d


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


def _cap_action(current: str, cap: str) -> str:
    """将 current action 限制到不超过 cap 的级别."""
    cur_rank = _ACTION_RANK.get(current, 1)
    cap_rank = _ACTION_RANK.get(cap, 1)
    if cur_rank <= cap_rank:
        return current
    return cap


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
    atr_pct: Optional[float] = None,
    config: Optional[SetupTimingConfig] = None,
) -> float:
    """评估回踩位置——低吸的核心维度.

    四个子指标，钟形曲线评分:
      - dist_ma10:      理想 -0.5%~+1%  (width 可自适应)
      - dist_ma20:      理想 0%~+5%     (width 可自适应)
      - range_pos:      理想 40%~65%
      - pullback_depth: 理想 3%~8%      (center/width 可自适应)

    v3.0: atr_pct 可驱动钟形曲线参数自适应，高波动票允许更宽回踩区间。
    """
    cfg = config or _DEFAULT_CONFIG
    use_adaptive = cfg.enable_volatility_adaptive_pullback and atr_pct is not None and atr_pct > 0

    # 计算自适应参数 (atr_pct 是百分数如 5.0 表示 5%, atr_ratio 是小数 0.05)
    if use_adaptive:
        atr_ratio = atr_pct / 100.0
        pb_center = _clamp(cfg.pullback_atr_multiplier * atr_ratio, cfg.pullback_min_center, cfg.pullback_max_center)
        pb_width = _clamp(1.5 * atr_ratio, 0.05, 0.12)
        dist_ma10_width = _clamp(0.8 * atr_ratio, 0.025, 0.06)
        dist_ma20_width = _clamp(1.2 * atr_ratio, 0.04, 0.09)
    else:
        # v2.1 固定参数 (fallback)
        pb_center = 0.05
        pb_width = 0.06
        dist_ma10_width = 0.03
        dist_ma20_width = 0.05

    scores = []
    weights = []

    # 1) 距离 MA10
    if ma10 is not None and ma10 > 0:
        dist_ma10 = (close - ma10) / ma10
        s = _bell_curve(dist_ma10, center=-0.005, width=dist_ma10_width)
        scores.append(s)
        weights.append(0.30)

    # 2) 距离 MA20
    if ma20 is not None and ma20 > 0:
        dist_ma20 = (close - ma20) / ma20
        s = _bell_curve(dist_ma20, center=0.02, width=dist_ma20_width)
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
        s = _bell_curve(pullback_depth, center=pb_center, width=pb_width)
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


def compute_enhanced_market_regime(
    index_closes_20d: Optional[List[float]],
    index_amounts_20d: Optional[List[float]] = None,
    config: Optional["SetupTimingConfig"] = None,
) -> Dict[str, Any]:
    """v3.0 增强版大盘环境判断.

    维度:
      1. index_trend: 指数趋势 (bull/neutral/bear)  — 复用原有逻辑
      2. turnover_state: 成交额状态 (expanding/neutral/shrinking)

    其余维度 (breadth/sentiment/sector) 待数据源就绪后扩展。

    Args:
        index_closes_20d: 指数近20日收盘价序列（升序）
        index_amounts_20d: 指数近20日成交额序列（升序，元）
        config: 配置

    Returns:
        dict with index_trend, turnover_state, final_regime, multiplier
    """
    cfg = config or _DEFAULT_CONFIG

    # 维度1: 指数趋势 (复用原有函数)
    index_trend, _ = compute_market_regime(index_closes_20d)

    # 维度2: 成交额状态
    turnover_state = "neutral"
    if index_amounts_20d and len(index_amounts_20d) >= 10:
        amounts = [a for a in index_amounts_20d if a and a > 0]
        if len(amounts) >= 10:
            avg_all = sum(amounts) / len(amounts)
            # 近3日均值 vs 全期均值
            avg_recent_3 = sum(amounts[-3:]) / 3
            ratio = avg_recent_3 / avg_all if avg_all > 0 else 1.0
            if ratio > 1.15:
                turnover_state = "expanding"
            elif ratio < 0.80:
                turnover_state = "shrinking"

    # 综合判断
    if index_trend == "bull" and turnover_state != "shrinking":
        final_regime = "risk_on"
        multiplier = cfg.risk_on_multiplier
    elif index_trend == "bear":
        final_regime = "risk_off"
        multiplier = cfg.risk_off_multiplier
    elif index_trend == "neutral" and turnover_state == "shrinking":
        final_regime = "risk_off"
        multiplier = cfg.risk_off_multiplier
    else:
        final_regime = "neutral"
        multiplier = MARKET_MULT_NEUTRAL

    return {
        "index_trend": index_trend,
        "turnover_state": turnover_state,
        "final_regime": final_regime,
        "multiplier": multiplier,
    }


# ════════════════════════════════════════════════════════
# 参考价位计算
# ════════════════════════════════════════════════════════

def _find_swing_lows(rows: List[Dict[str, Any]], lookback: int = 2) -> List[float]:
    """检测近20日的 swing low (局部最低点).

    swing low: 某根K线的low比前后各 lookback 根K线的low都低。
    """
    swing_lows = []
    for i in range(lookback, len(rows) - lookback):
        low_i = rows[i].get("low", float("inf"))
        is_swing = True
        for j in range(1, lookback + 1):
            if rows[i - j].get("low", 0) <= low_i:
                is_swing = False
                break
            if rows[i + j].get("low", 0) <= low_i:
                is_swing = False
                break
        if is_swing and low_i < float("inf"):
            swing_lows.append(low_i)
    return swing_lows


def _find_divergence_low(rows_10d: List[Dict[str, Any]]) -> Optional[float]:
    """找到近10日分歧日的低点 (与 score_repair 共享分歧检测逻辑)."""
    for i in range(len(rows_10d)):
        r = rows_10d[i]
        h = r.get("high", 0)
        l = r.get("low", 0)
        c = r.get("close", 0)
        if h > 0 and l > 0:
            amp = (h - l) / l * 100
            reversal = (h - c) / (h - l) if h != l else 0
            if amp > 6.0 or (reversal > 0.45 and amp > 3.0):
                return l
    return None


def _find_box_low(rows: List[Dict[str, Any]], close: float) -> Optional[float]:
    """找箱体下沿: 近20日低点中聚集的区域.

    将近20日所有low按价格排序, 寻找距离 ±1.5% 内有多个低点的聚集区。
    """
    lows = sorted([r.get("low", 0) for r in rows if r.get("low", 0) > 0])
    if len(lows) < 3:
        return None

    best_cluster_center = None
    best_count = 0
    for i in range(len(lows)):
        center = lows[i]
        if center <= 0:
            continue
        count = sum(1 for l in lows if abs(l - center) / center <= 0.015)
        if count > best_count:
            best_count = count
            best_cluster_center = center

    if best_count >= 3 and best_cluster_center is not None:
        return best_cluster_center
    return None


def _cluster_support_levels(
    candidates: List[Dict[str, Any]],
    close: float,
) -> Tuple[Optional[float], Optional[float], str, List[Dict[str, Any]]]:
    """从候选支撑位中找到最佳支撑聚集区.

    Args:
        candidates: [{"price": float, "source": str, "confidence": float}]
        close: 当前收盘价

    Returns:
        (zone_low, zone_high, basis_str, sorted_candidates)
    """
    if not candidates:
        return None, None, "", []

    # 过滤: 支撑位应该在收盘价下方或附近 (不超过 close+3%)
    valid = [c for c in candidates if c["price"] > 0 and c["price"] <= close * 1.03]
    if not valid:
        valid = candidates  # fallback: 全用

    # 按价格排序
    valid.sort(key=lambda c: c["price"])

    # 给每个候选找到 ±1.5% 范围内的其他候选数量
    best_idx = 0
    best_score = 0.0
    for i, c in enumerate(valid):
        price = c["price"]
        cluster_score = sum(
            other["confidence"]
            for other in valid
            if abs(other["price"] - price) / price <= 0.015
        )
        # 加权: 越接近 close 越好 (但要在下方)
        dist_pct = abs(close - price) / close if close > 0 else 0
        proximity_bonus = max(0, 1.0 - dist_pct * 5)  # 0~20% 内有bonus
        total = cluster_score + proximity_bonus * 0.3
        if total > best_score:
            best_score = total
            best_idx = i

    center = valid[best_idx]["price"]
    # 聚集区: 中心 ±1.5%
    cluster = [c for c in valid if abs(c["price"] - center) / center <= 0.015]
    prices = [c["price"] for c in cluster]
    zone_low = round(min(prices), 2)
    zone_high = round(max(prices), 2)

    # basis: 聚集区内最高置信度的来源
    best_source = max(cluster, key=lambda c: c["confidence"])["source"]
    sources = sorted(set(c["source"] for c in cluster))
    basis = "+".join(sources) if len(sources) > 1 else best_source

    return zone_low, zone_high, basis, valid


def compute_reference_levels(
    close: float,
    ma10: Optional[float],
    ma20: Optional[float],
    atr: Optional[float],
    high_20d: float,
    low_20d: float,
    rows: Optional[List[Dict[str, Any]]] = None,
    config: Optional[SetupTimingConfig] = None,
) -> Dict[str, Any]:
    """计算关键参考价位.

    v3.0: 当 enable_enhanced_reference_levels=True 且 rows 可用时,
    使用多来源候选支撑体系 (swing low / 分歧日低点 / 箱体下沿 / 近5日低点)。
    否则 fallback 到 v2.1 MA10/MA20 逻辑。

    Returns:
        dict with support_zone_low, support_zone_high, invalidation_level,
              resistance_1, resistance_2, ref_reward_risk, support_basis,
              candidate_support_levels (v3.0)
    """
    cfg = config or _DEFAULT_CONFIG
    result: Dict[str, Any] = {}
    use_enhanced = cfg.enable_enhanced_reference_levels and rows and len(rows) >= 10

    if use_enhanced:
        # 构建候选支撑位列表
        candidates: List[Dict[str, Any]] = []

        # 来源1: MA10
        if ma10 is not None and ma10 > 0:
            candidates.append({"price": ma10, "source": "ma10", "confidence": 0.8})

        # 来源2: MA20
        if ma20 is not None and ma20 > 0:
            candidates.append({"price": ma20, "source": "ma20", "confidence": 0.85})

        # 来源3: swing lows
        recent_20 = rows[-20:] if len(rows) >= 20 else rows
        swing_lows = _find_swing_lows(recent_20)
        for sl in swing_lows:
            candidates.append({"price": sl, "source": "swing_low", "confidence": 0.75})

        # 来源4: 分歧日低点
        rows_10d = rows[-10:] if len(rows) >= 10 else rows
        div_low = _find_divergence_low(rows_10d)
        if div_low is not None:
            candidates.append({"price": div_low, "source": "divergence_low", "confidence": 0.70})

        # 来源5: 近5日最低点
        recent_5 = rows[-5:] if len(rows) >= 5 else rows
        low_5d = min((r.get("low", float("inf")) for r in recent_5), default=None)
        if low_5d is not None and low_5d < float("inf"):
            candidates.append({"price": low_5d, "source": "recent_5d_low", "confidence": 0.65})

        # 来源6: 箱体下沿
        box_low = _find_box_low(recent_20, close)
        if box_low is not None:
            candidates.append({"price": box_low, "source": "box_low", "confidence": 0.70})

        # 聚集区检测
        zone_low, zone_high, basis, sorted_cands = _cluster_support_levels(candidates, close)

        if zone_low is not None and zone_high is not None:
            result["support_zone_low"] = zone_low
            result["support_zone_high"] = zone_high
            result["support_basis"] = basis
        else:
            # fallback to MA
            result.update(_basic_support_zone(close, ma10, ma20, low_20d))

        # 记录候选
        result["candidate_support_levels"] = [
            {
                "price": round(c["price"], 2),
                "source": c["source"],
                "distance_to_close": round((close - c["price"]) / close, 4) if close > 0 else 0,
                "confidence": c["confidence"],
            }
            for c in sorted_cands
        ]
    else:
        # v2.1 原始逻辑
        result.update(_basic_support_zone(close, ma10, ma20, low_20d))
        result["candidate_support_levels"] = []

    # 失效位: 多来源取最合理
    invalidation_candidates = []
    if ma20 is not None and atr is not None and atr > 0:
        invalidation_candidates.append(ma20 - 1.5 * atr)
    if use_enhanced:
        # swing low 下方 1×ATR
        swing_lows = _find_swing_lows(rows[-20:] if len(rows) >= 20 else rows)
        if swing_lows and atr is not None and atr > 0:
            nearest_swing = min(swing_lows, key=lambda sl: abs(sl - close))
            invalidation_candidates.append(nearest_swing - atr)
        # 支撑聚集区下沿
        sz_low = result.get("support_zone_low")
        if sz_low is not None and atr is not None and atr > 0:
            invalidation_candidates.append(sz_low - 0.5 * atr)

    if invalidation_candidates:
        # 取中位数 — 不取最高(太近)也不取最低(太远)
        invalidation_candidates.sort()
        mid_idx = len(invalidation_candidates) // 2
        result["invalidation_level"] = round(invalidation_candidates[mid_idx], 2)
    elif ma20 is not None:
        result["invalidation_level"] = round(ma20 * 0.95, 2)
    else:
        result["invalidation_level"] = round(low_20d * 0.97, 2)

    # 压力位1: 近20日高点
    result["resistance_1"] = round(high_20d, 2)

    # 压力位2 (v3.0): 前高 / 近期平台上沿
    resistance_2 = None
    if use_enhanced and rows:
        recent_40 = rows[-40:] if len(rows) >= 40 else rows
        highs_40 = [r.get("high", 0) for r in recent_40]
        if highs_40:
            max_40 = max(highs_40)
            if max_40 > high_20d * 1.01:  # 只有明显高于20日高点才有意义
                resistance_2 = round(max_40, 2)
    result["resistance_2"] = resistance_2

    # 参考盈亏比
    inv = result.get("invalidation_level")
    if inv is not None and close > inv:
        risk = close - inv
        reward = high_20d - close
        result["ref_reward_risk"] = round(reward / risk, 2) if risk > 0 else 0.0
    else:
        result["ref_reward_risk"] = 0.0

    return result


def _basic_support_zone(
    close: float,
    ma10: Optional[float],
    ma20: Optional[float],
    low_20d: float,
) -> Dict[str, Any]:
    """v2.1 原始支撑区间计算 (fallback)."""
    if ma10 is not None and ma20 is not None:
        return {
            "support_zone_low": round(min(ma10, ma20) * 0.99, 2),
            "support_zone_high": round(max(ma10, ma20) * 1.01, 2),
            "support_basis": "ma10" if abs(close - ma10) < abs(close - ma20) else "ma20",
        }
    elif ma20 is not None:
        return {
            "support_zone_low": round(ma20 * 0.98, 2),
            "support_zone_high": round(ma20 * 1.02, 2),
            "support_basis": "ma20",
        }
    elif ma10 is not None:
        return {
            "support_zone_low": round(ma10 * 0.98, 2),
            "support_zone_high": round(ma10 * 1.02, 2),
            "support_basis": "ma10",
        }
    else:
        return {
            "support_zone_low": round(low_20d, 2),
            "support_zone_high": round(low_20d * 1.03, 2),
            "support_basis": "box_low",
        }


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
# Action Cap 体系 (v3.0)
# ════════════════════════════════════════════════════════

def apply_action_caps(
    signal: SetupSignal,
    metrics: Dict[str, Any],
    upper_reversal_count_5d: Optional[int],
    limit_board_count_5d: Optional[int],
    config: Optional[SetupTimingConfig] = None,
) -> Tuple[str, List[str]]:
    """应用 action cap 硬规则，限制高风险结构的 action 上限.

    不修改 timing_score，只限制最终 action。
    所有触发的 cap 原因记录到 cap_reasons。

    Args:
        signal: 已完成评分和初始 action 映射的 SetupSignal
        metrics: _extract_eod_metrics 返回的中间指标
        upper_reversal_count_5d: 近5日上影线天数
        limit_board_count_5d: 近5日一字板天数
        config: 配置

    Returns:
        (capped_action, cap_reasons)
    """
    cfg = config or _DEFAULT_CONFIG
    if not cfg.enable_action_caps:
        return signal.action, []

    action = signal.action
    cap_reasons: List[str] = []

    # ── 规则 1-2: risk_score 过低 ──────────────────────
    if signal.risk_score < cfg.cap_min_risk_score_critical:
        new = _cap_action(action, "avoid_chase")
        if new != action:
            cap_reasons.append(f"risk_score={signal.risk_score:.2f}<{cfg.cap_min_risk_score_critical}→avoid_chase")
            action = new
    elif signal.risk_score < cfg.cap_min_risk_score_for_ready:
        new = _cap_action(action, "wait")
        if new != action:
            cap_reasons.append(f"risk_score={signal.risk_score:.2f}<{cfg.cap_min_risk_score_for_ready}→最高wait")
            action = new

    # ── 规则 3-4: level_confidence 过低 ────────────────
    if signal.level_confidence == "low":
        rr = signal.ref_reward_risk
        if rr is not None and rr < cfg.cap_min_reward_risk_for_ready:
            new = _cap_action(action, "wait")
            if new != action:
                cap_reasons.append(f"level_confidence=low且ref_rr={rr:.2f}<{cfg.cap_min_reward_risk_for_ready}→最高wait")
                action = new
        else:
            new = _cap_action(action, "watch")
            if new != action:
                cap_reasons.append("level_confidence=low→最高watch")
                action = new

    # ── 规则 5-6: 盈亏比不足 ──────────────────────────
    rr = signal.ref_reward_risk
    if rr is not None:
        if rr < cfg.cap_min_reward_risk_critical:
            new = _cap_action(action, "wait")
            if new != action:
                cap_reasons.append(f"ref_reward_risk={rr:.2f}<{cfg.cap_min_reward_risk_critical}→最高wait")
                action = new
        elif rr < cfg.cap_min_reward_risk_for_ready:
            new = _cap_action(action, "watch")
            if new != action:
                cap_reasons.append(f"ref_reward_risk={rr:.2f}<{cfg.cap_min_reward_risk_for_ready}→最高watch")
                action = new

    # ── 规则 7-8: 价格远离 MA20 ──────────────────────
    ma20 = metrics.get("ma20")
    close = metrics.get("close", 0)
    if ma20 is not None and ma20 > 0 and close > 0:
        dist_ma20_pct = abs(close - ma20) / ma20
        if dist_ma20_pct > cfg.cap_max_dist_ma20_critical:
            new = _cap_action(action, "wait")
            if new != action:
                cap_reasons.append(f"距MA20={dist_ma20_pct:.1%}>{cfg.cap_max_dist_ma20_critical:.0%}→最高wait")
                action = new
        elif dist_ma20_pct > cfg.cap_max_dist_ma20_for_ready:
            new = _cap_action(action, "watch")
            if new != action:
                cap_reasons.append(f"距MA20={dist_ma20_pct:.1%}>{cfg.cap_max_dist_ma20_for_ready:.0%}→最高watch")
                action = new

    # ── 规则 9: 高波动 + 高回撤 ──────────────────────
    atr_pct = metrics.get("atr_pct")
    max_dd = metrics.get("max_drawdown_5d")
    if atr_pct is not None and max_dd is not None:
        if atr_pct > cfg.cap_atr_drawdown_atr_threshold and abs(max_dd) > cfg.cap_atr_drawdown_dd_threshold:
            new = _cap_action(action, "wait")
            if new != action:
                cap_reasons.append(
                    f"ATR%={atr_pct:.1f}%>{cfg.cap_atr_drawdown_atr_threshold}%"
                    f"且回撤={abs(max_dd):.1f}%>{cfg.cap_atr_drawdown_dd_threshold}%→最高wait"
                )
                action = new

    # ── 规则 10: ATR% 极高 ───────────────────────────
    if atr_pct is not None and atr_pct > cfg.cap_max_atr_pct_for_ready:
        new = _cap_action(action, "watch")
        if new != action:
            cap_reasons.append(f"ATR%={atr_pct:.1f}%>{cfg.cap_max_atr_pct_for_ready}%→最高watch")
            action = new

    # ── 规则 11: 上影线密集 ──────────────────────────
    if upper_reversal_count_5d is not None and upper_reversal_count_5d > cfg.cap_max_upper_shadow_days:
        new = _cap_action(action, "wait")
        if new != action:
            cap_reasons.append(
                f"近5日上影线={upper_reversal_count_5d}天>{cfg.cap_max_upper_shadow_days}天→最高wait"
            )
            action = new

    # ── 规则 12-13: 弱市环境 (I5: 统一 bear/risk_off/ice_point) ──
    if signal.market_regime in WEAK_REGIMES:
        if signal.timing_score < cfg.cap_bear_ready_threshold and action == "setup_ready":
            cap_reasons.append(
                f"bear市且timing={signal.timing_score:.1f}<{cfg.cap_bear_ready_threshold}→不允许setup_ready"
            )
            action = _cap_action(action, "watch")
        if signal.risk_score < cfg.cap_bear_min_risk_score:
            new = _cap_action(action, "wait")
            if new != action:
                cap_reasons.append(
                    f"bear市且risk_score={signal.risk_score:.2f}<{cfg.cap_bear_min_risk_score}→最高wait"
                )
                action = new

    # ── 规则 14: 近期一字板频繁 ──────────────────────
    if limit_board_count_5d is not None and limit_board_count_5d > cfg.cap_max_limit_board_days:
        new = _cap_action(action, "watch")
        if new != action:
            cap_reasons.append(
                f"近5日一字板={limit_board_count_5d}天>{cfg.cap_max_limit_board_days}天→最高watch"
            )
            action = new

    return action, cap_reasons


# ════════════════════════════════════════════════════════
# Stage 1 Context 联动 (v3.0)
# ════════════════════════════════════════════════════════

def apply_stage1_context(
    signal: SetupSignal,
    stage1_context: Optional[Dict[str, Any]],
    config: Optional[SetupTimingConfig] = None,
) -> Tuple[str, Optional[str], bool]:
    """根据 Stage 1 四轴分数调整 action.

    Args:
        signal: 已完成 action cap 的 SetupSignal
        stage1_context: Stage 1 上下文, 包含:
            - hot_theme_score, trend_flow_score, liquidity_execution_score,
              risk_control_score, total_score, stage1_rank_pctile (0~1, 越大越靠前)
        config: 配置

    Returns:
        (adjusted_action, adjustment_reason, is_high_priority)
    """
    cfg = config or _DEFAULT_CONFIG
    if not cfg.enable_stage1_context or not stage1_context:
        return signal.action, None, False

    action = signal.action
    reasons: List[str] = []
    is_high_priority = False

    ht = stage1_context.get("hot_theme_score")
    le = stage1_context.get("liquidity_execution_score")
    rc = stage1_context.get("risk_control_score")
    rank_pctile = stage1_context.get("stage1_rank_pctile")  # 0~1, 越大越靠前

    # 主题强度联动
    if ht is not None and ht < cfg.min_hot_theme_for_ready:
        if action == "setup_ready":
            action = "watch"
            reasons.append(f"hot_theme={ht:.2f}<{cfg.min_hot_theme_for_ready}→降为watch")

    # high_priority 标记
    if (
        ht is not None and ht >= cfg.high_priority_hot_theme
        and signal.timing_score >= cfg.high_priority_timing
    ):
        is_high_priority = True

    # 流动性联动
    if le is not None:
        if le < cfg.min_liquidity_critical:
            new = _cap_action(action, "wait")
            if new != action:
                reasons.append(f"liquidity={le:.2f}<{cfg.min_liquidity_critical}→最高wait")
                action = new
        elif le < cfg.min_liquidity_for_ready:
            if action == "setup_ready":
                action = "watch"
                reasons.append(f"liquidity={le:.2f}<{cfg.min_liquidity_for_ready}→降为watch")

    # 风控联动
    if rc is not None:
        if rc < cfg.min_stage1_risk_critical:
            new = _cap_action(action, "wait")
            if new != action:
                reasons.append(f"stage1_risk={rc:.2f}<{cfg.min_stage1_risk_critical}→最高wait")
                action = new
        elif rc < cfg.min_stage1_risk_for_ready:
            if action == "setup_ready":
                action = "watch"
                reasons.append(f"stage1_risk={rc:.2f}<{cfg.min_stage1_risk_for_ready}→降为watch")

    # 总分排序联动: 后50%需要更高timing才能setup_ready
    if rank_pctile is not None and rank_pctile < 0.5:
        if action == "setup_ready" and signal.timing_score < cfg.stage1_bottom_pctile_timing:
            action = "watch"
            reasons.append(
                f"stage1排名后{(1-rank_pctile)*100:.0f}%"
                f"且timing={signal.timing_score:.1f}<{cfg.stage1_bottom_pctile_timing}→降为watch"
            )

    reason_str = "; ".join(reasons) if reasons else None
    return action, reason_str, is_high_priority


# ════════════════════════════════════════════════════════
# 理由生成 (v3.0 增强)
# ════════════════════════════════════════════════════════

def generate_reason(signal: SetupSignal) -> str:
    """生成多维度理由描述."""
    parts = []

    # 趋势描述
    if signal.trend_score >= 0.7:
        parts.append("趋势多头")
    elif signal.trend_score >= 0.4:
        parts.append("趋势偏多")
    else:
        parts.append("趋势走弱")

    # 回踩描述
    if signal.pullback_score >= 0.7:
        parts.append("回踩到位")
    elif signal.pullback_score >= 0.4:
        parts.append("回踩接近")
    else:
        parts.append("位置偏高或回撤过深")

    # 量价描述
    if signal.volume_score >= 0.7:
        parts.append("回调缩量确认")
    elif signal.volume_score >= 0.4:
        parts.append("量能尚可")
    elif signal.volume_score < 0.3:
        parts.append("下跌放量需谨慎")

    # 修复描述
    if signal.repair_score >= 0.7:
        parts.append("分歧后修复")
    elif signal.repair_score <= 0.3:
        parts.append("分歧后未修复")

    # 风险描述
    if signal.risk_score < 0.3:
        parts.append("风险偏高")
    elif signal.risk_score < 0.5:
        parts.append("波动较大")

    return "，".join(parts)


def generate_score_commentary(signal: SetupSignal) -> Dict[str, str]:
    """为每个评分维度生成一句话解释."""
    commentary: Dict[str, str] = {}

    # 趋势
    if signal.trend_score >= 0.7:
        commentary["trend"] = "均线多头排列，趋势向上"
    elif signal.trend_score >= 0.5:
        commentary["trend"] = "趋势偏多但正在回踩"
    elif signal.trend_score >= 0.3:
        commentary["trend"] = "趋势转弱，均线开始走平"
    else:
        commentary["trend"] = "趋势走弱，均线空头排列"

    # 回踩
    if signal.pullback_score >= 0.7:
        commentary["pullback"] = "回踩到MA10/MA20附近，位置理想"
    elif signal.pullback_score >= 0.5:
        commentary["pullback"] = "回踩接近关键支撑，但未到最佳区域"
    elif signal.pullback_score >= 0.3:
        commentary["pullback"] = "位置偏高或回撤过深"
    else:
        commentary["pullback"] = "远离理想回踩区域"

    # 量价
    if signal.volume_score >= 0.7:
        commentary["volume"] = "回调缩量明显，量价配合良好"
    elif signal.volume_score >= 0.5:
        commentary["volume"] = "量能尚可，无明显异常"
    elif signal.volume_score >= 0.3:
        commentary["volume"] = "量能偏弱或下跌放量"
    else:
        commentary["volume"] = "量能异常，下跌放量或底部无承接"

    # 修复
    if signal.repair_score >= 0.7:
        commentary["repair"] = "分歧后缩量守住支撑并出现修复阳线"
    elif signal.repair_score > 0.5:
        commentary["repair"] = "有分歧结构但修复尚不充分"
    elif signal.repair_score < 0.3 and signal.repair_score != 0.5:
        commentary["repair"] = "分歧后未能守住关键支撑"
    else:
        commentary["repair"] = "无明显分歧结构"

    # 风险
    if signal.risk_score >= 0.7:
        commentary["risk"] = "波动适中，风险可控"
    elif signal.risk_score >= 0.4:
        commentary["risk"] = "波动较大或有上影线"
    else:
        commentary["risk"] = "高波动/大回撤/上影线密集，风险偏高"

    return commentary


# ════════════════════════════════════════════════════════
# 生成警告 (v3.0 增强)
# ════════════════════════════════════════════════════════

def generate_warnings(
    signal: SetupSignal,
    metrics: Optional[Dict[str, Any]] = None,
) -> List[str]:
    """生成风险提示列表.

    v3.0: 增加更具体的警告内容，包括 action cap 原因。
    """
    warns = []
    if signal.risk_score < 0.3:
        warns.append(f"风险评分偏低({signal.risk_score:.2f})，注意控制仓位")
    if signal.level_confidence == "low":
        warns.append("参考价位置信度低，仅供参考")
    if signal.market_regime in WEAK_REGIMES:
        warns.append(f"大盘环境偏弱({signal.market_regime})，谨慎操作")
    if signal.ref_reward_risk is not None and signal.ref_reward_risk < 1.0:
        warns.append(f"盈亏比不足1:1(当前{signal.ref_reward_risk:.2f})")

    # v3.0 新增: 基于 metrics 的具体警告
    if metrics:
        atr_pct = metrics.get("atr_pct")
        if atr_pct is not None and atr_pct > 8.0:
            warns.append(f"ATR%={atr_pct:.1f}%，波动较大")
        dist_ma20 = None
        ma20 = metrics.get("ma20")
        close = metrics.get("close", 0)
        if ma20 and ma20 > 0 and close > 0:
            dist_ma20 = abs(close - ma20) / ma20
            if dist_ma20 > 0.12:
                warns.append(f"距MA20={dist_ma20:.1%}，偏离较远不适合低吸")
        max_dd = metrics.get("max_drawdown_5d")
        if max_dd is not None and abs(max_dd) > 10.0:
            warns.append(f"近5日最大回撤={abs(max_dd):.1f}%")

    # action cap 原因合并到 warnings
    for reason in signal.action_cap_reasons:
        warns.append(f"[action cap] {reason}")

    # stage1 调整原因
    if signal.stage1_adjustment_reason:
        warns.append(f"[stage1联动] {signal.stage1_adjustment_reason}")

    return warns


# ════════════════════════════════════════════════════════
# 盘中执行提示 (v3.0)
# ════════════════════════════════════════════════════════

def generate_intraday_hint(signal: SetupSignal) -> Tuple[bool, str]:
    """生成盘中执行确认提示.

    仅对 setup_ready 和 watch 级别生成提示。
    """
    if signal.action == "avoid_chase":
        return False, ""
    if signal.action == "wait":
        return False, ""

    hints = []
    if signal.action == "setup_ready":
        if signal.risk_score < 0.5:
            hints.append("需确认盘中波动可控")
        if signal.pullback_score >= 0.6:
            hints.append("需确认回踩支撑不破")
        else:
            hints.append("需观察是否企稳")
        if signal.volume_score >= 0.6:
            hints.append("需确认放量修复")
        else:
            hints.append("需观察量能配合")
        hints.append("不适合竞价追高")
    elif signal.action == "watch":
        hints.append("需观察是否回踩到位")
        hints.append("需确认量价配合")

    return True, "；".join(hints)


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
    stage1_context: Optional[Dict[str, Any]] = None,
    config: Optional[SetupTimingConfig] = None,
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
        stage1_context: Stage 1 四轴分数 (v3.0)
        config: SetupTimingConfig (v3.0)

    Returns:
        SetupSignal
    """
    cfg = config or _DEFAULT_CONFIG
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
        atr_pct=metrics["atr_pct"],
        config=cfg,
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

    # ── 动作映射 (基础) ─────────────────────────────────

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
        rows=eod_rows,
        config=cfg,
    )
    signal.support_zone_low = ref_levels.get("support_zone_low")
    signal.support_zone_high = ref_levels.get("support_zone_high")
    signal.invalidation_level = ref_levels.get("invalidation_level")
    signal.resistance_1 = ref_levels.get("resistance_1")
    signal.resistance_2 = ref_levels.get("resistance_2")
    signal.ref_reward_risk = ref_levels.get("ref_reward_risk")
    signal.support_basis = ref_levels.get("support_basis", "")
    signal.candidate_support_levels = ref_levels.get("candidate_support_levels", [])

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

    # ── Action Cap (v3.0) ──────────────────────────────

    capped_action, cap_reasons = apply_action_caps(
        signal=signal,
        metrics=metrics,
        upper_reversal_count_5d=upper_reversal_count_5d,
        limit_board_count_5d=limit_board_count_5d,
        config=cfg,
    )
    signal.action = capped_action
    signal.action_cap_reasons = cap_reasons

    # ── Stage 1 Context (v3.0) ─────────────────────────

    signal.final_action_before_stage1_cap = signal.action
    if stage1_context and cfg.enable_stage1_context:
        signal.stage1_context_used = True
        adjusted_action, adj_reason, high_pri = apply_stage1_context(
            signal=signal,
            stage1_context=stage1_context,
            config=cfg,
        )
        signal.action = adjusted_action
        signal.stage1_adjustment_reason = adj_reason
        signal.high_priority_watch = high_pri
    signal.final_action_after_stage1_cap = signal.action

    # ── 评分维度解释 (v3.0) ────────────────────────────

    signal.score_components_commentary = generate_score_commentary(signal)

    # ── 盘中执行提示 (v3.0) ────────────────────────────

    needs_confirm, hint = generate_intraday_hint(signal)
    signal.requires_intraday_confirmation = needs_confirm
    signal.intraday_check_hint = hint

    # ── 理由 + 警告 ────────────────────────────────────

    signal.reason = generate_reason(signal)
    signal.warnings = generate_warnings(signal, metrics=metrics)

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
    index_amounts_20d: Optional[List[float]] = None,
    config: Optional[SetupTimingConfig] = None,
) -> List[SetupSignal]:
    """对 tradeable 的 details 批量运行观察时机评估.

    Args:
        details: 通过 pass_stage1 的 HotStockDetail 列表
        tushare_client: TushareClient 实例 (用于获取日线数据)
        trade_cal: TradeCalendar 实例
        trade_date_str: 运行日期 (YYYY-MM-DD)
        index_closes_20d: 指数近20日收盘价序列
        index_amounts_20d: 指数近20日成交额序列 (v3.0, 元)
        config: SetupTimingConfig (v3.0)

    Returns:
        List[SetupSignal]
    """
    import datetime as _dt

    cfg = config or _DEFAULT_CONFIG

    # 大盘环境
    if cfg.enable_enhanced_market_regime:
        regime_detail = compute_enhanced_market_regime(
            index_closes_20d, index_amounts_20d, config=cfg,
        )
        regime = regime_detail["final_regime"]
        multiplier = regime_detail["multiplier"]
        logger.info(
            "[setup_timing] 增强大盘环境: %s (乘数=%.2f) index_trend=%s turnover=%s",
            regime, multiplier, regime_detail["index_trend"], regime_detail["turnover_state"],
        )
    else:
        regime, multiplier = compute_market_regime(index_closes_20d)
        logger.info("[setup_timing] 大盘环境: %s (乘数=%.2f)", regime, multiplier)

    # v3.0: 预计算 Stage 1 排名百分位 (用于 stage1_context 联动)
    total_scores = []
    for d in details:
        ts = getattr(d, "total_score", None)
        if ts is not None and isinstance(ts, (int, float)):
            total_scores.append(ts)
    total_scores.sort()
    total_count = len(total_scores)

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

            # v3.0: 构建 stage1_context
            stage1_ctx = None
            if cfg.enable_stage1_context:
                ts_val = getattr(detail, "total_score", None)
                rank_pctile = None
                # I10: 候选池样本数达到阈値才使用排名百分位，避免小样本误伤
                rank_sample_ok = total_count >= cfg.min_stage1_rank_sample_size
                if ts_val is not None and isinstance(ts_val, (int, float)) and rank_sample_ok:
                    # 用 bisect 计算百分位 (越大越靠前)
                    import bisect
                    pos = bisect.bisect_right(total_scores, ts_val)
                    rank_pctile = pos / total_count

                def _safe_score(val: Any) -> Any:
                    """Only pass through numeric scores, not MagicMock etc."""
                    return val if isinstance(val, (int, float)) else None

                stage1_ctx = {
                    "hot_theme_score": _safe_score(getattr(detail, "hot_theme_score", None)),
                    "trend_flow_score": _safe_score(getattr(detail, "trend_flow_score", None)),
                    "liquidity_execution_score": _safe_score(getattr(detail, "liquidity_execution_score", None)),
                    "risk_control_score": _safe_score(getattr(detail, "risk_control_score", None)),
                    "total_score": ts_val if isinstance(ts_val, (int, float)) else None,
                    "stage1_rank_pctile": rank_pctile,
                }

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
                stage1_context=stage1_ctx,
                config=cfg,
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
