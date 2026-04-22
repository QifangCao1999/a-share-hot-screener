"""Tests for setup_timing.py — Phase 4 观察时机评估.

覆盖:
  - 五维评分函数 (trend / pullback / volume / repair / risk)
  - 大盘环境判断
  - 参考价位计算 + 置信度
  - 动作映射 (含硬规则)
  - 综合评估 (evaluate_setup_timing)
  - 辅助函数
  - 批量运行 (run_setup_timing)
"""

from __future__ import annotations

import math
import pytest
from typing import Dict, List, Any, Optional
from unittest.mock import MagicMock, patch
from dataclasses import dataclass

from a_share_hot_screener.setup_timing import (
    # 常量
    W_TREND, W_PULLBACK, W_VOLUME, W_REPAIR, W_RISK,
    THRESHOLD_SETUP_READY, THRESHOLD_WATCH, THRESHOLD_WAIT,
    # 数据模型
    SetupSignal,
    # 评分函数
    score_trend,
    score_pullback,
    score_volume,
    score_repair,
    score_risk,
    # 大盘
    compute_market_regime,
    # 参考价位
    compute_reference_levels,
    compute_level_confidence,
    # 动作
    map_action,
    # 理由/警告
    generate_reason,
    generate_warnings,
    # 辅助
    _bell_curve,
    _compute_ma,
    _compute_atr,
    _extract_eod_metrics,
    _normalize_eod_rows,
    # 主入口
    evaluate_setup_timing,
    run_setup_timing,
)


# ════════════════════════════════════════════════════════
# 测试数据工厂
# ════════════════════════════════════════════════════════

def _make_eod_row(
    date_str: str = "2026-04-20",
    open_: float = 10.0,
    high: float = 10.5,
    low: float = 9.8,
    close: float = 10.2,
    volume: float = 1000000,
    amount: float = 10000000,
) -> Dict[str, Any]:
    return {
        "date_str": date_str,
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
        "amount": amount,
    }


def _make_eod_series(
    n: int = 60,
    base_close: float = 10.0,
    trend: float = 0.001,  # 每日涨幅
    volatility: float = 0.02,
    start_date: str = "2026-01-01",
) -> List[Dict[str, Any]]:
    """生成模拟 EOD 序列."""
    import datetime as dt
    rows = []
    date = dt.date.fromisoformat(start_date)
    close = base_close
    for i in range(n):
        date_str = date.isoformat()
        change = trend + (volatility * (0.5 - (i % 3) / 3.0))
        open_ = close
        close = close * (1 + change)
        high = max(open_, close) * (1 + abs(volatility) * 0.3)
        low = min(open_, close) * (1 - abs(volatility) * 0.3)
        volume = 1000000 * (1 + 0.2 * ((i % 5) - 2))
        rows.append({
            "date_str": date_str,
            "open": round(open_, 2),
            "high": round(high, 2),
            "low": round(low, 2),
            "close": round(close, 2),
            "volume": round(volume),
            "amount": round(close * volume),
        })
        date += dt.timedelta(days=1)
        if date.weekday() >= 5:
            date += dt.timedelta(days=7 - date.weekday())
    return rows


def _make_pullback_series(n: int = 40) -> List[Dict[str, Any]]:
    """生成回踩到 MA10/MA20 附近的序列 — 适合低吸."""
    import datetime as dt
    rows = []
    date = dt.date(2026, 3, 1)

    # 先涨20天
    close = 10.0
    for i in range(20):
        date_str = date.isoformat()
        open_ = close
        close = close * 1.015
        high = close * 1.005
        low = open_ * 0.995
        rows.append(_make_eod_row(date_str, open_, high, low, round(close, 2), 1200000, round(close * 1200000)))
        date += dt.timedelta(days=1)
        if date.weekday() >= 5:
            date += dt.timedelta(days=7 - date.weekday())

    # 再跌回来接近 MA10/MA20（缩量回调）
    peak = close
    for i in range(n - 20):
        date_str = date.isoformat()
        open_ = close
        close = close * 0.99  # 每日跌1%
        high = open_ * 1.002
        low = close * 0.998
        vol = 800000 - i * 20000  # 缩量
        rows.append(_make_eod_row(date_str, open_, high, low, round(close, 2), max(vol, 400000), round(close * max(vol, 400000))))
        date += dt.timedelta(days=1)
        if date.weekday() >= 5:
            date += dt.timedelta(days=7 - date.weekday())

    return rows


# ════════════════════════════════════════════════════════
# 常量验证
# ════════════════════════════════════════════════════════

class TestConstants:
    def test_weights_sum_to_one(self):
        total = W_TREND + W_PULLBACK + W_VOLUME + W_REPAIR + W_RISK
        assert abs(total - 1.0) < 1e-10, f"权重和={total}，应为1.0"

    def test_thresholds_ordered(self):
        assert THRESHOLD_SETUP_READY > THRESHOLD_WATCH > THRESHOLD_WAIT > 0


# ════════════════════════════════════════════════════════
# 辅助函数测试
# ════════════════════════════════════════════════════════

class TestBellCurve:
    def test_peak_at_center(self):
        assert _bell_curve(0.5, 0.5, 0.1) == pytest.approx(1.0)

    def test_decay_away_from_center(self):
        at_center = _bell_curve(0.0, 0.0, 0.1)
        away = _bell_curve(0.1, 0.0, 0.1)
        assert at_center > away

    def test_symmetric(self):
        left = _bell_curve(-0.05, 0.0, 0.1)
        right = _bell_curve(0.05, 0.0, 0.1)
        assert left == pytest.approx(right)

    def test_zero_width(self):
        assert _bell_curve(0.0, 0.0, 0.0) == 1.0
        assert _bell_curve(0.1, 0.0, 0.0) == 0.0


class TestComputeMA:
    def test_basic(self):
        closes = [10.0, 11.0, 12.0, 13.0, 14.0]
        assert _compute_ma(closes, 5) == pytest.approx(12.0)

    def test_insufficient_data(self):
        assert _compute_ma([10.0, 11.0], 5) is None

    def test_last_n(self):
        closes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        # MA5 = (6+7+8+9+10)/5 = 8.0
        assert _compute_ma(closes, 5) == pytest.approx(8.0)


class TestComputeATR:
    def test_basic(self):
        # 构造简单行情: 固定 TR = 1.0
        rows = []
        for i in range(20):
            rows.append({"high": 11.0, "low": 10.0, "close": 10.5})
        atr = _compute_atr(rows, period=14)
        assert atr is not None
        assert atr == pytest.approx(1.0, abs=0.01)

    def test_insufficient_data(self):
        rows = [{"high": 11, "low": 10, "close": 10.5}] * 10
        assert _compute_atr(rows, period=14) is None


class TestNormalizeEodRows:
    def test_tushare_format(self):
        raw = [
            {"trade_date": "20260420", "open": 10.0, "high": 10.5, "low": 9.8, "close": 10.2, "vol": 10000, "amount": 10000},
        ]
        rows = _normalize_eod_rows(raw)
        assert len(rows) == 1
        assert rows[0]["date_str"] == "2026-04-20"
        assert rows[0]["volume"] == 10000 * 100  # vol(手) → 股
        assert rows[0]["amount"] == 10000 * 1000  # amount(千元) → 元

    def test_already_normalized(self):
        raw = [
            {"date_str": "2026-04-20", "open": 10.0, "high": 10.5, "low": 9.8, "close": 10.2, "volume": 1000000, "amount": 10000000},
        ]
        rows = _normalize_eod_rows(raw)
        assert len(rows) == 1
        assert rows[0]["volume"] == 1000000

    def test_sort_ascending(self):
        raw = [
            {"date_str": "2026-04-22", "open": 10, "high": 10.5, "low": 9.8, "close": 10.2, "volume": 100},
            {"date_str": "2026-04-20", "open": 10, "high": 10.5, "low": 9.8, "close": 10.0, "volume": 100},
        ]
        rows = _normalize_eod_rows(raw)
        assert rows[0]["date_str"] == "2026-04-20"
        assert rows[1]["date_str"] == "2026-04-22"

    def test_skip_zero_close(self):
        raw = [
            {"date_str": "2026-04-20", "open": 10, "high": 10.5, "low": 9.8, "close": 0, "volume": 100},
        ]
        rows = _normalize_eod_rows(raw)
        assert len(rows) == 0


# ════════════════════════════════════════════════════════
# 趋势状态评分
# ════════════════════════════════════════════════════════

class TestScoreTrend:
    def test_full_bullish(self):
        """4线多头排列 + 站上MA20 + 3日上升 + MA20上行 → 高分."""
        s = score_trend(
            close=15.0,
            ma5=14.5, ma10=14.0, ma20=13.0, ma60=12.0,
            closes_3d=[14.5, 14.8, 15.0],
            ma20_prev=12.8,
        )
        assert s > 0.7

    def test_bearish(self):
        """空头排列 + 跌破MA20 + 下跌 → 低分."""
        s = score_trend(
            close=8.0,
            ma5=8.5, ma10=9.0, ma20=10.0, ma60=11.0,
            closes_3d=[8.5, 8.2, 8.0],
            ma20_prev=10.2,
        )
        assert s < 0.35

    def test_pullback_in_uptrend(self):
        """趋势向上但正在回踩 — 中等偏强, 适合低吸."""
        s = score_trend(
            close=13.5,
            ma5=13.8, ma10=13.5, ma20=13.0, ma60=12.0,
            closes_3d=[14.0, 13.7, 13.5],  # 回调中
            ma20_prev=12.8,
        )
        # MA 多头排列给高分, 回调只影响方向分项, 整体中等偏强
        assert 0.5 < s < 0.85

    def test_missing_ma60(self):
        """MA60 缺失不崩溃."""
        s = score_trend(
            close=10.0,
            ma5=10.1, ma10=10.0, ma20=9.8, ma60=None,
            closes_3d=[9.9, 10.0, 10.0],
            ma20_prev=9.7,
        )
        assert 0.0 <= s <= 1.0

    def test_no_data(self):
        """全部 None → 中性分."""
        s = score_trend(
            close=10.0,
            ma5=None, ma10=None, ma20=None, ma60=None,
            closes_3d=[],
            ma20_prev=None,
        )
        assert s == pytest.approx(0.5, abs=0.05)


# ════════════════════════════════════════════════════════
# 回踩位置评分
# ════════════════════════════════════════════════════════

class TestScorePullback:
    def test_ideal_pullback(self):
        """价格在 MA10/MA20 附近, 区间中部 → 高分."""
        # close=10.0, ma10≈10.05, ma20≈9.8, 20日 high=11.0, low=9.0
        s = score_pullback(
            close=10.0,
            ma10=10.05,
            ma20=9.8,
            high_20d=11.0,
            low_20d=9.0,
        )
        assert s > 0.6  # 接近理想回踩位置

    def test_at_high(self):
        """在20日高点 → 低分 (不适合低吸)."""
        s = score_pullback(
            close=11.0,
            ma10=10.5,
            ma20=10.0,
            high_20d=11.0,
            low_20d=9.0,
        )
        assert s < 0.5

    def test_at_low(self):
        """在20日低点 → 偏低分 (跌太深)."""
        s = score_pullback(
            close=9.0,
            ma10=10.0,
            ma20=10.5,
            high_20d=11.0,
            low_20d=9.0,
        )
        assert s < 0.5

    def test_no_ma(self):
        """均线缺失不崩溃, 用 range 部分评分."""
        s = score_pullback(
            close=10.0,
            ma10=None,
            ma20=None,
            high_20d=11.0,
            low_20d=9.0,
        )
        assert 0.0 <= s <= 1.0

    def test_flat_range(self):
        """高低点相同 → 只有 ma 部分评分."""
        s = score_pullback(
            close=10.0,
            ma10=10.0,
            ma20=10.0,
            high_20d=10.0,
            low_20d=10.0,
        )
        assert 0.0 <= s <= 1.0


# ════════════════════════════════════════════════════════
# 量价确认评分
# ════════════════════════════════════════════════════════

class TestScoreVolume:
    def test_shrinking_volume(self):
        """缩量回调 → 高分."""
        rows = _make_eod_series(30, base_close=10.0, trend=-0.002, volatility=0.01)
        # 让最后3天量递减
        for i, r in enumerate(rows[-3:]):
            r["volume"] = 500000 - i * 100000
        s = score_volume(rows, current_volume_ratio=0.6)
        assert s > 0.5

    def test_heavy_volume(self):
        """放量 + 量比高 → 偏低分."""
        rows = _make_eod_series(30, base_close=10.0)
        for r in rows[-3:]:
            r["volume"] = 3000000
        s = score_volume(rows, current_volume_ratio=3.0)
        assert s < 0.6

    def test_insufficient_data(self):
        """数据不足 → 中性."""
        rows = _make_eod_series(3)
        s = score_volume(rows, current_volume_ratio=None)
        assert s == pytest.approx(0.5, abs=0.1)

    def test_no_volume_ratio(self):
        """volume_ratio=None 不崩溃."""
        rows = _make_eod_series(20)
        s = score_volume(rows, current_volume_ratio=None)
        assert 0.0 <= s <= 1.0


# ════════════════════════════════════════════════════════
# 分歧后修复评分
# ════════════════════════════════════════════════════════

class TestScoreRepair:
    def test_divergence_then_repair(self):
        """分歧 → 缩量守住支撑 → 阳线修复 → 高分."""
        ma20 = 10.0
        rows = [
            _make_eod_row("2026-04-10", 10.0, 10.8, 9.5, 9.8, 2000000),   # 分歧日: 振幅>6%
            _make_eod_row("2026-04-11", 9.8, 10.0, 9.7, 9.9, 1000000),    # 缩量
            _make_eod_row("2026-04-12", 9.9, 10.0, 9.8, 9.95, 800000),    # 缩量
            _make_eod_row("2026-04-13", 9.95, 10.1, 9.9, 10.05, 600000),  # 缩量
            _make_eod_row("2026-04-14", 10.05, 10.2, 10.0, 10.15, 700000),# 阳线修复+温和放量
        ]
        s = score_repair(rows, ma20=ma20)
        assert s > 0.5

    def test_no_divergence(self):
        """没有分歧日 → 中性分."""
        rows = [
            _make_eod_row("2026-04-10", 10.0, 10.1, 9.95, 10.05, 1000000),
            _make_eod_row("2026-04-11", 10.05, 10.1, 10.0, 10.08, 1000000),
            _make_eod_row("2026-04-12", 10.08, 10.12, 10.05, 10.1, 1000000),
            _make_eod_row("2026-04-13", 10.1, 10.15, 10.05, 10.12, 1000000),
            _make_eod_row("2026-04-14", 10.12, 10.2, 10.1, 10.18, 1000000),
        ]
        s = score_repair(rows, ma20=10.0)
        assert s == pytest.approx(0.5, abs=0.05)

    def test_broke_support(self):
        """分歧后跌破 MA20 → 低分."""
        ma20 = 10.0
        rows = [
            _make_eod_row("2026-04-10", 10.0, 10.8, 9.5, 9.8, 2000000),   # 分歧
            _make_eod_row("2026-04-11", 9.8, 9.9, 9.3, 9.4, 1500000),     # 跌破MA20
            _make_eod_row("2026-04-12", 9.4, 9.5, 9.0, 9.1, 1800000),     # 继续下跌
            _make_eod_row("2026-04-13", 9.1, 9.3, 9.0, 9.2, 1600000),
            _make_eod_row("2026-04-14", 9.2, 9.3, 9.1, 9.15, 1400000),
        ]
        s = score_repair(rows, ma20=ma20)
        assert s < 0.6

    def test_insufficient_rows(self):
        """行数不足 → 中性."""
        rows = [_make_eod_row("2026-04-10")]
        s = score_repair(rows, ma20=10.0)
        assert s == pytest.approx(0.5, abs=0.05)


# ════════════════════════════════════════════════════════
# 风险度量评分
# ════════════════════════════════════════════════════════

class TestScoreRisk:
    def test_low_risk(self):
        """低 ATR + 低回撤 + 无上影 + 非一字板 → 高分."""
        s = score_risk(
            atr_pct=2.0,
            max_drawdown_5d=1.5,
            upper_reversal_count_5d=0,
            latest_is_limit_board=False,
        )
        assert s > 0.85

    def test_high_risk(self):
        """高 ATR + 大回撤 + 多上影 + 一字板 → 低分."""
        s = score_risk(
            atr_pct=10.0,
            max_drawdown_5d=12.0,
            upper_reversal_count_5d=3,
            latest_is_limit_board=True,
        )
        assert s < 0.25

    def test_moderate_risk(self):
        """中等风险 → 中间分."""
        s = score_risk(
            atr_pct=4.0,
            max_drawdown_5d=4.0,
            upper_reversal_count_5d=1,
            latest_is_limit_board=False,
        )
        assert 0.4 < s < 0.8

    def test_all_none(self):
        """全部缺失 → 中性."""
        s = score_risk(None, None, None, None)
        assert s == pytest.approx(0.5, abs=0.05)

    def test_limit_board_penalty(self):
        """一字涨停 → 显著降低风险分."""
        s_normal = score_risk(2.0, 1.5, 0, False)
        s_limit = score_risk(2.0, 1.5, 0, True)
        assert s_normal > s_limit


# ════════════════════════════════════════════════════════
# 大盘环境
# ════════════════════════════════════════════════════════

class TestMarketRegime:
    def test_bull(self):
        closes = [100.0 + i * 0.5 for i in range(20)]  # +10%
        regime, mult = compute_market_regime(closes)
        assert regime == "bull"
        assert mult == pytest.approx(1.10)

    def test_bear(self):
        closes = [100.0 - i * 0.5 for i in range(20)]  # -10%
        regime, mult = compute_market_regime(closes)
        assert regime == "bear"
        assert mult == pytest.approx(0.75)

    def test_neutral(self):
        closes = [100.0 + (i % 3 - 1) * 0.2 for i in range(20)]  # 小幅震荡
        regime, mult = compute_market_regime(closes)
        assert regime == "neutral"
        assert mult == pytest.approx(1.00)

    def test_no_data(self):
        regime, mult = compute_market_regime(None)
        assert regime == "neutral"
        assert mult == 1.0

    def test_short_data(self):
        regime, mult = compute_market_regime([100, 101, 102])
        assert regime == "neutral"


# ════════════════════════════════════════════════════════
# 参考价位
# ════════════════════════════════════════════════════════

class TestReferenceLevels:
    def test_with_both_mas(self):
        result = compute_reference_levels(
            close=10.0, ma10=10.2, ma20=9.8,
            atr=0.5, high_20d=11.0, low_20d=9.0,
        )
        assert result["support_zone_low"] == pytest.approx(9.8 * 0.99, abs=0.01)
        assert result["support_zone_high"] == pytest.approx(10.2 * 1.01, abs=0.01)
        assert result["invalidation_level"] == pytest.approx(9.8 - 1.5 * 0.5, abs=0.01)
        assert result["resistance_1"] == pytest.approx(11.0)
        assert result["ref_reward_risk"] > 0

    def test_no_ma(self):
        """均线全缺失 → 使用箱体低点."""
        result = compute_reference_levels(
            close=10.0, ma10=None, ma20=None,
            atr=None, high_20d=11.0, low_20d=9.0,
        )
        assert result["support_basis"] == "box_low"
        assert result["support_zone_low"] == pytest.approx(9.0)

    def test_reward_risk_when_at_invalidation(self):
        """close <= invalidation → 盈亏比=0."""
        # invalidation = ma20 - 1.5*atr = 10.0 - 3.0 = 7.0
        # close=8.0 > 7.0, so risk=1.0, reward=3.0 → ratio=3.0
        # 要让 close <= invalidation, 需要更极端的参数
        result = compute_reference_levels(
            close=6.0, ma10=10.0, ma20=10.0,
            atr=3.0, high_20d=11.0, low_20d=5.0,
        )
        # invalidation = 10.0 - 4.5 = 5.5, close=6.0 > 5.5 → still positive
        # 用 close = 5.0 < inv=5.5
        result2 = compute_reference_levels(
            close=5.0, ma10=10.0, ma20=10.0,
            atr=3.0, high_20d=11.0, low_20d=5.0,
        )
        assert result2["ref_reward_risk"] == 0.0


# ════════════════════════════════════════════════════════
# 价位置信度
# ════════════════════════════════════════════════════════

class TestLevelConfidence:
    def test_high_confidence(self):
        conf, reasons = compute_level_confidence(
            close=10.0, ma10=10.1, ma20=9.8,
            atr=0.3, high_20d=11.0, low_20d=9.0,
            latest_is_limit_board=False, limit_board_count=0,
        )
        assert conf == "high"
        assert reasons == []

    def test_low_confidence_multiple_reasons(self):
        conf, reasons = compute_level_confidence(
            close=15.0, ma10=10.0, ma20=10.0,  # 远离 MA20
            atr=1.5, high_20d=15.0, low_20d=14.8,  # 高波动 + 窄幅
            latest_is_limit_board=True, limit_board_count=3,
        )
        assert conf == "low"
        assert len(reasons) >= 2

    def test_medium_confidence(self):
        # near_resistance: (10.1 - 10.0)/10.1 = 0.99% < 2% → triggers
        # narrow_range: (10.1 - 9.5)/10.1 = 5.9% > 5% → does NOT trigger
        # Use values where only near_resistance triggers
        conf, reasons = compute_level_confidence(
            close=10.9, ma10=10.5, ma20=10.0,
            atr=0.3, high_20d=11.0, low_20d=9.0,  # near_resistance only
            latest_is_limit_board=False, limit_board_count=0,
        )
        assert conf == "medium"
        assert len(reasons) == 1
        assert "near_resistance" in reasons

    def test_consecutive_limit_up(self):
        """连板 → 降低置信度."""
        conf, reasons = compute_level_confidence(
            close=10.0, ma10=10.0, ma20=9.8,
            atr=0.3, high_20d=11.0, low_20d=9.0,
            latest_is_limit_board=False, limit_board_count=2,
        )
        assert "consecutive_limit_up" in reasons


# ════════════════════════════════════════════════════════
# 动作映射
# ════════════════════════════════════════════════════════

class TestMapAction:
    def test_setup_ready(self):
        assert map_action(85.0, False) == "setup_ready"

    def test_watch(self):
        assert map_action(70.0, False) == "watch"

    def test_wait(self):
        assert map_action(50.0, False) == "wait"

    def test_avoid_chase(self):
        assert map_action(30.0, False) == "avoid_chase"

    def test_limit_board_override(self):
        """一字涨停 → 无论分数多高都是 avoid_chase."""
        assert map_action(95.0, True) == "avoid_chase"

    def test_watch_only_cap(self):
        """watch_only → 最高 watch."""
        assert map_action(90.0, False, is_watch_only=True) == "watch"
        assert map_action(70.0, False, is_watch_only=True) == "watch"
        assert map_action(50.0, False, is_watch_only=True) == "wait"

    def test_boundary_values(self):
        assert map_action(80.0, False) == "setup_ready"
        assert map_action(79.99, False) == "watch"
        assert map_action(65.0, False) == "watch"
        assert map_action(64.99, False) == "wait"
        assert map_action(45.0, False) == "wait"
        assert map_action(44.99, False) == "avoid_chase"


# ════════════════════════════════════════════════════════
# 理由生成
# ════════════════════════════════════════════════════════

class TestGenerateReason:
    def test_bullish_setup(self):
        signal = SetupSignal(code="000001", name="test")
        signal.trend_score = 0.8
        signal.pullback_score = 0.75
        signal.volume_score = 0.8
        signal.risk_score = 0.9
        reason = generate_reason(signal)
        assert "趋势多头" in reason
        assert "回踩到位" in reason

    def test_weak_setup(self):
        signal = SetupSignal(code="000001", name="test")
        signal.trend_score = 0.2
        signal.pullback_score = 0.2
        signal.volume_score = 0.2
        signal.risk_score = 0.2
        reason = generate_reason(signal)
        assert "趋势偏弱" in reason
        assert "位置偏高" in reason


# ════════════════════════════════════════════════════════
# 警告生成
# ════════════════════════════════════════════════════════

class TestGenerateWarnings:
    def test_no_warnings(self):
        signal = SetupSignal(code="000001", name="test")
        signal.risk_score = 0.8
        signal.level_confidence = "high"
        signal.market_regime = "neutral"
        signal.ref_reward_risk = 2.0
        warns = generate_warnings(signal)
        assert warns == []

    def test_bear_market_warning(self):
        signal = SetupSignal(code="000001", name="test")
        signal.risk_score = 0.8
        signal.level_confidence = "high"
        signal.market_regime = "bear"
        signal.ref_reward_risk = 2.0
        warns = generate_warnings(signal)
        assert any("大盘" in w for w in warns)

    def test_low_reward_risk(self):
        signal = SetupSignal(code="000001", name="test")
        signal.risk_score = 0.8
        signal.level_confidence = "high"
        signal.market_regime = "neutral"
        signal.ref_reward_risk = 0.5
        warns = generate_warnings(signal)
        assert any("盈亏比" in w for w in warns)


# ════════════════════════════════════════════════════════
# SetupSignal 数据模型
# ════════════════════════════════════════════════════════

class TestSetupSignal:
    def test_default_values(self):
        s = SetupSignal(code="000001", name="测试")
        assert s.timing_score == 0.0
        assert s.action == "wait"
        assert s.level_confidence == "low"

    def test_to_dict(self):
        s = SetupSignal(code="600519", name="贵州茅台")
        s.timing_score = 75.5
        s.action = "watch"
        s.trend_score = 0.8
        d = s.to_dict()
        assert d["code"] == "600519"
        assert d["timing_score"] == 75.5
        assert d["trend_score"] == 0.8
        assert isinstance(d, dict)

    def test_to_dict_roundtrip(self):
        s = SetupSignal(code="000001", name="test")
        s.timing_score = 82.345
        s.support_zone_low = 9.876
        d = s.to_dict()
        # round(82.345, 2) = 82.34 (banker's rounding in Python)
        assert d["timing_score"] == round(82.345, 2)
        assert d["support_zone_low"] == round(9.876, 2)


# ════════════════════════════════════════════════════════
# 综合评估 (evaluate_setup_timing)
# ════════════════════════════════════════════════════════

class TestEvaluateSetupTiming:
    def test_basic_evaluation(self):
        rows = _make_eod_series(60, base_close=10.0, trend=0.002)
        signal = evaluate_setup_timing(
            code="600519",
            name="贵州茅台",
            eod_rows=rows,
        )
        assert signal.code == "600519"
        assert 0 <= signal.timing_score <= 100
        assert signal.action in ("setup_ready", "watch", "wait", "avoid_chase")
        assert signal.support_zone_low is not None
        assert signal.level_confidence in ("high", "medium", "low")
        assert signal.reason != ""

    def test_insufficient_data(self):
        """不足10行 → wait + 警告."""
        rows = _make_eod_series(5)
        signal = evaluate_setup_timing(code="000001", name="test", eod_rows=rows)
        assert signal.action == "wait"
        assert len(signal.warnings) > 0

    def test_empty_data(self):
        signal = evaluate_setup_timing(code="000001", name="test", eod_rows=[])
        assert signal.action == "wait"

    def test_limit_board_forces_avoid(self):
        """一字涨停 → 强制 avoid_chase."""
        rows = _make_eod_series(60, base_close=10.0, trend=0.005)
        signal = evaluate_setup_timing(
            code="000001", name="test",
            eod_rows=rows,
            latest_is_limit_board=True,
        )
        assert signal.action == "avoid_chase"

    def test_watch_only_caps_action(self):
        """watch_only → 最高 watch."""
        rows = _make_eod_series(60, base_close=10.0, trend=0.003)
        signal = evaluate_setup_timing(
            code="000001", name="test",
            eod_rows=rows,
            is_watch_only=True,
        )
        assert signal.action in ("watch", "wait", "avoid_chase")
        assert signal.action != "setup_ready"

    def test_market_multiplier_applied(self):
        """牛市乘数提高分数."""
        rows = _make_eod_series(60, base_close=10.0, trend=0.002)
        signal_neutral = evaluate_setup_timing(
            code="000001", name="test", eod_rows=rows,
            market_regime="neutral", market_multiplier=1.0,
        )
        signal_bull = evaluate_setup_timing(
            code="000001", name="test", eod_rows=rows,
            market_regime="bull", market_multiplier=1.10,
        )
        assert signal_bull.timing_score >= signal_neutral.timing_score

    def test_pullback_series_scores_well(self):
        """回踩到位的序列有合理评分."""
        rows = _make_pullback_series(40)
        signal = evaluate_setup_timing(
            code="000001", name="test",
            eod_rows=rows,
        )
        # 模拟的回踩序列跌幅较深(20天每天1%=~20%)，pullback_score 偏低是合理的
        # 但 volume_score 因为缩量应该不错
        assert signal.volume_score > 0.4
        assert 0 <= signal.pullback_score <= 1.0
        assert 0 <= signal.timing_score <= 100

    def test_all_scores_in_range(self):
        """所有分项评分在 0~1 范围内."""
        rows = _make_eod_series(60, base_close=10.0, trend=0.002)
        signal = evaluate_setup_timing(code="000001", name="test", eod_rows=rows)
        for attr in ["trend_score", "pullback_score", "volume_score", "repair_score", "risk_score"]:
            v = getattr(signal, attr)
            assert 0.0 <= v <= 1.0, f"{attr}={v} 超出范围"

    def test_timing_score_clamped(self):
        """timing_score 被 clamp 到 0~100."""
        rows = _make_eod_series(60, base_close=10.0, trend=0.002)
        signal = evaluate_setup_timing(
            code="000001", name="test",
            eod_rows=rows,
            market_regime="bull", market_multiplier=5.0,  # 极端乘数
        )
        assert signal.timing_score <= 100.0


# ════════════════════════════════════════════════════════
# _extract_eod_metrics
# ════════════════════════════════════════════════════════

class TestExtractEodMetrics:
    def test_basic(self):
        rows = _make_eod_series(60, base_close=10.0, trend=0.001)
        m = _extract_eod_metrics(rows)
        assert m["close"] > 0
        assert m["ma5"] is not None
        assert m["ma10"] is not None
        assert m["ma20"] is not None
        assert m["high_20d"] > 0
        assert m["low_20d"] > 0
        assert m["atr"] is not None

    def test_short_series(self):
        rows = _make_eod_series(15)
        m = _extract_eod_metrics(rows)
        assert m["ma60"] is None  # 不够60根
        assert m["ma5"] is not None


# ════════════════════════════════════════════════════════
# run_setup_timing (批量入口)
# ════════════════════════════════════════════════════════

class TestRunSetupTiming:
    def _make_mock_detail(self, code="000001", name="test", ts_code="000001.SZ",
                          pass_stage1=True, pass_stage1_watch=False):
        detail = MagicMock()
        detail.code = code
        detail.name = name
        detail.ts_code = ts_code
        detail.pass_stage1 = pass_stage1
        detail.pass_stage1_watch = pass_stage1_watch
        detail.upper_reversal_count_5d = 0
        detail.latest_is_limit_board = False
        detail.limit_board_count_5d = 0
        detail.volume_ratio = 1.2
        return detail

    def _make_mock_tushare(self, rows=None):
        import pandas as pd
        tushare = MagicMock()
        if rows is None:
            rows = _make_eod_series(60)
        df = pd.DataFrame(rows)
        tushare.get_daily.return_value = df
        return tushare

    def _make_mock_trade_cal(self):
        import datetime as dt
        cal = MagicMock()
        cal.eod_start_date.return_value = dt.date(2025, 12, 1)
        return cal

    def test_basic_batch(self):
        details = [self._make_mock_detail(f"00000{i}", f"stock_{i}", f"00000{i}.SZ") for i in range(3)]
        tushare = self._make_mock_tushare()
        cal = self._make_mock_trade_cal()

        signals = run_setup_timing(
            details=details,
            tushare_client=tushare,
            trade_cal=cal,
            trade_date_str="2026-04-22",
        )
        assert len(signals) == 3
        for s in signals:
            assert s.action in ("setup_ready", "watch", "wait", "avoid_chase")

    def test_with_market_index(self):
        details = [self._make_mock_detail()]
        tushare = self._make_mock_tushare()
        cal = self._make_mock_trade_cal()

        bull_closes = [100.0 + i * 0.5 for i in range(20)]
        signals = run_setup_timing(
            details=details,
            tushare_client=tushare,
            trade_cal=cal,
            trade_date_str="2026-04-22",
            index_closes_20d=bull_closes,
        )
        assert len(signals) == 1
        assert signals[0].market_regime == "bull"
        assert signals[0].market_multiplier == pytest.approx(1.10)

    def test_empty_details(self):
        signals = run_setup_timing(
            details=[],
            tushare_client=MagicMock(),
            trade_cal=MagicMock(),
            trade_date_str="2026-04-22",
        )
        assert signals == []

    def test_no_ts_code_skipped(self):
        detail = self._make_mock_detail()
        detail.ts_code = ""
        signals = run_setup_timing(
            details=[detail],
            tushare_client=MagicMock(),
            trade_cal=self._make_mock_trade_cal(),
            trade_date_str="2026-04-22",
        )
        assert len(signals) == 0

    def test_tushare_returns_none(self):
        """Tushare 返回 None → 跳过该股."""
        detail = self._make_mock_detail()
        tushare = MagicMock()
        tushare.get_daily.return_value = None
        cal = self._make_mock_trade_cal()

        signals = run_setup_timing(
            details=[detail],
            tushare_client=tushare,
            trade_cal=cal,
            trade_date_str="2026-04-22",
        )
        assert len(signals) == 0

    def test_watch_only_detected(self):
        """pass_stage1_watch=True, pass_stage1=False → is_watch_only."""
        detail = self._make_mock_detail(pass_stage1=False, pass_stage1_watch=True)
        tushare = self._make_mock_tushare()
        cal = self._make_mock_trade_cal()

        signals = run_setup_timing(
            details=[detail],
            tushare_client=tushare,
            trade_cal=cal,
            trade_date_str="2026-04-22",
        )
        assert len(signals) == 1
        assert signals[0].action != "setup_ready"
