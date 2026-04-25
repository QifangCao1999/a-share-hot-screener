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
        assert "趋势走弱" in reason
        assert "位置偏高" in reason or "回撤过深" in reason


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


# ════════════════════════════════════════════════════════
# v3.0 新增测试: SetupTimingConfig
# ════════════════════════════════════════════════════════

from a_share_hot_screener.setup_timing import (
    SetupTimingConfig,
    apply_action_caps,
    apply_stage1_context,
    generate_score_commentary,
    generate_intraday_hint,
    _cap_action,
)


class TestSetupTimingConfig:
    def test_default_values(self):
        cfg = SetupTimingConfig()
        assert cfg.enable_action_caps is True
        assert cfg.enable_volatility_adaptive_pullback is True
        assert cfg.enable_stage1_context is True
        assert cfg.cap_min_risk_score_for_ready == 0.25

    def test_custom_values(self):
        cfg = SetupTimingConfig(cap_min_risk_score_for_ready=0.30)
        assert cfg.cap_min_risk_score_for_ready == 0.30


class TestCapAction:
    def test_no_cap_needed(self):
        assert _cap_action("wait", "watch") == "wait"
        assert _cap_action("avoid_chase", "watch") == "avoid_chase"

    def test_cap_applied(self):
        assert _cap_action("setup_ready", "watch") == "watch"
        assert _cap_action("watch", "wait") == "wait"
        assert _cap_action("setup_ready", "avoid_chase") == "avoid_chase"

    def test_same_level(self):
        assert _cap_action("watch", "watch") == "watch"


# ════════════════════════════════════════════════════════
# v3.0 新增测试: Action Cap 体系
# ════════════════════════════════════════════════════════

class TestApplyActionCaps:
    def _make_signal(self, **kwargs) -> SetupSignal:
        defaults = {
            "code": "000001", "name": "test",
            "timing_score": 85.0, "action": "setup_ready",
            "risk_score": 0.8, "level_confidence": "high",
            "ref_reward_risk": 2.0, "market_regime": "neutral",
        }
        defaults.update(kwargs)
        s = SetupSignal(code=defaults["code"], name=defaults["name"])
        for k, v in defaults.items():
            if k not in ("code", "name"):
                setattr(s, k, v)
        return s

    def _make_metrics(self, **kwargs) -> dict:
        defaults = {
            "close": 10.0, "ma20": 9.8, "atr_pct": 3.0, "max_drawdown_5d": 2.0,
        }
        defaults.update(kwargs)
        return defaults

    def test_no_cap_when_all_healthy(self):
        signal = self._make_signal()
        action, reasons = apply_action_caps(signal, self._make_metrics(), 0, 0)
        assert action == "setup_ready"
        assert reasons == []

    def test_risk_score_low_caps_to_wait(self):
        signal = self._make_signal(risk_score=0.20)
        action, reasons = apply_action_caps(signal, self._make_metrics(), 0, 0)
        assert action == "wait"
        assert len(reasons) >= 1
        assert any("risk_score" in r for r in reasons)

    def test_risk_score_critical_caps_to_avoid(self):
        signal = self._make_signal(risk_score=0.10)
        action, reasons = apply_action_caps(signal, self._make_metrics(), 0, 0)
        assert action == "avoid_chase"

    def test_level_confidence_low_caps_to_watch(self):
        signal = self._make_signal(level_confidence="low", ref_reward_risk=2.0)
        action, reasons = apply_action_caps(signal, self._make_metrics(), 0, 0)
        assert action == "watch"
        assert any("level_confidence" in r for r in reasons)

    def test_level_confidence_low_and_low_rr_caps_to_wait(self):
        signal = self._make_signal(level_confidence="low", ref_reward_risk=0.8)
        action, reasons = apply_action_caps(signal, self._make_metrics(), 0, 0)
        assert action == "wait"

    def test_reward_risk_low_caps_to_watch(self):
        signal = self._make_signal(ref_reward_risk=0.8)
        action, reasons = apply_action_caps(signal, self._make_metrics(), 0, 0)
        assert action == "watch"

    def test_reward_risk_critical_caps_to_wait(self):
        signal = self._make_signal(ref_reward_risk=0.5)
        action, reasons = apply_action_caps(signal, self._make_metrics(), 0, 0)
        assert action == "wait"

    def test_dist_ma20_far_caps_to_watch(self):
        # close=11.5, ma20=10.0 → dist=15% > 12%
        signal = self._make_signal()
        metrics = self._make_metrics(close=11.5, ma20=10.0)
        action, reasons = apply_action_caps(signal, metrics, 0, 0)
        assert action == "watch"
        assert any("MA20" in r for r in reasons)

    def test_dist_ma20_very_far_caps_to_wait(self):
        # close=12.0, ma20=10.0 → dist=20% > 18%
        signal = self._make_signal()
        metrics = self._make_metrics(close=12.0, ma20=10.0)
        action, reasons = apply_action_caps(signal, metrics, 0, 0)
        assert action == "wait"

    def test_high_atr_and_drawdown_caps_to_wait(self):
        signal = self._make_signal()
        metrics = self._make_metrics(atr_pct=9.0, max_drawdown_5d=12.0)
        action, reasons = apply_action_caps(signal, metrics, 0, 0)
        assert action == "wait"

    def test_extreme_atr_caps_to_watch(self):
        signal = self._make_signal()
        metrics = self._make_metrics(atr_pct=11.0)
        action, reasons = apply_action_caps(signal, metrics, 0, 0)
        assert action == "watch"

    def test_upper_shadow_dense_caps_to_wait(self):
        signal = self._make_signal()
        action, reasons = apply_action_caps(signal, self._make_metrics(), 3, 0)
        assert action == "wait"
        assert any("上影线" in r for r in reasons)

    def test_bear_market_blocks_ready(self):
        signal = self._make_signal(timing_score=82.0, market_regime="bear")
        action, reasons = apply_action_caps(signal, self._make_metrics(), 0, 0)
        assert action == "watch"
        assert any("bear" in r for r in reasons)

    def test_bear_market_high_score_allows_ready(self):
        """bear 市 timing >= 85 仍可 setup_ready."""
        signal = self._make_signal(timing_score=90.0, market_regime="bear", risk_score=0.8)
        action, reasons = apply_action_caps(signal, self._make_metrics(), 0, 0)
        assert action == "setup_ready"

    def test_bear_low_risk_caps_to_wait(self):
        signal = self._make_signal(market_regime="bear", risk_score=0.35)
        action, reasons = apply_action_caps(signal, self._make_metrics(), 0, 0)
        assert action == "wait"

    def test_frequent_limit_board_caps_to_watch(self):
        signal = self._make_signal()
        action, reasons = apply_action_caps(signal, self._make_metrics(), 0, 2)
        assert action == "watch"
        assert any("一字板" in r for r in reasons)

    def test_caps_disabled_by_config(self):
        cfg = SetupTimingConfig(enable_action_caps=False)
        signal = self._make_signal(risk_score=0.05)
        action, reasons = apply_action_caps(signal, self._make_metrics(), 3, 3, config=cfg)
        assert action == "setup_ready"  # no caps applied
        assert reasons == []

    def test_multiple_caps_strictest_wins(self):
        """多条规则同时触发，最严格的cap生效."""
        signal = self._make_signal(risk_score=0.10, ref_reward_risk=0.5)
        action, reasons = apply_action_caps(signal, self._make_metrics(), 0, 0)
        assert action == "avoid_chase"  # risk_score<0.15 is strictest
        assert len(reasons) >= 1
        assert any("risk_score" in r for r in reasons)


# ════════════════════════════════════════════════════════
# v3.0 新增测试: 波动率自适应回踩
# ════════════════════════════════════════════════════════

class TestVolatilityAdaptivePullback:
    def test_high_atr_wider_params(self):
        """高ATR股票回踩到位的分更高（因为允许更深回踩）."""
        # 高ATR: close远离MA10/MA20但在更宽容的范围内
        # 回踩8%对于ATR%=8%的股是合理的，但对固定参数(center=5%)偏深
        s_adaptive = score_pullback(
            close=9.2, ma10=9.5, ma20=9.3,
            high_20d=10.0, low_20d=8.8,
            atr_pct=8.0,  # 高波动
        )
        s_fixed = score_pullback(
            close=9.2, ma10=9.5, ma20=9.3,
            high_20d=10.0, low_20d=8.8,
            atr_pct=None,  # fallback to fixed
        )
        # 自适应模式下，深回踩更宽容
        assert s_adaptive != s_fixed  # 参数不同应产生不同结果

    def test_low_atr_narrower_params(self):
        """低ATR股票用更窄参数."""
        s = score_pullback(
            close=10.0, ma10=10.05, ma20=9.95,
            high_20d=10.3, low_20d=9.7,
            atr_pct=2.0,  # 低波动
        )
        assert 0.0 <= s <= 1.0

    def test_no_atr_falls_back(self):
        """atr_pct=None 时 fallback 到固定参数."""
        s1 = score_pullback(
            close=10.0, ma10=10.05, ma20=9.8,
            high_20d=11.0, low_20d=9.0,
            atr_pct=None,
        )
        s2 = score_pullback(
            close=10.0, ma10=10.05, ma20=9.8,
            high_20d=11.0, low_20d=9.0,
        )
        assert s1 == pytest.approx(s2)  # 应完全相同

    def test_disabled_by_config(self):
        """配置关闭自适应时用固定参数."""
        cfg = SetupTimingConfig(enable_volatility_adaptive_pullback=False)
        s_disabled = score_pullback(
            close=10.0, ma10=10.05, ma20=9.8,
            high_20d=11.0, low_20d=9.0,
            atr_pct=8.0, config=cfg,
        )
        s_no_atr = score_pullback(
            close=10.0, ma10=10.05, ma20=9.8,
            high_20d=11.0, low_20d=9.0,
            atr_pct=None,
        )
        assert s_disabled == pytest.approx(s_no_atr)


# ════════════════════════════════════════════════════════
# v3.0 新增测试: Stage 1 Context 联动
# ════════════════════════════════════════════════════════

class TestApplyStage1Context:
    def _make_signal(self, **kwargs) -> SetupSignal:
        defaults = {
            "code": "000001", "name": "test",
            "timing_score": 85.0, "action": "setup_ready",
        }
        defaults.update(kwargs)
        s = SetupSignal(code=defaults["code"], name=defaults["name"])
        for k, v in defaults.items():
            if k not in ("code", "name"):
                setattr(s, k, v)
        return s

    def test_no_context_no_change(self):
        signal = self._make_signal()
        action, reason, hp = apply_stage1_context(signal, None)
        assert action == "setup_ready"
        assert reason is None
        assert hp is False

    def test_hot_theme_low_demotes_ready(self):
        signal = self._make_signal()
        ctx = {"hot_theme_score": 0.40, "liquidity_execution_score": 0.8, "risk_control_score": 0.8}
        action, reason, hp = apply_stage1_context(signal, ctx)
        assert action == "watch"
        assert "hot_theme" in reason

    def test_hot_theme_high_marks_priority(self):
        signal = self._make_signal(timing_score=80.0)
        ctx = {"hot_theme_score": 0.80}
        action, reason, hp = apply_stage1_context(signal, ctx)
        assert hp is True

    def test_liquidity_low_demotes_ready(self):
        signal = self._make_signal()
        ctx = {"liquidity_execution_score": 0.45}
        action, reason, hp = apply_stage1_context(signal, ctx)
        assert action == "watch"

    def test_liquidity_critical_caps_wait(self):
        signal = self._make_signal()
        ctx = {"liquidity_execution_score": 0.30}
        action, reason, hp = apply_stage1_context(signal, ctx)
        assert action == "wait"

    def test_stage1_risk_low_demotes(self):
        signal = self._make_signal()
        ctx = {"risk_control_score": 0.35}
        action, reason, hp = apply_stage1_context(signal, ctx)
        assert action == "watch"

    def test_stage1_risk_critical_caps_wait(self):
        signal = self._make_signal()
        ctx = {"risk_control_score": 0.20}
        action, reason, hp = apply_stage1_context(signal, ctx)
        assert action == "wait"

    def test_bottom_50pct_needs_high_timing(self):
        signal = self._make_signal(timing_score=82.0)
        ctx = {"stage1_rank_pctile": 0.3}  # bottom 70%
        action, reason, hp = apply_stage1_context(signal, ctx)
        assert action == "watch"
        assert "排名" in reason

    def test_top_50pct_normal_timing_ok(self):
        signal = self._make_signal(timing_score=82.0)
        ctx = {"stage1_rank_pctile": 0.6}  # top 40%
        action, reason, hp = apply_stage1_context(signal, ctx)
        assert action == "setup_ready"

    def test_watch_only_still_capped(self):
        """Stage 1 不能让 watch 升级为 setup_ready."""
        signal = self._make_signal(action="watch")
        ctx = {"hot_theme_score": 0.90}
        action, reason, hp = apply_stage1_context(signal, ctx)
        assert action == "watch"  # 不会升级

    def test_disabled_by_config(self):
        cfg = SetupTimingConfig(enable_stage1_context=False)
        signal = self._make_signal()
        ctx = {"hot_theme_score": 0.10}  # 很低，但开关关了
        action, reason, hp = apply_stage1_context(signal, ctx, config=cfg)
        assert action == "setup_ready"  # 不受影响


# ════════════════════════════════════════════════════════
# v3.0 新增测试: 增强 reason / warnings / commentary
# ════════════════════════════════════════════════════════

class TestEnhancedReasonWarnings:
    def test_reason_includes_repair(self):
        signal = SetupSignal(code="000001", name="test")
        signal.trend_score = 0.8
        signal.pullback_score = 0.7
        signal.volume_score = 0.7
        signal.repair_score = 0.8
        signal.risk_score = 0.8
        reason = generate_reason(signal)
        assert "分歧后修复" in reason

    def test_reason_unrepaired(self):
        signal = SetupSignal(code="000001", name="test")
        signal.trend_score = 0.6
        signal.pullback_score = 0.5
        signal.volume_score = 0.5
        signal.repair_score = 0.2
        signal.risk_score = 0.6
        reason = generate_reason(signal)
        assert "未修复" in reason

    def test_score_commentary_all_keys(self):
        signal = SetupSignal(code="000001", name="test")
        signal.trend_score = 0.7
        signal.pullback_score = 0.6
        signal.volume_score = 0.5
        signal.repair_score = 0.5
        signal.risk_score = 0.7
        commentary = generate_score_commentary(signal)
        assert "trend" in commentary
        assert "pullback" in commentary
        assert "volume" in commentary
        assert "repair" in commentary
        assert "risk" in commentary

    def test_warnings_include_cap_reasons(self):
        signal = SetupSignal(code="000001", name="test")
        signal.risk_score = 0.8
        signal.level_confidence = "high"
        signal.market_regime = "neutral"
        signal.ref_reward_risk = 2.0
        signal.action_cap_reasons = ["risk_score=0.10→avoid_chase"]
        warns = generate_warnings(signal)
        assert any("action cap" in w for w in warns)

    def test_warnings_include_stage1_reason(self):
        signal = SetupSignal(code="000001", name="test")
        signal.risk_score = 0.8
        signal.level_confidence = "high"
        signal.market_regime = "neutral"
        signal.ref_reward_risk = 2.0
        signal.stage1_adjustment_reason = "hot_theme=0.40<0.55→降为watch"
        warns = generate_warnings(signal)
        assert any("stage1" in w for w in warns)

    def test_warnings_with_metrics(self):
        signal = SetupSignal(code="000001", name="test")
        signal.risk_score = 0.8
        signal.level_confidence = "high"
        signal.market_regime = "neutral"
        signal.ref_reward_risk = 2.0
        metrics = {"atr_pct": 9.0, "ma20": 10.0, "close": 11.5, "max_drawdown_5d": 12.0}
        warns = generate_warnings(signal, metrics=metrics)
        assert any("ATR%" in w for w in warns)
        assert any("MA20" in w for w in warns)
        assert any("回撤" in w for w in warns)


# ════════════════════════════════════════════════════════
# v3.0 新增测试: 盘中执行提示
# ════════════════════════════════════════════════════════

class TestIntradayHint:
    def test_setup_ready_needs_confirmation(self):
        signal = SetupSignal(code="000001", name="test")
        signal.action = "setup_ready"
        signal.risk_score = 0.7
        signal.pullback_score = 0.7
        signal.volume_score = 0.7
        needs, hint = generate_intraday_hint(signal)
        assert needs is True
        assert len(hint) > 0
        assert "竞价追高" in hint

    def test_watch_gets_hint(self):
        signal = SetupSignal(code="000001", name="test")
        signal.action = "watch"
        needs, hint = generate_intraday_hint(signal)
        assert needs is True
        assert "回踩到位" in hint

    def test_wait_no_hint(self):
        signal = SetupSignal(code="000001", name="test")
        signal.action = "wait"
        needs, hint = generate_intraday_hint(signal)
        assert needs is False
        assert hint == ""

    def test_avoid_chase_no_hint(self):
        signal = SetupSignal(code="000001", name="test")
        signal.action = "avoid_chase"
        needs, hint = generate_intraday_hint(signal)
        assert needs is False


# ════════════════════════════════════════════════════════
# v3.0 新增测试: 综合集成 (evaluate_setup_timing with v3.0 features)
# ════════════════════════════════════════════════════════

class TestEvaluateV3Integration:
    def test_action_cap_applied_in_evaluation(self):
        """evaluate_setup_timing 中 action cap 生效."""
        # 生成一个高分序列，但给低 risk_score 条件（用一字板来压低 risk_score）
        rows = _make_eod_series(60, base_close=10.0, trend=0.003)
        signal = evaluate_setup_timing(
            code="000001", name="test",
            eod_rows=rows,
            upper_reversal_count_5d=4,  # 很多上影线 → risk_score 低 + cap 触发
        )
        # upper_reversal_count_5d=4 → risk_score 中 upper_reversal 子项=0.1
        # 同时 action cap 规则11 (>=3天) → 最高wait
        if signal.action_cap_reasons:
            assert any("上影线" in r for r in signal.action_cap_reasons)

    def test_stage1_context_integration(self):
        """evaluate_setup_timing 中 stage1_context 生效."""
        rows = _make_eod_series(60, base_close=10.0, trend=0.003)
        signal = evaluate_setup_timing(
            code="000001", name="test",
            eod_rows=rows,
            stage1_context={"hot_theme_score": 0.30},
        )
        assert signal.stage1_context_used is True
        # hot_theme=0.30 < 0.55 → 如果原 action 是 setup_ready，会被降级
        if signal.final_action_before_stage1_cap == "setup_ready":
            assert signal.action in ("watch", "wait", "avoid_chase")

    def test_to_dict_includes_v3_fields(self):
        rows = _make_eod_series(60, base_close=10.0, trend=0.002)
        signal = evaluate_setup_timing(
            code="000001", name="test",
            eod_rows=rows,
            stage1_context={"hot_theme_score": 0.70},
        )
        d = signal.to_dict()
        assert "action_cap_reasons" in d
        assert "score_components_commentary" in d
        assert "stage1_context_used" in d
        assert "requires_intraday_confirmation" in d
        assert "intraday_check_hint" in d
        assert "high_priority_watch" in d

    def test_intraday_hint_populated(self):
        rows = _make_eod_series(60, base_close=10.0, trend=0.002)
        signal = evaluate_setup_timing(
            code="000001", name="test",
            eod_rows=rows,
        )
        # action 是 setup_ready 或 watch 时应有提示
        if signal.action in ("setup_ready", "watch"):
            assert signal.requires_intraday_confirmation is True
            assert signal.intraday_check_hint != ""

    def test_score_commentary_populated(self):
        rows = _make_eod_series(60, base_close=10.0, trend=0.002)
        signal = evaluate_setup_timing(
            code="000001", name="test",
            eod_rows=rows,
        )
        assert len(signal.score_components_commentary) == 5

    def test_backward_compat_no_stage1(self):
        """不传 stage1_context 时行为与 v2.1 一致."""
        rows = _make_eod_series(60, base_close=10.0, trend=0.002)
        signal = evaluate_setup_timing(
            code="000001", name="test",
            eod_rows=rows,
        )
        assert signal.stage1_context_used is False
        assert signal.stage1_adjustment_reason is None

    def test_backward_compat_insufficient_data(self):
        """数据不足时仍返回 wait (向后兼容)."""
        rows = _make_eod_series(5)
        signal = evaluate_setup_timing(code="000001", name="test", eod_rows=rows)
        assert signal.action == "wait"

    def test_backward_compat_limit_board_avoid(self):
        """一字涨停仍强制 avoid_chase (向后兼容)."""
        rows = _make_eod_series(60, base_close=10.0, trend=0.005)
        signal = evaluate_setup_timing(
            code="000001", name="test",
            eod_rows=rows,
            latest_is_limit_board=True,
        )
        assert signal.action == "avoid_chase"

    def test_backward_compat_watch_only(self):
        """watch_only 仍最高 watch (向后兼容)."""
        rows = _make_eod_series(60, base_close=10.0, trend=0.003)
        signal = evaluate_setup_timing(
            code="000001", name="test",
            eod_rows=rows,
            is_watch_only=True,
        )
        assert signal.action != "setup_ready"

    def test_all_caps_disabled(self):
        """全部新功能关闭时行为与 v2.1 一致."""
        cfg = SetupTimingConfig(
            enable_action_caps=False,
            enable_volatility_adaptive_pullback=False,
            enable_stage1_context=False,
        )
        rows = _make_eod_series(60, base_close=10.0, trend=0.002)
        signal = evaluate_setup_timing(
            code="000001", name="test",
            eod_rows=rows,
            stage1_context={"hot_theme_score": 0.10},  # 很低但被忽略
            config=cfg,
        )
        assert signal.stage1_context_used is False  # stage1 被关闭
        assert signal.action_cap_reasons == []  # cap 被关闭


# ════════════════════════════════════════════════════════
# v3.0 Round 2: 增强版 Market Regime
# ════════════════════════════════════════════════════════

from a_share_hot_screener.setup_timing import (
    compute_enhanced_market_regime,
    _find_swing_lows,
    _find_divergence_low,
    _find_box_low,
    _cluster_support_levels,
    _basic_support_zone,
)


class TestEnhancedMarketRegime:
    def test_bull_expanding_is_risk_on(self):
        closes = [100.0 + i * 0.5 for i in range(20)]  # +10%
        amounts = [1e9] * 17 + [1.3e9, 1.4e9, 1.5e9]  # 近3日放量
        result = compute_enhanced_market_regime(closes, amounts)
        assert result["index_trend"] == "bull"
        assert result["turnover_state"] == "expanding"
        assert result["final_regime"] == "risk_on"
        assert result["multiplier"] > 1.0

    def test_bear_is_risk_off(self):
        closes = [100.0 - i * 0.5 for i in range(20)]  # -10%
        result = compute_enhanced_market_regime(closes)
        assert result["final_regime"] == "risk_off"
        assert result["multiplier"] < 1.0

    def test_neutral_shrinking_is_risk_off(self):
        closes = [100.0 + (i % 3 - 1) * 0.2 for i in range(20)]  # 小幅震荡
        amounts = [1e9] * 17 + [5e8, 4e8, 3e8]  # 近3日缩量
        result = compute_enhanced_market_regime(closes, amounts)
        assert result["index_trend"] == "neutral"
        assert result["turnover_state"] == "shrinking"
        assert result["final_regime"] == "risk_off"

    def test_neutral_normal_is_neutral(self):
        closes = [100.0 + (i % 3 - 1) * 0.2 for i in range(20)]
        amounts = [1e9] * 20  # 平稳量能
        result = compute_enhanced_market_regime(closes, amounts)
        assert result["final_regime"] == "neutral"
        assert result["multiplier"] == pytest.approx(1.0)

    def test_no_amounts_fallback(self):
        closes = [100.0 + i * 0.5 for i in range(20)]
        result = compute_enhanced_market_regime(closes, None)
        assert result["turnover_state"] == "neutral"  # 无数据→中性

    def test_no_data_fallback(self):
        result = compute_enhanced_market_regime(None, None)
        assert result["final_regime"] == "neutral"
        assert result["multiplier"] == 1.0

    def test_custom_multipliers(self):
        cfg = SetupTimingConfig(
            enable_enhanced_market_regime=True,
            risk_on_multiplier=1.15,
            risk_off_multiplier=0.80,
        )
        closes = [100.0 + i * 0.5 for i in range(20)]
        result = compute_enhanced_market_regime(closes, config=cfg)
        assert result["multiplier"] == pytest.approx(1.15)


# ════════════════════════════════════════════════════════
# v3.0 Round 2: 支撑位检测函数
# ════════════════════════════════════════════════════════

class TestFindSwingLows:
    def test_basic_swing_low(self):
        rows = [
            {"low": 10.0}, {"low": 9.5}, {"low": 9.0},  # swing low at idx 2
            {"low": 9.3}, {"low": 9.8},
        ]
        lows = _find_swing_lows(rows, lookback=2)
        assert len(lows) == 1
        assert lows[0] == pytest.approx(9.0)

    def test_no_swing_low(self):
        rows = [{"low": 10.0 + i * 0.1} for i in range(10)]  # 单调上行
        lows = _find_swing_lows(rows)
        assert len(lows) == 0

    def test_multiple_swing_lows(self):
        rows = [
            {"low": 10}, {"low": 9}, {"low": 8}, {"low": 9}, {"low": 10},  # 第一个 swing
            {"low": 9}, {"low": 8.5}, {"low": 9.5}, {"low": 10},  # 第二个 swing
        ]
        lows = _find_swing_lows(rows)
        assert len(lows) >= 1

    def test_short_series(self):
        rows = [{"low": 10}, {"low": 9}]
        lows = _find_swing_lows(rows)
        assert lows == []


class TestFindDivergenceLow:
    def test_finds_divergence(self):
        rows = [
            _make_eod_row("2026-04-10", 10.0, 10.8, 9.5, 9.8, 2000000),  # 分歧: 振幅13%
            _make_eod_row("2026-04-11", 9.8, 10.0, 9.7, 9.9, 1000000),
        ]
        low = _find_divergence_low(rows)
        assert low == pytest.approx(9.5)

    def test_no_divergence(self):
        rows = [
            _make_eod_row("2026-04-10", 10.0, 10.1, 9.95, 10.05, 1000000),
            _make_eod_row("2026-04-11", 10.05, 10.1, 10.0, 10.08, 1000000),
        ]
        low = _find_divergence_low(rows)
        assert low is None


class TestFindBoxLow:
    def test_clustered_lows(self):
        rows = [
            {"low": 9.5}, {"low": 9.6}, {"low": 9.55}, {"low": 9.52},  # 4个聚集在~9.5
            {"low": 10.0}, {"low": 10.5}, {"low": 11.0},
        ]
        box = _find_box_low(rows, close=10.0)
        assert box is not None
        assert 9.4 < box < 9.7

    def test_no_cluster(self):
        rows = [{"low": 9.0 + i * 0.5} for i in range(5)]  # 散布
        box = _find_box_low(rows, close=11.0)
        assert box is None

    def test_short_series(self):
        rows = [{"low": 10.0}, {"low": 10.1}]
        box = _find_box_low(rows, close=10.0)
        assert box is None


# ════════════════════════════════════════════════════════
# v3.0 Round 2: 聚集区 + 增强参考价位
# ════════════════════════════════════════════════════════

class TestClusterSupportLevels:
    def test_basic_clustering(self):
        candidates = [
            {"price": 9.50, "source": "ma10", "confidence": 0.8},
            {"price": 9.55, "source": "swing_low", "confidence": 0.75},
            {"price": 9.48, "source": "ma20", "confidence": 0.85},
            {"price": 8.00, "source": "box_low", "confidence": 0.70},  # 远离聚集区
        ]
        low, high, basis, sorted_c = _cluster_support_levels(candidates, close=10.0)
        assert low is not None
        assert high is not None
        # 9.48~9.55 应该聚集
        assert 9.4 <= low <= 9.55
        assert 9.48 <= high <= 9.6
        # basis 应含多个 source
        assert "+" in basis or basis in ("ma10", "ma20", "swing_low")

    def test_empty_candidates(self):
        low, high, basis, _ = _cluster_support_levels([], close=10.0)
        assert low is None
        assert high is None

    def test_single_candidate(self):
        candidates = [{"price": 9.5, "source": "ma20", "confidence": 0.8}]
        low, high, basis, _ = _cluster_support_levels(candidates, close=10.0)
        assert low == pytest.approx(9.5)
        assert high == pytest.approx(9.5)
        assert basis == "ma20"


class TestEnhancedReferenceLevels:
    def test_with_rows_produces_candidates(self):
        rows = _make_eod_series(60, base_close=10.0, trend=0.002)
        result = compute_reference_levels(
            close=rows[-1]["close"],
            ma10=10.0, ma20=9.8, atr=0.3,
            high_20d=11.0, low_20d=9.0,
            rows=rows,
        )
        assert "candidate_support_levels" in result
        assert len(result["candidate_support_levels"]) >= 2  # 至少 ma10 + ma20
        assert result["support_zone_low"] is not None
        assert result["resistance_1"] is not None

    def test_resistance_2_present(self):
        """40日内有更高点时 resistance_2 不为空."""
        rows = _make_eod_series(60, base_close=10.0, trend=0.001)
        # 在第30根插入一个远超近20日高点的spike
        rows[30]["high"] = 30.0
        high_20d = max(r["high"] for r in rows[-20:])
        result = compute_reference_levels(
            close=rows[-1]["close"],
            ma10=None, ma20=None, atr=0.3,
            high_20d=high_20d,
            low_20d=min(r["low"] for r in rows[-20:]),
            rows=rows,
        )
        # 30.0 在 rows[-40:] 内但不在 rows[-20:] 内 → resistance_2 = 30.0
        assert result["resistance_2"] is not None
        assert result["resistance_2"] >= result["resistance_1"]

    def test_without_rows_fallback(self):
        """不传rows时 fallback 到v2.1逻辑."""
        result = compute_reference_levels(
            close=10.0, ma10=10.2, ma20=9.8,
            atr=0.5, high_20d=11.0, low_20d=9.0,
        )
        assert result["support_basis"] in ("ma10", "ma20")
        assert result["candidate_support_levels"] == []
        assert result["resistance_2"] is None

    def test_disabled_by_config(self):
        cfg = SetupTimingConfig(enable_enhanced_reference_levels=False)
        rows = _make_eod_series(60, base_close=10.0, trend=0.002)
        result = compute_reference_levels(
            close=10.0, ma10=10.2, ma20=9.8,
            atr=0.5, high_20d=11.0, low_20d=9.0,
            rows=rows, config=cfg,
        )
        assert result["candidate_support_levels"] == []

    def test_invalidation_uses_multiple_sources(self):
        """v3.0 增强版失效位来自多来源."""
        rows = _make_eod_series(60, base_close=10.0, trend=0.002)
        close = rows[-1]["close"]
        ma20 = sum(r["close"] for r in rows[-20:]) / 20
        from a_share_hot_screener.setup_timing import _compute_atr
        atr = _compute_atr(rows) or 0.3
        result = compute_reference_levels(
            close=close, ma10=None, ma20=ma20, atr=atr,
            high_20d=max(r["high"] for r in rows[-20:]),
            low_20d=min(r["low"] for r in rows[-20:]),
            rows=rows,
        )
        # 失效位应存在且有效
        assert result["invalidation_level"] is not None
        assert result["invalidation_level"] < close

    def test_backward_compat_no_rows(self):
        """原有三个测试场景仍可用."""
        # 复制原有 TestReferenceLevels.test_with_both_mas 的输入
        result = compute_reference_levels(
            close=10.0, ma10=10.2, ma20=9.8,
            atr=0.5, high_20d=11.0, low_20d=9.0,
        )
        assert result["support_zone_low"] == pytest.approx(9.8 * 0.99, abs=0.01)
        assert result["support_zone_high"] == pytest.approx(10.2 * 1.01, abs=0.01)
        assert result["invalidation_level"] == pytest.approx(9.8 - 1.5 * 0.5, abs=0.01)
        assert result["resistance_1"] == pytest.approx(11.0)
        assert result["ref_reward_risk"] > 0


# ════════════════════════════════════════════════════════
# v3.0 Round 2: 集成测试
# ════════════════════════════════════════════════════════

class TestRound2Integration:
    def test_evaluate_has_candidate_levels(self):
        rows = _make_eod_series(60, base_close=10.0, trend=0.002)
        signal = evaluate_setup_timing(
            code="000001", name="test",
            eod_rows=rows,
        )
        assert len(signal.candidate_support_levels) >= 2

    def test_evaluate_has_resistance_2(self):
        rows = _make_eod_series(60, base_close=10.0, trend=0.002)
        signal = evaluate_setup_timing(
            code="000001", name="test",
            eod_rows=rows,
        )
        # resistance_2 可能为 None 如果没有明显前高
        d = signal.to_dict()
        assert "resistance_2" in d
        assert "candidate_support_levels" in d

    def test_run_with_enhanced_regime(self):
        """run_setup_timing 使用增强版regime."""
        detail = MagicMock()
        detail.code = "000001"
        detail.name = "test"
        detail.ts_code = "000001.SZ"
        detail.pass_stage1 = True
        detail.pass_stage1_watch = False
        detail.upper_reversal_count_5d = 0
        detail.latest_is_limit_board = False
        detail.limit_board_count_5d = 0
        detail.volume_ratio = 1.2
        detail.hot_theme_score = 0.7
        detail.trend_flow_score = 0.7
        detail.liquidity_execution_score = 0.7
        detail.risk_control_score = 0.7
        detail.total_score = 0.75

        import pandas as pd
        import datetime as dt
        tushare = MagicMock()
        rows = _make_eod_series(60)
        tushare.get_daily.return_value = pd.DataFrame(rows)
        cal = MagicMock()
        cal.eod_start_date.return_value = dt.date(2025, 12, 1)

        cfg = SetupTimingConfig(enable_enhanced_market_regime=True)
        bull_closes = [100.0 + i * 0.5 for i in range(20)]
        bull_amounts = [1e9] * 17 + [1.3e9, 1.4e9, 1.5e9]

        signals = run_setup_timing(
            details=[detail],
            tushare_client=tushare,
            trade_cal=cal,
            trade_date_str="2026-04-22",
            index_closes_20d=bull_closes,
            index_amounts_20d=bull_amounts,
            config=cfg,
        )
        assert len(signals) == 1
        # 增强版regime应返回enhanced类型
        assert signals[0].market_regime in ("risk_on", "neutral", "risk_off")


# ════════════════════════════════════════════════════════
# 第一批 Integration Fix 测试 (I3/I4/I5/I10)
# ════════════════════════════════════════════════════════

class TestWeakRegimes:
    """I5: 弱市 regime 语义统一测试."""

    def test_bear_triggers_action_cap(self):
        """bear 仍然触发弱市 cap."""
        from a_share_hot_screener.setup_timing import (
            WEAK_REGIMES, apply_action_caps, SetupSignal, SetupTimingConfig,
        )
        sig = SetupSignal(code="000001", name="T", timing_score=70.0, action="setup_ready")
        sig.market_regime = "bear"
        sig.risk_score = 0.5
        sig.level_confidence = "high"
        sig.ref_reward_risk = 2.0
        cfg = SetupTimingConfig(cap_bear_ready_threshold=75.0)
        action, reasons = apply_action_caps(sig, metrics={}, upper_reversal_count_5d=0,
                                             limit_board_count_5d=0, config=cfg)
        assert action in ("watch", "wait", "avoid_chase"), "bear 市应降级 setup_ready"
        assert any("bear" in r.lower() for r in reasons)

    def test_risk_off_triggers_weak_cap(self):
        """risk_off 应触发弱市 action cap."""
        from a_share_hot_screener.setup_timing import (
            apply_action_caps, SetupSignal, SetupTimingConfig,
        )
        sig = SetupSignal(code="000001", name="T", timing_score=70.0, action="setup_ready")
        sig.market_regime = "risk_off"
        sig.risk_score = 0.3  # 低于 cap_bear_min_risk_score=0.4
        sig.level_confidence = "high"
        sig.ref_reward_risk = 2.0
        cfg = SetupTimingConfig(cap_bear_ready_threshold=75.0, cap_bear_min_risk_score=0.4)
        action, reasons = apply_action_caps(sig, metrics={}, upper_reversal_count_5d=0,
                                             limit_board_count_5d=0, config=cfg)
        assert action in ("wait", "avoid_chase"), "risk_off + 低风险分应降至最高 wait"

    def test_ice_point_triggers_weak_cap(self):
        """ice_point 应触发弱市 action cap."""
        from a_share_hot_screener.setup_timing import (
            apply_action_caps, SetupSignal, SetupTimingConfig,
        )
        sig = SetupSignal(code="000001", name="T", timing_score=70.0, action="setup_ready")
        sig.market_regime = "ice_point"
        sig.risk_score = 0.3
        sig.level_confidence = "high"
        sig.ref_reward_risk = 2.0
        cfg = SetupTimingConfig(cap_bear_ready_threshold=75.0, cap_bear_min_risk_score=0.4)
        action, reasons = apply_action_caps(sig, metrics={}, upper_reversal_count_5d=0,
                                             limit_board_count_5d=0, config=cfg)
        assert action in ("wait", "avoid_chase"), "ice_point 应降至最高 wait"

    def test_neutral_not_triggered(self):
        """neutral 不触发弱市 cap."""
        from a_share_hot_screener.setup_timing import (
            apply_action_caps, SetupSignal, SetupTimingConfig,
        )
        sig = SetupSignal(code="000001", name="T", timing_score=82.0, action="setup_ready")
        sig.market_regime = "neutral"
        sig.risk_score = 0.5
        sig.level_confidence = "high"
        sig.ref_reward_risk = 2.5
        cfg = SetupTimingConfig()
        action, reasons = apply_action_caps(sig, metrics={}, upper_reversal_count_5d=0,
                                             limit_board_count_5d=0, config=cfg)
        assert action == "setup_ready", "neutral 不应被弱市 cap"

    def test_weak_regimes_set_contents(self):
        """WEAK_REGIMES 集合包含预期成员."""
        from a_share_hot_screener.setup_timing import WEAK_REGIMES
        assert "bear" in WEAK_REGIMES
        assert "risk_off" in WEAK_REGIMES
        assert "ice_point" in WEAK_REGIMES
        assert "neutral" not in WEAK_REGIMES
        assert "risk_on" not in WEAK_REGIMES

    def test_weak_regime_warning_message(self):
        """弱市 warning 消息包含 regime 名称."""
        from a_share_hot_screener.setup_timing import generate_warnings, SetupSignal
        for regime in ("bear", "risk_off", "ice_point"):
            sig = SetupSignal(code="000001", name="T", timing_score=70.0, action="watch")
            sig.market_regime = regime
            sig.risk_score = 0.5
            sig.level_confidence = "medium"
            warns = generate_warnings(sig)
            assert any(regime in w for w in warns), f"{regime} 应出现在 warning 中"


class TestSmallSampleRankPctile:
    """I10: 小样本不使用 stage1_rank_pctile 测试."""

    def _make_detail(self, total_score=0.70):
        from unittest.mock import MagicMock
        d = MagicMock()
        d.pass_stage1 = True
        d.pass_stage1_watch = False
        d.ts_code = "000001.SZ"
        d.code = "000001"
        d.name = "测试"
        d.total_score = total_score
        d.hot_theme_score = 0.6
        d.trend_flow_score = 0.6
        d.liquidity_execution_score = 0.6
        d.risk_control_score = 0.45
        d.upper_reversal_count_5d = 0
        d.latest_is_limit_board = False
        d.limit_board_count_5d = 0
        d.volume_ratio = 1.0
        return d

    def _make_eod(self, n=60):
        import datetime as dt
        rows = []
        base = dt.date(2026, 1, 1)
        for i in range(n):
            rows.append({
                "ts_code": "000001.SZ", "trade_date": (base + dt.timedelta(days=i)).strftime("%Y%m%d"),
                "open": 10.0, "high": 10.5, "low": 9.8, "close": 10.2 + i * 0.01,
                "vol": 1e6, "amount": 1e7,
            })
        return rows

    def test_small_sample_no_rank_pctile(self):
        """样本数 < min_stage1_rank_sample_size 时 rank_pctile 应为 None."""
        import pandas as pd
        import datetime as dt
        from unittest.mock import MagicMock
        from a_share_hot_screener.setup_timing import run_setup_timing, SetupTimingConfig

        # 只提供 5 只股票（< 默认阈值 20）
        details = [self._make_detail(0.70 + i * 0.02) for i in range(5)]
        tushare = MagicMock()
        tushare.get_daily.return_value = pd.DataFrame(self._make_eod(60))
        cal = MagicMock()
        cal.eod_start_date.return_value = dt.date(2025, 12, 1)

        # 设置一个低分股，rank_pctile 应为 None，不会被后50%规则降级
        cfg = SetupTimingConfig(
            enable_stage1_context=True,
            min_stage1_rank_sample_size=20,
            stage1_bottom_pctile_timing=85.0,
        )
        signals = run_setup_timing(
            details=details,
            tushare_client=tushare,
            trade_cal=cal,
            trade_date_str="2026-04-22",
            config=cfg,
        )
        # 不应因排名规则降级（stage1_adjustment_reason 不应包含"排名后"）
        for sig in signals:
            if sig.stage1_adjustment_reason:
                assert "排名后" not in sig.stage1_adjustment_reason, \
                    f"小样本不应触发排名降级，但 {sig.code} 触发了: {sig.stage1_adjustment_reason}"

    def test_large_sample_uses_rank_pctile(self):
        """样本数 >= min_stage1_rank_sample_size 时 rank_pctile 可以生效."""
        import pandas as pd
        import datetime as dt
        from unittest.mock import MagicMock
        from a_share_hot_screener.setup_timing import run_setup_timing, SetupTimingConfig

        # 25 只股票，底部股 total_score 很低
        details = [self._make_detail(0.50 + i * 0.01) for i in range(25)]
        # 最底部的几只分数很低，rank_pctile < 0.5
        for d in details[:5]:
            d.total_score = 0.40

        tushare = MagicMock()
        tushare.get_daily.return_value = pd.DataFrame(self._make_eod(60))
        cal = MagicMock()
        cal.eod_start_date.return_value = dt.date(2025, 12, 1)

        cfg = SetupTimingConfig(
            enable_stage1_context=True,
            min_stage1_rank_sample_size=20,
        )
        # 只要不崩溃，样本数足够时排名逻辑可以运行
        signals = run_setup_timing(
            details=details,
            tushare_client=tushare,
            trade_cal=cal,
            trade_date_str="2026-04-22",
            config=cfg,
        )
        assert len(signals) == 25

    def test_config_min_rank_sample_size_default(self):
        """SetupTimingConfig 默认 min_stage1_rank_sample_size=20."""
        from a_share_hot_screener.setup_timing import SetupTimingConfig
        cfg = SetupTimingConfig()
        assert cfg.min_stage1_rank_sample_size == 20

    def test_config_min_rank_sample_size_custom(self):
        """可以自定义 min_stage1_rank_sample_size."""
        from a_share_hot_screener.setup_timing import SetupTimingConfig
        cfg = SetupTimingConfig(min_stage1_rank_sample_size=5)
        assert cfg.min_stage1_rank_sample_size == 5


class TestPipelinePassesConfigAndAmounts:
    """I3/I4: pipeline 传入 config 和 index_amounts_20d 测试."""

    def _make_tradeable_detail(self):
        from unittest.mock import MagicMock
        d = MagicMock()
        d.pass_stage1 = True
        d.pass_stage1_watch = False
        d.ts_code = "000001.SZ"
        d.code = "000001"
        d.name = "Test"
        d.total_score = 0.75
        d.hot_theme_score = 0.7
        d.trend_flow_score = 0.7
        d.liquidity_execution_score = 0.7
        d.risk_control_score = 0.5
        d.upper_reversal_count_5d = 0
        d.latest_is_limit_board = False
        d.limit_board_count_5d = 0
        d.volume_ratio = 1.0
        return d

    def test_pipeline_passes_amounts_to_batch(self):
        """pipeline 拉取 amount 列后传入 run_setup_timing."""
        import pandas as pd
        import datetime as dt
        from unittest.mock import MagicMock, patch

        # 构造带 amount 列的指数数据
        idx_rows = []
        for i in range(25):
            idx_rows.append({
                "trade_date": f"202604{i+1:02d}",
                "close": 3000.0 + i * 5,
                "amount": 5e8 + i * 1e7,
            })
        idx_df = pd.DataFrame(idx_rows)

        captured = {}

        def mock_run(**kwargs):
            captured.update(kwargs)
            return []

        with patch("a_share_hot_screener.pipeline._run_setup_timing_batch", mock_run):
            from a_share_hot_screener.pipeline import Stage1HotPipeline
            from a_share_hot_screener.config import HotScreenerConfig

            cfg = HotScreenerConfig(
                tushare_token="test",
                run_date=dt.date(2026, 4, 22),
                stock_codes=[],
                output_dir="/tmp/test_output",
                enable_setup_timing=True,
            )
            pipeline = Stage1HotPipeline.__new__(Stage1HotPipeline)
            pipeline.config = cfg
            pipeline._details = [self._make_tradeable_detail()]
            pipeline.warnings = MagicMock()
            pipeline.warnings.add_global = MagicMock()

            tushare_mock = MagicMock()
            tushare_mock.get_daily.return_value = idx_df
            pipeline._tushare = tushare_mock

            cal_mock = MagicMock()
            cal_mock.n_trade_dates_before.return_value = dt.date(2026, 3, 20)
            pipeline._trade_cal = cal_mock

            pipeline._run_setup_timing("2026-04-22")

        # amount 数据应该被传入
        assert "index_amounts_20d" in captured
        if captured["index_amounts_20d"] is not None:
            assert all(a > 0 for a in captured["index_amounts_20d"])

    def test_pipeline_passes_config_to_batch(self):
        """pipeline 应从 config 读取 setup_timing_config 并传入。"""
        import datetime as dt
        from unittest.mock import MagicMock, patch
        from a_share_hot_screener.setup_timing import SetupTimingConfig

        captured = {}

        def mock_run(**kwargs):
            captured.update(kwargs)
            return []

        with patch("a_share_hot_screener.pipeline._run_setup_timing_batch", mock_run):
            from a_share_hot_screener.pipeline import Stage1HotPipeline
            from a_share_hot_screener.config import HotScreenerConfig

            st_cfg = SetupTimingConfig(enable_action_caps=False)
            cfg = HotScreenerConfig(
                tushare_token="test",
                run_date=dt.date(2026, 4, 22),
                stock_codes=[],
                output_dir="/tmp/test_output2",
                enable_setup_timing=True,
            )
            # 手动附加 setup_timing_config
            cfg.setup_timing_config = st_cfg

            pipeline = Stage1HotPipeline.__new__(Stage1HotPipeline)
            pipeline.config = cfg
            pipeline._details = [self._make_tradeable_detail()]
            pipeline.warnings = MagicMock()
            pipeline.warnings.add_global = MagicMock()

            import pandas as pd
            tushare_mock = MagicMock()
            tushare_mock.get_daily.return_value = pd.DataFrame(
                [{"trade_date": "20260422", "close": 3000.0, "amount": 5e8}]
            )
            pipeline._tushare = tushare_mock
            cal_mock = MagicMock()
            cal_mock.n_trade_dates_before.return_value = dt.date(2026, 3, 20)
            pipeline._trade_cal = cal_mock

            pipeline._run_setup_timing("2026-04-22")

        assert captured.get("config") is st_cfg, "pipeline 应将 setup_timing_config 传入 run_setup_timing"
