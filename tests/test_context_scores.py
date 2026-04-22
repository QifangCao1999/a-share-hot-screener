"""Phase 3 测试 — HT8/HT9/HT10 Context Scores.

覆盖:
  - HT8 市场确认度 (5 个级别 + 降级)
  - HT9 板块扩散度 (边界 + cap + 不适用 + 辅助输出)
  - HT10 板块内辨识度 (完整 + 降级 + 类型分类)
  - 综合入口 compute_context_scores
  - Config 3-flag 控制
  - hot_theme_score 集成 (use_context_scores)
  - ContextScoresResult.to_dict 序列化
"""

from __future__ import annotations

import pytest
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set
from unittest.mock import MagicMock

from a_share_hot_screener.context_scores import (
    ContextScoresResult,
    compute_context_scores,
    compute_ht8,
    compute_ht9,
    compute_ht10,
    _classify_position_type,
    _calc_amount_breadth_ratio,
    _calc_top5_amount_concentration,
)
from a_share_hot_screener.models import HotStockDetail
from a_share_hot_screener.scoring import AxisScore, ScoreItem, ScoringPool
from a_share_hot_screener.scorers.hot_theme import compute_hot_theme_score


# ════════════════════════════════════════════════════════
# Fixtures
# ════════════════════════════════════════════════════════

def _make_detail(**overrides) -> HotStockDetail:
    """构造 HotStockDetail 测试实例."""
    defaults = dict(
        code="002436",
        name="兴森科技",
        ts_code="002436.SZ",
        exchange="SZ",
        passed_hard_filter=True,
        return_5d=15.0,
        return_10d=25.0,
        amount_avg_5d=500_000_000.0,
        amount_ratio_5d_to_20d=2.5,
        limit_up_count_5d=2,
        limit_up_count_10d=3,
        lhb_count_20d=1,
        concept_heat_pctile_5d=0.92,
        concept_names=["PCB", "AI算力"],
        advanced_concept_module_available=True,
        industry_heat_pctile_5d=0.75,
        big_up_count_10d=3,
        consec_up_days=2,
        flags={},
    )
    defaults.update(overrides)
    return HotStockDetail(**defaults)


def _make_event_ctx(
    zt_available: bool = True,
    concept_mode: str = "ths_daily_full",
    concept_cons_map: Optional[Dict[str, List[str]]] = None,
) -> MagicMock:
    """构造 EventLayerContext mock."""
    ctx = MagicMock()
    ctx.zt_pool_available = zt_available
    ctx.concept_heat_mode = concept_mode
    ctx.concept_cons_map = concept_cons_map or {}
    ctx.industry_cons_map = {}
    return ctx


def _make_sector_members(
    n: int = 30,
    strong_pct: float = 0.30,
    target_ts_code: str = "002436.SZ",
    target_return: float = 15.0,
    target_amount: float = 2_500_000_000.0,
    target_limit_up: int = 2,
) -> Dict[str, Dict[str, float]]:
    """构造板块成分股日线数据."""
    members = {}
    strong_count = int(n * strong_pct)

    for i in range(n):
        ts_code = f"00{1000+i}.SZ"
        if i < strong_count:
            ret = 8.0 + i * 0.5  # >5%
        else:
            ret = 1.0 + i * 0.1  # <5%

        members[ts_code] = {
            "return_5d": ret,
            "amount_ratio": 1.5 + (i % 3) * 0.3,
            "amount_5d": 100_000_000.0 * (n - i),  # 递减
            "limit_up_count_5d": 1 if i < 3 else 0,
        }

    # 插入目标股票
    members[target_ts_code] = {
        "return_5d": target_return,
        "amount_ratio": 2.5,
        "amount_5d": target_amount,
        "limit_up_count_5d": target_limit_up,
    }

    return members


def _make_pool() -> ScoringPool:
    """构造 ScoringPool."""
    pool = ScoringPool()
    pool.stock_count = 100
    pool.pool_return_5d = sorted([float(i) for i in range(-10, 40)])
    pool.pool_return_10d = sorted([float(i) for i in range(-15, 60)])
    pool.pool_amount_avg_5d = sorted([float(i * 1e7) for i in range(1, 51)])
    pool.pool_volume_ratio_20d = sorted([0.5 + i * 0.1 for i in range(50)])
    pool.pool_close_position_20d = sorted([i / 50.0 for i in range(50)])
    pool.pool_clv_latest = sorted([i / 50.0 for i in range(50)])
    pool.pool_amount_ratio_5d_to_20d = sorted([0.5 + i * 0.1 for i in range(50)])
    return pool


# ════════════════════════════════════════════════════════
# HT8 Tests
# ════════════════════════════════════════════════════════

class TestHT8:
    """HT8 市场确认度测试."""

    def test_multi_confirmed(self):
        """龙虎榜 + 涨停 + 概念Top10 → 1.0."""
        detail = _make_detail(
            lhb_count_20d=2,
            limit_up_count_5d=3,
            concept_heat_pctile_5d=0.95,
        )
        ctx = _make_event_ctx(concept_mode="ths_daily_full")
        score, level, signals, degraded = compute_ht8(detail, ctx)
        assert score == 1.0
        assert level == "multi_confirmed"
        assert "lhb" in signals
        assert "limit_up" in signals
        assert "concept_top10" in signals
        assert not degraded

    def test_sector_resonance(self):
        """涨停 + 概念Top20 → 0.80."""
        detail = _make_detail(
            lhb_count_20d=0,
            limit_up_count_5d=1,
            concept_heat_pctile_5d=0.85,
        )
        ctx = _make_event_ctx(concept_mode="ths_daily_full")
        score, level, signals, degraded = compute_ht8(detail, ctx)
        assert score == 0.80
        assert level == "sector_resonance"

    def test_single_anomaly_lhb_only(self):
        """龙虎榜无涨停 → 0.60."""
        detail = _make_detail(
            lhb_count_20d=1,
            limit_up_count_5d=0,
            concept_heat_pctile_5d=0.50,
        )
        ctx = _make_event_ctx(concept_mode="ths_daily_full")
        score, level, signals, degraded = compute_ht8(detail, ctx)
        assert score == 0.60
        assert level == "single_anomaly"

    def test_single_anomaly_limit_up_no_concept(self):
        """涨停但板块不活跃 → 0.60."""
        detail = _make_detail(
            lhb_count_20d=0,
            limit_up_count_5d=2,
            concept_heat_pctile_5d=0.50,
        )
        ctx = _make_event_ctx(concept_mode="ths_daily_full")
        score, level, signals, degraded = compute_ht8(detail, ctx)
        assert score == 0.60
        assert level == "single_anomaly"

    def test_volume_anomaly(self):
        """成交额放大+涨幅>5%但无涨停/龙虎 → 0.40."""
        detail = _make_detail(
            lhb_count_20d=0,
            limit_up_count_5d=0,
            concept_heat_pctile_5d=0.50,
            amount_ratio_5d_to_20d=2.5,
            return_5d=8.0,
        )
        ctx = _make_event_ctx(concept_mode="ths_daily_full")
        score, level, signals, degraded = compute_ht8(detail, ctx)
        assert score == 0.40
        assert level == "volume_anomaly"

    def test_no_confirmation(self):
        """无信号 → 0.10."""
        detail = _make_detail(
            lhb_count_20d=0,
            limit_up_count_5d=0,
            concept_heat_pctile_5d=0.30,
            amount_ratio_5d_to_20d=1.0,
            return_5d=2.0,
        )
        ctx = _make_event_ctx(concept_mode="ths_daily_full")
        score, level, signals, degraded = compute_ht8(detail, ctx)
        assert score == 0.10
        assert level == "no_confirmation"
        assert len(signals) == 0

    def test_degraded_caps_at_060(self):
        """Level 3 不可用时 cap 0.60."""
        detail = _make_detail(
            lhb_count_20d=2,
            limit_up_count_5d=3,
            concept_heat_pctile_5d=0.95,
        )
        ctx = _make_event_ctx(concept_mode="none")  # Level 3 不可用
        score, level, signals, degraded = compute_ht8(detail, ctx)
        assert score == 0.60
        assert degraded
        # 降级后级别应为 single_anomaly（不能是 multi_confirmed）
        assert level == "single_anomaly"

    def test_degraded_below_060_unchanged(self):
        """Level 3 不可用但原本分<0.60，不变."""
        detail = _make_detail(
            lhb_count_20d=0,
            limit_up_count_5d=0,
            concept_heat_pctile_5d=None,
            amount_ratio_5d_to_20d=2.5,
            return_5d=8.0,
        )
        ctx = _make_event_ctx(concept_mode="none")
        score, level, signals, degraded = compute_ht8(detail, ctx)
        assert score == 0.40
        assert degraded


# ════════════════════════════════════════════════════════
# HT9 Tests
# ════════════════════════════════════════════════════════

class TestHT9:
    """HT9 板块扩散度测试."""

    def test_high_breadth(self):
        """50%+走强 → 1.0."""
        members = _make_sector_members(n=30, strong_pct=0.60)
        ctx = _make_event_ctx()
        detail = _make_detail()
        score, ratio, applicable, cap, _, size, _, _ = compute_ht9(detail, ctx, members)
        assert applicable
        assert score == 1.0
        assert ratio >= 50.0

    def test_mid_breadth(self):
        """~30% 走强 → ~0.70."""
        members = _make_sector_members(n=30, strong_pct=0.30)
        ctx = _make_event_ctx()
        detail = _make_detail()
        score, ratio, applicable, cap, _, size, _, _ = compute_ht9(detail, ctx, members)
        assert applicable
        # With 30% strong + target (also strong), should be close to 0.70
        assert 0.50 <= score <= 0.90

    def test_low_breadth(self):
        """<=10% 走强 → 0.0."""
        members = _make_sector_members(n=30, strong_pct=0.05)
        ctx = _make_event_ctx()
        detail = _make_detail()
        score, ratio, applicable, cap, _, size, _, _ = compute_ht9(detail, ctx, members)
        assert applicable
        # Very few strong, ratio close to 10% or below
        assert score <= 0.35  # At most slightly above 0

    def test_small_sector_cap(self):
        """10-29 成分 → cap 0.80."""
        members = _make_sector_members(n=15, strong_pct=0.80)
        ctx = _make_event_ctx()
        detail = _make_detail()
        score, ratio, applicable, cap, _, size, _, _ = compute_ht9(detail, ctx, members)
        assert applicable
        assert cap  # cap_applied=True
        assert score <= 0.80

    def test_tiny_sector_not_applicable(self):
        """<10 成分 → 不适用."""
        members = _make_sector_members(n=5, strong_pct=0.80)
        ctx = _make_event_ctx()
        detail = _make_detail()
        score, ratio, applicable, cap, _, size, _, _ = compute_ht9(detail, ctx, members)
        assert not applicable
        assert score is None

    def test_no_data_not_applicable(self):
        """无板块数据 → 不适用."""
        ctx = _make_event_ctx()
        detail = _make_detail()
        score, ratio, applicable, _, _, _, _, _ = compute_ht9(detail, ctx, None)
        assert not applicable
        assert score is None

    def test_amount_breadth_ratio(self):
        """辅助指标: 成交额放大>50%占比."""
        members = {
            f"00{i}.SZ": {"return_5d": 8.0, "amount_ratio": 1.8 if i < 5 else 1.2, "amount_5d": 1e8}
            for i in range(20)
        }
        ratio = _calc_amount_breadth_ratio(members)
        assert ratio is not None
        assert ratio == 25.0  # 5/20 = 25%

    def test_top5_amount_concentration(self):
        """辅助指标: Top5成交额占比."""
        members = {}
        for i in range(20):
            members[f"00{i}.SZ"] = {
                "return_5d": 5.0,
                "amount_ratio": 1.5,
                "amount_5d": float(100_000 * (20 - i)),
            }
        conc = _calc_top5_amount_concentration(members)
        assert conc is not None
        # Top5 amounts: 2e6, 1.9e6, 1.8e6, 1.7e6, 1.6e6 = 9e6
        # Total = sum of 100000*(20-i) for i in 0..19 = 100000 * sum(20,19,...,1) = 100000 * 210 = 2.1e7
        # conc = 9e6 / 2.1e7 * 100 ≈ 42.86
        assert 40.0 <= conc <= 50.0


# ════════════════════════════════════════════════════════
# HT10 Tests
# ════════════════════════════════════════════════════════

class TestHT10:
    """HT10 板块内辨识度测试."""

    def test_full_formula(self):
        """完整公式: rank + amount_share + first_zt."""
        members = _make_sector_members(
            n=30, strong_pct=0.30,
            target_ts_code="002436.SZ",
            target_return=20.0,          # 板块内高涨幅
            target_amount=5_000_000_000.0,  # 大成交额
            target_limit_up=3,
        )
        ctx = _make_event_ctx(zt_available=True)
        detail = _make_detail()
        (score, rank_pctile, amount_share, first_zt,
         applicable, pos_type, confidence, degraded) = compute_ht10(detail, ctx, members)

        assert applicable
        assert score is not None
        assert confidence == "high"
        assert not degraded
        assert 0.0 <= score <= 1.0
        assert rank_pctile is not None
        assert amount_share is not None
        assert first_zt is not None

    def test_degraded_no_zt_pool(self):
        """Level 2 缺失: 重新归一 + cap 0.80."""
        members = _make_sector_members(
            n=30, strong_pct=0.30,
            target_ts_code="002436.SZ",
            target_return=20.0,
            target_amount=5_000_000_000.0,
            target_limit_up=3,
        )
        ctx = _make_event_ctx(zt_available=False)
        detail = _make_detail()
        (score, rank_pctile, amount_share, first_zt,
         applicable, pos_type, confidence, degraded) = compute_ht10(detail, ctx, members)

        assert applicable
        assert degraded
        assert confidence == "medium"
        assert score <= 0.80
        assert first_zt is None  # 无法计算

    def test_small_sector_not_applicable(self):
        """<10 成分 → 不适用."""
        members = _make_sector_members(n=5, strong_pct=0.50)
        ctx = _make_event_ctx()
        detail = _make_detail()
        (score, _, _, _, applicable, _, _, _) = compute_ht10(detail, ctx, members)
        assert not applicable
        assert score is None

    def test_no_data_not_applicable(self):
        """无板块数据 → 不适用."""
        ctx = _make_event_ctx()
        detail = _make_detail()
        (score, _, _, _, applicable, _, _, _) = compute_ht10(detail, ctx, None)
        assert not applicable

    def test_frontline_like_classification(self):
        """高涨幅排名 + 高 first_zt → frontline_like."""
        pos = _classify_position_type(0.85, 0.50, 0.90)
        assert pos == "frontline_like"

    def test_capacity_core_classification(self):
        """高成交额占比 + 中等涨幅 → capacity_core_like."""
        pos = _classify_position_type(0.50, 0.70, 0.30)
        assert pos == "capacity_core_like"

    def test_follower_classification(self):
        """低涨幅 + 低成交额 → follower_like."""
        pos = _classify_position_type(0.20, 0.10, 0.0)
        assert pos == "follower_like"

    def test_unknown_classification(self):
        """中间地带 → unknown."""
        pos = _classify_position_type(0.50, 0.40, 0.50)
        assert pos == "unknown"


# ════════════════════════════════════════════════════════
# 综合入口测试
# ════════════════════════════════════════════════════════

class TestComputeContextScores:
    """compute_context_scores 综合测试."""

    def test_all_scores_computed(self):
        """三个分数全部输出."""
        detail = _make_detail()
        ctx = _make_event_ctx(concept_mode="ths_daily_full")
        members = _make_sector_members(n=30, strong_pct=0.30)
        result = compute_context_scores(detail, ctx, members, "PCB")

        assert result.ht8_score is not None
        assert result.ht9_score is not None
        assert result.ht10_score is not None
        assert result.ht9_sector_name == "PCB"

    def test_no_sector_data(self):
        """无板块数据: HT8 可算, HT9/HT10 不适用."""
        detail = _make_detail()
        ctx = _make_event_ctx(concept_mode="ths_daily_full")
        result = compute_context_scores(detail, ctx, None)

        assert result.ht8_score is not None
        assert result.ht9_score is None
        assert not result.ht9_is_applicable
        assert result.ht10_score is None
        assert not result.ht10_is_applicable

    def test_to_dict_roundtrip(self):
        """to_dict 序列化."""
        detail = _make_detail()
        ctx = _make_event_ctx(concept_mode="ths_daily_full")
        members = _make_sector_members(n=30, strong_pct=0.30)
        result = compute_context_scores(detail, ctx, members, "PCB")
        d = result.to_dict()

        assert isinstance(d, dict)
        assert "ht8_score" in d
        assert "ht9_score" in d
        assert "ht10_score" in d
        assert "ht8_confirmation_level" in d
        assert "ht10_position_type" in d
        assert "ht9_amount_breadth_ratio" in d


# ════════════════════════════════════════════════════════
# Config 3-flag 控制测试
# ════════════════════════════════════════════════════════

class TestConfigFlags:
    """3-flag 控制测试."""

    def test_config_defaults(self):
        """默认配置: compute=True, use=False, show=False."""
        from a_share_hot_screener.config import HotScreenerConfig
        import datetime as dt

        cfg = HotScreenerConfig(
            tushare_token="test",
            run_date=dt.date(2026, 4, 22),
            stock_codes=["600519"],
            output_dir="/tmp/test",
        )
        assert cfg.compute_context_scores is True
        assert cfg.use_context_scores_in_total is False
        assert cfg.show_context_scores_in_discord is False


# ════════════════════════════════════════════════════════
# hot_theme_score 集成测试
# ════════════════════════════════════════════════════════

class TestHotThemeIntegration:
    """hot_theme_score 集成 context scores."""

    def _make_detail_with_context(self) -> HotStockDetail:
        detail = _make_detail(
            flags={"sector_momentum_signal": "neutral"},
        )
        detail.context_scores = {
            "ht8_score": 0.80,
            "ht8_confirmation_level": "sector_resonance",
            "ht8_signals": ["limit_up", "concept_top20"],
            "ht8_degraded": False,
            "ht9_score": 0.70,
            "ht9_breadth_ratio": 30.0,
            "ht9_is_applicable": True,
            "ht9_cap_applied": False,
            "ht9_sector_name": "PCB",
            "ht9_sector_size": 30,
            "ht10_score": 0.65,
            "ht10_rank_pctile": 0.80,
            "ht10_amount_share_score": 0.60,
            "ht10_first_zt_score": 0.50,
            "ht10_is_applicable": True,
            "ht10_position_type": "capacity_core_like",
            "ht10_confidence": "high",
            "ht10_degraded": False,
        }
        return detail

    def test_without_context_7_items(self):
        """use_context_scores=False → 7 个 items (HT1-HT7)."""
        detail = self._make_detail_with_context()
        pool = _make_pool()
        axis = compute_hot_theme_score(detail, pool, use_context_scores=False)

        assert axis.score is not None
        # HT1-HT7 = 7 items
        assert len(axis.items) == 7

    def test_with_context_10_items(self):
        """use_context_scores=True → 10 个 items (HT1-HT10)."""
        detail = self._make_detail_with_context()
        pool = _make_pool()
        axis = compute_hot_theme_score(detail, pool, use_context_scores=True)

        assert axis.score is not None
        # HT1-HT7 + HT8/9/10 = 10 items
        assert len(axis.items) == 10

        # 验证新增的 3 个 item 名称
        names = [item.name for item in axis.items]
        assert "market_confirmation_score" in names
        assert "sector_breadth_score" in names
        assert "sector_position_score" in names

    def test_context_scores_affect_total(self):
        """use_context_scores=True 时 HT8/9/10 参与计算，改变总分."""
        detail = self._make_detail_with_context()
        pool = _make_pool()

        axis_without = compute_hot_theme_score(detail, pool, use_context_scores=False)
        axis_with = compute_hot_theme_score(detail, pool, use_context_scores=True)

        # 分数应该不同（因为加入了新指标）
        assert axis_without.score != axis_with.score

    def test_context_scores_empty_no_crash(self):
        """context_scores 为空 dict → use_context_scores=True 不崩溃."""
        detail = _make_detail(flags={"sector_momentum_signal": "neutral"})
        detail.context_scores = {}
        pool = _make_pool()
        axis = compute_hot_theme_score(detail, pool, use_context_scores=True)

        assert axis.score is not None
        # HT8/9/10 subscore=None → is_data_available=False，但不崩溃
        assert len(axis.items) == 10

    def test_ht9_not_applicable_excluded(self):
        """HT9 is_applicable=False → 不参与 HT 轴总分."""
        detail = self._make_detail_with_context()
        detail.context_scores["ht9_is_applicable"] = False
        detail.context_scores["ht9_score"] = None
        pool = _make_pool()
        axis = compute_hot_theme_score(detail, pool, use_context_scores=True)

        # HT9 item 存在但 is_applicable=False
        ht9_item = [i for i in axis.items if i.name == "sector_breadth_score"][0]
        assert not ht9_item.is_applicable


# ════════════════════════════════════════════════════════
# ContextScoresResult 序列化测试
# ════════════════════════════════════════════════════════

class TestContextScoresResult:
    """ContextScoresResult 数据模型测试."""

    def test_default_values(self):
        """默认值合理."""
        r = ContextScoresResult()
        assert r.ht8_score is None
        assert r.ht9_is_applicable is True
        assert r.ht10_confidence == "high"
        assert r.ht8_degraded is False

    def test_to_dict_keys(self):
        """to_dict 包含所有必要 key."""
        r = ContextScoresResult(
            ht8_score=0.80,
            ht8_confirmation_level="sector_resonance",
            ht9_score=0.70,
            ht10_score=0.65,
        )
        d = r.to_dict()
        expected_keys = {
            "ht8_score", "ht8_confirmation_level", "ht8_signals", "ht8_degraded",
            "ht9_score", "ht9_breadth_ratio", "ht9_is_applicable", "ht9_cap_applied",
            "ht9_sector_name", "ht9_sector_size",
            "ht9_amount_breadth_ratio", "ht9_top5_amount_concentration",
            "ht10_score", "ht10_rank_pctile", "ht10_amount_share_score",
            "ht10_first_zt_score", "ht10_is_applicable",
            "ht10_position_type", "ht10_confidence", "ht10_degraded",
        }
        assert set(d.keys()) == expected_keys

    def test_to_dict_rounding(self):
        """to_dict 中 float 值保留 4 位小数."""
        r = ContextScoresResult(ht8_score=0.123456789)
        d = r.to_dict()
        assert d["ht8_score"] == 0.1235

    def test_to_dict_none_preserved(self):
        """to_dict 中 None 值保持 None."""
        r = ContextScoresResult()
        d = r.to_dict()
        assert d["ht8_score"] is None
        assert d["ht9_score"] is None


# ════════════════════════════════════════════════════════
# Summary 集成测试
# ════════════════════════════════════════════════════════

class TestSummaryIntegration:
    """HotStockSummary.from_detail 包含 context scores."""

    def test_summary_has_context_fields(self):
        """Summary 包含 HT8/9/10 字段."""
        from a_share_hot_screener.models import HotStockSummary

        detail = _make_detail()
        detail.context_scores = {
            "ht8_score": 0.80,
            "ht8_confirmation_level": "sector_resonance",
            "ht9_score": 0.70,
            "ht9_breadth_ratio": 30.0,
            "ht9_sector_name": "PCB",
            "ht10_score": 0.65,
            "ht10_position_type": "capacity_core_like",
            "ht10_confidence": "high",
        }
        # Need to set scores for from_detail to work
        detail.total_score = 0.75
        detail.data_coverage = 0.90

        summary = HotStockSummary.from_detail(detail)
        assert summary.ht8_score == 0.80
        assert summary.ht8_confirmation_level == "sector_resonance"
        assert summary.ht9_score == 0.70
        assert summary.ht9_breadth_ratio == 30.0
        assert summary.ht9_sector_name == "PCB"
        assert summary.ht10_score == 0.65
        assert summary.ht10_position_type == "capacity_core_like"
        assert summary.ht10_confidence == "high"

    def test_summary_empty_context(self):
        """context_scores 为空 → Summary 字段为 None/空."""
        from a_share_hot_screener.models import HotStockSummary

        detail = _make_detail()
        detail.context_scores = {}
        detail.total_score = 0.75
        detail.data_coverage = 0.90

        summary = HotStockSummary.from_detail(detail)
        assert summary.ht8_score is None
        assert summary.ht8_confirmation_level == ""
        assert summary.ht10_position_type == ""
