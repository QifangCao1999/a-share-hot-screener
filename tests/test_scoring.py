"""Session 5 — 通用评分函数单元测试.

覆盖范围：
  TestScoreItem           — ScoreItem 序列化
  TestAxisScore           — AxisScore 加权聚合逻辑
  TestScoreLowerBound     — 下限型
  TestScoreUpperBound     — 上限型
  TestScoreClampLinear    — 区间线性型
  TestScoreDiscrete       — 离散型
  TestScorePercentile     — 百分位型
  TestScoreBool           — 布尔型
  TestScoringPool         — 横截面 pool 构建
  TestHotThemeScore       — hot_theme_score 完整路径
  TestTrendFlowScore      — trend_flow_score 完整路径
  TestScoreIntegration    — 综合降级场景
"""

from __future__ import annotations

import math
from typing import List, Optional
from unittest.mock import MagicMock

import pytest

from a_share_hot_screener.scoring import (
    AxisScore,
    ScoreItem,
    ScoringPool,
    _append_if_not_none,
    score_bool,
    score_clamp_linear,
    score_discrete,
    score_lower_bound,
    score_percentile,
    score_upper_bound,
)


# ════════════════════════════════════════════════════════
# 测试辅助
# ════════════════════════════════════════════════════════

def _make_detail(**kwargs):
    """构造 HotStockDetail Mock 对象."""
    from a_share_hot_screener.models import HotStockDetail
    defaults = dict(
        code="600519",
        name="贵州茅台",
        passed_hard_filter=True,
        return_5d=None,
        return_10d=None,
        limit_up_count_10d=None,
        limit_up_source="",
        strong_pool_entry_count_3d=None,
        strong_pool_source="",
        industry_heat_pctile_5d=None,
        industry_heat_source="",
        close_position_20d=None,
        abs_distance_to_ma10=None,
        volume_ratio_20d=None,
        clv_latest=None,
        amount_ratio_5d_to_20d=None,
        limit_up_count_5d=None,
    )
    defaults.update(kwargs)
    return HotStockDetail(**defaults)


def _make_pool(**kwargs) -> ScoringPool:
    pool = ScoringPool()
    for k, v in kwargs.items():
        setattr(pool, k, v)
    return pool


# ════════════════════════════════════════════════════════
# TestScoreItem
# ════════════════════════════════════════════════════════

class TestScoreItem:

    def test_to_dict_basic(self):
        item = ScoreItem(name="x", subscore=0.75, weight=2.0)
        d = item.to_dict()
        assert d["name"] == "x"
        assert d["subscore"] == 0.75
        assert d["weight"] == 2.0

    def test_to_dict_none_subscore(self):
        item = ScoreItem(name="x")
        d = item.to_dict()
        assert d["subscore"] is None
        assert d["weighted_score"] is None

    def test_to_dict_nan_safe(self):
        item = ScoreItem(name="x", raw_value=float("nan"))
        d = item.to_dict()
        assert d["raw_value"] is None  # nan → None

    def test_to_dict_inf_safe(self):
        item = ScoreItem(name="x", raw_value=float("inf"))
        d = item.to_dict()
        assert d["raw_value"] is None


# ════════════════════════════════════════════════════════
# TestAxisScore
# ════════════════════════════════════════════════════════

class TestAxisScore:

    def _make_item(self, subscore, weight, applicable=True, available=True) -> ScoreItem:
        item = ScoreItem(name="x", weight=weight,
                         is_applicable=applicable,
                         is_data_available=available)
        item.subscore = subscore
        return item

    def test_basic_weighted_average(self):
        axis = AxisScore(axis_name="test")
        axis.items = [
            self._make_item(0.8, 2.0),
            self._make_item(0.4, 1.0),
        ]
        axis.compute()
        # (0.8*2 + 0.4*1) / (2+1) = 2.0/3 ≈ 0.6667
        assert axis.score == pytest.approx(2.0 / 3, abs=1e-3)
        assert axis.coverage == pytest.approx(1.0)

    def test_all_unavailable(self):
        axis = AxisScore(axis_name="test")
        axis.items = [
            self._make_item(None, 1.0, available=False),
            self._make_item(None, 2.0, available=False),
        ]
        axis.compute()
        assert axis.score is None
        assert axis.coverage == 0.0

    def test_partial_available(self):
        """1个可用，1个不可用 → coverage=0.5, score=仅用可用项计算."""
        axis = AxisScore(axis_name="test")
        axis.items = [
            self._make_item(0.6, 1.0, available=True),
            self._make_item(None, 1.0, available=False),
        ]
        axis.compute()
        assert axis.score == pytest.approx(0.6)
        assert axis.coverage == pytest.approx(0.5)

    def test_not_applicable_excluded_from_denominator(self):
        """is_applicable=False 的项不参与分母，也不影响 coverage."""
        axis = AxisScore(axis_name="test")
        axis.items = [
            self._make_item(0.8, 2.0, applicable=True),
            self._make_item(0.0, 1.0, applicable=False),  # 不适用，不参与
        ]
        axis.compute()
        # 只有 weight=2 参与
        assert axis.score == pytest.approx(0.8)
        assert axis.coverage == pytest.approx(1.0)

    def test_coverage_weighted(self):
        """coverage 是权重加权的，不是简单计数."""
        axis = AxisScore(axis_name="test")
        axis.items = [
            self._make_item(0.9, 3.0, available=True),  # 权重 3
            self._make_item(None, 1.0, available=False),  # 权重 1，缺失
        ]
        axis.compute()
        # coverage = 3 / (3+1) = 0.75
        assert axis.coverage == pytest.approx(0.75)

    def test_empty_items(self):
        axis = AxisScore(axis_name="test")
        axis.compute()
        assert axis.score is None
        assert axis.coverage == 0.0

    def test_to_dict_structure(self):
        axis = AxisScore(axis_name="hot_theme_score")
        axis.items.append(ScoreItem(name="i1", subscore=0.5, weight=1.0))
        axis.compute()
        d = axis.to_dict()
        assert d["axis"] == "hot_theme_score"
        assert "score" in d
        assert "coverage" in d
        assert "items" in d
        assert len(d["items"]) == 1


# ════════════════════════════════════════════════════════
# TestScoreLowerBound
# ════════════════════════════════════════════════════════

class TestScoreLowerBound:

    def test_at_good_threshold(self):
        item = score_lower_bound("x", 15.0, bad_threshold=-5.0, good_threshold=15.0)
        assert item.subscore == pytest.approx(1.0)
        assert item.is_data_available is True

    def test_above_good_threshold(self):
        item = score_lower_bound("x", 20.0, bad_threshold=-5.0, good_threshold=15.0)
        assert item.subscore == pytest.approx(1.0)

    def test_at_bad_threshold(self):
        item = score_lower_bound("x", -5.0, bad_threshold=-5.0, good_threshold=15.0)
        assert item.subscore == pytest.approx(0.0)

    def test_below_bad_threshold(self):
        item = score_lower_bound("x", -10.0, bad_threshold=-5.0, good_threshold=15.0)
        assert item.subscore == pytest.approx(0.0)

    def test_linear_interpolation(self):
        # value=8.0: (8-(-5))/(15-(-5)) = 13/20 = 0.65
        item = score_lower_bound("x", 8.0, bad_threshold=-5.0, good_threshold=15.0)
        assert item.subscore == pytest.approx(0.65)

    def test_none_value(self):
        item = score_lower_bound("x", None, bad_threshold=0.0, good_threshold=10.0)
        assert item.is_data_available is False
        assert item.subscore is None

    def test_nan_value(self):
        item = score_lower_bound("x", float("nan"), bad_threshold=0.0, good_threshold=10.0)
        assert item.is_data_available is False

    def test_not_applicable(self):
        item = score_lower_bound("x", 5.0, bad_threshold=0.0, good_threshold=10.0,
                                 is_applicable=False)
        assert item.is_applicable is False
        assert item.is_data_available is False

    def test_weight_preserved(self):
        item = score_lower_bound("x", 5.0, bad_threshold=0.0, good_threshold=10.0, weight=3.0)
        assert item.weight == 3.0


# ════════════════════════════════════════════════════════
# TestScoreUpperBound
# ════════════════════════════════════════════════════════

class TestScoreUpperBound:

    def test_at_good_threshold(self):
        item = score_upper_bound("x", 20.0, good_threshold=20.0, bad_threshold=100.0)
        assert item.subscore == pytest.approx(1.0)

    def test_below_good_threshold(self):
        item = score_upper_bound("x", 10.0, good_threshold=20.0, bad_threshold=100.0)
        assert item.subscore == pytest.approx(1.0)

    def test_at_bad_threshold(self):
        item = score_upper_bound("x", 100.0, good_threshold=20.0, bad_threshold=100.0)
        assert item.subscore == pytest.approx(0.0)

    def test_linear_interpolation(self):
        # value=50: (100-50)/(100-20) = 50/80 = 0.625
        item = score_upper_bound("x", 50.0, good_threshold=20.0, bad_threshold=100.0)
        assert item.subscore == pytest.approx(0.625)

    def test_none_value(self):
        item = score_upper_bound("x", None, good_threshold=10.0, bad_threshold=100.0)
        assert item.is_data_available is False


# ════════════════════════════════════════════════════════
# TestScoreClampLinear
# ════════════════════════════════════════════════════════

class TestScoreClampLinear:

    def test_at_lo(self):
        item = score_clamp_linear("x", 0.0, lo=0.0, hi=1.0)
        assert item.subscore == pytest.approx(0.0)

    def test_at_hi(self):
        item = score_clamp_linear("x", 1.0, lo=0.0, hi=1.0)
        assert item.subscore == pytest.approx(1.0)

    def test_middle(self):
        item = score_clamp_linear("x", 0.5, lo=0.0, hi=1.0)
        assert item.subscore == pytest.approx(0.5)

    def test_clamp_below(self):
        item = score_clamp_linear("x", -1.0, lo=0.0, hi=1.0)
        assert item.subscore == pytest.approx(0.0)

    def test_clamp_above(self):
        item = score_clamp_linear("x", 2.0, lo=0.0, hi=1.0)
        assert item.subscore == pytest.approx(1.0)

    def test_reverse(self):
        item = score_clamp_linear("x", 0.3, lo=0.0, hi=1.0, reverse=True)
        assert item.subscore == pytest.approx(0.7)

    def test_reverse_extremes(self):
        assert score_clamp_linear("x", 0.0, lo=0.0, hi=1.0, reverse=True).subscore == pytest.approx(1.0)
        assert score_clamp_linear("x", 1.0, lo=0.0, hi=1.0, reverse=True).subscore == pytest.approx(0.0)

    def test_close_position_passthrough(self):
        """close_position_20d = 0.72 → subscore = 0.72."""
        item = score_clamp_linear("close_position_20d", 0.72, lo=0.0, hi=1.0)
        assert item.subscore == pytest.approx(0.72)

    def test_none_value(self):
        item = score_clamp_linear("x", None, lo=0.0, hi=1.0)
        assert item.is_data_available is False


# ════════════════════════════════════════════════════════
# TestScoreDiscrete
# ════════════════════════════════════════════════════════

class TestScoreDiscrete:

    _MAP = {0: 0.0, 1: 0.5, 2: 0.85, 3: 1.0}

    def test_exact_match_0(self):
        item = score_discrete("x", 0, mapping=self._MAP)
        assert item.subscore == pytest.approx(0.0)

    def test_exact_match_2(self):
        item = score_discrete("x", 2, mapping=self._MAP)
        assert item.subscore == pytest.approx(0.85)

    def test_exact_match_3(self):
        item = score_discrete("x", 3, mapping=self._MAP)
        assert item.subscore == pytest.approx(1.0)

    def test_clamped_above_max(self):
        """值超过最大 key → clamp 到最大 key."""
        item = score_discrete("x", 5, mapping=self._MAP)
        assert item.subscore == pytest.approx(1.0)
        assert "clamped_to_3" in item.note

    def test_none_value(self):
        item = score_discrete("x", None, mapping=self._MAP)
        assert item.is_data_available is False

    def test_default_score_used(self):
        # int clamp: value=99 >= max_key(1) → clamp to mapping[1]=1.0
        item = score_discrete("x", 99, mapping={0: 0.0, 1: 1.0}, default_score=0.5)
        assert item.subscore == pytest.approx(1.0)   # int clamp to max_key=1
        assert "clamped_to_1" in item.note

    def test_default_score_used_string_key(self):
        # 非 int key，不触发 clamp，走 default_score
        item = score_discrete("x", "unknown", mapping={"a": 0.0, "b": 1.0}, default_score=0.5)
        assert item.subscore == pytest.approx(0.5)
        assert "default_score_used" in item.note

    def test_no_default_no_match(self):
        # 非 int key，无命中，default_score=None → is_data_available=False
        item = score_discrete("x", "unknown", mapping={"a": 0.0}, default_score=None)
        assert item.is_data_available is False


# ════════════════════════════════════════════════════════
# TestScorePercentile
# ════════════════════════════════════════════════════════

class TestScorePercentile:

    _POOL = [1.0, 2.0, 3.0, 4.0, 5.0]

    def test_highest_value(self):
        item = score_percentile("x", 5.0, pool=self._POOL, ascending=True)
        assert item.subscore == pytest.approx(1.0)

    def test_lowest_value(self):
        item = score_percentile("x", 1.0, pool=self._POOL, ascending=True)
        assert item.subscore == pytest.approx(0.2)   # bisect_right([1,2,3,4,5], 1) = 1 → 1/5

    def test_middle_value(self):
        item = score_percentile("x", 3.0, pool=self._POOL, ascending=True)
        assert item.subscore == pytest.approx(0.6)   # bisect_right = 3 → 3/5

    def test_ascending_false(self):
        # 最低值 → 最高分
        item = score_percentile("x", 1.0, pool=self._POOL, ascending=False)
        assert item.subscore == pytest.approx(0.8)   # 1 - 0.2 = 0.8

    def test_ascending_false_highest(self):
        item = score_percentile("x", 5.0, pool=self._POOL, ascending=False)
        assert item.subscore == pytest.approx(0.0)   # 1 - 1.0

    def test_value_below_pool_min(self):
        item = score_percentile("x", 0.0, pool=self._POOL, ascending=True)
        assert item.subscore == pytest.approx(0.0)   # bisect_right = 0 → 0/5

    def test_value_above_pool_max(self):
        item = score_percentile("x", 10.0, pool=self._POOL, ascending=True)
        assert item.subscore == pytest.approx(1.0)   # bisect_right = 5 → 5/5

    def test_none_value(self):
        item = score_percentile("x", None, pool=self._POOL)
        assert item.is_data_available is False

    def test_none_pool(self):
        item = score_percentile("x", 3.0, pool=None)
        assert item.is_data_available is False
        assert "pool_too_small" in item.note

    def test_pool_too_small(self):
        item = score_percentile("x", 3.0, pool=[1.0])
        assert item.is_data_available is False

    def test_pool_derived_value(self):
        """derived_value 应记录分位值."""
        item = score_percentile("x", 3.0, pool=self._POOL, ascending=True)
        assert item.derived_value == pytest.approx(0.6)

    def test_no_none_in_pool(self):
        """pool 中不应有 None（调用方负责过滤）."""
        # 正常 pool，全数字
        pool = [1.0, 2.0, 3.0]
        item = score_percentile("x", 2.0, pool=pool, ascending=True)
        assert item.subscore is not None

    def test_duplicate_pool_values(self):
        """pool 中有重复值的情况."""
        pool = [3.0, 3.0, 3.0, 3.0, 3.0]
        # value=3.0 → bisect_right = 5 → 5/5 = 1.0
        item = score_percentile("x", 3.0, pool=pool)
        assert item.subscore == pytest.approx(1.0)


# ════════════════════════════════════════════════════════
# TestScoreBool
# ════════════════════════════════════════════════════════

class TestScoreBool:

    def test_true_default(self):
        item = score_bool("x", True)
        assert item.subscore == pytest.approx(1.0)

    def test_false_default(self):
        item = score_bool("x", False)
        assert item.subscore == pytest.approx(0.0)

    def test_custom_scores(self):
        item = score_bool("x", True, true_score=0.8, false_score=0.2)
        assert item.subscore == pytest.approx(0.8)
        item2 = score_bool("x", False, true_score=0.8, false_score=0.2)
        assert item2.subscore == pytest.approx(0.2)

    def test_none_value(self):
        item = score_bool("x", None)
        assert item.is_data_available is False


# ════════════════════════════════════════════════════════
# TestScoringPool
# ════════════════════════════════════════════════════════

class TestScoringPool:

    def test_build_filters_none(self):
        """None 值不进入 pool."""
        lst: list = []
        _append_if_not_none(lst, None)
        _append_if_not_none(lst, float("nan"))
        _append_if_not_none(lst, 1.0)
        assert lst == [1.0]

    def test_build_from_details(self):
        from a_share_hot_screener.models import HotStockDetail
        details = [
            HotStockDetail(code="A", passed_hard_filter=True,
                           return_5d=5.0, return_10d=8.0,
                           limit_up_count_10d=2, strong_pool_entry_count_3d=1,
                           close_position_20d=0.7, volume_ratio_20d=1.5,
                           clv_latest=0.6, amount_ratio_5d_to_20d=1.2),
            HotStockDetail(code="B", passed_hard_filter=True,
                           return_5d=3.0, return_10d=None,
                           limit_up_count_10d=0, strong_pool_entry_count_3d=None,
                           close_position_20d=0.3, volume_ratio_20d=None,
                           clv_latest=0.4, amount_ratio_5d_to_20d=0.9),
            HotStockDetail(code="C", passed_hard_filter=False,  # 未通过硬筛，不参与
                           return_5d=20.0),
        ]
        pool = ScoringPool.build(details)
        assert pool.stock_count == 2   # 只有 A, B
        assert sorted(pool.pool_return_5d) == [3.0, 5.0]
        assert pool.pool_return_10d == [8.0]  # B 的 return_10d=None 被过滤
        assert sorted(pool.pool_limit_up_count_10d) == [0.0, 2.0]
        assert pool.pool_strong_pool_3d == [1.0]   # B 的 None 被过滤
        assert sorted(pool.pool_close_position_20d) == [0.3, 0.7]
        assert pool.pool_volume_ratio_20d == [1.5]  # B 的 None 被过滤
        assert sorted(pool.pool_clv_latest) == [0.4, 0.6]
        assert sorted(pool.pool_amount_ratio_5d_to_20d) == [0.9, 1.2]

    def test_build_empty(self):
        from a_share_hot_screener.models import HotStockDetail
        pool = ScoringPool.build([])
        assert pool.stock_count == 0
        assert pool.pool_return_5d == []


# ════════════════════════════════════════════════════════
# TestHotThemeScore
# ════════════════════════════════════════════════════════

class TestHotThemeScore:

    from a_share_hot_screener.scorers.hot_theme import compute_hot_theme_score

    def _make_full_pool(self) -> ScoringPool:
        return _make_pool(
            pool_return_5d=[1.0, 3.0, 5.0, 8.0, 10.0],
            pool_return_10d=[2.0, 5.0, 8.0, 12.0, 15.0],
            pool_limit_up_count_10d=[0.0, 0.0, 1.0, 2.0, 3.0],
            pool_strong_pool_3d=[0.0, 0.0, 1.0, 2.0],
        )

    def test_all_available(self):
        from a_share_hot_screener.scorers.hot_theme import compute_hot_theme_score
        detail = _make_detail(
            return_5d=8.0,                    # 高分位
            return_10d=12.0,
            big_up_count_10d=2,               # S11: HT3 大涨天数
            consec_up_days=3,                 # S11: HT4 连续上涨天数
            industry_heat_pctile_5d=0.85,
            industry_heat_source="hist_em_5d",
            flags={"sector_momentum_signal": "steady_strong"},  # HT7
        )
        pool = self._make_full_pool()
        axis = compute_hot_theme_score(detail, pool)
        assert axis.score is not None
        assert 0 <= axis.score <= 1.0
        assert axis.coverage == pytest.approx(1.0)
        assert len(axis.items) == 6  # HT1-HT5 + HT7 (HT6 off)
        assert axis.axis_name == "hot_theme_score"

    def test_industry_heat_missing_reduces_coverage(self):
        from a_share_hot_screener.scorers.hot_theme import compute_hot_theme_score
        detail = _make_detail(
            return_5d=5.0,
            return_10d=8.0,
            big_up_count_10d=1,               # S11: HT3
            consec_up_days=0,                 # S11: HT4
            industry_heat_pctile_5d=None,    # 缺失
        )
        pool = self._make_full_pool()
        axis = compute_hot_theme_score(detail, pool)
        assert axis.score is not None
        # industry_heat 权重=7，总权重=8+6+8+6+7=35，available=8+6+8+6=28
        assert axis.coverage == pytest.approx(28.0 / 35.0, abs=1e-3)

    def test_pool_too_small_pctile_items_unavailable(self):
        """pool 只有 1 个元素 → 百分位项全部 is_data_available=False."""
        from a_share_hot_screener.scorers.hot_theme import compute_hot_theme_score
        detail = _make_detail(
            return_5d=5.0,
            return_10d=8.0,
            limit_up_count_10d=1,
            strong_pool_entry_count_3d=1,
            industry_heat_pctile_5d=0.7,
        )
        pool = _make_pool(
            pool_return_5d=[5.0],          # 只有 1 个
            pool_return_10d=[8.0],
            pool_limit_up_count_10d=[1.0],
        )
        axis = compute_hot_theme_score(detail, pool)
        # #2: pool只有1个元素 → 绝对涨幅 fallback（is_data_available=True，但子分受限）
        ht1 = next(i for i in axis.items if i.name == "return_5d_pctile")
        assert ht1.is_data_available is True
        assert "abs_fallback" in (ht1.note or "")
        assert ht1.subscore <= 0.70  # TINY_POOL_SUBSCORE_CAP

    def test_ht4_consec_up_days_fallback(self):
        """HT4 consec_up_days=None 时 is_data_available=False（S11 改为 consec_up_days）."""
        from a_share_hot_screener.scorers.hot_theme import compute_hot_theme_score
        detail = _make_detail(
            return_5d=5.0,
            return_10d=8.0,
            big_up_count_10d=1,
            consec_up_days=None,              # 缺失
            industry_heat_pctile_5d=0.5,
        )
        pool = self._make_full_pool()
        axis = compute_hot_theme_score(detail, pool)
        ht4 = next(i for i in axis.items if i.name == "consec_up_days")
        assert ht4.is_data_available is False

    def test_all_missing_returns_none_score(self):
        """所有字段均缺失 → score=None, coverage=0."""
        from a_share_hot_screener.scorers.hot_theme import compute_hot_theme_score
        detail = _make_detail()  # 所有评分字段均 None
        pool = _make_pool()      # 空 pool
        axis = compute_hot_theme_score(detail, pool)
        assert axis.score is None
        assert axis.coverage == 0.0

    def test_subscore_precision(self):
        """子分保留 4 位小数。"""
        from a_share_hot_screener.scorers.hot_theme import compute_hot_theme_score
        detail = _make_detail(
            return_5d=3.7,
            industry_heat_pctile_5d=0.666666,
        )
        pool = _make_pool(pool_return_5d=[1.0, 2.0, 3.0, 3.7, 5.0])
        axis = compute_hot_theme_score(detail, pool)
        for item in axis.items:
            if item.subscore is not None:
                assert item.subscore == round(item.subscore, 4)

    def test_ht4_consec_0_gets_zero(self):
        """HT4 S11: consec_up_days=0 → 0.0."""
        from a_share_hot_screener.scorers.hot_theme import compute_hot_theme_score
        detail = _make_detail(consec_up_days=0)
        pool = _make_pool()
        axis = compute_hot_theme_score(detail, pool)
        ht4 = next(i for i in axis.items if i.name == "consec_up_days")
        assert ht4.subscore == pytest.approx(0.0)

    def test_ht4_consec_5_gets_one(self):
        """HT4 S11: consec_up_days=5 → 1.0."""
        from a_share_hot_screener.scorers.hot_theme import compute_hot_theme_score
        detail = _make_detail(consec_up_days=5)
        pool = _make_pool()
        axis = compute_hot_theme_score(detail, pool)
        ht4 = next(i for i in axis.items if i.name == "consec_up_days")
        assert ht4.subscore == pytest.approx(1.0)

    # ── HT6: 概念板块热度（Session 16 新增）──────────────────

    def test_ht6_included_when_concept_available(self):
        """HT6: 概念模块可用时纳入评分（共 7 项：HT1-HT6 + HT7）."""
        from a_share_hot_screener.scorers.hot_theme import compute_hot_theme_score
        detail = _make_detail(
            return_5d=8.0, return_10d=12.0,
            big_up_count_10d=2, consec_up_days=3,
            industry_heat_pctile_5d=0.8,
            flags={"sector_momentum_signal": "neutral"},
        )
        detail.advanced_concept_module_available = True
        detail.concept_heat_pctile_5d = 0.85
        detail.concept_heat_source = "concept_spot_em_degraded"
        pool = self._make_full_pool()
        axis = compute_hot_theme_score(detail, pool)
        assert len(axis.items) == 7  # HT1-HT7
        ht6 = next(i for i in axis.items if i.name == "concept_heat_pctile_5d")
        assert ht6.is_data_available is True
        assert ht6.weight == 7.0
        assert ht6.subscore is not None
        assert ht6.subscore > 0.5

    def test_ht6_excluded_when_concept_unavailable(self):
        """HT6: 概念模块不可用时不纳入评分（共 6 项：HT1-HT5 + HT7）."""
        from a_share_hot_screener.scorers.hot_theme import compute_hot_theme_score
        detail = _make_detail(
            return_5d=8.0, return_10d=12.0,
            big_up_count_10d=2, consec_up_days=3,
            industry_heat_pctile_5d=0.8,
            flags={"sector_momentum_signal": "neutral"},
        )
        detail.advanced_concept_module_available = False
        detail.concept_heat_pctile_5d = None
        pool = self._make_full_pool()
        axis = compute_hot_theme_score(detail, pool)
        assert len(axis.items) == 6  # HT1-HT5 + HT7 (no HT6)
        assert all(i.name != "concept_heat_pctile_5d" for i in axis.items)

    def test_ht6_high_concept_heat_score(self):
        """HT6: 概念热度 95 百分位 → 接近满分."""
        from a_share_hot_screener.scorers.hot_theme import compute_hot_theme_score
        detail = _make_detail(
            return_5d=8.0, return_10d=12.0,
            big_up_count_10d=2, consec_up_days=3,
            industry_heat_pctile_5d=0.8,
        )
        detail.advanced_concept_module_available = True
        detail.concept_heat_pctile_5d = 0.95  # 95百分位
        pool = self._make_full_pool()
        axis = compute_hot_theme_score(detail, pool)
        ht6 = next(i for i in axis.items if i.name == "concept_heat_pctile_5d")
        # 95*100=95 >= H=90 → score_lower_bound 应给接近 1.0
        assert ht6.subscore >= 0.9

    def test_ht6_low_concept_heat_score(self):
        """HT6: 概念热度 40 百分位 → 0 分."""
        from a_share_hot_screener.scorers.hot_theme import compute_hot_theme_score
        detail = _make_detail(
            return_5d=8.0, return_10d=12.0,
            big_up_count_10d=2, consec_up_days=3,
            industry_heat_pctile_5d=0.8,
        )
        detail.advanced_concept_module_available = True
        detail.concept_heat_pctile_5d = 0.40  # 40百分位
        pool = self._make_full_pool()
        axis = compute_hot_theme_score(detail, pool)
        ht6 = next(i for i in axis.items if i.name == "concept_heat_pctile_5d")
        # 40*100=40 < L=55 → score=0
        assert ht6.subscore == pytest.approx(0.0)

    def test_ht6_coverage_with_concept(self):
        """概念模块可用时 coverage 应反映 7 个指标."""
        from a_share_hot_screener.scorers.hot_theme import compute_hot_theme_score
        detail = _make_detail(
            return_5d=8.0, return_10d=12.0,
            big_up_count_10d=2, consec_up_days=3,
            industry_heat_pctile_5d=0.8,
            flags={"sector_momentum_signal": "steady_strong"},
        )
        detail.advanced_concept_module_available = True
        detail.concept_heat_pctile_5d = 0.75
        pool = self._make_full_pool()
        axis = compute_hot_theme_score(detail, pool)
        # 7 个指标全有数据，coverage 应为 1.0
        assert axis.coverage == pytest.approx(1.0)
        # 总权重: 8+6+8+6+7+7+5=47
        total_weight = sum(i.weight for i in axis.items)
        assert total_weight == pytest.approx(47.0)


# ════════════════════════════════════════════════════════
# TestTrendFlowScore
# ════════════════════════════════════════════════════════

class TestTrendFlowScore:

    def _make_full_pool(self) -> ScoringPool:
        return _make_pool(
            pool_close_position_20d=[0.2, 0.4, 0.6, 0.8],
            pool_volume_ratio_20d=[0.5, 1.0, 1.5, 2.5],
            pool_clv_latest=[0.3, 0.5, 0.7, 0.9],
            pool_amount_ratio_5d_to_20d=[0.8, 1.0, 1.2, 1.5],
        )

    def test_all_available(self):
        from a_share_hot_screener.scorers.trend_flow import compute_trend_flow_score
        detail = _make_detail(
            close_position_20d=0.75,
            latest_price=10.0,
            ma5=9.5, ma10=9.2, ma20=9.0,  # TF2: 3条件满足
            volume_ratio_20d=1.8,
            clv_latest=0.7,
            amount_ratio_5d_to_20d=1.3,
            flags={                         # TF6/TF7: Session 22 新增
                "net_main_inflow_ratio_5d": 5.0,
                "margin_buy_net_ratio_5d": 2.0,
                "is_margin_eligible": True,
            },
        )
        pool = self._make_full_pool()
        axis = compute_trend_flow_score(detail, pool)
        assert axis.score is not None
        assert 0 <= axis.score <= 1.0
        assert axis.coverage == pytest.approx(1.0)
        assert len(axis.items) == 7
        assert axis.axis_name == "trend_flow_score"

    def test_tf1_close_position_three_seg(self):
        """close_position_20d 三段型（L=0.55,T=0.75,H=0.95）."""
        from a_share_hot_screener.scorers.trend_flow import compute_trend_flow_score
        # 0.95 >= H → 1.0
        detail = _make_detail(close_position_20d=0.95)
        pool = _make_pool()
        axis = compute_trend_flow_score(detail, pool)
        tf1 = next(i for i in axis.items if i.name == "close_position_20d")
        assert tf1.subscore == pytest.approx(1.0)

    def test_tf2_full_bull(self):
        """close>ma5, ma5>ma10, ma10>ma20 → 3条件 → 1.0."""
        from a_share_hot_screener.scorers.trend_flow import compute_trend_flow_score
        detail = _make_detail(latest_price=10.0, ma5=9.5, ma10=9.0, ma20=8.5)
        pool = _make_pool()
        axis = compute_trend_flow_score(detail, pool)
        tf2 = next(i for i in axis.items if i.name == "ma_bullish_alignment")
        assert tf2.subscore == pytest.approx(1.0)

    def test_tf2_zero_conditions(self):
        """close<ma5, ma5<ma10, ma10<ma20 → 0条件 → 0.0."""
        from a_share_hot_screener.scorers.trend_flow import compute_trend_flow_score
        detail = _make_detail(latest_price=8.0, ma5=9.0, ma10=10.0, ma20=11.0)
        pool = _make_pool()
        axis = compute_trend_flow_score(detail, pool)
        tf2 = next(i for i in axis.items if i.name == "ma_bullish_alignment")
        assert tf2.subscore == pytest.approx(0.0)

    def test_tf2_two_conditions(self):
        """close>ma5, ma5>ma10, ma10<ma20 → 2条件 → 0.70."""
        from a_share_hot_screener.scorers.trend_flow import compute_trend_flow_score
        detail = _make_detail(latest_price=10.0, ma5=9.5, ma10=9.0, ma20=9.5)
        pool = _make_pool()
        axis = compute_trend_flow_score(detail, pool)
        tf2 = next(i for i in axis.items if i.name == "ma_bullish_alignment")
        assert tf2.subscore == pytest.approx(0.70)

    def test_tf2_one_condition(self):
        """only close>ma5 → 1条件 → 0.35."""
        from a_share_hot_screener.scorers.trend_flow import compute_trend_flow_score
        detail = _make_detail(latest_price=10.0, ma5=9.5, ma10=10.5, ma20=11.0)
        pool = _make_pool()
        axis = compute_trend_flow_score(detail, pool)
        tf2 = next(i for i in axis.items if i.name == "ma_bullish_alignment")
        assert tf2.subscore == pytest.approx(0.35)

    def test_tf3_volume_ratio_clamp(self):
        """量比超过 5.0 → clamp 到 2.5（good_threshold）→ 满分."""
        from a_share_hot_screener.scorers.trend_flow import compute_trend_flow_score
        detail = _make_detail(volume_ratio_20d=10.0)
        pool = _make_pool()
        axis = compute_trend_flow_score(detail, pool)
        tf3 = next(i for i in axis.items if i.name == "volume_ratio_20d")
        assert tf3.subscore == pytest.approx(1.0)

    def test_tf3_low_volume_ratio(self):
        from a_share_hot_screener.scorers.trend_flow import compute_trend_flow_score
        # Session 10: L=0.8（原 1.1），0.9 > L=0.8，线性段 (0.9-0.8)/(1.8-0.8)*0.70 = 0.07
        detail = _make_detail(volume_ratio_20d=0.9)   # L=0.8 < 0.9 < T=1.8 → 0.07
        pool = _make_pool()
        axis = compute_trend_flow_score(detail, pool)
        tf3 = next(i for i in axis.items if i.name == "volume_ratio_20d")
        assert tf3.subscore == pytest.approx(0.07, abs=1e-3)
        # 0.7 时真正缩量，仍接近 0
        detail2 = _make_detail(volume_ratio_20d=0.7)  # < L=0.8 → 0.0
        axis2 = compute_trend_flow_score(detail2, pool)
        tf3b = next(i for i in axis2.items if i.name == "volume_ratio_20d")
        assert tf3b.subscore == pytest.approx(0.0)

    def test_tf4_clv_three_segment(self):
        """clv_latest 三段型（L=0.40,T=0.70,H=0.90）."""
        from a_share_hot_screener.scorers.trend_flow import compute_trend_flow_score
        # 0.90 >= H → 1.0
        detail = _make_detail(clv_latest=0.90)
        pool = _make_pool()
        axis = compute_trend_flow_score(detail, pool)
        tf4 = next(i for i in axis.items if i.name == "clv_latest")
        assert tf4.subscore == pytest.approx(1.0)

    def test_tf5_amount_ratio_above_good(self):
        from a_share_hot_screener.scorers.trend_flow import compute_trend_flow_score
        detail = _make_detail(amount_ratio_5d_to_20d=2.5)  # >= H=2.5
        pool = _make_pool()
        axis = compute_trend_flow_score(detail, pool)
        tf5 = next(i for i in axis.items if i.name == "amount_ratio_5d_to_20d")
        assert tf5.subscore == pytest.approx(1.0)

    def test_tf5_amount_ratio_clamp(self):
        """amount_ratio > 4.0 → clamp → >= H=2.5 → 满分."""
        from a_share_hot_screener.scorers.trend_flow import compute_trend_flow_score
        detail = _make_detail(amount_ratio_5d_to_20d=5.0)
        pool = _make_pool()
        axis = compute_trend_flow_score(detail, pool)
        tf5 = next(i for i in axis.items if i.name == "amount_ratio_5d_to_20d")
        assert tf5.subscore == pytest.approx(1.0)

    def test_all_missing(self):
        from a_share_hot_screener.scorers.trend_flow import compute_trend_flow_score
        detail = _make_detail()
        pool = _make_pool()
        axis = compute_trend_flow_score(detail, pool)
        assert axis.score is None
        assert axis.coverage == 0.0


# ════════════════════════════════════════════════════════
# TestScoreIntegration — 综合降级与字段写入
# ════════════════════════════════════════════════════════

class TestScoreIntegration:

    def test_pipeline_writes_scores_to_detail(self):
        """pipeline 的 _apply_two_axis_scores 正确写入 detail 字段."""
        from a_share_hot_screener.models import HotStockDetail
        from a_share_hot_screener.scoring_aggregator import apply_two_axis_scores as _apply_two_axis_scores
        from a_share_hot_screener.logger import WarningsCollector

        detail = HotStockDetail(
            code="600519",
            passed_hard_filter=True,
            return_5d=8.0, return_10d=12.0,
            close_position_20d=0.75,
            abs_distance_to_ma10=0.04,
            volume_ratio_20d=1.8,
            clv_latest=0.7,
            amount_ratio_5d_to_20d=1.3,
            limit_up_count_10d=2,
            limit_up_source="zt_pool_em",
            strong_pool_entry_count_3d=1,
            strong_pool_source="zt_pool_strong_em",
            industry_heat_pctile_5d=0.8,
            industry_heat_source="hist_em_5d",
        )
        pool = ScoringPool()
        pool.pool_return_5d = [2.0, 5.0, 8.0, 10.0, 12.0]
        pool.pool_return_10d = [3.0, 7.0, 12.0, 15.0]
        pool.pool_limit_up_count_10d = [0.0, 1.0, 2.0, 3.0]
        pool.pool_strong_pool_3d = [0.0, 1.0, 2.0]

        wc = WarningsCollector()
        _apply_two_axis_scores(detail, pool, wc)

        assert detail.hot_theme_score is not None
        assert detail.hot_theme_coverage is not None
        assert isinstance(detail.hot_theme_subscores, dict)
        assert "items" in detail.hot_theme_subscores

        assert detail.trend_flow_score is not None
        assert detail.trend_flow_coverage is not None
        assert isinstance(detail.trend_flow_subscores, dict)
        assert "items" in detail.trend_flow_subscores

    def test_scores_in_01_range(self):
        """所有 score 和 subscore 均在 [0, 1] 范围内。"""
        from a_share_hot_screener.models import HotStockDetail
        from a_share_hot_screener.scoring_aggregator import apply_two_axis_scores as _apply_two_axis_scores
        from a_share_hot_screener.logger import WarningsCollector

        detail = HotStockDetail(
            code="000858",
            passed_hard_filter=True,
            return_5d=3.5, return_10d=6.0,
            close_position_20d=0.55,
            abs_distance_to_ma10=0.02,
            volume_ratio_20d=1.2,
            clv_latest=0.55,
            amount_ratio_5d_to_20d=1.1,
            limit_up_count_10d=1,
            limit_up_source="zt_pool_em",
            strong_pool_entry_count_3d=0,
            industry_heat_pctile_5d=0.6,
        )
        pool = ScoringPool()
        pool.pool_return_5d = [1.0, 2.0, 3.5, 5.0, 7.0]
        pool.pool_return_10d = [2.0, 4.0, 6.0, 9.0]
        pool.pool_limit_up_count_10d = [0.0, 0.0, 1.0, 2.0]

        wc = WarningsCollector()
        _apply_two_axis_scores(detail, pool, wc)

        def check_axis(score, subscores_dict):
            if score is not None:
                assert 0.0 <= score <= 1.0, f"score={score} 超范围"
            for item_d in subscores_dict.get("items", []):
                s = item_d.get("subscore")
                if s is not None:
                    assert 0.0 <= s <= 1.0, f"subscore={s} 超范围 ({item_d['name']})"

        check_axis(detail.hot_theme_score, detail.hot_theme_subscores)
        check_axis(detail.trend_flow_score, detail.trend_flow_subscores)

    def test_exception_in_scorer_does_not_crash_pipeline(self):
        """评分模块抛异常不导致 pipeline 崩溃。"""
        from a_share_hot_screener.models import HotStockDetail
        from a_share_hot_screener.scoring_aggregator import apply_two_axis_scores as _apply_two_axis_scores
        from a_share_hot_screener.logger import WarningsCollector
        from unittest.mock import patch

        detail = HotStockDetail(code="600519", passed_hard_filter=True)
        pool = ScoringPool()
        wc = WarningsCollector()

        with patch("a_share_hot_screener.scoring_aggregator.compute_hot_theme_score",
                   side_effect=RuntimeError("mock error")):
            _apply_two_axis_scores(detail, pool, wc)  # 不 crash

        # hot_theme_score 应为 None（异常路径），trend_flow 可能正常
        assert detail.hot_theme_score is None
        # warning 应有记录
        warns = wc.get("600519")
        assert any("hot_theme_score" in w for w in warns)

    def test_detail_subscores_serializable(self):
        """hot_theme_subscores / trend_flow_subscores 可以 JSON 序列化。"""
        import json
        from a_share_hot_screener.models import HotStockDetail
        from a_share_hot_screener.scoring_aggregator import apply_two_axis_scores as _apply_two_axis_scores
        from a_share_hot_screener.logger import WarningsCollector

        detail = HotStockDetail(
            code="600519", passed_hard_filter=True,
            return_5d=5.0, close_position_20d=0.6,
        )
        pool = ScoringPool()
        pool.pool_return_5d = [1.0, 3.0, 5.0, 8.0]
        wc = WarningsCollector()
        _apply_two_axis_scores(detail, pool, wc)

        # 不应抛出 JSON 序列化异常
        json.dumps(detail.hot_theme_subscores)
        json.dumps(detail.trend_flow_subscores)
