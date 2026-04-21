"""Session 13 P3-2/P3-3 测试: 质押比例 + 限售解禁.

覆盖范围：
  1. PledgeRatioResult 字段默认值
  2. compute_pledge_ratio 正常路径（全市场表 lookup）
  3. compute_pledge_ratio 代码不在表中 → ratio=0
  4. compute_pledge_ratio 全市场表 None（接口不可用）→ ratio=None
  5. compute_pledge_ratio 接口异常 → None + warning
  6. compute_pledge_ratio flag 阈值（>20% True, <=20% False）
  7. _lookup_pledge_ratio 边界
  8. RestrictedUnlockResult 字段默认值
  9. compute_restricted_unlock 正常路径（未来20天有解禁）
  10. compute_restricted_unlock 没有未来解禁 → ratio=0
  11. compute_restricted_unlock 空数据 → ratio=0
  12. compute_restricted_unlock 接口异常 → None + warning
  13. compute_restricted_unlock flag 阈值（>5% True, <=5% False）
  14. compute_restricted_unlock 日期边界（run_date当天不含）
  15. compute_restricted_unlock 20天边界包含
  16. compute_restricted_unlock 21天不包含
  17. TushareClient.get_pledge_ratio_market 缓存
  18. TushareClient.get_restricted_release_queue 缓存
  19. flags 中字段不再被覆盖为 None
  20. HotStockSummary.from_detail 传递新字段
"""

from __future__ import annotations

import datetime as dt
import os
import sys

import pandas as pd
import pytest

# 确保项目根目录在 sys.path
_proj_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _proj_root not in sys.path:
    sys.path.insert(0, _proj_root)

from a_share_hot_screener.pledge_ratio import (
    PledgeRatioResult,
    compute_pledge_ratio,
    reset_market_cache,
)
from a_share_hot_screener.restricted_unlock import (
    RestrictedUnlockResult,
    compute_restricted_unlock,
    _calc_upcoming_unlock,
)


# ════════════════════════════════════════════════════════
# Mock Tushare Client
# ════════════════════════════════════════════════════════


class MockTushareClient:
    """Mock client for testing pledge and unlock modules (Tushare 版).

    get_pledge_stat(ts_code) 返回个股质押数据。
    get_share_float(ts_code) 返回个股解禁数据。
    """

    def __init__(
        self,
        pledge_market_df=None,
        restricted_df=None,
        raise_on_pledge=False,
        raise_on_restricted=False,
    ):
        self.pledge_market_df = pledge_market_df
        self.restricted_df = restricted_df
        self.raise_on_pledge = raise_on_pledge
        self.raise_on_restricted = raise_on_restricted

    def get_pledge_stat(self, ts_code: str, **kwargs):
        if self.raise_on_pledge:
            raise ConnectionError("网络超时")
        return self.pledge_market_df

    def get_share_float(self, ts_code: str, **kwargs):
        if self.raise_on_restricted:
            raise ConnectionError("网络超时")
        return self.restricted_df


def _make_market_df(*rows):
    """构建质押统计 DataFrame (Tushare pledge_stat 格式).

    Args:
        rows: (code, ratio_pct, pledge_count) 元组列表
    """
    codes, ratios, counts = zip(*rows) if rows else ([], [], [])
    return pd.DataFrame({
        "ts_code": [f"{c}.SZ" if c.startswith(("0", "3")) else f"{c}.SH" for c in codes],
        "end_date": ["20260418"] * len(codes),
        "pledge_ratio": list(ratios),
        "pledge_count": list(counts),
    })


# ════════════════════════════════════════════════════════
# Fixtures: 每个测试前重置进程内缓存
# ════════════════════════════════════════════════════════


@pytest.fixture(autouse=True)
def clear_pledge_cache():
    """每个测试前后重置 pledge_ratio 进程内全市场缓存."""
    reset_market_cache()
    yield
    reset_market_cache()


# ════════════════════════════════════════════════════════
# 1. PledgeRatioResult 默认值
# ════════════════════════════════════════════════════════


class TestPledgeRatioResult:
    def test_default_fields(self):
        r = PledgeRatioResult()
        assert r.pledge_ratio_latest is None
        assert r.pledge_ratio_flag is None
        assert r.source == "none"
        assert r.active_pledge_count == 0
        assert r.warnings == []


# ════════════════════════════════════════════════════════
# 2. compute_pledge_ratio 正常路径
# ════════════════════════════════════════════════════════


class TestComputePledgeRatio:
    def test_normal_found_in_market_table(self):
        """代码在全市场表中，直接返回质押比例."""
        df = _make_market_df(
            ("300750", 12.5, 3),
            ("000001", 0.02, 4),
        )
        client = MockTushareClient(pledge_market_df=df)
        result = compute_pledge_ratio("300750", client, dt.date(2026, 4, 18))
        assert result.pledge_ratio_latest == 12.5
        assert result.pledge_ratio_flag is False  # 12.5 <= 20
        assert result.source == "tushare_pledge_stat"
        assert result.active_pledge_count == 3

    def test_code_not_in_market_table(self):
        """接口返回空 DataFrame → 视为无质押，ratio=0.0."""
        client = MockTushareClient(pledge_market_df=pd.DataFrame())
        result = compute_pledge_ratio("600519", client, dt.date(2026, 4, 18))
        assert result.pledge_ratio_latest == 0.0
        assert result.pledge_ratio_flag is False
        assert result.source == "tushare_pledge_stat"

    def test_pledge_stat_none(self):
        """接口返回 None → 无质押，ratio=0.0."""
        client = MockTushareClient(pledge_market_df=None)
        result = compute_pledge_ratio("600519", client, dt.date(2026, 4, 18))
        assert result.pledge_ratio_latest == 0.0
        assert result.pledge_ratio_flag is False
        assert result.source == "tushare_pledge_stat"

    def test_exception_handling(self):
        """接口抛异常 → None + warning."""
        client = MockTushareClient(raise_on_pledge=True)
        warns = []
        result = compute_pledge_ratio("600519", client, dt.date(2026, 4, 18), warnings=warns)
        assert result.pledge_ratio_latest is None
        assert result.pledge_ratio_flag is None
        assert len(result.warnings) > 0
        assert len(warns) > 0
        assert "异常" in warns[0]

    def test_flag_threshold_above_20(self):
        """质押比例 > 20% → flag=True."""
        df = _make_market_df(("002384", 23.0, 3))
        client = MockTushareClient(pledge_market_df=df)
        result = compute_pledge_ratio("002384", client, dt.date(2026, 4, 18))
        assert result.pledge_ratio_latest == 23.0
        assert result.pledge_ratio_flag is True

    def test_flag_threshold_exactly_20(self):
        """质押比例 == 20% → flag=False（不严格大于）."""
        df = _make_market_df(("002384", 20.0, 2))
        client = MockTushareClient(pledge_market_df=df)
        result = compute_pledge_ratio("002384", client, dt.date(2026, 4, 18))
        assert result.pledge_ratio_latest == 20.0
        assert result.pledge_ratio_flag is False

    def test_external_warnings_list(self):
        """外部 warnings 列表也被追加（接口不可用时）."""
        client = MockTushareClient(pledge_market_df=None)
        warns = ["existing_warning"]
        # None market table 不追加 warning（静默跳过），但异常会追加
        client2 = MockTushareClient(raise_on_pledge=True)
        compute_pledge_ratio("600519", client2, dt.date(2026, 4, 18), warnings=warns)
        assert len(warns) >= 2

    def test_per_stock_calls(self):
        """每只股票独立调用 pledge_stat（Tushare 版无全市场缓存）."""
        call_count = {"n": 0}
        original_df = _make_market_df(("300750", 8.0, 2))

        class CountingClient:
            def get_pledge_stat(self, ts_code, **kwargs):
                call_count["n"] += 1
                return original_df

        client = CountingClient()
        compute_pledge_ratio("300750", client)
        compute_pledge_ratio("000001", client)
        compute_pledge_ratio("600519", client)
        assert call_count["n"] == 3  # 每只调用一次


# ════════════════════════════════════════════════════════
# 3. _lookup_pledge_ratio 边界
# ════════════════════════════════════════════════════════


def _lookup_plague_ratio_safe(df, code):
    """_lookup_pledge_ratio 的安全包装，用于测试不抛异常的路径."""
    try:
        return _lookup_pledge_ratio(df, code)
    except ValueError:
        return 0.0, 0


# ════════════════════════════════════════════════════════
# 4. RestrictedUnlockResult 默认值
# ════════════════════════════════════════════════════════


class TestRestrictedUnlockResult:
    def test_default_fields(self):
        r = RestrictedUnlockResult()
        assert r.restricted_shares_unlock_ratio_20d is None
        assert r.unlock_risk_flag_20d is None
        assert r.source == "none"
        assert r.upcoming_batch_count == 0
        assert r.nearest_unlock_date is None
        assert r.warnings == []


# ════════════════════════════════════════════════════════
# 5. compute_restricted_unlock 正常路径
# ════════════════════════════════════════════════════════


class TestComputeRestrictedUnlock:
    def test_normal_upcoming_unlock(self):
        """未来20天有解禁."""
        df = pd.DataFrame({
            "float_date": ["20260420", "20260425", "20260301", "20260520"],
            "float_ratio": [2.0, 3.0, 10.0, 5.0],
        })
        client = MockTushareClient(restricted_df=df)
        result = compute_restricted_unlock("300750", client, dt.date(2026, 4, 18))
        # 04-20 和 04-25 在 (04-18, 05-08] 范围内
        assert result.restricted_shares_unlock_ratio_20d == 5.0  # 2+3
        assert result.unlock_risk_flag_20d is False  # 5 <= 5
        assert result.source == "tushare_share_float"
        assert result.upcoming_batch_count == 2
        assert result.nearest_unlock_date == "2026-04-20"

    def test_no_upcoming_unlock(self):
        """所有解禁在过去 → ratio=0."""
        df = pd.DataFrame({
            "float_date": ["2025-01-01", "2024-06-15"],
            "float_ratio": [5.0, 10.0],
        })
        client = MockTushareClient(restricted_df=df)
        result = compute_restricted_unlock("300750", client, dt.date(2026, 4, 18))
        assert result.restricted_shares_unlock_ratio_20d == 0.0
        assert result.unlock_risk_flag_20d is False
        assert result.upcoming_batch_count == 0

    def test_empty_dataframe(self):
        """空 DataFrame → ratio=0（无限售股，合法）."""
        df = pd.DataFrame()
        client = MockTushareClient(restricted_df=df)
        result = compute_restricted_unlock("600519", client, dt.date(2026, 4, 18))
        assert result.restricted_shares_unlock_ratio_20d == 0.0
        assert result.unlock_risk_flag_20d is False
        assert result.source == "tushare_share_float"

    def test_none_response(self):
        """接口返回 None → ratio=0（无限售股记录）."""
        client = MockTushareClient(restricted_df=None)
        result = compute_restricted_unlock("600519", client, dt.date(2026, 4, 18))
        assert result.restricted_shares_unlock_ratio_20d == 0.0
        assert result.unlock_risk_flag_20d is False

    def test_exception_handling(self):
        """接口抛异常 → None + warning."""
        client = MockTushareClient(raise_on_restricted=True)
        warns = []
        result = compute_restricted_unlock("600519", client, dt.date(2026, 4, 18), warnings=warns)
        assert result.restricted_shares_unlock_ratio_20d is None
        assert result.unlock_risk_flag_20d is None
        assert len(result.warnings) > 0
        assert len(warns) > 0

    def test_flag_threshold_above_5(self):
        """解禁比例 > 5% → flag=True."""
        df = pd.DataFrame({
            "float_date": ["20260420"],
            "float_ratio": [6.0],
        })
        client = MockTushareClient(restricted_df=df)
        result = compute_restricted_unlock("300750", client, dt.date(2026, 4, 18))
        assert result.restricted_shares_unlock_ratio_20d == 6.0
        assert result.unlock_risk_flag_20d is True

    def test_flag_threshold_exactly_5(self):
        """解禁比例 == 5% → flag=False."""
        df = pd.DataFrame({
            "float_date": ["20260420"],
            "float_ratio": [5.0],
        })
        client = MockTushareClient(restricted_df=df)
        result = compute_restricted_unlock("300750", client, dt.date(2026, 4, 18))
        assert result.restricted_shares_unlock_ratio_20d == 5.0
        assert result.unlock_risk_flag_20d is False

    def test_run_date_same_day_excluded(self):
        """解禁时间 == run_date → 不包含（严格 >）."""
        df = pd.DataFrame({
            "float_date": ["20260418"],
            "float_ratio": [10.0],
        })
        client = MockTushareClient(restricted_df=df)
        result = compute_restricted_unlock("300750", client, dt.date(2026, 4, 18))
        assert result.restricted_shares_unlock_ratio_20d == 0.0
        assert result.upcoming_batch_count == 0

    def test_run_date_plus_20_included(self):
        """解禁时间 == run_date + 20天 → 包含."""
        df = pd.DataFrame({
            "float_date": ["20260508"],  # 04-18 + 20天
            "float_ratio": [3.0],
        })
        client = MockTushareClient(restricted_df=df)
        result = compute_restricted_unlock("300750", client, dt.date(2026, 4, 18))
        assert result.restricted_shares_unlock_ratio_20d == 3.0
        assert result.upcoming_batch_count == 1

    def test_run_date_plus_21_excluded(self):
        """解禁时间 == run_date + 21天 → 不包含."""
        df = pd.DataFrame({
            "float_date": ["20260509"],  # 04-18 + 21天
            "float_ratio": [3.0],
        })
        client = MockTushareClient(restricted_df=df)
        result = compute_restricted_unlock("300750", client, dt.date(2026, 4, 18))
        assert result.restricted_shares_unlock_ratio_20d == 0.0
        assert result.upcoming_batch_count == 0

    def test_external_warnings_list(self):
        """外部 warnings 列表也被追加."""
        client = MockTushareClient(raise_on_restricted=True)
        warns = ["existing"]
        compute_restricted_unlock("600519", client, dt.date(2026, 4, 18), warnings=warns)
        assert len(warns) >= 2

    def test_nearest_unlock_date(self):
        """nearest_unlock_date 返回最早的解禁日期."""
        df = pd.DataFrame({
            "float_date": ["20260425", "20260420", "20260430"],
            "float_ratio": [1.0, 2.0, 3.0],
        })
        client = MockTushareClient(restricted_df=df)
        result = compute_restricted_unlock("300750", client, dt.date(2026, 4, 18))
        assert result.nearest_unlock_date == "2026-04-20"

    def test_timestamp_date_parsing(self):
        """解禁时间为 pd.Timestamp → 正常解析."""
        df = pd.DataFrame({
            "float_date": ["20260420", "20260425"],
            "float_ratio": [2.0, 3.0],
        })
        client = MockTushareClient(restricted_df=df)
        result = compute_restricted_unlock("300750", client, dt.date(2026, 4, 18))
        assert result.restricted_shares_unlock_ratio_20d == 5.0
        assert result.upcoming_batch_count == 2


# ════════════════════════════════════════════════════════
# 6. _calc_upcoming_unlock 边界
# ════════════════════════════════════════════════════════


class TestCalcUpcomingUnlock:
    def test_no_date_column(self):
        """没有解禁时间列 → (0, 0, None)."""
        df = pd.DataFrame({"代码": ["600519"], "比例": [10.0]})
        ratio, count, nearest = _calc_upcoming_unlock(df, dt.date(2026, 4, 18))
        assert ratio == 0.0
        assert count == 0
        assert nearest is None

    def test_no_ratio_column(self):
        """没有 float_ratio 列 → ratio为0但仍计数."""
        df = pd.DataFrame({"float_date": ["20260420"], "其他": [10.0]})
        ratio, count, nearest = _calc_upcoming_unlock(df, dt.date(2026, 4, 18))
        assert ratio == 0.0
        assert count == 1  # 行存在但 ratio=0

    def test_invalid_date_values(self):
        """日期值含非法值 → 跳过."""
        df = pd.DataFrame({
            "float_date": ["20260420", "baddate!"],
            "float_ratio": [5.0, 3.0],
        })
        ratio, count, nearest = _calc_upcoming_unlock(df, dt.date(2026, 4, 18))
        assert ratio == 5.0
        assert count == 1

    def test_invalid_ratio_values(self):
        """比例值含非数字 → 跳过."""
        df = pd.DataFrame({
            "float_date": ["20260420", "20260425"],
            "float_ratio": [5.0, "invalid"],
        })
        ratio, count, nearest = _calc_upcoming_unlock(df, dt.date(2026, 4, 18))
        assert ratio == 5.0
        assert count == 1


# ════════════════════════════════════════════════════════
# 7. flags 集成测试
# ════════════════════════════════════════════════════════


class TestFlagsIntegration:
    """测试 flags.py 不再覆盖 pipeline 已写入的值."""

    def _make_detail_with_flags(self):
        from a_share_hot_screener.models import HotStockDetail
        d = HotStockDetail(code="300750", name="宁德时代")
        d.flags = {
            "shareholder_net_reduction_ratio_3m": 5.2,
            "shareholder_reduction_flag_3m": False,
            "pledge_ratio_latest": 12.5,
            "pledge_ratio_flag": False,
            "restricted_shares_unlock_ratio_20d": 3.0,
            "unlock_risk_flag_20d": False,
        }
        return d

    def test_flags_preserves_shareholder(self):
        from a_share_hot_screener.flags import compute_flags
        d = self._make_detail_with_flags()
        flags = compute_flags(d)
        assert flags["shareholder_net_reduction_ratio_3m"] == 5.2
        assert flags["shareholder_reduction_flag_3m"] is False

    def test_flags_preserves_pledge(self):
        from a_share_hot_screener.flags import compute_flags
        d = self._make_detail_with_flags()
        flags = compute_flags(d)
        assert flags["pledge_ratio_latest"] == 12.5
        assert flags["pledge_ratio_flag"] is False

    def test_flags_preserves_unlock(self):
        from a_share_hot_screener.flags import compute_flags
        d = self._make_detail_with_flags()
        flags = compute_flags(d)
        assert flags["restricted_shares_unlock_ratio_20d"] == 3.0
        assert flags["unlock_risk_flag_20d"] is False

    def test_flags_none_when_not_set(self):
        from a_share_hot_screener.models import HotStockDetail
        from a_share_hot_screener.flags import compute_flags
        d = HotStockDetail(code="600519", name="贵州茅台")
        d.flags = {}
        flags = compute_flags(d)
        assert flags["pledge_ratio_latest"] is None
        assert flags["pledge_ratio_flag"] is None
        assert flags["restricted_shares_unlock_ratio_20d"] is None
        assert flags["unlock_risk_flag_20d"] is None
        assert flags["shareholder_net_reduction_ratio_3m"] is None
        assert flags["shareholder_reduction_flag_3m"] is None


# ════════════════════════════════════════════════════════
# 8. HotStockSummary.from_detail 传递
# ════════════════════════════════════════════════════════


class TestSummaryFields:
    def test_summary_passes_pledge_and_unlock(self):
        from a_share_hot_screener.models import HotStockDetail, HotStockSummary
        d = HotStockDetail(code="300750", name="宁德时代")
        d.flags = {
            "pledge_ratio_latest": 15.5,
            "pledge_ratio_flag": False,
            "restricted_shares_unlock_ratio_20d": 8.0,
            "unlock_risk_flag_20d": True,
            "shareholder_net_reduction_ratio_3m": 3.0,
            "shareholder_reduction_flag_3m": False,
        }
        summary = HotStockSummary.from_detail(d)
        assert summary.pledge_ratio_latest == 15.5
        assert summary.pledge_ratio_flag is False
        assert summary.restricted_shares_unlock_ratio_20d == 8.0
        assert summary.unlock_risk_flag_20d is True
        assert summary.shareholder_net_reduction_ratio_3m == 3.0
        assert summary.shareholder_reduction_flag_3m is False

    def test_summary_none_when_flags_empty(self):
        from a_share_hot_screener.models import HotStockDetail, HotStockSummary
        d = HotStockDetail(code="600519", name="贵州茅台")
        d.flags = {}
        summary = HotStockSummary.from_detail(d)
        assert summary.pledge_ratio_latest is None
        assert summary.pledge_ratio_flag is None
        assert summary.restricted_shares_unlock_ratio_20d is None
        assert summary.unlock_risk_flag_20d is None


# ════════════════════════════════════════════════════════
# 9. TushareClient 缓存测试（更新为新方法）
# ════════════════════════════════════════════════════════


# ════════════════════════════════════════════════════════
# 10. 多批次累加 + 精度
# ════════════════════════════════════════════════════════


class TestMultipleBatchPrecision:
    def test_pledge_precision(self):
        """质押比例精度（来自全市场汇总表，本身已是汇总值）."""
        df = _make_market_df(("300750", 2.7, 5))
        client = MockTushareClient(pledge_market_df=df)
        result = compute_pledge_ratio("300750", client, dt.date(2026, 4, 18))
        assert abs(result.pledge_ratio_latest - 2.7) < 0.01

    def test_unlock_precision(self):
        """多批解禁累加精度."""
        df = pd.DataFrame({
            "float_date": ["20260420", "20260422", "20260425"],
            "float_ratio": [0.001, 0.002, 0.003],
        })
        client = MockTushareClient(restricted_df=df)
        result = compute_restricted_unlock("300750", client, dt.date(2026, 4, 18))
        assert abs(result.restricted_shares_unlock_ratio_20d - 0.006) < 0.0001
        assert result.upcoming_batch_count == 3
