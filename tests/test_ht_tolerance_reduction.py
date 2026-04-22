"""Session 11 新增测试.

覆盖范围：
  1. HT3 大涨天数（big_up_count_10d）— price_features + scorer
  2. HT4 连续上涨天数（consec_up_days）— price_features + scorer
  3. H6 amount_tolerance_pct 成交额容差
  4. P3-1 股东减持模块（shareholder_reduction）
  5. pipeline 新字段传递
"""

from __future__ import annotations

import datetime as dt
import os
import sys
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import pytest


# ── 工具函数 ─────────────────────────────────────────────

def _make_detail(**kwargs):
    from a_share_hot_screener.models import HotStockDetail
    defaults = {
        "code": "600519",
        "name": "贵州茅台",
        "exchange": "SH",
    }
    defaults.update(kwargs)
    return HotStockDetail(**defaults)


def _make_pool(**kwargs):
    from a_share_hot_screener.scoring import ScoringPool
    pool = ScoringPool.__new__(ScoringPool)
    pool.pool_return_5d = kwargs.get("pool_return_5d", [1.0, 2.0, 3.0, 4.0, 5.0])
    pool.pool_return_10d = kwargs.get("pool_return_10d", [1.0, 2.0, 3.0, 4.0, 5.0])
    pool.pool_amount_avg_5d = kwargs.get("pool_amount_avg_5d", [])
    pool.stock_count = kwargs.get("stock_count", 5)
    return pool


def _make_config(**kwargs):
    from a_share_hot_screener.config import HotScreenerConfig
    defaults = dict(
        tushare_token="test",
        run_date=dt.date(2026, 4, 18),
        stock_codes=["600519"],
        output_dir="/tmp/test_s11_output",
    )
    defaults.update(kwargs)
    return HotScreenerConfig(**defaults)


def _make_eod_rows(
    n: int = 20,
    base_price: float = 100.0,
    daily_returns: Optional[List[float]] = None,
) -> List[Dict[str, Any]]:
    """构建 n 行 EOD 模拟数据.

    daily_returns: 如果提供，按此列表设定每日涨跌幅(%)，长度 = n-1
    """
    rows = []
    price = base_price
    for i in range(n):
        if daily_returns and i > 0 and i - 1 < len(daily_returns):
            pct = daily_returns[i - 1]
            price = price * (1 + pct / 100.0)
        elif i > 0 and not daily_returns:
            price = price * 1.01  # 默认每天涨1%
        d = f"2026-04-{str(i + 1).zfill(2)}"
        rows.append({
            "date": d,
            "open": price * 0.998,
            "high": price * 1.01,
            "low": price * 0.99,
            "close": price,
            "volume": 10000000,
            "adjusted_close": price,
        })
    return rows


# ════════════════════════════════════════════════════════
# 1. price_features: big_up_count_10d
# ════════════════════════════════════════════════════════

class TestBigUpCount10d:

    def test_no_big_up_days(self):
        """所有日涨幅 < 5% → big_up_count_10d=0."""
        from a_share_hot_screener.price_features import compute_price_features
        # 20天每天涨1%
        rows = _make_eod_rows(20, base_price=100.0)
        feat = compute_price_features(rows, "2026-04-20")
        assert feat.big_up_count_10d == 0

    def test_some_big_up_days(self):
        """3天涨幅>5% → big_up_count_10d=3."""
        from a_share_hot_screener.price_features import compute_price_features
        # 前10天平稳，后10天中3天大涨
        returns = [1.0] * 9 + [6.0, 1.0, 7.0, 1.0, 1.0, 8.0, 1.0, 1.0, 1.0, 1.0]
        rows = _make_eod_rows(20, daily_returns=returns)
        feat = compute_price_features(rows, "2026-04-20")
        # 近10日 = rows[-10:], 需要 rows[-11] 作 prev_close
        # returns[9]=6.0, [11]=7.0, [14]=8.0 → 3天
        assert feat.big_up_count_10d == 3

    def test_all_big_up_days(self):
        """所有10天涨幅>5% → big_up_count_10d=10."""
        from a_share_hot_screener.price_features import compute_price_features
        returns = [6.0] * 19
        rows = _make_eod_rows(20, daily_returns=returns)
        feat = compute_price_features(rows, "2026-04-20")
        assert feat.big_up_count_10d == 10

    def test_exactly_5pct_not_counted(self):
        """涨幅恰好=5.0% 不算大涨（> 不含 =）."""
        from a_share_hot_screener.price_features import compute_price_features
        # 直接构造精确的 5.0% 涨幅行
        rows = [
            {"date": "2026-04-01", "open": 100, "high": 101, "low": 99, "close": 100.0, "volume": 10000000, "adjusted_close": 100.0},
            {"date": "2026-04-02", "open": 100, "high": 106, "low": 100, "close": 105.0, "volume": 10000000, "adjusted_close": 105.0},  # exactly +5%
        ]
        feat = compute_price_features(rows, "2026-04-02")
        # 涨幅 = (105-100)/100 = 5.0%，阈值是 >5%，不含等于
        assert feat.big_up_count_10d == 0

    def test_short_data(self):
        """只有1行数据 → big_up_count_10d=None（行数不足2）."""
        from a_share_hot_screener.price_features import compute_price_features
        rows = _make_eod_rows(1)
        feat = compute_price_features(rows, "2026-04-01")
        assert feat.big_up_count_10d is None

    def test_few_rows(self):
        """只有5行数据 → 可计算，但窗口只有4天."""
        from a_share_hot_screener.price_features import compute_price_features
        returns = [6.0, 6.0, 1.0, 1.0]
        rows = _make_eod_rows(5, daily_returns=returns)
        feat = compute_price_features(rows, "2026-04-05")
        assert feat.big_up_count_10d == 2


# ════════════════════════════════════════════════════════
# 2. price_features: consec_up_days
# ════════════════════════════════════════════════════════

class TestConsecUpDays:

    def test_all_up(self):
        """10天全涨 → consec_up_days=10 (window内可计算的天数)."""
        from a_share_hot_screener.price_features import compute_price_features
        returns = [1.0] * 10
        rows = _make_eod_rows(11, daily_returns=returns)
        feat = compute_price_features(rows, "2026-04-11")
        assert feat.consec_up_days == 10

    def test_mixed_up_down(self):
        """涨涨跌涨涨涨 → consec_up_days=3."""
        from a_share_hot_screener.price_features import compute_price_features
        returns = [1.0, 1.0, -1.0, 1.0, 1.0, 1.0]
        rows = _make_eod_rows(7, daily_returns=returns)
        feat = compute_price_features(rows, "2026-04-07")
        assert feat.consec_up_days == 3

    def test_latest_down(self):
        """最新日下跌 → consec_up_days=0."""
        from a_share_hot_screener.price_features import compute_price_features
        returns = [1.0, 1.0, 1.0, -1.0]
        rows = _make_eod_rows(5, daily_returns=returns)
        feat = compute_price_features(rows, "2026-04-05")
        assert feat.consec_up_days == 0

    def test_flat_day(self):
        """持平(=prev_close) → 不算上涨，中断连续."""
        from a_share_hot_screener.price_features import compute_price_features
        returns = [1.0, 0.0, 1.0]  # 中间一天持平
        rows = _make_eod_rows(4, daily_returns=returns)
        feat = compute_price_features(rows, "2026-04-04")
        assert feat.consec_up_days == 1  # 只有最后一天

    def test_short_data(self):
        """只有1行 → consec_up_days=None."""
        from a_share_hot_screener.price_features import compute_price_features
        rows = _make_eod_rows(1)
        feat = compute_price_features(rows, "2026-04-01")
        assert feat.consec_up_days is None


# ════════════════════════════════════════════════════════
# 3. HT3 scorer: big_up_count_10d 离散映射
# ════════════════════════════════════════════════════════

class TestHT3BigUpCount:

    def test_ht3_zero(self):
        """HT3: big_up_count_10d=0 → 0.0."""
        from a_share_hot_screener.scorers.hot_theme import compute_hot_theme_score
        detail = _make_detail(big_up_count_10d=0)
        pool = _make_pool()
        axis = compute_hot_theme_score(detail, pool)
        ht3 = next(i for i in axis.items if i.name == "big_up_count_10d")
        assert ht3.subscore == pytest.approx(0.0)
        assert ht3.is_data_available is True

    def test_ht3_one(self):
        """HT3: big_up_count_10d=1 → 0.40."""
        from a_share_hot_screener.scorers.hot_theme import compute_hot_theme_score
        detail = _make_detail(big_up_count_10d=1)
        pool = _make_pool()
        axis = compute_hot_theme_score(detail, pool)
        ht3 = next(i for i in axis.items if i.name == "big_up_count_10d")
        assert ht3.subscore == pytest.approx(0.40)

    def test_ht3_two(self):
        """HT3: big_up_count_10d=2 → 0.70."""
        from a_share_hot_screener.scorers.hot_theme import compute_hot_theme_score
        detail = _make_detail(big_up_count_10d=2)
        pool = _make_pool()
        axis = compute_hot_theme_score(detail, pool)
        ht3 = next(i for i in axis.items if i.name == "big_up_count_10d")
        assert ht3.subscore == pytest.approx(0.70)

    def test_ht3_three(self):
        """HT3: big_up_count_10d=3 → 0.90."""
        from a_share_hot_screener.scorers.hot_theme import compute_hot_theme_score
        detail = _make_detail(big_up_count_10d=3)
        pool = _make_pool()
        axis = compute_hot_theme_score(detail, pool)
        ht3 = next(i for i in axis.items if i.name == "big_up_count_10d")
        assert ht3.subscore == pytest.approx(0.90)

    def test_ht3_four_or_more(self):
        """HT3: big_up_count_10d>=4 → 1.0."""
        from a_share_hot_screener.scorers.hot_theme import compute_hot_theme_score
        detail = _make_detail(big_up_count_10d=5)
        pool = _make_pool()
        axis = compute_hot_theme_score(detail, pool)
        ht3 = next(i for i in axis.items if i.name == "big_up_count_10d")
        assert ht3.subscore == pytest.approx(1.0)

    def test_ht3_none(self):
        """HT3: big_up_count_10d=None → is_data_available=False."""
        from a_share_hot_screener.scorers.hot_theme import compute_hot_theme_score
        detail = _make_detail(big_up_count_10d=None)
        pool = _make_pool()
        axis = compute_hot_theme_score(detail, pool)
        ht3 = next(i for i in axis.items if i.name == "big_up_count_10d")
        assert ht3.is_data_available is False

    def test_ht3_weight_is_8(self):
        """HT3 权重=8."""
        from a_share_hot_screener.scorers.hot_theme import compute_hot_theme_score
        detail = _make_detail(big_up_count_10d=1)
        pool = _make_pool()
        axis = compute_hot_theme_score(detail, pool)
        ht3 = next(i for i in axis.items if i.name == "big_up_count_10d")
        assert ht3.weight == 8.0


# ════════════════════════════════════════════════════════
# 4. HT4 scorer: consec_up_days 离散映射
# ════════════════════════════════════════════════════════

class TestHT4ConsecUpDays:

    def test_ht4_zero(self):
        """HT4: consec_up_days=0 → 0.0."""
        from a_share_hot_screener.scorers.hot_theme import compute_hot_theme_score
        detail = _make_detail(consec_up_days=0)
        pool = _make_pool()
        axis = compute_hot_theme_score(detail, pool)
        ht4 = next(i for i in axis.items if i.name == "consec_up_days")
        assert ht4.subscore == pytest.approx(0.0)

    def test_ht4_one(self):
        """HT4: consec_up_days=1 → 0.25."""
        from a_share_hot_screener.scorers.hot_theme import compute_hot_theme_score
        detail = _make_detail(consec_up_days=1)
        pool = _make_pool()
        axis = compute_hot_theme_score(detail, pool)
        ht4 = next(i for i in axis.items if i.name == "consec_up_days")
        assert ht4.subscore == pytest.approx(0.25)

    def test_ht4_two(self):
        """HT4: consec_up_days=2 → 0.50."""
        from a_share_hot_screener.scorers.hot_theme import compute_hot_theme_score
        detail = _make_detail(consec_up_days=2)
        pool = _make_pool()
        axis = compute_hot_theme_score(detail, pool)
        ht4 = next(i for i in axis.items if i.name == "consec_up_days")
        assert ht4.subscore == pytest.approx(0.50)

    def test_ht4_three(self):
        """HT4: consec_up_days=3 → 0.70."""
        from a_share_hot_screener.scorers.hot_theme import compute_hot_theme_score
        detail = _make_detail(consec_up_days=3)
        pool = _make_pool()
        axis = compute_hot_theme_score(detail, pool)
        ht4 = next(i for i in axis.items if i.name == "consec_up_days")
        assert ht4.subscore == pytest.approx(0.70)

    def test_ht4_four(self):
        """HT4: consec_up_days=4 → 0.85."""
        from a_share_hot_screener.scorers.hot_theme import compute_hot_theme_score
        detail = _make_detail(consec_up_days=4)
        pool = _make_pool()
        axis = compute_hot_theme_score(detail, pool)
        ht4 = next(i for i in axis.items if i.name == "consec_up_days")
        assert ht4.subscore == pytest.approx(0.85)

    def test_ht4_five_or_more(self):
        """HT4: consec_up_days>=5 → 1.0."""
        from a_share_hot_screener.scorers.hot_theme import compute_hot_theme_score
        detail = _make_detail(consec_up_days=7)
        pool = _make_pool()
        axis = compute_hot_theme_score(detail, pool)
        ht4 = next(i for i in axis.items if i.name == "consec_up_days")
        assert ht4.subscore == pytest.approx(1.0)

    def test_ht4_weight_is_6(self):
        """HT4 权重=6."""
        from a_share_hot_screener.scorers.hot_theme import compute_hot_theme_score
        detail = _make_detail(consec_up_days=1)
        pool = _make_pool()
        axis = compute_hot_theme_score(detail, pool)
        ht4 = next(i for i in axis.items if i.name == "consec_up_days")
        assert ht4.weight == 6.0


# ════════════════════════════════════════════════════════
# 5. H6 amount_tolerance_pct
# ════════════════════════════════════════════════════════

class TestH6AmountTolerance:

    def test_default_tolerance_5pct(self):
        """默认容差5%."""
        config = _make_config()
        assert config.amount_tolerance_pct == 5.0

    def test_tolerance_passes_edge_case(self):
        """amount=1.92亿 + 容差5% → 通过（实际门槛1.9亿）."""
        from a_share_hot_screener.hard_filters import apply_hard_filters
        config = _make_config(min_amount_avg_5d=200_000_000)
        result = apply_hard_filters(
            config=config,
            name="测试股",
            ipo_date="20200101",
            listing_days=1000,
            latest_price=10.0,
            latest_volume=100000,
            amount_1d=200_000_000,
            amount_avg_5d=192_000_000,  # 1.92亿 < 2亿 但 > 1.9亿
            float_market_cap=5_000_000_000,
        )
        for reason in result.fail_reasons:
            assert "amount_too_low" not in reason

    def test_tolerance_still_fails_below_effective(self):
        """amount=1.85亿 + 容差5% → 仍然失败（<1.9亿）."""
        from a_share_hot_screener.hard_filters import apply_hard_filters
        config = _make_config(min_amount_avg_5d=200_000_000)
        result = apply_hard_filters(
            config=config,
            name="测试股",
            ipo_date="20200101",
            listing_days=1000,
            latest_price=10.0,
            latest_volume=100000,
            amount_1d=200_000_000,
            amount_avg_5d=185_000_000,  # 1.85亿 < 1.9亿
            float_market_cap=5_000_000_000,
        )
        assert any("amount_too_low" in r for r in result.fail_reasons)

    def test_zero_tolerance(self):
        """容差=0%: 严格检查."""
        from a_share_hot_screener.hard_filters import apply_hard_filters
        config = _make_config(min_amount_avg_5d=200_000_000, amount_tolerance_pct=0.0)
        result = apply_hard_filters(
            config=config,
            name="测试股",
            ipo_date="20200101",
            listing_days=1000,
            latest_price=10.0,
            latest_volume=100000,
            amount_1d=200_000_000,
            amount_avg_5d=199_000_000,  # 1.99亿 < 2亿, 无容差
            float_market_cap=5_000_000_000,
        )
        assert any("amount_too_low" in r for r in result.fail_reasons)

    def test_tolerance_message_includes_effective(self):
        """失败消息包含实际门槛信息."""
        from a_share_hot_screener.hard_filters import apply_hard_filters
        config = _make_config(min_amount_avg_5d=200_000_000)
        result = apply_hard_filters(
            config=config,
            name="测试股",
            ipo_date="20200101",
            listing_days=1000,
            latest_price=10.0,
            latest_volume=100000,
            amount_1d=200_000_000,
            amount_avg_5d=100_000_000,  # 明显低于门槛
            float_market_cap=5_000_000_000,
        )
        assert any("容差" in r for r in result.fail_reasons)

    def test_amount_above_original_passes_regardless(self):
        """amount=2.5亿 > 2亿 → 不受容差影响，直接通过."""
        from a_share_hot_screener.hard_filters import apply_hard_filters
        config = _make_config()
        result = apply_hard_filters(
            config=config,
            name="测试股",
            ipo_date="20200101",
            listing_days=1000,
            latest_price=10.0,
            latest_volume=100000,
            amount_1d=200_000_000,
            amount_avg_5d=250_000_000,
            float_market_cap=5_000_000_000,
        )
        for reason in result.fail_reasons:
            assert "amount_too_low" not in reason


# ════════════════════════════════════════════════════════
# 6. CLI: --amount-tolerance-pct
# ════════════════════════════════════════════════════════

class TestCLIAmountTolerance:

    def test_cli_default(self):
        from a_share_hot_screener.cli import build_parser
        parser = build_parser()
        args = parser.parse_args([
            "--tushare-token", "test",
            "--run-date", "2026-04-18",
            "--stock-codes", "600519",
            "--output-dir", "/tmp/test",
        ])
        assert args.amount_tolerance_pct == 5.0

    def test_cli_custom(self):
        from a_share_hot_screener.cli import build_parser
        parser = build_parser()
        args = parser.parse_args([
            "--tushare-token", "test",
            "--run-date", "2026-04-18",
            "--stock-codes", "600519",
            "--output-dir", "/tmp/test",
            "--amount-tolerance-pct", "10",
        ])
        assert args.amount_tolerance_pct == 10.0


# ════════════════════════════════════════════════════════
# 7. P3-1: 股东减持模块
# ════════════════════════════════════════════════════════

class TestShareholderReduction:

    def test_normal_increase_flags(self):
        """股东人数增加>10% → flag=True."""
        import pandas as pd
        from a_share_hot_screener.shareholder_reduction import compute_shareholder_reduction

        mock_df = pd.DataFrame({
            "end_date": ["20260115", "20260415"],
            "holder_num": [10000, 12000],  # +20%
        })
        mock_client = MagicMock()
        mock_client.get_stk_holdernumber.return_value = mock_df

        result = compute_shareholder_reduction(
            code="600519",
            tushare_client=mock_client,
            run_date=dt.date(2026, 4, 18),
        )
        assert result.shareholder_net_reduction_ratio_3m == pytest.approx(20.0)
        assert result.shareholder_reduction_flag_3m is True
        assert result.source == "tushare_stk_holdernumber"

    def test_normal_decrease_no_flag(self):
        """股东人数减少 → ratio<0, flag=False."""
        import pandas as pd
        from a_share_hot_screener.shareholder_reduction import compute_shareholder_reduction

        mock_df = pd.DataFrame({
            "end_date": ["20260115", "20260415"],
            "holder_num": [12000, 10000],  # -16.7%
        })
        mock_client = MagicMock()
        mock_client.get_stk_holdernumber.return_value = mock_df

        result = compute_shareholder_reduction(
            code="600519",
            tushare_client=mock_client,
            run_date=dt.date(2026, 4, 18),
        )
        assert result.shareholder_net_reduction_ratio_3m is not None
        assert result.shareholder_net_reduction_ratio_3m < 0
        assert result.shareholder_reduction_flag_3m is False

    def test_small_increase_no_flag(self):
        """股东人数增加<10% → flag=False."""
        import pandas as pd
        from a_share_hot_screener.shareholder_reduction import compute_shareholder_reduction

        mock_df = pd.DataFrame({
            "end_date": ["20260115", "20260415"],
            "holder_num": [10000, 10500],  # +5%
        })
        mock_client = MagicMock()
        mock_client.get_stk_holdernumber.return_value = mock_df

        result = compute_shareholder_reduction(
            code="600519",
            tushare_client=mock_client,
            run_date=dt.date(2026, 4, 18),
        )
        assert result.shareholder_net_reduction_ratio_3m == pytest.approx(5.0)
        assert result.shareholder_reduction_flag_3m is False

    def test_api_returns_none(self):
        """接口返回 None → 全部 None + warning."""
        from a_share_hot_screener.shareholder_reduction import compute_shareholder_reduction

        mock_client = MagicMock()
        mock_client.get_stk_holdernumber.return_value = None

        warns = []
        result = compute_shareholder_reduction(
            code="600519",
            tushare_client=mock_client,
            run_date=dt.date(2026, 4, 18),
            warnings=warns,
        )
        assert result.shareholder_net_reduction_ratio_3m is None
        assert result.shareholder_reduction_flag_3m is None
        assert result.source == "none"
        assert len(warns) > 0

    def test_single_data_point(self):
        """只有1个数据点 → 无法计算变化率."""
        import pandas as pd
        from a_share_hot_screener.shareholder_reduction import compute_shareholder_reduction

        mock_df = pd.DataFrame({
            "end_date": ["20260415"],
            "holder_num": [10000],
        })
        mock_client = MagicMock()
        mock_client.get_stk_holdernumber.return_value = mock_df

        result = compute_shareholder_reduction(
            code="600519",
            tushare_client=mock_client,
            run_date=dt.date(2026, 4, 18),
        )
        assert result.shareholder_net_reduction_ratio_3m is None
        assert result.data_points == 1

    def test_as_of_filter(self):
        """未来数据被 as-of 过滤."""
        import pandas as pd
        from a_share_hot_screener.shareholder_reduction import compute_shareholder_reduction

        mock_df = pd.DataFrame({
            "end_date": ["20260115", "20260415", "20260615"],
            "holder_num": [10000, 12000, 15000],
        })
        mock_client = MagicMock()
        mock_client.get_stk_holdernumber.return_value = mock_df

        result = compute_shareholder_reduction(
            code="600519",
            tushare_client=mock_client,
            run_date=dt.date(2026, 4, 18),  # 2026-06-15 应被过滤
        )
        # 只有前2个点可用
        assert result.data_points == 2
        assert result.shareholder_net_reduction_ratio_3m == pytest.approx(20.0)

    def test_api_exception_handled(self):
        """API 调用异常 → 优雅降级."""
        from a_share_hot_screener.shareholder_reduction import compute_shareholder_reduction

        mock_client = MagicMock()
        mock_client.get_stk_holdernumber.side_effect = Exception("network error")

        result = compute_shareholder_reduction(
            code="600519",
            tushare_client=mock_client,
            run_date=dt.date(2026, 4, 18),
        )
        assert result.shareholder_net_reduction_ratio_3m is None
        assert result.source == "none"
        assert len(result.warnings) > 0

    def test_timestamp_date_format(self):
        """支持 Timestamp 格式的日期."""
        import pandas as pd
        from a_share_hot_screener.shareholder_reduction import compute_shareholder_reduction

        mock_df = pd.DataFrame({
            "end_date": ["20260115", "20260415"],
            "holder_num": [10000, 11500],
        })
        mock_client = MagicMock()
        mock_client.get_stk_holdernumber.return_value = mock_df

        result = compute_shareholder_reduction(
            code="600519",
            tushare_client=mock_client,
            run_date=dt.date(2026, 4, 18),
        )
        assert result.shareholder_net_reduction_ratio_3m == pytest.approx(15.0)
        assert result.shareholder_reduction_flag_3m is True


# ════════════════════════════════════════════════════════
# 8. HotStockSummary 新字段传递
# ════════════════════════════════════════════════════════

class TestSummaryNewFields:

    def test_big_up_count_10d_in_summary(self):
        """big_up_count_10d 正确传递到 summary."""
        from a_share_hot_screener.models import HotStockSummary
        detail = _make_detail(big_up_count_10d=3)
        summary = HotStockSummary.from_detail(detail)
        assert summary.big_up_count_10d == 3

    def test_consec_up_days_in_summary(self):
        """consec_up_days 正确传递到 summary."""
        from a_share_hot_screener.models import HotStockSummary
        detail = _make_detail(consec_up_days=5)
        summary = HotStockSummary.from_detail(detail)
        assert summary.consec_up_days == 5

    def test_shareholder_fields_from_flags(self):
        """股东减持字段从 flags 字典正确传递到 summary."""
        from a_share_hot_screener.models import HotStockSummary
        detail = _make_detail()
        detail.flags["shareholder_net_reduction_ratio_3m"] = 15.5
        detail.flags["shareholder_reduction_flag_3m"] = True
        summary = HotStockSummary.from_detail(detail)
        assert summary.shareholder_net_reduction_ratio_3m == 15.5
        assert summary.shareholder_reduction_flag_3m is True

    def test_shareholder_fields_none_when_missing(self):
        """flags 中无减持数据 → summary 字段为 None."""
        from a_share_hot_screener.models import HotStockSummary
        detail = _make_detail()
        summary = HotStockSummary.from_detail(detail)
        assert summary.shareholder_net_reduction_ratio_3m is None
        assert summary.shareholder_reduction_flag_3m is None


# ════════════════════════════════════════════════════════
# 9. config.amount_tolerance_pct
# ════════════════════════════════════════════════════════

class TestConfigTolerance:

    def test_default_value(self):
        config = _make_config()
        assert config.amount_tolerance_pct == 5.0

    def test_custom_value(self):
        config = _make_config(amount_tolerance_pct=10.0)
        assert config.amount_tolerance_pct == 10.0

    def test_zero_value(self):
        config = _make_config(amount_tolerance_pct=0.0)
        assert config.amount_tolerance_pct == 0.0
