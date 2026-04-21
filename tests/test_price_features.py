"""test_price_features.py – 测试日线派生特征计算."""

import pytest
from a_share_hot_screener.price_features import (
    PriceFeatures,
    _amp_avg,
    _count_limit_board_proxy,
    _count_upper_shadow,
    _return_nd,
    compute_price_features,
)


def make_rows(n: int, base_close=100.0, vol=1_000_000, step=0.5) -> list:
    """生成 n 行模拟 EOD 数据（等差收盘价，成交量固定）."""
    rows = []
    for i in range(n):
        c = round(base_close + i * step, 2)
        rows.append({
            "date": f"2026-03-{i+1:02d}" if i < 31 else f"2026-04-{i-30:02d}",
            "open": round(c - 0.5, 2),
            "high": round(c + 1.0, 2),
            "low": round(c - 1.0, 2),
            "close": c,
            "adjusted_close": c,
            "volume": vol,
        })
    return rows


class TestReturnNd:
    def test_basic_5d(self):
        rows = make_rows(10, base_close=100.0, step=1.0)
        # close[-1]=109, close[-6]=104 → (109-104)/104*100 ≈ 4.81%
        r = _return_nd(rows, n=5)
        assert r is not None
        assert abs(r - (109 - 104) / 104 * 100) < 0.01

    def test_insufficient_rows(self):
        rows = make_rows(4)
        assert _return_nd(rows, n=5) is None

    def test_exact_rows(self):
        rows = make_rows(6, base_close=100.0, step=2.0)
        r = _return_nd(rows, n=5)
        # close[-1]=110, close[-6]=100 → 10%
        assert r is not None
        assert abs(r - 10.0) < 0.01


class TestComputePriceFeatures:
    def test_full_data_30_rows(self):
        rows = make_rows(30, base_close=100.0, step=0.5)
        trade_date = rows[-1]["date"]
        feat = compute_price_features(rows, trade_date)

        assert feat.data_rows == 30
        assert feat.latest_close == pytest.approx(100.0 + 29 * 0.5)
        assert feat.return_3d is not None
        assert feat.return_5d is not None
        assert feat.return_10d is not None
        assert feat.amount_avg_5d is not None
        assert feat.amount_avg_20d is not None
        assert feat.volume_ratio_20d is not None
        assert feat.close_position_20d is not None
        assert 0.0 <= feat.close_position_20d <= 1.0
        assert feat.clv_latest is not None
        assert feat.amount_ratio_5d_to_20d is not None
        assert feat.abs_distance_to_ma10 is not None
        assert feat.amp_norm_avg_5d is not None
        assert feat.upper_shadow_count_5d is not None
        assert feat.limit_board_count_5d is not None

    def test_as_of_cutoff(self):
        """as-of 截断：晚于 trade_date_used 的行被忽略."""
        rows = make_rows(10)
        # 截断到第 5 行
        trade_date = rows[4]["date"]
        feat = compute_price_features(rows, trade_date)
        assert feat.data_rows == 5

    def test_empty_rows(self):
        feat = compute_price_features([], "2026-04-17")
        assert feat.data_rows == 0
        assert feat.latest_close is None
        assert len(feat.coverage_notes) > 0

    def test_insufficient_20d(self):
        rows = make_rows(10)
        trade_date = rows[-1]["date"]
        feat = compute_price_features(rows, trade_date)
        # 10行 → amount_avg_20d 不可用
        assert feat.amount_avg_20d is None
        assert feat.volume_ratio_20d is None
        assert feat.close_position_20d is None
        assert any("20" in n for n in feat.coverage_notes)

    def test_clv_one_bar_equal_hl(self):
        """high == low 时 CLV 取 0.5."""
        rows = [{
            "date": "2026-04-17",
            "open": 10.0, "high": 10.0, "low": 10.0,
            "close": 10.0, "adjusted_close": 10.0, "volume": 1_000,
        }]
        feat = compute_price_features(rows, "2026-04-17")
        assert feat.clv_latest == 0.5

    def test_amount_approx(self):
        """amount ≈ close * volume."""
        rows = [{
            "date": "2026-04-17",
            "open": 100.0, "high": 101.0, "low": 99.0,
            "close": 100.0, "adjusted_close": 100.0, "volume": 500_000,
        }]
        feat = compute_price_features(rows, "2026-04-17")
        assert feat.latest_amount == pytest.approx(100.0 * 500_000)

    def test_warnings_propagated(self):
        """coverage_notes 应传播到外部 warnings 列表."""
        rows = make_rows(5)  # 只有5行，20d 特征全缺
        trade_date = rows[-1]["date"]
        warns = []
        compute_price_features(rows, trade_date, warnings=warns)
        assert any("price_features" in w for w in warns)


class TestUpperShadow:
    def test_no_shadow(self):
        rows = [{"close": 100.0, "open": 100.0, "high": 100.0, "low": 99.0}]
        assert _count_upper_shadow(rows, 0.02) == 0

    def test_has_shadow(self):
        # high=105, max(open,close)=100 → upper_shadow=5 → 5% > 2%
        rows = [{"close": 100.0, "open": 100.0, "high": 105.0, "low": 99.0}]
        assert _count_upper_shadow(rows, 0.02) == 1


class TestLimitBoardProxy:
    def test_one_bar_board(self):
        # open≈close, high≈low → 一字板
        rows = [{"close": 100.0, "open": 100.1, "high": 100.2, "low": 100.0}]
        assert _count_limit_board_proxy(rows) == 1

    def test_normal_bar_not_board(self):
        rows = [{"close": 100.0, "open": 98.0, "high": 103.0, "low": 97.0}]
        assert _count_limit_board_proxy(rows) == 0


class TestAmpAvg:
    def test_single_row(self):
        rows = [{"high": 102.0, "low": 98.0, "open": 100.0, "close": 100.0}]
        result = _amp_avg(rows)
        # amp = (102-98)/100*100 = 4.0%
        assert result is not None
        assert abs(result - 4.0) < 0.01
