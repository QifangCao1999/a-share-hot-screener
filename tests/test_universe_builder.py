"""Phase 2 Universe Builder 测试.

覆盖:
  - 静态底仓加载 + 刷新
  - 涨停池扩展
  - 龙虎榜扩展
  - 成交额 Top N 扩展
  - 热门板块成分扩展 (ths_daily + ths_member)
  - 过滤: ST/停牌/上市天数/行业
  - 降级场景 (Level 2/3 不可用)
  - 输出文件读写
  - 来源标记正确性
"""

import datetime as dt_mod
import os
import tempfile
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from a_share_hot_screener.universe_builder import UniverseBuilder, UniverseResult


# ════════════════════════════════════════════════════════
# Fixtures
# ════════════════════════════════════════════════════════

def _make_mock_calendar():
    """Mock TradeCalendar: 周一~周五为交易日."""
    cal = MagicMock()

    def is_trade_date(d):
        if isinstance(d, str):
            d = dt_mod.datetime.strptime(d.replace("-", ""), "%Y%m%d").date()
        return d.weekday() < 5

    cal.is_trade_date = is_trade_date
    return cal


def _make_mock_client():
    """Mock TushareClient with basic data."""
    client = MagicMock()

    # stock_basic: 有 ST 和正常股
    basic_df = pd.DataFrame([
        {"ts_code": "600519.SH", "symbol": "600519", "name": "贵州茅台", "industry": "白酒", "list_date": "20010827", "market": "主板"},
        {"ts_code": "000858.SZ", "symbol": "000858", "name": "五粮液", "industry": "白酒", "list_date": "19980427", "market": "主板"},
        {"ts_code": "002436.SZ", "symbol": "002436", "name": "兴森科技", "industry": "元器件", "list_date": "20100318", "market": "中小板"},
        {"ts_code": "300750.SZ", "symbol": "300750", "name": "宁德时代", "industry": "电气设备", "list_date": "20180611", "market": "创业板"},
        {"ts_code": "601398.SH", "symbol": "601398", "name": "工商银行", "industry": "银行", "list_date": "20061027", "market": "主板"},
        {"ts_code": "000001.SZ", "symbol": "000001", "name": "平安银行", "industry": "银行", "list_date": "19910403", "market": "主板"},
        {"ts_code": "600000.SH", "symbol": "600000", "name": "*ST信用", "industry": "银行", "list_date": "19991110", "market": "主板"},
        {"ts_code": "688001.SH", "symbol": "688001", "name": "华兴源创", "industry": "专用设备", "list_date": "20260401", "market": "科创板"},  # 新股 ~21天
    ])
    client.get_stock_basic.return_value = basic_df

    # index_weight: CSI300 有 600519, 000858
    def mock_index_weight(index_code, **kwargs):
        if "000300" in index_code:
            return pd.DataFrame([
                {"index_code": index_code, "con_code": "600519.SH", "trade_date": "20260420", "weight": 2.5},
                {"index_code": index_code, "con_code": "000858.SZ", "trade_date": "20260420", "weight": 1.2},
            ])
        elif "000905" in index_code:
            return pd.DataFrame([
                {"index_code": index_code, "con_code": "002436.SZ", "trade_date": "20260420", "weight": 0.3},
                {"index_code": index_code, "con_code": "300750.SZ", "trade_date": "20260420", "weight": 0.5},
            ])
        elif "000852" in index_code:
            return pd.DataFrame([
                {"index_code": index_code, "con_code": "601398.SH", "trade_date": "20260420", "weight": 0.1},
            ])
        return pd.DataFrame()

    client.get_index_weight.side_effect = mock_index_weight

    # limit_list_d: 一天有涨停
    client.get_limit_list.return_value = pd.DataFrame([
        {"ts_code": "002436.SZ", "name": "兴森科技", "close": 30.0, "pct_chg": 10.0, "limit_times": 1},
        {"ts_code": "000001.SZ", "name": "平安银行", "close": 15.0, "pct_chg": 10.0, "limit_times": 1},
    ])

    # top_list: 龙虎榜
    client.get_top_list.return_value = pd.DataFrame([
        {"trade_date": "20260420", "ts_code": "300750.SZ", "name": "宁德时代", "l_buy": 1e8, "l_sell": 5e7},
    ])

    # daily (全市场): 成交额排序
    client.get_daily_by_date.return_value = pd.DataFrame([
        {"ts_code": "600519.SH", "trade_date": "20260422", "open": 1800, "high": 1820, "low": 1790, "close": 1810, "pre_close": 1800, "change": 10, "pct_chg": 0.56, "vol": 50000, "amount": 900000},
        {"ts_code": "000858.SZ", "trade_date": "20260422", "open": 150, "high": 155, "low": 148, "close": 153, "pre_close": 150, "change": 3, "pct_chg": 2.0, "vol": 200000, "amount": 800000},
        {"ts_code": "002436.SZ", "trade_date": "20260422", "open": 28, "high": 30, "low": 27, "close": 29.5, "pre_close": 28, "change": 1.5, "pct_chg": 5.36, "vol": 500000, "amount": 700000},
        {"ts_code": "300750.SZ", "trade_date": "20260422", "open": 200, "high": 210, "low": 195, "close": 208, "pre_close": 200, "change": 8, "pct_chg": 4.0, "vol": 300000, "amount": 600000},
        {"ts_code": "601398.SH", "trade_date": "20260422", "open": 5.5, "high": 5.6, "low": 5.4, "close": 5.55, "pre_close": 5.5, "change": 0.05, "pct_chg": 0.91, "vol": 1000000, "amount": 500000},
        {"ts_code": "000001.SZ", "trade_date": "20260422", "open": 14.5, "high": 15.2, "low": 14.3, "close": 15.0, "pre_close": 14.5, "change": 0.5, "pct_chg": 3.45, "vol": 800000, "amount": 400000},
    ])

    # ths: 不可用默认
    client.get_ths_index.return_value = None
    client.get_ths_daily.return_value = None
    client.get_ths_member.return_value = None

    return client


# ════════════════════════════════════════════════════════
# 基础测试
# ════════════════════════════════════════════════════════

class TestUniverseBuilderBasic:
    """基本构建流程测试."""

    def test_build_basic(self):
        """全流程构建 (Level 3 不可用)."""
        client = _make_mock_client()
        cal = _make_mock_calendar()

        with tempfile.TemporaryDirectory() as tmpdir:
            builder = UniverseBuilder(
                client, cal,
                universe_dir=tmpdir,
                amount_top_n=5,
                min_listing_days=20,
            )
            result = builder.build("20260422")

            assert result.total_count > 0
            assert result.run_date == "20260422"
            assert len(result.codes) == result.total_count
            assert result.ths_hot_available is False  # ths_index 返回 None

    def test_source_tags(self):
        """来源标记正确."""
        client = _make_mock_client()
        cal = _make_mock_calendar()

        with tempfile.TemporaryDirectory() as tmpdir:
            builder = UniverseBuilder(
                client, cal,
                universe_dir=tmpdir,
                amount_top_n=200,
                min_listing_days=20,
            )
            result = builder.build("20260422")

            # 600519 应该在 csi300 + amount_top
            tags_600519 = result.source_tags.get("600519", set())
            assert "csi300" in tags_600519

            # 002436 应该在 csi500 + zt_pool + amount_top
            tags_002436 = result.source_tags.get("002436", set())
            assert "csi500" in tags_002436
            assert "zt_pool" in tags_002436

    def test_st_filtered(self):
        """ST 股票被过滤."""
        client = _make_mock_client()
        cal = _make_mock_calendar()

        # 让 *ST信用 出现在成交额 Top N 中
        daily_df = client.get_daily_by_date.return_value.copy()
        st_row = pd.DataFrame([{
            "ts_code": "600000.SH", "trade_date": "20260422",
            "open": 5, "high": 5.1, "low": 4.9, "close": 5,
            "pre_close": 5, "change": 0, "pct_chg": 0,
            "vol": 100000, "amount": 300000,
        }])
        client.get_daily_by_date.return_value = pd.concat([daily_df, st_row], ignore_index=True)

        with tempfile.TemporaryDirectory() as tmpdir:
            builder = UniverseBuilder(
                client, cal,
                universe_dir=tmpdir,
                amount_top_n=200,
                min_listing_days=0,  # 不过滤上市天数
            )
            result = builder.build("20260422")

            # *ST信用 (600000) 应该被过滤
            assert "600000" not in result.codes
            assert result.filtered_st >= 1

    def test_new_stock_filtered(self):
        """上市天数不足的新股被过滤."""
        client = _make_mock_client()
        cal = _make_mock_calendar()

        with tempfile.TemporaryDirectory() as tmpdir:
            builder = UniverseBuilder(
                client, cal,
                universe_dir=tmpdir,
                amount_top_n=5,
                min_listing_days=30,  # 30天
            )
            result = builder.build("20260422")

            # 688001 上市 ~21天, 应该被过滤
            assert "688001" not in result.codes

    def test_industry_filter(self):
        """行业过滤."""
        client = _make_mock_client()
        cal = _make_mock_calendar()

        with tempfile.TemporaryDirectory() as tmpdir:
            builder = UniverseBuilder(
                client, cal,
                universe_dir=tmpdir,
                amount_top_n=5,
                min_listing_days=0,
                excluded_industries=["银行"],
            )
            result = builder.build("20260422")

            # 银行股应该被过滤 (601398, 000001)
            assert "601398" not in result.codes
            assert result.filtered_industry >= 1


# ════════════════════════════════════════════════════════
# 降级测试
# ════════════════════════════════════════════════════════

class TestUniverseBuilderDegradation:
    """降级场景测试."""

    def test_zt_pool_unavailable(self):
        """涨停池不可用 → 标记 zt_pool_available=False."""
        client = _make_mock_client()
        cal = _make_mock_calendar()
        client.get_limit_list.return_value = None  # Level 2 不可用

        with tempfile.TemporaryDirectory() as tmpdir:
            builder = UniverseBuilder(client, cal, universe_dir=tmpdir, amount_top_n=5)
            result = builder.build("20260422")

            assert result.zt_pool_available is False
            assert result.total_count > 0  # 仍然有其他来源

    def test_lhb_unavailable(self):
        """龙虎榜不可用 → 标记 top_list_available=False."""
        client = _make_mock_client()
        cal = _make_mock_calendar()
        client.get_top_list.return_value = None

        with tempfile.TemporaryDirectory() as tmpdir:
            builder = UniverseBuilder(client, cal, universe_dir=tmpdir, amount_top_n=5)
            result = builder.build("20260422")

            assert result.top_list_available is False
            assert result.total_count > 0

    def test_daily_unavailable_raises(self):
        """daily 全市场不可用 → 抛异常."""
        client = _make_mock_client()
        cal = _make_mock_calendar()
        client.get_daily_by_date.return_value = None

        with tempfile.TemporaryDirectory() as tmpdir:
            builder = UniverseBuilder(client, cal, universe_dir=tmpdir, amount_top_n=5)
            with pytest.raises(RuntimeError, match="daily"):
                builder.build("20260422")

    def test_stock_basic_unavailable_raises(self):
        """stock_basic 不可用 → 抛异常."""
        client = _make_mock_client()
        cal = _make_mock_calendar()
        client.get_stock_basic.return_value = None

        with tempfile.TemporaryDirectory() as tmpdir:
            builder = UniverseBuilder(client, cal, universe_dir=tmpdir)
            with pytest.raises(RuntimeError, match="stock_basic"):
                builder.build("20260422")

    def test_index_weight_unavailable_uses_stale(self):
        """index_weight 不可用 → 使用旧文件, 标记 stale."""
        client = _make_mock_client()
        cal = _make_mock_calendar()
        client.get_index_weight.side_effect = None  # 清除 side_effect
        client.get_index_weight.return_value = None  # 全部刷新失败

        with tempfile.TemporaryDirectory() as tmpdir:
            # 预先创建所有旧的静态文件 (60天前)
            for fname in ["static_csi300.txt", "static_csi500.txt", "static_csi1000.txt"]:
                fpath = os.path.join(tmpdir, fname)
                with open(fpath, "w") as f:
                    f.write("600519\n000858\n")
                old_time = dt_mod.datetime.now().timestamp() - 60 * 86400
                os.utime(fpath, (old_time, old_time))

            builder = UniverseBuilder(client, cal, universe_dir=tmpdir, amount_top_n=5)
            result = builder.build("20260422")

            assert result.stale_static_universe is True
            # 仍然有 600519 (来自旧文件 + amount_top)
            assert "600519" in result.codes


# ════════════════════════════════════════════════════════
# 文件 I/O 测试
# ════════════════════════════════════════════════════════

class TestUniverseFileIO:
    """文件读写测试."""

    def test_daily_file_written(self):
        """每日文件正确写入."""
        client = _make_mock_client()
        cal = _make_mock_calendar()

        with tempfile.TemporaryDirectory() as tmpdir:
            builder = UniverseBuilder(client, cal, universe_dir=tmpdir, amount_top_n=5)
            result = builder.build("20260422")

            daily_path = os.path.join(tmpdir, "daily_20260422.txt")
            assert os.path.exists(daily_path)

            # 加载回来
            loaded = UniverseBuilder.load_daily_file(daily_path)
            assert len(loaded) == result.total_count
            assert set(loaded) == set(result.codes)

    def test_static_file_refresh(self):
        """静态文件刷新."""
        client = _make_mock_client()
        cal = _make_mock_calendar()

        with tempfile.TemporaryDirectory() as tmpdir:
            builder = UniverseBuilder(client, cal, universe_dir=tmpdir, amount_top_n=5)
            builder.build("20260422")

            # CSI300 静态文件应该被创建
            csi300_path = os.path.join(tmpdir, "static_csi300.txt")
            assert os.path.exists(csi300_path)

            codes = builder._read_static_file(csi300_path)
            assert "600519" in codes
            assert "000858" in codes

    def test_load_daily_file_with_tags(self):
        """加载带标签的每日文件."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "daily_test.txt")
            with open(path, "w") as f:
                f.write("# Test\n")
                f.write("600519\tcsi300,amount_top\n")
                f.write("002436\tcsi500,zt_pool\n")

            codes = UniverseBuilder.load_daily_file(path)
            assert codes == ["600519", "002436"]


# ════════════════════════════════════════════════════════
# 热门板块扩展测试
# ════════════════════════════════════════════════════════

class TestThsHotExpansion:
    """热门板块成分扩展测试."""

    def test_ths_hot_with_data(self):
        """Level 3 数据可用时的板块扩展."""
        client = _make_mock_client()
        cal = _make_mock_calendar()

        # 启用 ths 数据
        client.get_ths_index.return_value = pd.DataFrame([
            {"ts_code": "885600.TI", "name": "AI概念", "type": "N"},
            {"ts_code": "885601.TI", "name": "半导体", "type": "N"},
        ])

        def mock_ths_daily(ts_code, start_date, end_date, **kwargs):
            return pd.DataFrame([
                {"ts_code": ts_code, "trade_date": "20260418", "close": 100},
                {"ts_code": ts_code, "trade_date": "20260422", "close": 110},
            ])

        client.get_ths_daily.side_effect = mock_ths_daily

        client.get_ths_member.return_value = pd.DataFrame([
            {"con_code": "002436.SZ"},
            {"con_code": "300750.SZ"},
        ])

        with tempfile.TemporaryDirectory() as tmpdir:
            builder = UniverseBuilder(client, cal, universe_dir=tmpdir, amount_top_n=5, hot_sector_top_n=2)
            result = builder.build("20260422")

            assert result.ths_hot_available is True
            assert result.ths_hot_count > 0
            # 002436 应该有 ths_concept_hot 标签
            tags_002436 = result.source_tags.get("002436", set())
            assert "ths_concept_hot" in tags_002436

    def test_ths_hot_unavailable(self):
        """Level 3 不可用 → 跳过."""
        client = _make_mock_client()
        cal = _make_mock_calendar()
        # 默认 ths 返回 None

        with tempfile.TemporaryDirectory() as tmpdir:
            builder = UniverseBuilder(client, cal, universe_dir=tmpdir, amount_top_n=5)
            result = builder.build("20260422")

            assert result.ths_hot_available is False
            assert result.ths_hot_count == 0
            # 仍然有其他来源的结果
            assert result.total_count > 0


# ════════════════════════════════════════════════════════
# UniverseResult 测试
# ════════════════════════════════════════════════════════

class TestUniverseResult:
    """UniverseResult 数据结构测试."""

    def test_default_values(self):
        result = UniverseResult(run_date="20260422")
        assert result.total_count == 0
        assert result.zt_pool_available is True
        assert result.stale_static_universe is False

    def test_counts_consistent(self):
        """各项计数一致性."""
        client = _make_mock_client()
        cal = _make_mock_calendar()

        with tempfile.TemporaryDirectory() as tmpdir:
            builder = UniverseBuilder(client, cal, universe_dir=tmpdir, amount_top_n=5)
            result = builder.build("20260422")

            assert result.total_count == len(result.codes)
            assert result.total_count == len(result.source_tags)
            # pre_filter >= total (过滤会减少)
            assert result.pre_filter_count >= result.total_count
