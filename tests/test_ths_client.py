"""TushareClient ths_* 方法单元测试 (6000pt 同花顺板块接口).

全部使用 Mock，不调用真实网络。
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from a_share_hot_screener.clients.tushare_client import TushareClient


@pytest.fixture
def mock_cache():
    cache = MagicMock()
    cache.get.return_value = None
    return cache


@pytest.fixture
def client(mock_cache):
    """创建带 mock cache 的 TushareClient（跳过真实 tushare 初始化）。"""
    with patch.dict("sys.modules", {"tushare": MagicMock()}):
        import importlib
        mock_pro = MagicMock()
        c = TushareClient(token="fake_token", cache=mock_cache)
        c.pro = mock_pro
        return c


class TestGetThsIndex:
    def test_success(self, client, mock_cache):
        expected = pd.DataFrame({
            "ts_code": ["885001.TI", "885002.TI"],
            "name": ["半导体", "白酒"],
            "count": [50, 30],
            "exchange": ["A", "A"],
            "list_date": ["20200101", "20200101"],
            "type": ["I", "I"],
        })
        client.pro.ths_index.return_value = expected

        result = client.get_ths_index(exchange="A", type_="I")
        assert result is not None
        assert len(result) == 2
        assert result.iloc[0]["name"] == "半导体"
        mock_cache.put.assert_called_once()

    def test_no_permission(self, client, mock_cache):
        """积分不足时 _call 返回 None（内部已处理无权限错误）。"""
        client.pro.ths_index.side_effect = Exception("没有接口访问权限")

        result = client.get_ths_index(exchange="A", type_="I")
        assert result is None

    def test_cache_hit(self, client, mock_cache):
        cached_data = [{"ts_code": "885001.TI", "name": "半导体", "count": 50}]
        mock_cache.get.return_value = cached_data

        result = client.get_ths_index(exchange="A", type_="I")
        assert result is not None
        assert len(result) == 1
        # API 不应被调用
        client.pro.ths_index.assert_not_called()

    def test_type_none_returns_all(self, client, mock_cache):
        expected = pd.DataFrame({
            "ts_code": ["885001.TI"],
            "name": ["半导体"],
            "count": [50],
            "exchange": ["A"],
            "list_date": ["20200101"],
            "type": ["I"],
        })
        client.pro.ths_index.return_value = expected

        result = client.get_ths_index(exchange="A", type_=None)
        assert result is not None
        # 调用时不应传 type 参数
        call_kwargs = client.pro.ths_index.call_args[1]
        assert "type" not in call_kwargs


class TestGetThsDaily:
    def test_by_trade_date(self, client, mock_cache):
        expected = pd.DataFrame({
            "ts_code": ["885001.TI", "885002.TI"],
            "trade_date": ["20260418", "20260418"],
            "close": [1200.0, 800.0],
            "open": [1190.0, 795.0],
            "high": [1210.0, 810.0],
            "low": [1185.0, 790.0],
            "pct_change": [1.5, -0.3],
            "vol": [10000.0, 8000.0],
        })
        client.pro.ths_daily.return_value = expected

        result = client.get_ths_daily(trade_date="20260418")
        assert result is not None
        assert len(result) == 2
        mock_cache.put.assert_called_once()

    def test_by_ts_code_range(self, client, mock_cache):
        expected = pd.DataFrame({
            "ts_code": ["885001.TI"] * 3,
            "trade_date": ["20260416", "20260417", "20260418"],
            "close": [1100.0, 1150.0, 1200.0],
        })
        client.pro.ths_daily.return_value = expected

        result = client.get_ths_daily(
            ts_code="885001.TI", start_date="20260416", end_date="20260418"
        )
        assert result is not None
        assert len(result) == 3

    def test_no_permission(self, client, mock_cache):
        client.pro.ths_daily.side_effect = Exception("没有接口访问权限")
        result = client.get_ths_daily(trade_date="20260418")
        assert result is None

    def test_cache_hit(self, client, mock_cache):
        mock_cache.get.return_value = [
            {"ts_code": "885001.TI", "trade_date": "20260418", "close": 1200.0}
        ]
        result = client.get_ths_daily(trade_date="20260418")
        assert result is not None
        client.pro.ths_daily.assert_not_called()


class TestGetThsMember:
    def test_by_con_code(self, client, mock_cache):
        expected = pd.DataFrame({
            "ts_code": ["885001.TI", "885002.TI"],
            "con_code": ["600519.SH", "600519.SH"],
            "con_name": ["贵州茅台", "贵州茅台"],
        })
        client.pro.ths_member.return_value = expected

        result = client.get_ths_member(con_code="600519.SH")
        assert result is not None
        assert len(result) == 2
        mock_cache.put.assert_called_once()

    def test_by_ts_code(self, client, mock_cache):
        expected = pd.DataFrame({
            "ts_code": ["885001.TI"] * 3,
            "con_code": ["600519.SH", "000858.SZ", "300750.SZ"],
            "con_name": ["贵州茅台", "五粮液", "宁德时代"],
        })
        client.pro.ths_member.return_value = expected

        result = client.get_ths_member(ts_code="885001.TI")
        assert result is not None
        assert len(result) == 3

    def test_no_args_returns_none(self, client, mock_cache):
        result = client.get_ths_member()
        assert result is None

    def test_no_permission(self, client, mock_cache):
        client.pro.ths_member.side_effect = Exception("没有接口访问权限")
        result = client.get_ths_member(con_code="600519.SH")
        assert result is None

    def test_cache_hit(self, client, mock_cache):
        mock_cache.get.return_value = [
            {"ts_code": "885001.TI", "con_code": "600519.SH", "con_name": "贵州茅台"}
        ]
        result = client.get_ths_member(con_code="600519.SH")
        assert result is not None
        client.pro.ths_member.assert_not_called()
