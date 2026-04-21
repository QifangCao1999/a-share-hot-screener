"""test_ticker_mapping.py – Ticker 映射工具测试."""

from a_share_hot_screener.ticker_mapping import (
    code_to_tushare,
    infer_exchange,
    tushare_to_code,
    code_to_prefix_format,
)


class TestInferExchange:
    def test_sh_600(self):
        assert infer_exchange("600519") == "SH"

    def test_sh_601(self):
        assert infer_exchange("601318") == "SH"

    def test_sh_688(self):
        assert infer_exchange("688981") == "SH"

    def test_sz_000(self):
        assert infer_exchange("000858") == "SZ"

    def test_sz_002(self):
        assert infer_exchange("002594") == "SZ"

    def test_sz_300(self):
        assert infer_exchange("300750") == "SZ"

    def test_bj(self):
        assert infer_exchange("833979") == "BJ"

    def test_unknown(self):
        assert infer_exchange("999999") == "UNKNOWN"


class TestCodeToTushare:
    def test_sh(self):
        assert code_to_tushare("600519") == "600519.SH"

    def test_sz(self):
        assert code_to_tushare("000858") == "000858.SZ"

    def test_bj(self):
        assert code_to_tushare("833979") == "833979.BJ"

    def test_explicit_exchange(self):
        assert code_to_tushare("600519", exchange="SH") == "600519.SH"

    def test_unknown_returns_none(self):
        assert code_to_tushare("999999") is None


class TestTushareToCode:
    def test_sh(self):
        assert tushare_to_code("600519.SH") == "600519"

    def test_sz(self):
        assert tushare_to_code("000858.SZ") == "000858"

    def test_invalid(self):
        assert tushare_to_code("INVALID") is None


class TestPrefixFormat:
    def test_sh(self):
        assert code_to_prefix_format("600519") == "SH600519"

    def test_sz(self):
        assert code_to_prefix_format("000858") == "SZ000858"
