"""test_stock_codes.py – 测试 stock_codes 解析函数."""

import pytest

from a_share_hot_screener.stock_codes import (
    classify_exchange,
    filter_beijing,
    parse_stock_codes,
)


class TestParseStockCodes:
    def test_comma_separated_string(self):
        valid, invalid = parse_stock_codes("600519,000858,300750")
        assert valid == ["600519", "000858", "300750"]
        assert invalid == []

    def test_list_input(self):
        valid, invalid = parse_stock_codes(["600519", "000858"])
        assert valid == ["600519", "000858"]
        assert invalid == []

    def test_sh_prefix(self):
        valid, _ = parse_stock_codes("SH600519")
        assert valid == ["600519"]

    def test_sz_prefix(self):
        valid, _ = parse_stock_codes("SZ000858")
        assert valid == ["000858"]

    def test_tushare_fallback_format(self):
        valid, _ = parse_stock_codes("600519.SHG")
        assert valid == ["600519"]

    def test_dedup_preserve_order(self):
        valid, _ = parse_stock_codes("600519,000858,600519,300750")
        assert valid == ["600519", "000858", "300750"]

    def test_invalid_code(self):
        valid, invalid = parse_stock_codes("600519,ABCDEF,300750")
        assert "600519" in valid
        assert "ABCDEF" in invalid

    def test_semicolon_separator(self):
        valid, _ = parse_stock_codes("600519;000858;300750")
        assert valid == ["600519", "000858", "300750"]

    def test_empty_string(self):
        valid, invalid = parse_stock_codes("")
        assert valid == []
        assert invalid == []

    def test_spaces_stripped(self):
        valid, _ = parse_stock_codes(" 600519 , 000858 ")
        assert valid == ["600519", "000858"]


class TestClassifyExchange:
    def test_sh_600(self):
        assert classify_exchange("600519") == "SH"

    def test_sh_688(self):
        assert classify_exchange("688111") == "SH"

    def test_sz_000(self):
        assert classify_exchange("000858") == "SZ"

    def test_sz_300(self):
        assert classify_exchange("300750") == "SZ"

    def test_bj_8(self):
        assert classify_exchange("833979") == "BJ"


class TestFilterBeijing:
    def test_include_true(self):
        codes = ["600519", "833979", "000858"]
        kept, removed = filter_beijing(codes, include=True)
        assert kept == codes
        assert removed == []

    def test_include_false(self):
        codes = ["600519", "833979", "000858"]
        kept, removed = filter_beijing(codes, include=False)
        assert "833979" not in kept
        assert "833979" in removed
        assert "600519" in kept
