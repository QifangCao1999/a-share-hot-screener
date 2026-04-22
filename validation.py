"""股票池校验层 – Tushare daily_basic 全市场快照 + 代码映射.

校验流程：
  1. 通过 Tushare daily_basic(trade_date) 拉取全市场单日指标
     - 用于：universe 校验 + 批量收集 spot 字段（PE/PB/换手率/市值等）
  2. 降级备用：Tushare stock_basic() 仅做代码存在性校验（无 spot 字段）
  3. 逐代码校验：格式合法 → 在 universe 中存在 → ts_code 可映射
  4. 北交所：include_beijing=True 时保留但跳过 spot 校验，标记 warning
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd

from a_share_hot_screener.cache import LocalCache
from a_share_hot_screener.clients.tushare_client import TushareClient
from a_share_hot_screener.logger import WarningsCollector
from a_share_hot_screener.models import RejectedRecord, ValidatedHotStock
from a_share_hot_screener.ticker_mapping import code_to_tushare, infer_exchange

logger = logging.getLogger("a_share_hot_screener.validation")

_SPOT_CACHE_TTL = 4 * 3600      # daily_basic 全市场：4 小时
_UNIVERSE_CACHE_TTL = 7 * 86400  # stock_basic 代码列表：7 天


class SpotUniverse:
    """封装 Tushare daily_basic 全市场表的加载 + 索引.

    一次拉取 trade_date 全市场 daily_basic，建立 code → spot_row dict，
    供 StockValidator 做 universe 校验和 spot 字段批量收集。

    降级策略：
      Primary:   Tushare daily_basic(trade_date=YYYYMMDD)
      Fallback:  Tushare stock_basic()（只能做校验，无 spot 字段）
    """

    def __init__(
        self,
        tushare: TushareClient,
        cache: LocalCache,
        warnings: WarningsCollector,
    ) -> None:
        self._tushare = tushare
        self._cache = cache
        self._warnings = warnings
        self._index: Dict[str, Dict] = {}
        self._universe: Set[str] = set()
        self._loaded: bool = False
        self._fallback_mode: bool = False

    def load(self, run_date_str: str) -> bool:
        """加载全市场指标表（带缓存）.

        Args:
            run_date_str: 'YYYY-MM-DD'
        """
        trade_date = run_date_str.replace("-", "")
        cache_key = f"spot_universe_{trade_date}"
        cached = self._cache.get("spot_universe", cache_key, ttl=_SPOT_CACHE_TTL)
        if cached is not None:
            self._index = cached
            self._universe = set(cached.keys())
            self._loaded = True
            logger.info("从缓存加载 spot universe: %d 只", len(self._universe))
            return True

        # Tushare daily_basic 全市场
        df = self._tushare.get_daily_basic_by_date(trade_date)

        if df is not None and not df.empty:
            self._build_index(df)
            self._cache.put("spot_universe", cache_key, self._index, ttl=_SPOT_CACHE_TTL)
            self._loaded = True
            self._fallback_mode = False
            logger.info("spot universe (daily_basic) 加载完成: %d 只", len(self._universe))
            return True

        # 降级：使用 stock_basic
        logger.warning("Tushare daily_basic 获取失败，降级到 stock_basic（无 spot 字段）")
        self._warnings.add_global(
            "Tushare daily_basic 不可用，已降级为 stock_basic 校验（spot 字段将缺失）"
        )
        ok = self._load_stock_basic_fallback()
        if ok:
            self._loaded = True
            self._fallback_mode = True
        return ok

    def _build_index(self, df: pd.DataFrame) -> None:
        """从 daily_basic DataFrame 建立 code → 标准化 dict 索引.

        Tushare daily_basic 字段:
          ts_code, close, turnover_rate, volume_ratio, pe, pe_ttm, pb,
          total_mv(万元), circ_mv(万元), ...
        """
        for _, row in df.iterrows():
            ts_code = str(row.get("ts_code", ""))
            if "." not in ts_code:
                continue
            code = ts_code.split(".")[0]
            if not (len(code) == 6 and code.isdigit()):
                continue

            entry: Dict = {"code": code, "ts_code": ts_code}
            entry["name"] = ""  # daily_basic 不含 name，后续从 stock_basic 补充
            entry["latest_price"] = _safe_float(row.get("close"))
            entry["turnover_rate"] = _safe_float(row.get("turnover_rate"))
            entry["turnover_rate_f"] = _safe_float(row.get("turnover_rate_f"))  # P0-B: 自由流通换手率
            entry["volume_ratio"] = _safe_float(row.get("volume_ratio"))
            entry["pe_ttm"] = _safe_float(row.get("pe_ttm"))
            entry["pb"] = _safe_float(row.get("pb"))
            # 单位转换：万元 → 元
            total_mv = _safe_float(row.get("total_mv"))
            circ_mv = _safe_float(row.get("circ_mv"))
            entry["market_cap"] = total_mv * 10000 if total_mv is not None else None
            entry["float_market_cap"] = circ_mv * 10000 if circ_mv is not None else None

            self._index[code] = entry
        self._universe = set(self._index.keys())

        # 补充 name：从 stock_basic 拉取
        self._enrich_names()

    def _enrich_names(self) -> None:
        """从 stock_basic 补充股票名称."""
        basic_df = self._tushare.get_stock_basic()
        if basic_df is None or basic_df.empty:
            return
        for _, row in basic_df.iterrows():
            ts_code = str(row.get("ts_code", ""))
            code = ts_code.split(".")[0] if "." in ts_code else ""
            if code in self._index:
                self._index[code]["name"] = str(row.get("name", ""))
                if "industry" not in self._index[code]:
                    self._index[code]["industry"] = str(row.get("industry", ""))
                if "list_date" not in self._index[code]:
                    self._index[code]["list_date"] = str(row.get("list_date", ""))

    def _load_stock_basic_fallback(self) -> bool:
        """降级：从 stock_basic 构建 universe（无 spot 字段）."""
        df = self._tushare.get_stock_basic()
        if df is None or df.empty:
            return False
        codes: Set[str] = set()
        for _, row in df.iterrows():
            ts_code = str(row.get("ts_code", ""))
            code = ts_code.split(".")[0] if "." in ts_code else ""
            if len(code) == 6 and code.isdigit():
                codes.add(code)
                self._index[code] = {
                    "code": code,
                    "ts_code": ts_code,
                    "name": str(row.get("name", "")),
                    "industry": str(row.get("industry", "")),
                    "list_date": str(row.get("list_date", "")),
                }
        self._universe = codes
        logger.info("stock_basic 降级 universe: %d 只", len(codes))
        return bool(codes)

    def contains(self, code: str) -> bool:
        return code in self._universe

    def get_spot(self, code: str) -> Optional[Dict]:
        return self._index.get(code)

    @property
    def is_fallback(self) -> bool:
        return self._fallback_mode

    @property
    def loaded(self) -> bool:
        return self._loaded


class StockValidator:
    """股票代码校验器."""

    def __init__(
        self,
        spot_universe: SpotUniverse,
        warnings: WarningsCollector,
        include_beijing: bool = False,
    ) -> None:
        self._spot = spot_universe
        self._warnings = warnings
        self._include_beijing = include_beijing

    def validate(
        self,
        valid_codes: List[str],
        invalid_codes: List[str],
    ) -> Tuple[List[ValidatedHotStock], List[RejectedRecord]]:
        validated: List[ValidatedHotStock] = []
        rejected: List[RejectedRecord] = []

        for raw in invalid_codes:
            rejected.append(RejectedRecord(
                code=raw,
                reject_stage="code_parse",
                reject_reason="invalid_code_format",
                reject_detail=f"无法解析为6位数字代码: {raw!r}",
            ))

        for order, code in enumerate(valid_codes):
            exchange = infer_exchange(code)

            if exchange == "UNKNOWN":
                rejected.append(RejectedRecord(
                    code=code,
                    input_order=order,
                    reject_stage="validation",
                    reject_reason="unsupported_code",
                    reject_detail=f"无法识别交易所，前缀 {code[:3]!r} 不在已知范围",
                ))
                continue

            if exchange == "BJ":
                if not self._include_beijing:
                    rejected.append(RejectedRecord(
                        code=code,
                        input_order=order,
                        reject_stage="validation",
                        reject_reason="beijing_exchange_excluded",
                        reject_detail="北交所代码，include_beijing=False",
                    ))
                    continue
                else:
                    self._warnings.add(code, "北交所股票，跳过 spot universe 校验")
                    vs = ValidatedHotStock(
                        code=code,
                        exchange="BJ",
                        ts_code=code_to_tushare(code, "BJ") or "",
                        input_order=order,
                        validation_status="valid",
                    )
                    validated.append(vs)
                    continue

            if self._spot.loaded and not self._spot.contains(code):
                rejected.append(RejectedRecord(
                    code=code,
                    input_order=order,
                    reject_stage="validation",
                    reject_reason="not_in_a_share_spot_universe",
                    reject_detail=(
                        f"{code} 不在 Tushare daily_basic 全市场列表中"
                        f"（{'降级stock_basic' if self._spot.is_fallback else 'daily_basic'}）"
                    ),
                ))
                continue
            elif not self._spot.loaded:
                self._warnings.add(code, "spot universe 未加载，跳过 universe 校验")

            ts_code = code_to_tushare(code, exchange)
            if ts_code is None:
                rejected.append(RejectedRecord(
                    code=code,
                    input_order=order,
                    reject_stage="validation",
                    reject_reason="ticker_mapping_failed",
                    reject_detail=f"无法映射 Tushare ts_code: code={code}, exchange={exchange}",
                ))
                continue

            spot = self._spot.get_spot(code) if self._spot.loaded else {}
            spot = spot or {}

            vs = ValidatedHotStock(
                code=code,
                name=spot.get("name", ""),
                exchange=exchange,
                ts_code=ts_code,
                input_order=order,
                validation_status="valid",
                latest_price=spot.get("latest_price"),
                turnover_rate=spot.get("turnover_rate"),
                turnover_rate_f=spot.get("turnover_rate_f"),  # P0-B
                pe_ttm=spot.get("pe_ttm"),
                pb=spot.get("pb"),
                market_cap=spot.get("market_cap"),
                float_market_cap=spot.get("float_market_cap"),
                volume_ratio=spot.get("volume_ratio"),
            )
            validated.append(vs)

        logger.info(
            "校验完成: 通过=%d, 淘汰=%d (格式非法=%d)",
            len(validated), len(rejected), len(invalid_codes),
        )
        return validated, rejected


def _safe_float(val) -> Optional[float]:
    if val is None:
        return None
    try:
        import math
        f = float(val)
        return None if math.isnan(f) else f
    except (ValueError, TypeError):
        return None
