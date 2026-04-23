"""标签生成器 — 为历史 run 的全量 scored stocks 生成未来收益标签.

标签定义 (DESIGN_v2.md §1.2):
  return_t1       T+1 日收益率 (%)
  mfe_3d          未来 3 日最大有利偏移 (%)
  mfe_5d          未来 5 日最大有利偏移 (%)
  mae_5d          未来 5 日最大不利偏移 (%)  (负数)
  hit_limit_up_3d 未来 3 日是否涨停 (bool)
  beat_index_5d   未来 5 日是否跑赢沪深300 (bool)

数据源: Tushare daily (120pt)
覆盖: 全量通过硬筛的股票 (不仅仅 pass_stage1)
"""

from __future__ import annotations

import csv
import logging
import os
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Sequence

import pandas as pd

logger = logging.getLogger("a_share_hot_screener.evaluation.label_generator")

# 沪深300 指数代码
_HS300_INDEX = "399300.SZ"  # 也可以用 000300.SH

# 涨停判定阈值 (主板/创业板通用近似: pct_chg >= 9.8%)
_LIMIT_UP_PCT = 9.8


@dataclass
class StockLabel:
    """一只股票在某个 run_date 的未来收益标签."""

    code: str
    run_date: str                  # YYYYMMDD or YYYY-MM-DD

    # 未来收益标签
    return_t1: Optional[float] = None         # T+1 收益 (%)
    mfe_3d: Optional[float] = None            # 3日 MFE (%)
    mfe_5d: Optional[float] = None            # 5日 MFE (%)
    mae_5d: Optional[float] = None            # 5日 MAE (%) (负数)
    hit_limit_up_3d: Optional[bool] = None    # 3日内是否涨停
    beat_index_5d: Optional[bool] = None      # 5日是否跑赢沪深300

    # ── Setup Timing 相关标签 (Round 3) ──
    touched_support_zone_5d: Optional[bool] = None    # 未来5日最低价是否触及支撑区间
    broke_invalidation_5d: Optional[bool] = None      # 未来5日最低价是否跌破失效位
    touched_resistance_5d: Optional[bool] = None      # 未来5日最高价是否触及压力位

    # 元数据
    close_on_run_date: Optional[float] = None # run_date 收盘价
    future_days_available: int = 0            # 实际可用的未来交易日数
    label_quality: str = "full"               # full / partial / missing


@dataclass
class LabelResult:
    """标签生成结果."""

    run_date: str
    labels: List[StockLabel] = field(default_factory=list)
    index_return_5d: Optional[float] = None   # 沪深300 5日收益
    total_requested: int = 0
    total_labeled: int = 0
    total_partial: int = 0
    total_missing: int = 0


class LabelGenerator:
    """为历史 run 的全量 scored stocks 生成未来收益标签.

    用法:
        gen = LabelGenerator(tushare_client, trade_calendar)
        result = gen.generate(run_date="20260418", stock_codes=["600519", ...])
        gen.save_csv(result, output_path)
    """

    def __init__(self, tushare_client, trade_calendar):
        """
        Args:
            tushare_client: TushareClient 实例 (需要 get_daily 方法)
            trade_calendar: TradeCalendar 实例 (需要 resolve / 日期工具)
        """
        self._client = tushare_client
        self._cal = trade_calendar

    def generate(
        self,
        run_date: str,
        stock_codes: Sequence[str],
        ts_code_map: Optional[Dict[str, str]] = None,
        timing_signals: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> LabelResult:
        """为一批股票生成标签.

        Args:
            run_date: 运行日期 (YYYYMMDD)
            stock_codes: 6位纯数字代码列表
            ts_code_map: code -> ts_code 映射 (如 {"600519": "600519.SH"})
                         如果不提供则自动推断
            timing_signals: code -> SetupSignal.to_dict() 映射
                            用于生成 timing 相关标签 (触及支撑/跌破失效/触及压力)

        Returns:
            LabelResult
        """
        run_date_norm = run_date.replace("-", "")
        result = LabelResult(run_date=run_date_norm, total_requested=len(stock_codes))

        # 获取 run_date 之后的交易日
        future_dates = self._get_future_trade_dates(run_date_norm, n=6)
        if not future_dates:
            logger.warning("No future trade dates after %s, cannot generate labels", run_date_norm)
            for code in stock_codes:
                label = StockLabel(code=code, run_date=run_date_norm, label_quality="missing")
                result.labels.append(label)
            result.total_missing = len(stock_codes)
            return result

        # 获取沪深300 指数 5 日收益
        result.index_return_5d = self._get_index_return(
            run_date_norm, future_dates, days=5
        )

        # 逐股生成标签
        for code in stock_codes:
            ts_code = self._resolve_ts_code(code, ts_code_map)
            timing_data = (timing_signals or {}).get(code)
            label = self._generate_single(
                code=code,
                ts_code=ts_code,
                run_date=run_date_norm,
                future_dates=future_dates,
                index_return_5d=result.index_return_5d,
                timing_signal=timing_data,
            )
            result.labels.append(label)

            if label.label_quality == "full":
                result.total_labeled += 1
            elif label.label_quality == "partial":
                result.total_partial += 1
            else:
                result.total_missing += 1

        logger.info(
            "[label_gen] run_date=%s: %d requested, %d full, %d partial, %d missing",
            run_date_norm,
            result.total_requested,
            result.total_labeled,
            result.total_partial,
            result.total_missing,
        )
        return result

    def _generate_single(
        self,
        code: str,
        ts_code: str,
        run_date: str,
        future_dates: List[str],
        index_return_5d: Optional[float],
        timing_signal: Optional[Dict[str, Any]] = None,
    ) -> StockLabel:
        """为单只股票生成标签."""
        label = StockLabel(code=code, run_date=run_date)

        # 获取 run_date 当日 + 未来 5 日行情
        all_dates = [run_date] + future_dates[:5]
        start_date = all_dates[0]
        end_date = all_dates[-1]

        df = self._client.get_daily(
            ts_code=ts_code,
            start_date=start_date,
            end_date=end_date,
            use_cache=True,
            cache_ttl=86400 * 30,  # 历史数据缓存 30 天
        )

        if df is None or df.empty:
            label.label_quality = "missing"
            return label

        # 确保按日期排序（升序）
        df = df.sort_values("trade_date").reset_index(drop=True)

        # 找到 run_date 当日行
        run_rows = df[df["trade_date"] == run_date]
        if run_rows.empty:
            label.label_quality = "missing"
            return label

        close_0 = float(run_rows.iloc[0]["close"])
        label.close_on_run_date = close_0

        # 未来行情（不含 run_date 当日）
        future_df = df[df["trade_date"] > run_date].sort_values("trade_date")
        label.future_days_available = len(future_df)

        if future_df.empty:
            label.label_quality = "missing"
            return label

        # T+1 收益
        if len(future_df) >= 1:
            close_t1 = float(future_df.iloc[0]["close"])
            label.return_t1 = (close_t1 / close_0 - 1) * 100

        # MFE/MAE 计算
        future_highs = future_df["high"].astype(float).values
        future_lows = future_df["low"].astype(float).values
        future_closes = future_df["close"].astype(float).values
        future_pct_chgs = future_df["pct_chg"].astype(float).values if "pct_chg" in future_df.columns else None

        # MFE 3d
        if len(future_highs) >= 3:
            max_high_3d = max(future_highs[:3])
            label.mfe_3d = (max_high_3d / close_0 - 1) * 100
        elif len(future_highs) >= 1:
            max_high = max(future_highs)
            label.mfe_3d = (max_high / close_0 - 1) * 100

        # MFE 5d
        if len(future_highs) >= 5:
            max_high_5d = max(future_highs[:5])
            label.mfe_5d = (max_high_5d / close_0 - 1) * 100
        elif len(future_highs) >= 1:
            max_high = max(future_highs)
            label.mfe_5d = (max_high / close_0 - 1) * 100

        # MAE 5d (负数，最大不利偏移)
        if len(future_lows) >= 5:
            min_low_5d = min(future_lows[:5])
            label.mae_5d = (min_low_5d / close_0 - 1) * 100
        elif len(future_lows) >= 1:
            min_low = min(future_lows)
            label.mae_5d = (min_low / close_0 - 1) * 100

        # hit_limit_up_3d
        if future_pct_chgs is not None and len(future_pct_chgs) >= 1:
            days_to_check = min(3, len(future_pct_chgs))
            label.hit_limit_up_3d = any(
                pct >= _LIMIT_UP_PCT for pct in future_pct_chgs[:days_to_check]
            )

        # beat_index_5d
        if len(future_closes) >= 5 and label.mfe_5d is not None:
            stock_return_5d = (float(future_closes[4]) / close_0 - 1) * 100
            if index_return_5d is not None:
                label.beat_index_5d = stock_return_5d > index_return_5d
        elif len(future_closes) >= 1:
            # 用实际可用天数
            stock_return = (float(future_closes[-1]) / close_0 - 1) * 100
            if index_return_5d is not None:
                label.beat_index_5d = stock_return > index_return_5d

        # ── Setup Timing 参考价位标签 (Round 3) ──
        if timing_signal and label.future_days_available >= 1:
            self._fill_timing_labels(label, future_lows, future_highs, timing_signal)

        # 标签质量
        if label.future_days_available >= 5:
            label.label_quality = "full"
        elif label.future_days_available >= 1:
            label.label_quality = "partial"
        else:
            label.label_quality = "missing"

        return label

    @staticmethod
    def _fill_timing_labels(
        label: StockLabel,
        future_lows,
        future_highs,
        timing_signal: Dict[str, Any],
    ) -> None:
        """根据 setup_timing 参考价位填充 timing 相关标签.

        Args:
            label: 待填充的 StockLabel
            future_lows: numpy array of future low prices
            future_highs: numpy array of future high prices
            timing_signal: SetupSignal.to_dict() 的输出
        """
        import numpy as np

        days = min(5, len(future_lows))
        if days < 1:
            return

        lows_5d = future_lows[:days]
        highs_5d = future_highs[:days]
        min_low = float(np.min(lows_5d))
        max_high = float(np.max(highs_5d))

        # 触及支撑区间: 最低价 <= support_zone_high
        sz_high = timing_signal.get("support_zone_high")
        if sz_high is not None:
            label.touched_support_zone_5d = min_low <= sz_high

        # 跌破失效位: 最低价 < invalidation_level
        inv = timing_signal.get("invalidation_level")
        if inv is not None:
            label.broke_invalidation_5d = min_low < inv

        # 触及压力位: 最高价 >= resistance_1
        res = timing_signal.get("resistance_1")
        if res is not None:
            label.touched_resistance_5d = max_high >= res

    def _get_future_trade_dates(self, run_date: str, n: int = 6) -> List[str]:
        """获取 run_date 之后的 n 个交易日 (YYYYMMDD).

        多返回 1 个以确保有足够的未来数据。
        """
        import datetime as dt_mod
        run_dt = dt_mod.datetime.strptime(run_date, "%Y%m%d").date()

        # 使用 trade_calendar 获取后续交易日
        dates = []
        current = run_dt
        attempts = 0
        max_attempts = n * 3 + 10  # 跳过周末和节假日的容错

        while len(dates) < n and attempts < max_attempts:
            current = current + dt_mod.timedelta(days=1)
            attempts += 1
            if self._cal.is_trade_date(current):
                dates.append(current.strftime("%Y%m%d"))

        return dates

    def _get_index_return(
        self, run_date: str, future_dates: List[str], days: int = 5
    ) -> Optional[float]:
        """获取沪深300指数 N 日收益率."""
        if len(future_dates) < days:
            return None

        end_date = future_dates[days - 1]
        df = self._client.get_daily(
            ts_code=_HS300_INDEX,
            start_date=run_date,
            end_date=end_date,
            use_cache=True,
            cache_ttl=86400 * 30,
        )
        if df is None or df.empty:
            # 沪深300 用 000300.SH 重试
            df = self._client.get_daily(
                ts_code="000300.SH",
                start_date=run_date,
                end_date=end_date,
                use_cache=True,
                cache_ttl=86400 * 30,
            )

        if df is None or df.empty:
            return None

        df = df.sort_values("trade_date")
        run_rows = df[df["trade_date"] == run_date]
        end_rows = df[df["trade_date"] == end_date]

        if run_rows.empty or end_rows.empty:
            return None

        close_0 = float(run_rows.iloc[0]["close"])
        close_n = float(end_rows.iloc[0]["close"])
        return (close_n / close_0 - 1) * 100

    def _resolve_ts_code(
        self, code: str, ts_code_map: Optional[Dict[str, str]]
    ) -> str:
        """将6位代码转为 Tushare ts_code."""
        if ts_code_map and code in ts_code_map:
            return ts_code_map[code]
        # 自动推断
        from a_share_hot_screener.ticker_mapping import code_to_tushare
        return code_to_tushare(code)

    @staticmethod
    def save_csv(result: LabelResult, output_path: str) -> str:
        """将标签结果保存为 CSV.

        Returns:
            实际写入的文件路径
        """
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

        fieldnames = [
            "code", "run_date", "return_t1", "mfe_3d", "mfe_5d",
            "mae_5d", "hit_limit_up_3d", "beat_index_5d",
            "touched_support_zone_5d", "broke_invalidation_5d",
            "touched_resistance_5d",
            "close_on_run_date", "future_days_available", "label_quality",
        ]

        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for label in result.labels:
                row = asdict(label)
                # 只写需要的字段
                filtered = {k: row.get(k, "") for k in fieldnames}
                # None → 空字符串
                for k, v in filtered.items():
                    if v is None:
                        filtered[k] = ""
                    elif isinstance(v, float):
                        filtered[k] = f"{v:.4f}"
                    elif isinstance(v, bool):
                        filtered[k] = str(v)
                writer.writerow(filtered)

        logger.info("[label_gen] Saved %d labels to %s", len(result.labels), output_path)
        return output_path

    @staticmethod
    def load_csv(csv_path: str) -> LabelResult:
        """从 CSV 加载标签结果."""
        labels = []
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                label = StockLabel(
                    code=row["code"],
                    run_date=row["run_date"],
                )
                for fld in ["return_t1", "mfe_3d", "mfe_5d", "mae_5d", "close_on_run_date"]:
                    val = row.get(fld, "")
                    setattr(label, fld, float(val) if val else None)

                for fld in ["hit_limit_up_3d", "beat_index_5d",
                            "touched_support_zone_5d", "broke_invalidation_5d",
                            "touched_resistance_5d"]:
                    val = row.get(fld, "")
                    setattr(label, fld, val == "True" if val else None)

                label.future_days_available = int(row.get("future_days_available", 0) or 0)
                label.label_quality = row.get("label_quality", "missing")
                labels.append(label)

        run_date = labels[0].run_date if labels else ""
        return LabelResult(
            run_date=run_date,
            labels=labels,
            total_requested=len(labels),
            total_labeled=sum(1 for l in labels if l.label_quality == "full"),
            total_partial=sum(1 for l in labels if l.label_quality == "partial"),
            total_missing=sum(1 for l in labels if l.label_quality == "missing"),
        )
