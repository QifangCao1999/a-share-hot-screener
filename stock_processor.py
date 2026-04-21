"""单股处理模块 — 从 pipeline.py 提取（重构，不改业务逻辑）.

负责单只股票从 ValidatedHotStock → HotStockDetail 的完整处理流程：
  - Tushare daily 拉取 + price_features 计算
  - spot 基础信息解析
  - listing_days 计算
  - hard_filters 应用
  - 事件层字段填充
  - 股东减持 / 质押 / 限售解禁检测
"""

from __future__ import annotations

import datetime as dt
import logging
from typing import Dict, List, Optional

from a_share_hot_screener.clients.tushare_client import TushareClient
from a_share_hot_screener.config import HotScreenerConfig
from a_share_hot_screener.event_layer import EventLayerProcessor, EventLayerResult
from a_share_hot_screener.hard_filters import apply_hard_filters
from a_share_hot_screener.models import HotStockDetail, ValidatedHotStock
from a_share_hot_screener.pledge_ratio import compute_pledge_ratio
from a_share_hot_screener.price_features import PriceFeatures, compute_price_features
from a_share_hot_screener.restricted_unlock import compute_restricted_unlock
from a_share_hot_screener.shareholder_reduction import compute_shareholder_reduction
from a_share_hot_screener.trade_calendar import last_n_trade_dates
from a_share_hot_screener.validation import SpotUniverse

logger = logging.getLogger("a_share_hot_screener.stock_processor")


# ════════════════════════════════════════════════════════
# 工具函数
# ════════════════════════════════════════════════════════

def _safe_float(val) -> Optional[float]:
    if val is None:
        return None
    try:
        f = float(val)
        return None if (f != f) else f
    except (ValueError, TypeError):
        return None


# ════════════════════════════════════════════════════════
# Detail 字段填充 — 从各阶段中间结果写入 HotStockDetail
# ════════════════════════════════════════════════════════

def apply_price_features(detail: HotStockDetail, feat: PriceFeatures) -> None:
    """将 PriceFeatures 写入 HotStockDetail."""
    detail.latest_eod_date = feat.latest_date
    detail.latest_volume = feat.latest_volume
    detail.latest_amount_approx = feat.latest_amount
    detail.return_3d = feat.return_3d
    detail.return_5d = feat.return_5d
    detail.return_10d = feat.return_10d
    detail.amount_avg_5d = feat.amount_avg_5d
    detail.amount_avg_20d = feat.amount_avg_20d
    detail.volume_ratio_20d = feat.volume_ratio_20d
    detail.close_position_20d = feat.close_position_20d
    detail.clv_latest = feat.clv_latest
    detail.amount_ratio_5d_to_20d = feat.amount_ratio_5d_to_20d
    detail.abs_distance_to_ma10 = feat.abs_distance_to_ma10
    detail.amp_norm_avg_5d = feat.amp_norm_avg_5d
    detail.upper_shadow_count_5d = feat.upper_shadow_count_5d
    detail.limit_board_count_5d = feat.limit_board_count_5d
    detail.eod_data_rows = feat.data_rows

    # 均线字段（Session 8 P1 新增）
    detail.ma5 = feat.ma5
    detail.ma10 = feat.ma10
    detail.ma20 = feat.ma20

    # 最新日一字板判定（Session 8 P1 新增）
    detail.latest_is_limit_board = feat.latest_is_limit_board
    detail.latest_pct_change = feat.latest_pct_change

    # 大涨天数 + 连续上涨天数（Session 11 新增）
    detail.big_up_count_10d = feat.big_up_count_10d
    detail.consec_up_days = feat.consec_up_days

    # 布尔趋势特征（基于均线）
    if feat.latest_close and feat.ma5 is not None:
        detail.above_ma5 = feat.latest_close > feat.ma5
    if feat.latest_close and feat.ma10 is not None:
        detail.above_ma10 = feat.latest_close > feat.ma10
    if feat.latest_close and feat.ma20 is not None:
        detail.above_ma20 = feat.latest_close > feat.ma20


def parse_spot_info(
    detail: HotStockDetail,
    spot: Dict,
    warns: List[str],
) -> None:
    """从 SpotUniverse 的 spot dict 解析基础字段."""
    try:
        if not detail.name:
            detail.name = spot.get("name", "")
        if not detail.industry:
            detail.industry = spot.get("industry", "")

        ipo_raw = spot.get("list_date", "")
        if ipo_raw and ipo_raw not in ("-", "None", ""):
            detail.ipo_date = str(ipo_raw).replace("-", "")

        if detail.market_cap is None:
            detail.market_cap = _safe_float(spot.get("market_cap"))
        if detail.float_market_cap is None:
            detail.float_market_cap = _safe_float(spot.get("float_market_cap"))
        if detail.pe_ttm is None:
            detail.pe_ttm = _safe_float(spot.get("pe_ttm"))
        if detail.pb is None:
            detail.pb = _safe_float(spot.get("pb"))

    except Exception as e:
        logger.warning("parse_spot_info(%s) 失败: %s", detail.code, e)
        warns.append(f"spot_info 解析异常: {e}")


def apply_event_layer(
    detail: HotStockDetail,
    ev: EventLayerResult,
) -> None:
    """将 EventLayerResult 写入 HotStockDetail."""
    detail.limit_up_count_5d = ev.limit_up_count_5d
    detail.limit_up_count_10d = ev.limit_up_count_10d
    detail.max_consecutive_limit_up_10d = ev.max_consecutive_limit_up_10d
    detail.limit_up_source = ev.limit_up_source
    detail.strong_pool_entry_count_3d = ev.strong_pool_entry_count_3d
    detail.strong_pool_source = ev.strong_pool_source
    detail.lhb_count_20d = ev.lhb_count_20d
    detail.lhb_on_board = ev.lhb_on_board
    detail.lhb_source = ev.lhb_source
    detail.industry_heat_pctile_5d = ev.industry_heat_pctile_5d
    detail.industry_pct_5d = ev.industry_pct_5d
    detail.industry_heat_source = ev.industry_heat_source
    detail.concept_heat_pctile_5d = ev.concept_heat_pctile_5d
    detail.concept_names = ev.concept_names
    detail.advanced_concept_module_available = ev.advanced_concept_module_available
    detail.concept_heat_source = ev.concept_heat_source


# ════════════════════════════════════════════════════════
# 单股处理主函数
# ════════════════════════════════════════════════════════

def process_single_stock(
    vs: ValidatedHotStock,
    trade_date_str: str,
    config: HotScreenerConfig,
    tushare_client: TushareClient,
    spot_universe: SpotUniverse,
    trade_cal,  # TradeCalendar instance
    event_proc: Optional[EventLayerProcessor] = None,
) -> HotStockDetail:
    """处理单只股票的完整流程（Step 3~6.8）.

    返回填充了 price_features / hard_filter / event_layer / flags 的 HotStockDetail。
    评分（Step 7+）不在此函数中执行（需要横截面 pool）。
    """
    code = vs.code
    warns: List[str] = []

    detail = HotStockDetail(
        code=code,
        name=vs.name,
        exchange=vs.exchange,
        ts_code=vs.ts_code,
        input_order=vs.input_order,
        # spot 字段（来自 Tushare daily_basic）
        latest_price=vs.latest_price,
        market_cap=vs.market_cap,
        float_market_cap=vs.float_market_cap,
        pe_ttm=vs.pe_ttm,
        pb=vs.pb,
        turnover_rate_1d=vs.turnover_rate,
        volume_ratio=vs.volume_ratio,
    )

    try:
        # Step 3a: Tushare daily 日线历史
        eod_data = None
        ts_code = vs.ts_code
        if ts_code:
            try:
                eod_start = trade_cal.eod_start_date(
                    dt.date.fromisoformat(trade_date_str),
                    window=60, extra=10,
                )
            except Exception:
                eod_start = dt.date.fromisoformat(trade_date_str) - dt.timedelta(days=100)

            daily_df = tushare_client.get_daily(
                ts_code=ts_code,
                start_date=eod_start.strftime("%Y%m%d"),
                end_date=trade_date_str.replace("-", ""),
            )
            if daily_df is not None and not daily_df.empty:
                eod_data = daily_df.to_dict("records")
            else:
                warns.append("Tushare daily 日线历史获取失败")

        # Step 3b: 日线派生特征
        if eod_data:
            feat = compute_price_features(
                raw_eod=eod_data,
                trade_date_used_str=trade_date_str,
                warnings=warns,
                code=code,
            )
            apply_price_features(detail, feat)
            if detail.latest_price is None and feat.latest_close:
                detail.latest_price = feat.latest_close
                warns.append("latest_price 来自 Tushare daily（daily_basic 缺失）")

        # Step 4: 个股基础信息（从 spot_universe 中获取，已包含 stock_basic 数据）
        spot_data = spot_universe.get_spot(code) if spot_universe.loaded else None
        if spot_data:
            parse_spot_info(detail, spot_data, warns)
        else:
            warns.append("spot 数据不可用，个股基础信息缺失")

        # Step 5: listing_days（从 ipo_date 计算日历天数）
        if detail.ipo_date:
            try:
                ipo = dt.date(
                    int(detail.ipo_date[:4]),
                    int(detail.ipo_date[4:6]),
                    int(detail.ipo_date[6:8]),
                )
                detail.listing_days = (dt.date.fromisoformat(trade_date_str) - ipo).days
            except Exception:
                warns.append(f"ipo_date 解析失败: {detail.ipo_date!r}")

        # Step 6: hard filters
        hf = apply_hard_filters(
            config=config,
            name=detail.name,
            industry=detail.industry or None,
            ipo_date=detail.ipo_date,
            listing_days=detail.listing_days,
            latest_price=detail.latest_price,
            latest_volume=detail.latest_volume,
            amount_1d=detail.amount_1d,
            amount_avg_5d=detail.amount_avg_5d,
            float_market_cap=detail.float_market_cap,
            daily_row_count=len(eod_data) if eod_data else None,  # #1: IPODate fallback
        )
        detail.passed_hard_filter = hf.passed
        detail.hard_fail_reasons = hf.fail_reasons
        detail.hard_filter_reason = hf.primary_reason
        detail.hard_filter_warnings = hf.data_warnings
        warns.extend(hf.data_warnings)

        # Step 6.5: 事件层字段填充
        if event_proc is not None:
            trade_date_dt = dt.date.fromisoformat(trade_date_str)
            td_sorted = trade_cal._trade_dates
            td_10d = last_n_trade_dates(td_sorted, trade_date_dt, 10)
            td_3d = td_10d[-3:] if len(td_10d) >= 3 else td_10d
            td_20d = last_n_trade_dates(td_sorted, trade_date_dt, 20)
            ev_result = event_proc.process(
                code=code,
                industry=detail.industry or "",
                run_date=trade_date_dt,
                trade_dates_10d=td_10d,
                trade_dates_3d=td_3d,
                trade_dates_20d=td_20d,
                limit_board_count_5d_proxy=detail.limit_board_count_5d,
            )
            apply_event_layer(detail, ev_result)
            warns.extend(ev_result.warnings)

        # Step 6.6: 股东减持检测
        try:
            sr_result = compute_shareholder_reduction(
                code=code,
                tushare_client=tushare_client,
                run_date=dt.date.fromisoformat(trade_date_str),
                ts_code=ts_code,
                warnings=warns,
            )
            detail.flags["shareholder_net_reduction_ratio_3m"] = sr_result.shareholder_net_reduction_ratio_3m
            detail.flags["shareholder_reduction_flag_3m"] = sr_result.shareholder_reduction_flag_3m
        except Exception as e:
            logger.warning("shareholder_reduction(%s) 异常: %s", code, e)
            warns.append(f"[shareholder_reduction] 异常: {e}")

        # Step 6.7: 质押比例检测
        try:
            pr_result = compute_pledge_ratio(
                code=code,
                tushare_client=tushare_client,
                run_date=dt.date.fromisoformat(trade_date_str),
                ts_code=ts_code,
                warnings=warns,
            )
            detail.flags["pledge_ratio_latest"] = pr_result.pledge_ratio_latest
            detail.flags["pledge_ratio_flag"] = pr_result.pledge_ratio_flag
        except Exception as e:
            logger.warning("pledge_ratio(%s) 异常: %s", code, e)
            warns.append(f"[pledge_ratio] 异常: {e}")

        # Step 6.8: 限售解禁检测
        try:
            ru_result = compute_restricted_unlock(
                code=code,
                tushare_client=tushare_client,
                run_date=dt.date.fromisoformat(trade_date_str),
                ts_code=ts_code,
                warnings=warns,
            )
            detail.flags["restricted_shares_unlock_ratio_20d"] = ru_result.restricted_shares_unlock_ratio_20d
            detail.flags["unlock_risk_flag_20d"] = ru_result.unlock_risk_flag_20d
        except Exception as e:
            logger.warning("restricted_unlock(%s) 异常: %s", code, e)
            warns.append(f"[restricted_unlock] 异常: {e}")

        # 评分由 pipeline 在并发完成后统一执行
        detail.pass_stage1 = False

    except Exception as e:
        logger.error("process_single_stock(%s) 异常: %s", code, e, exc_info=True)
        warns.append(f"处理异常: {e}")

    detail.warnings = warns
    return detail
