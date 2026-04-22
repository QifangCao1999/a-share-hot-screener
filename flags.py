"""Structured Flags 模块（Session 6）.

将 HotStockDetail 中的关键字段汇聚为一个结构化的 flags dict，
供 summary 输出和第二阶段消费。

设计原则：
  1. flags 不参与 total_score 计算（仅供信息展示和第二阶段参考）
  2. flags 必须进入 summary 输出
  3. 尊重模块开关（enable_lhb_module / enable_unlock_risk_module /
     enable_concept_heat_module）
  4. 所有事件字段严格遵守 run_date as-of 口径（已由上游保证）
  5. 接口不稳定时允许 None + warnings，不编造

Flag 字段速查（共 20 项）：
  来自 event_layer：
    limit_up_count_5d              近5日涨停池入选次数
    limit_up_count_10d             近10日涨停池入选次数
    max_consecutive_limit_up_10d   近10日最大连续涨停天数
    strong_pool_entry_count_3d     近3日强势股池入选次数
    lhb_count_20d                  近20日龙虎榜上榜次数（enable_lhb=True）
  来自 price_features：
    one_word_limit_up_latest       最新日疑似一字涨停（proxy）
    one_word_limit_down_latest     最新日疑似一字跌停（proxy）
    amount_avg_5d                  近5日均成交额（tushare_amount，元）
    amp_norm_avg_5d                近5日均振幅（%）
    upper_shadow_count_5d          近5日上影线 proxy 次数
  来自换手率推算：
    turnover_avg_5d                近5日均换手率推算（%，approx）
  来自 event_layer（概念热度）：
    industry_heat_pctile_5d        行业近5日强度分位（0~1，enable时）
    concept_heat_pctile_5d         概念近5日强度分位（0~1，enable时）
  来自基础信息：
    new_stock_flag                 次新股标记（上市 < 120 天）
  模块占位（当前 Session 不实现，设为 None）：
    shareholder_net_reduction_ratio_3m   重要股东3个月净减持比例
    shareholder_reduction_flag_3m        减持预警标记
    restricted_shares_unlock_ratio_20d   未来20日解禁占流通盘比例
    unlock_risk_flag_20d                 解禁风险标记（enable_unlock_risk_module）
    pledge_ratio_latest                  质押比例
    pledge_ratio_flag                    质押风险标记

计算时机：
  在评分（_apply_four_axis_scores）完成后，pipeline 主线程统一调用
  compute_flags(detail, config) → 写入 detail.flags

注意：
  flags dict 中的 key 即为 summary 中对应的列名，
  OutputWriter 会将 detail.flags 展开为 summary 中的独立字段。
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, Optional

if TYPE_CHECKING:
    from a_share_hot_screener.models import HotStockDetail
    from a_share_hot_screener.config import HotScreenerConfig

from a_share_hot_screener.limit_rules import infer_limit_pct

logger = logging.getLogger("a_share_hot_screener.flags")

# 次新股阈值（日历天数）
_NEW_STOCK_DAYS_THRESHOLD = 120


def compute_flags(
    detail: "HotStockDetail",
    enable_lhb_module: bool = True,
    enable_unlock_risk_module: bool = False,
    enable_concept_heat_module: bool = False,
) -> Dict[str, Any]:
    """计算 structured flags 并返回 dict.

    Args:
        detail:                    已填充全量字段的 HotStockDetail
        enable_lhb_module:         是否启用龙虎榜模块
        enable_unlock_risk_module: 是否启用解禁风险模块（当前 session 未实现，始终返回 None）
        enable_concept_heat_module: 是否启用概念热度模块

    Returns:
        flags dict，key 为标准字段名
    """
    flags: Dict[str, Any] = {}

    # ── 事件层 flags（涨停/强势股池）──────────────────────
    flags["limit_up_count_5d"] = detail.limit_up_count_5d
    flags["limit_up_count_10d"] = detail.limit_up_count_10d
    flags["max_consecutive_limit_up_10d"] = detail.max_consecutive_limit_up_10d
    flags["strong_pool_entry_count_3d"] = detail.strong_pool_entry_count_3d

    # ── 龙虎榜（尊重模块开关）────────────────────────────
    if enable_lhb_module:
        flags["lhb_count_20d"] = detail.lhb_count_20d
    else:
        flags["lhb_count_20d"] = None   # 模块未启用，不展示（None 表示不可得）

    # ── 价格行为 flags（来自 price_features）─────────────
    flags["one_word_limit_up_latest"] = _detect_one_word_limit_up(detail)
    flags["one_word_limit_down_latest"] = _detect_one_word_limit_down(detail)
    flags["amount_avg_5d"] = detail.amount_avg_5d
    flags["amp_norm_avg_5d"] = detail.amp_norm_avg_5d
    flags["upper_shadow_count_5d"] = detail.upper_shadow_count_5d

    # ── 换手率推算 ────────────────────────────────────────
    flags["turnover_avg_5d"] = _calc_turnover_approx(detail)

    # ── 行业/概念热度 ─────────────────────────────────────
    flags["industry_heat_pctile_5d"] = detail.industry_heat_pctile_5d
    if enable_concept_heat_module:
        flags["concept_heat_pctile_5d"] = detail.concept_heat_pctile_5d
    else:
        flags["concept_heat_pctile_5d"] = None

    # ── 次新股标记 ────────────────────────────────────────
    flags["new_stock_flag"] = _is_new_stock(detail)

    # ── 减持/质押/解禁模块（保留 pipeline 已写入 detail.flags 的值，无则 fallback None）──
    # pipeline Step 6.6/6.7/6.8 已将结果写入 detail.flags，此处保留而非覆盖
    flags["shareholder_net_reduction_ratio_3m"] = detail.flags.get("shareholder_net_reduction_ratio_3m")
    flags["shareholder_reduction_flag_3m"] = detail.flags.get("shareholder_reduction_flag_3m")
    flags["restricted_shares_unlock_ratio_20d"] = detail.flags.get("restricted_shares_unlock_ratio_20d")
    flags["unlock_risk_flag_20d"] = detail.flags.get("unlock_risk_flag_20d")
    flags["pledge_ratio_latest"] = detail.flags.get("pledge_ratio_latest")
    flags["pledge_ratio_flag"] = detail.flags.get("pledge_ratio_flag")

    # ── Session 22 新增：资金流向/股东增减持/融资融券/板块轮动 ────
    flags["net_main_inflow_ratio_5d"] = detail.flags.get("net_main_inflow_ratio_5d")
    flags["net_holder_reduction_ratio_30d"] = detail.flags.get("net_holder_reduction_ratio_30d")
    flags["margin_buy_net_ratio_5d"] = detail.flags.get("margin_buy_net_ratio_5d")
    flags["short_sell_ratio_change_5d"] = detail.flags.get("short_sell_ratio_change_5d")
    flags["is_margin_eligible"] = detail.flags.get("is_margin_eligible")
    flags["sector_momentum_signal"] = detail.flags.get("sector_momentum_signal")

    # ── Phase 3: Context Scores (HT8/HT9/HT10 experimental) ────
    _cs = detail.context_scores
    if _cs:
        flags["ht8_score"] = _cs.get("ht8_score")
        flags["ht8_confirmation_level"] = _cs.get("ht8_confirmation_level")
        flags["ht9_score"] = _cs.get("ht9_score")
        flags["ht9_breadth_ratio"] = _cs.get("ht9_breadth_ratio")
        flags["ht9_sector_name"] = _cs.get("ht9_sector_name")
        flags["ht10_score"] = _cs.get("ht10_score")
        flags["ht10_position_type"] = _cs.get("ht10_position_type")
        flags["ht10_confidence"] = _cs.get("ht10_confidence")

    return flags


# ── 内部工具 ─────────────────────────────────────────────

def _detect_one_word_limit_up(detail: "HotStockDetail") -> Optional[bool]:
    """检测最新交易日是否为疑似一字涨停（proxy）.

    条件：
      - limit_board_count_5d >= 1（近5日有一字板 proxy）AND
      - clv_latest 接近 0.5（一字板时 high==low → CLV=0.5）AND
      - pct_change_1d 接近涨停幅度
    若数据不足以判断，返回 None。

    NOTE: 这是 proxy，无法 100% 精确区分涨停一字板与停牌日。
    """
    if detail.limit_board_count_5d is None:
        return None
    if detail.clv_latest is None:
        return None
    if detail.pct_change_1d is None:
        return None

    limit_pct = _infer_limit_pct_from_code(detail.code)
    # 一字涨停条件：当日涨幅接近涨停幅度（±1.5%）且 CLV 接近 0.5（价格区间极窄）
    is_near_limit = abs(detail.pct_change_1d - limit_pct) < 1.5
    is_flat_candle = abs(detail.clv_latest - 0.5) < 0.02  # high ≈ low

    return bool(is_near_limit and is_flat_candle and detail.limit_board_count_5d >= 1)


def _detect_one_word_limit_down(detail: "HotStockDetail") -> Optional[bool]:
    """检测最新交易日是否为疑似一字跌停（proxy）.

    条件：
      - pct_change_1d 接近跌停幅度（负值）AND
      - clv_latest 接近 0.5（价格区间极窄）
    """
    if detail.clv_latest is None:
        return None
    if detail.pct_change_1d is None:
        return None

    limit_pct = _infer_limit_pct_from_code(detail.code)
    is_near_limit_down = abs(detail.pct_change_1d + limit_pct) < 1.5
    is_flat_candle = abs(detail.clv_latest - 0.5) < 0.02

    return bool(is_near_limit_down and is_flat_candle)


def _calc_turnover_approx(detail: "HotStockDetail") -> Optional[float]:
    """推算近5日均换手率（%）= amount_avg_5d / float_market_cap * 100."""
    amt = detail.amount_avg_5d
    fmc = detail.float_market_cap
    if amt is None or fmc is None or fmc <= 0:
        return None
    return round(amt / fmc * 100.0, 4)


def _is_new_stock(detail: "HotStockDetail") -> Optional[bool]:
    """判断是否为次新股（上市 < 120 天）."""
    if detail.listing_days is None:
        return None
    return detail.listing_days < _NEW_STOCK_DAYS_THRESHOLD


# NOTE: _infer_limit_pct_from_code 已统一迁移至 limit_rules.infer_limit_pct
#       此处保留别名供内部调用（向后兼容）。
_infer_limit_pct_from_code = infer_limit_pct
