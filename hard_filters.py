"""硬性筛除规则（Session 3 完整实现）.

规则列表（全部检查，不短路，所有失败原因都汇总到 hard_fail_reasons）：

  H1: ST / *ST / 退市整理板标记   → reason=st_stock
  H2: 退市状态（名称/行业含退市关键词 或 listing_days=0 的异常）→ reason=delisted
  H3: 上市天数不足 min_trading_days → reason=listing_too_young
  H4: 最新收盘价 < min_price       → reason=price_too_low
  H5: 停牌 / 无量（amount_1d=0 或 latest_volume=0 且 listing_days>30）→ reason=suspended_or_no_volume
  H6: 近5日均成交额 < min_amount_avg_5d → reason=amount_too_low
  H7: 流通市值 < min_float_market_cap   → reason=float_mc_too_small
  H8: IPODate 缺失                       → reason=ipo_date_missing（hard_fail）
  H9: 金融行业（银行/证券/保险/信托）    → reason=industry_finance

边界说明（rejected_before_scoring）：
  以下情况在 pipeline 中更早被拒，不进入 apply_hard_filters：
    - code_parse 阶段：代码格式非法（invalid_code_format）
    - validation 阶段：不在 spot universe / ticker 映射失败 / 北交所被排除

  以下情况在 apply_hard_filters 中被拒（hard_filter 阶段）：
    - H1~H9 任意一条命中
    - 多条同时命中时，全部记录到 hard_fail_reasons，primary_reason 取第一条

数据容错原则：
  - 核心字段（latest_price / amount_avg_5d / float_market_cap）缺失 → hard_fail
    这些是价格/流动性/市值的基本判据，缺失意味着无法可靠筛选
  - 非核心字段（listing_days / volume+amount_1d / industry）缺失 → warning，不强制淘汰
  - 若 ipo_date 缺失 → H8 强制 hard_fail（IPO 日期是核心防护字段）
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import List, Optional

from a_share_hot_screener.config import HotScreenerConfig

logger = logging.getLogger("a_share_hot_screener.hard_filters")

# ── 行业关键词 ────────────────────────────────────────────
_FINANCE_KEYWORDS = {
    "银行", "证券", "保险", "信托", "期货", "基金",
    "股权投资", "资产管理", "货币", "征信", "典当",
}

# ── ST / 退市名称关键词 ───────────────────────────────────
_ST_PATTERNS = re.compile(r"^\*?ST|^ST|^退市", re.IGNORECASE)
_DELIST_KEYWORDS = {"退市整理", "退市", "摘牌"}


# ════════════════════════════════════════════════════════
# 结果容器
# ════════════════════════════════════════════════════════

@dataclass
class HardFilterResult:
    """单只股票的硬筛结果."""

    passed: bool
    fail_reasons: List[str] = field(default_factory=list)
    data_warnings: List[str] = field(default_factory=list)  # 数据缺失但未强制淘汰的提示

    @property
    def primary_reason(self) -> str:
        return self.fail_reasons[0] if self.fail_reasons else ""


# ════════════════════════════════════════════════════════
# 主入口
# ════════════════════════════════════════════════════════

def apply_hard_filters(
    *,
    config: HotScreenerConfig,
    name: str = "",
    industry: Optional[str] = None,
    ipo_date: Optional[str] = None,       # 'YYYYMMDD' 或 None
    listing_days: Optional[int] = None,   # 已上市日历天数（近似）
    latest_price: Optional[float] = None,
    latest_volume: Optional[float] = None,
    amount_1d: Optional[float] = None,    # 当日成交额（spot 字段，元）
    amount_avg_5d: Optional[float] = None,  # 近5日均成交额（price_features，NOTE:approx）
    float_market_cap: Optional[float] = None,
    amount_tolerance_pct: Optional[float] = None,  # Session 11: 成交额容差%，优先使用 config 值
) -> HardFilterResult:
    """对单只股票应用所有硬筛规则（H1-H9），返回 HardFilterResult.

    所有规则不短路，全部检查后汇总。

    参数均为 keyword-only，从 pipeline 注入，不依赖 HotStockDetail 直接传入
    （避免 detail 在管道中间态引入不一致）。
    """
    fails: List[str] = []
    warns: List[str] = []

    # H1: ST / *ST / 退市整理板
    r = _check_st(name)
    if r:
        fails.append(r)

    # H2: 退市状态
    r = _check_delisted(name, industry)
    if r:
        fails.append(r)

    # H3: 上市天数
    r = _check_listing_days(listing_days, config.min_trading_days)
    if r == "_warn_":
        warns.append("listing_days 缺失，跳过上市天数检查（H3）")
    elif r:
        fails.append(r)

    # H4: 最低价格（核心字段：缺失 → hard_fail）
    r = _check_min_price(latest_price, config.min_price)
    if r:
        fails.append(r)

    # H5: 停牌 / 无量
    r = _check_suspended(
        latest_volume=latest_volume,
        amount_1d=amount_1d,
        listing_days=listing_days,
    )
    if r == "_warn_":
        warns.append("volume/amount 缺失，跳过停牌检查（H5）")
    elif r:
        fails.append(r)

    # H6: 近5日均成交额（核心字段：缺失 → hard_fail；Session 11: 支持容差）
    tol_pct = amount_tolerance_pct if amount_tolerance_pct is not None else config.amount_tolerance_pct
    r = _check_amount_avg_5d(amount_avg_5d, config.min_amount_avg_5d, tolerance_pct=tol_pct)
    if r:
        fails.append(r)

    # H7: 流通市值（核心字段：缺失 → hard_fail）
    r = _check_float_market_cap(float_market_cap, config.min_float_market_cap)
    if r:
        fails.append(r)

    # H8: IPODate 缺失（强制 hard_fail，不走 warn 分支）
    r = _check_ipo_date(ipo_date)
    if r:
        fails.append(r)

    # H9: 金融行业（include_finance=True 时跳过此检查）
    if not config.include_finance:
        r = _check_industry(industry)
        if r == "_warn_":
            warns.append("industry 缺失，跳过金融行业检查（H9）")
        elif r:
            fails.append(r)
    else:
        warns.append("include_finance=True，跳过金融行业硬筛（H9）")

    return HardFilterResult(
        passed=(len(fails) == 0),
        fail_reasons=fails,
        data_warnings=warns,
    )


# ════════════════════════════════════════════════════════
# 各规则独立实现（返回 reason str / None / "_warn_"）
# ════════════════════════════════════════════════════════

def _check_st(name: str) -> Optional[str]:
    """H1: ST/*ST/退市整理板.

    检测逻辑：股票简称中含 ST、*ST 或 退市整理。
    数据来源：Tushare daily_basic 表 name 字段。
    """
    if not name:
        return None   # 无法判断，不淘汰（降级兼容）
    clean = name.strip()
    # 匹配 *ST xxx / ST xxx（大小写不敏感）
    if _ST_PATTERNS.match(clean):
        return f"st_stock: 名称含 ST/*ST 标记（{clean!r}）"
    # 退市整理板（名称开头含"退市整理"）
    for kw in _DELIST_KEYWORDS:
        if clean.startswith(kw):
            return f"st_stock: 名称含退市整理标记（{clean!r}）"
    return None


def _check_delisted(
    name: str, industry: Optional[str]
) -> Optional[str]:
    """H2: 退市状态.

    退市股通常名称含"退市"二字，或行业被标记为退市类。
    此为 proxy 检查，后续 session 可接入更精确的状态字段。
    """
    if name and "退市" in name and not name.startswith("退市整理"):
        # 已经被 H1 处理过退市整理，此处处理"已退市"标记
        return f"delisted: 名称含退市标记（{name!r}）"
    return None


def _check_listing_days(
    listing_days: Optional[int], min_days: int
) -> Optional[str]:
    """H3: 上市天数不足.

    listing_days 是日历天数（非交易日），比 min_trading_days 宽松约 40%。
    以 min_trading_days * 1.5 作为日历天数等效下限（保守估计）。
    """
    if listing_days is None:
        return "_warn_"
    # min_trading_days 个交易日 ≈ min_trading_days * 1.5 个日历天（节假日系数）
    min_calendar = int(min_days * 1.5)
    if listing_days < min_calendar:
        return (
            f"listing_too_young: 已上市约 {listing_days} 天"
            f"（要求 ≥ {min_calendar} 日历天，对应约 {min_days} 个交易日）"
        )
    return None


def _check_min_price(
    price: Optional[float], min_price: float
) -> Optional[str]:
    """H4: 最新收盘价低于下限.

    核心字段：price 缺失直接 hard_fail（无法判断价格是否达标）。
    """
    if price is None:
        return "insufficient_core_data: latest_price 缺失，无法判断价格（H4 hard_fail）"
    if price < min_price:
        return f"price_too_low: 最新价 {price:.2f} < 下限 {min_price:.2f}"
    return None


def _check_suspended(
    latest_volume: Optional[float],
    amount_1d: Optional[float],
    listing_days: Optional[int],
) -> Optional[str]:
    """H5: 停牌 / 无量.

    判断依据：
    - latest_volume == 0（无成交量）且上市超过 30 天 → 停牌
    - amount_1d == 0（无成交额）且上市超过 30 天 → 停牌
    - 若 listing_days ≤ 30，可能是新股打新期间，不淘汰
    """
    if latest_volume is None and amount_1d is None:
        return "_warn_"
    # 排除新股（上市 ≤ 30 天）
    if listing_days is not None and listing_days <= 30:
        return None
    if latest_volume is not None and latest_volume == 0:
        return "suspended_or_no_volume: 最新交易日成交量为 0（疑似停牌）"
    if amount_1d is not None and amount_1d == 0:
        return "suspended_or_no_volume: 最新交易日成交额为 0（疑似停牌）"
    return None


def _check_amount_avg_5d(
    amount_avg_5d: Optional[float],
    min_amount: float,
    tolerance_pct: float = 0.0,
) -> Optional[str]:
    """H6: 近5日均成交额低于下限.

    Session 11: 新增 tolerance_pct 容差参数。
    实际门槛 = min_amount * (1 - tolerance_pct/100)。
    例如 min_amount=2亿, tolerance_pct=5 → 实际门槛=1.9亿。
    这解决了 成交额数据精度问题。

    NOTE: amount_avg_5d 来自 Tushare daily，为精确成交额。

    核心字段：amount_avg_5d 缺失直接 hard_fail（无法判断流动性）。
    """
    if amount_avg_5d is None:
        return "insufficient_core_data: amount_avg_5d 缺失，无法判断流动性（H6 hard_fail）"
    effective_min = min_amount * (1.0 - tolerance_pct / 100.0)
    if amount_avg_5d < effective_min:
        return (
            f"amount_too_low: 近5日均成交额 {amount_avg_5d/1e8:.2f}亿"
            f" < 下限 {min_amount/1e8:.2f}亿"
            f"（容差{tolerance_pct:.1f}%→实际门槛{effective_min/1e8:.2f}亿）"
            f"（）"
        )
    return None


def _check_float_market_cap(
    fmc: Optional[float], min_fmc: float
) -> Optional[str]:
    """H7: 流通市值低于下限.

    核心字段：float_market_cap 缺失直接 hard_fail（无法判断可执行性）。
    """
    if fmc is None:
        return "insufficient_core_data: float_market_cap 缺失，无法判断市值（H7 hard_fail）"
    if fmc < min_fmc:
        return (
            f"float_mc_too_small: 流通市值 {fmc/1e8:.1f}亿"
            f" < 下限 {min_fmc/1e8:.1f}亿"
        )
    return None


def _check_ipo_date(ipo_date: Optional[str]) -> Optional[str]:
    """H8: IPODate 缺失（强制 hard_fail）.

    IPO 日期是 listing_days 计算的依据，缺失则无法判断 H3，
    同时可能是数据异常信号，强制 hard_fail。
    """
    if not ipo_date:
        return "ipo_date_missing: 上市日期字段缺失，无法核验上市时间（hard_fail）"
    return None


def _check_industry(industry: Optional[str]) -> Optional[str]:
    """H9: 金融行业排除.

    判断依据：行业字段包含金融相关关键词。
    金融股的价格驱动逻辑与热点逻辑不同，排除降低误判。
    """
    if not industry:
        return "_warn_"
    for kw in _FINANCE_KEYWORDS:
        if kw in industry:
            return f"industry_finance: 行业含金融关键词（{industry!r}）"
    return None
