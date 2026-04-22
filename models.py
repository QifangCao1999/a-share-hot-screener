"""数据结构：HotStockDetail / HotStockSummary / RejectedRecord / RunMetadata.

Session 3 变更：
  - HotStockDetail：重新组织量价字段，明确区分 spot 字段 vs Tushare 日线字段
  - 新增 Session 3 所有日线派生特征字段
  - Tushare daily 提供精确成交额
  - HotStockSummary.from_detail 更新对应字段

Session 4 变更：
  - 新增 EventLayerData dataclass（事件层中间结果容器）
  - HotStockDetail 新增事件层字段（limit_up_count_5d / _10d / max_consecutive_limit_up_10d /
    strong_pool_entry_count_3d / lhb_count_20d / lhb_on_board /
    industry_heat_pctile_5d / concept_heat_pctile_5d / advanced_concept_module_available）
  - HotStockSummary.from_detail 同步新增事件层摘要字段

设计原则：
- 严格与 a_share_screener 的模型隔离，不继承、不复用同名 dataclass
- 所有字段均为 Optional（除 code），支持流程任意阶段的部分填充
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# ════════════════════════════════════════════════════════
# 0. ValidatedHotStock – 通过 stock_codes 校验后的基础字段
# ════════════════════════════════════════════════════════
@dataclass
class ValidatedHotStock:
    """经过 A 股 spot 校验后的单只短线候选股基础信息.

    通过 StockValidator 校验后产出，包含从 spot 全市场表收集的基础字段。
    spot 字段均为 Optional，fallback 模式时可能全为 None。
    """

    code: str                               # 6 位纯数字
    name: str = ""                          # 股票简称
    exchange: str = ""                      # SH / SZ / BJ
    ts_code: str = ""                  # Tushare 格式，如 600519.SH
    input_order: int = 0                    # 用户输入顺序（0-based）
    validation_status: str = "valid"        # valid / rejected

    # ── Spot 字段（来自 Tushare daily_basic）────
    # 注意：非交易时间返回的是前一日数据；停牌股 latest_price 可能为 None
    latest_price: Optional[float] = None        # 最新价（元）
    pct_change: Optional[float] = None          # 涨跌幅（%）
    price_change: Optional[float] = None        # 涨跌额（元）
    volume: Optional[float] = None              # 成交量（手）
    amount: Optional[float] = None              # 成交额（元）
    amplitude: Optional[float] = None           # 振幅（%）
    high: Optional[float] = None                # 最高价
    low: Optional[float] = None                 # 最低价
    open: Optional[float] = None                # 今开价
    prev_close: Optional[float] = None          # 昨收价
    volume_ratio: Optional[float] = None        # 量比
    turnover_rate: Optional[float] = None       # 换手率（%）
    turnover_rate_f: Optional[float] = None     # 自由流通换手率（%，P0-B）
    pe_ttm: Optional[float] = None              # 市盈率-动态
    pb: Optional[float] = None                  # 市净率
    market_cap: Optional[float] = None          # 总市值（元）
    float_market_cap: Optional[float] = None    # 流通市值（元）


# ════════════════════════════════════════════════════════
# 1. HotStockDetail – 单只股票的完整第一阶段明细
# ════════════════════════════════════════════════════════
@dataclass
class HotStockDetail:
    """一只股票经过第一阶段处理后的全量明细.

    字段分组：
      基础标识 → 市场行情(spot) → 量价字段 → Tushare 日线派生特征
      → 价格趋势布尔特征 → 基础公司信息 → hard filter 结果
      → 四轴评分 → pass_stage1 → structured flags (dict)
      → 事件层 → 时序 delta → warnings
    """

    # ── 基础标识 ──────────────────────────────────────────
    code: str
    name: str = ""
    exchange: str = ""
    ts_code: str = ""
    input_order: int = 0

    # ── 市场行情（来自 Tushare daily_basic）────
    latest_price: Optional[float] = None        # 最新收盘价（元）
    market_cap: Optional[float] = None          # 总市值（元）
    float_market_cap: Optional[float] = None    # 流通市值（元）
    pe_ttm: Optional[float] = None              # 市盈率-动态
    pb: Optional[float] = None                  # 市净率

    # ── spot 实时量价字段（来自 Tushare daily_basic，初始化时写入）──
    pct_change_1d: Optional[float] = None       # 当日涨跌幅（%，spot）
    turnover_rate_1d: Optional[float] = None    # 当日换手率（%，spot总股本）
    turnover_rate_f_1d: Optional[float] = None  # 当日自由流通换手率（%，spot，P0-B）
    amount_1d: Optional[float] = None           # 当日成交额（元，spot）
    amplitude: Optional[float] = None           # 当日振幅（%，spot）
    volume_ratio: Optional[float] = None        # 量比（spot）

    # ── Tushare 日线派生特征（Session 3，由 price_features 填充）──
    # Tushare daily 提供精确成交额
    latest_eod_date: Optional[str] = None       # 最新 EOD 数据日期（YYYY-MM-DD）
    latest_volume: Optional[float] = None       # 最新 EOD 成交量（股）
    latest_amount_approx: Optional[float] = None  # 最新 EOD 成交额近似（元）
    return_3d: Optional[float] = None           # 近3个交易日收益率（%）
    return_5d: Optional[float] = None           # 近5个交易日收益率（%）
    return_10d: Optional[float] = None          # 近10个交易日收益率（%）
    amount_avg_5d: Optional[float] = None       # 近5日均成交额近似（元）
    amount_avg_20d: Optional[float] = None      # 近20日均成交额近似（元）
    volume_ratio_20d: Optional[float] = None    # 量比（20日均量基准）
    close_position_20d: Optional[float] = None  # 收盘价在近20日高低区间的位置（0~1）
    clv_latest: Optional[float] = None          # 当日 CLV（0~1）
    amount_ratio_5d_to_20d: Optional[float] = None  # 近5日均额 / 近20日均额
    abs_distance_to_ma10: Optional[float] = None    # (close-MA10)/MA10
    amp_norm_avg_5d: Optional[float] = None     # 近5日平均振幅（%）
    upper_shadow_count_5d: Optional[int] = None # 近5日显著上影线天数（K线形态）
    upper_reversal_count_5d: Optional[int] = None # 近5日冲高回落天数（交易风险信号，P0-A）
    limit_board_count_5d: Optional[int] = None  # 近5日疑似一字板/T字板天数（proxy）
    eod_data_rows: int = 0                      # EOD 实际可用行数
    turnover_method: str = ""                    # P0-B: 换手率来源 turnover_rate_f|turnover_rate|amount_proxy

    # ── 均线字段（Session 8 P1 填充）────────────────────
    ma5: Optional[float] = None                 # MA5
    ma10: Optional[float] = None                # MA10
    ma20: Optional[float] = None                # MA20

    # ── 最新日一字板判定（Session 8 P1）────────────────
    latest_is_limit_board: Optional[bool] = None  # 最新日是否一字板（proxy）
    latest_pct_change: Optional[float] = None     # 最新日涨跌幅（%）

    # ── 价格趋势布尔特征（从 EOD 衍生，Session 3 填充）──────
    above_ma5: Optional[bool] = None
    above_ma10: Optional[bool] = None
    above_ma20: Optional[bool] = None
    new_high_20d: Optional[bool] = None         # 收盘价是否创近20日新高
    consec_up_days: Optional[int] = None        # 连续收涨天数

    # ── 大涨天数（Session 11 新增）────────────────────
    big_up_count_10d: Optional[int] = None      # 近10日单日涨幅>5%天数

    # ── 基础公司信息（Tushare stock_basic，hard filter 用）─
    industry: str = ""
    ipo_date: Optional[str] = None              # YYYYMMDD
    listing_days: Optional[int] = None          # 已上市日历天数（近似）
    total_shares: Optional[float] = None        # 总股本
    float_shares: Optional[float] = None        # 流通股本

    # ── hard filter 结果（Session 3 实现）────────────────────
    passed_hard_filter: bool = False
    hard_filter_reason: str = ""                # 主要失败原因（first of hard_fail_reasons）
    hard_fail_reasons: List[str] = field(default_factory=list)
    hard_filter_warnings: List[str] = field(default_factory=list)  # 数据缺失但未强制淘汰

    # ── 四轴评分（TODO: Session 4+，预留位）─────────────────
    hot_theme_score: Optional[float] = None
    hot_theme_coverage: Optional[float] = None
    hot_theme_subscores: Dict[str, Any] = field(default_factory=dict)

    trend_flow_score: Optional[float] = None
    trend_flow_coverage: Optional[float] = None
    trend_flow_subscores: Dict[str, Any] = field(default_factory=dict)

    liquidity_execution_score: Optional[float] = None
    liquidity_execution_coverage: Optional[float] = None
    liquidity_execution_subscores: Dict[str, Any] = field(default_factory=dict)

    risk_control_score: Optional[float] = None
    risk_control_coverage: Optional[float] = None
    risk_control_subscores: Dict[str, Any] = field(default_factory=dict)

    total_score: Optional[float] = None
    data_coverage: Optional[float] = None          # 总体覆盖率（向后兼容）
    core_data_coverage: Optional[float] = None     # P0-D: 核心数据覆盖率
    overall_data_coverage: Optional[float] = None  # P0-D: 全量数据覆盖率

    # ── pass_stage1 判定（TODO: Session 4+）─────────────────
    pass_stage1: bool = False                    # P0-C: = tradeable only
    pass_stage1_watch: bool = False              # P0-C: 观察池（强但不可交易）
    pass_stage1_any: bool = False                # P0-C: tradeable ∪ watch_only
    candidate_pool_type: str = ""                # P0-C: tradeable/watch_only/failed_score/insufficient_data/rejected_hard
    candidate_pool_reason: str = ""              # P0-C: 具体原因
    pass_stage1_reasons: List[str] = field(default_factory=list)
    blocked_by: List[str] = field(default_factory=list)
    crowding_cap_applied: Optional[List[str]] = None

    # ── structured flags（Session 6，由 compute_flags 填充）────
    # NOTE: flags 不直接进入 total_score，供第二阶段消费和 summary 输出
    flags: Dict[str, Any] = field(default_factory=dict)

    # ── 事件层字段（Session 4，由 event_layer.EventLayerProcessor 填充）────
    # 涨停池
    limit_up_count_5d: Optional[int] = None          # 近5个交易日涨停池入选次数
    limit_up_count_10d: Optional[int] = None         # 近10个交易日涨停池入选次数
    max_consecutive_limit_up_10d: Optional[int] = None  # 近10日最大连续涨停天数
    limit_up_source: str = ""                        # 数据来源口径
    # 强势股池
    strong_pool_entry_count_3d: Optional[int] = None  # 近3个交易日强势股池入选次数
    strong_pool_source: str = ""                     # 数据来源口径
    # 龙虎榜（enable_lhb_module=True 时填充）
    lhb_count_20d: Optional[int] = None              # 近20日龙虎榜上榜次数
    lhb_on_board: Optional[bool] = None              # 近20日是否上过榜
    lhb_source: str = ""
    # 行业热度
    industry_heat_pctile_5d: Optional[float] = None  # 行业近5日涨幅分位（0~1，全行业排名）
    industry_pct_5d: Optional[float] = None          # 行业近5日绝对涨幅（%）
    industry_heat_source: str = ""
    # 概念热度（enable_concept_heat_module=True 时填充）
    concept_heat_pctile_5d: Optional[float] = None   # 概念近5日涨幅分位（0~1）
    concept_names: List[str] = field(default_factory=list)  # 所属概念列表（最多5个）
    advanced_concept_module_available: bool = False  # 概念热度模块是否成功运行
    concept_heat_source: str = ""

    # ── Context Scores (Phase 3: HT8/HT9/HT10 experimental) ────
    context_scores: Dict[str, Any] = field(default_factory=dict)  # ContextScoresResult.to_dict()

    # ── 时序 delta（Session 14 P2-6，由 trend_compare 模块填充）────
    trend_delta: Dict[str, Any] = field(default_factory=dict)  # TrendDelta.to_dict() 快照

    # ── warnings ─────────────────────────────────────────────
    warnings: List[str] = field(default_factory=list)

    def to_cache_dict(self) -> Dict[str, Any]:
        """序列化为 JSON 兼容 dict（用于断点续跑缓存）."""
        import dataclasses
        return dataclasses.asdict(self)

    @classmethod
    def from_cache_dict(cls, d: Dict[str, Any]) -> "HotStockDetail":
        """从 dict 反序列化（缺失字段用默认值）."""
        import dataclasses
        valid_fields = {f.name for f in dataclasses.fields(cls)}
        filtered = {k: v for k, v in d.items() if k in valid_fields}
        return cls(**filtered)


# ════════════════════════════════════════════════════════
# 2. HotStockSummary – 写入 stage1_hot_summary.csv 的简化行
# ════════════════════════════════════════════════════════
@dataclass
class HotStockSummary:
    """stage1_hot_summary.csv 的一行.

    Session 9 P2-2: 补齐约 23 个缺失字段，包括：
      validation_status, ts_code, industry, market_cap, listing_days,
      board_limit_pct, return_3d, return_10d, return_pctile_5d, return_pctile_10d,
      amount_avg_20d, clv_latest, amount_ratio_5d_to_20d, abs_distance_to_ma10,
      ma_bullish_alignment_count, 减持/质押/解禁 6 个字段。
    """

    code: str
    name: str
    exchange: str = ""
    ts_code: str = ""                       # Tushare ts_code
    validation_status: str = "valid"              # Tushare ts_code
    input_order: int = 0
    pass_stage1: bool = False
    pass_stage1_watch: bool = False              # P0-C
    pass_stage1_any: bool = False                # P0-C
    candidate_pool_type: str = ""                # P0-C
    candidate_pool_reason: str = ""              # P0-C
    blocked_by: List[str] = field(default_factory=list)

    # ── 基础信息 ─────────────────────────────────
    industry: str = ""                            # Tushare ts_code
    listing_days: Optional[int] = None            # Tushare ts_code
    board_limit_pct: Optional[float] = None       # Tushare ts_code（涨跌停幅度 %）

    # ── 行情字段 ─────────────────────────────────
    latest_price: Optional[float] = None
    market_cap: Optional[float] = None            # Tushare ts_code（总市值）
    float_market_cap: Optional[float] = None
    pct_change_1d: Optional[float] = None
    turnover_rate_1d: Optional[float] = None

    # ── 价格特征 ─────────────────────────────────
    return_3d: Optional[float] = None             # Tushare ts_code
    return_5d: Optional[float] = None
    return_10d: Optional[float] = None            # Tushare ts_code
    amount_avg_5d: Optional[float] = None         # 
    amount_avg_20d: Optional[float] = None        # Tushare ts_code
    volume_ratio_20d: Optional[float] = None
    close_position_20d: Optional[float] = None
    clv_latest: Optional[float] = None            # Tushare ts_code
    amount_ratio_5d_to_20d: Optional[float] = None  # Tushare ts_code
    abs_distance_to_ma10: Optional[float] = None  # Tushare ts_code
    ma_bullish_alignment_count: Optional[int] = None  # Tushare ts_code（0/1/2/3）

    # ── 评分 ───────────────────────────────────
    total_score: Optional[float] = None
    hot_theme_score: Optional[float] = None
    trend_flow_score: Optional[float] = None
    liquidity_execution_score: Optional[float] = None
    risk_control_score: Optional[float] = None
    data_coverage: Optional[float] = None
    core_data_coverage: Optional[float] = None     # P0-D
    overall_data_coverage: Optional[float] = None  # P0-D
    hot_theme_coverage: Optional[float] = None
    trend_flow_coverage: Optional[float] = None
    liquidity_execution_coverage: Optional[float] = None
    risk_control_coverage: Optional[float] = None
    turnover_method: str = ""                      # P0-B

    # ── 硬筛结果 ─────────────────────────────────
    passed_hard_filter: bool = False
    hard_filter_reason: str = ""
    eod_data_rows: int = 0

    # ── 事件层摘要 ────────────────────────────────
    limit_up_count_5d: Optional[int] = None
    limit_up_count_10d: Optional[int] = None
    max_consecutive_limit_up_10d: Optional[int] = None
    strong_pool_entry_count_3d: Optional[int] = None
    lhb_count_20d: Optional[int] = None
    lhb_on_board: Optional[bool] = None
    industry_heat_pctile_5d: Optional[float] = None
    concept_heat_pctile_5d: Optional[float] = None
    concept_names_str: str = ""                   # 所属概念板块（逗号分隔，最多5个）
    advanced_concept_module_available: bool = False

    # ── flags ─────────────────────────────────────
    one_word_limit_up_latest: Optional[bool] = None
    one_word_limit_down_latest: Optional[bool] = None
    turnover_avg_5d: Optional[float] = None
    amp_norm_avg_5d: Optional[float] = None
    upper_shadow_count_5d: Optional[int] = None
    new_stock_flag: Optional[bool] = None

    # ── Session 11 新增字段 ───────────────────────
    big_up_count_10d: Optional[int] = None       # 近10日大涨天数（单日>5%）
    consec_up_days: Optional[int] = None         # 连续收涨天数

    # ── 减持/质押/解禁（P2-2 新增，S11 减持已实现）──────────
    shareholder_net_reduction_ratio_3m: Optional[float] = None
    shareholder_reduction_flag_3m: Optional[bool] = None
    restricted_shares_unlock_ratio_20d: Optional[float] = None
    unlock_risk_flag_20d: Optional[bool] = None
    pledge_ratio_latest: Optional[float] = None
    pledge_ratio_flag: Optional[bool] = None

    # ── Context Scores (Phase 3: HT8/HT9/HT10 experimental) ────
    ht8_score: Optional[float] = None               # 市场确认度
    ht8_confirmation_level: str = ""                  # 确认级别
    ht9_score: Optional[float] = None               # 板块扩散度
    ht9_breadth_ratio: Optional[float] = None       # 板块内涨幅>5%股票占比
    ht9_sector_name: str = ""                        # 板块名称
    ht10_score: Optional[float] = None              # 板块内辨识度
    ht10_position_type: str = ""                     # frontline_like/capacity_core_like/follower_like/unknown
    ht10_confidence: str = ""                        # high/medium

    # ── 时序 delta（Session 14 P2-6）────────────────────
    prev_run_date: str = ""                              # 上次运行日期
    total_score_delta: Optional[float] = None            # total_score 变化量
    hot_theme_score_delta: Optional[float] = None
    trend_flow_score_delta: Optional[float] = None
    liquidity_execution_score_delta: Optional[float] = None
    risk_control_score_delta: Optional[float] = None
    pass_stage1_change: str = ""                          # new_pass/lost_pass/keep_pass/keep_fail/new_entry
    score_accelerating: Optional[bool] = None
    score_decelerating: Optional[bool] = None

    warnings_count: int = 0

    @classmethod
    def from_detail(cls, d: "HotStockDetail") -> "HotStockSummary":
        """从 HotStockDetail 自动投影生成 Summary.

        策略：
          1. 同名字段自动复制（反射式匹配）
          2. 显式覆写：计算字段 / flags dict / trend_delta dict
        """
        import dataclasses as _dc
        from a_share_hot_screener.limit_rules import infer_limit_pct

        # ── 1. 自动投影：Detail 与 Summary 同名字段直接复制 ──
        detail_vals = {f.name: getattr(d, f.name) for f in _dc.fields(d)}
        summary_field_names = {f.name for f in _dc.fields(cls)}
        kwargs = {k: v for k, v in detail_vals.items() if k in summary_field_names}

        # ── 2. 显式覆写：计算字段 ─────────────────────────────
        kwargs["validation_status"] = "valid"
        kwargs["blocked_by"] = list(d.blocked_by)
        kwargs["industry"] = d.industry or ""
        kwargs["board_limit_pct"] = infer_limit_pct(d.code)
        kwargs["ma_bullish_alignment_count"] = _calc_ma_bullish_count(d)
        kwargs["concept_names_str"] = ",".join(d.concept_names[:5]) if d.concept_names else ""
        kwargs["warnings_count"] = len(d.warnings)

        # ── 3. 显式覆写：flags dict 来源字段 ──────────────────
        _flags_keys = [
            "one_word_limit_up_latest", "one_word_limit_down_latest",
            "turnover_avg_5d", "new_stock_flag",
            "shareholder_net_reduction_ratio_3m", "shareholder_reduction_flag_3m",
            "restricted_shares_unlock_ratio_20d", "unlock_risk_flag_20d",
            "pledge_ratio_latest", "pledge_ratio_flag",
        ]
        for fk in _flags_keys:
            kwargs[fk] = d.flags.get(fk)

        # ── 4. 显式覆写：trend_delta dict 来源字段 ───────────
        _td = d.trend_delta
        kwargs["prev_run_date"] = _td.get("prev_run_date", "")
        kwargs["total_score_delta"] = _td.get("total_score_delta")
        kwargs["hot_theme_score_delta"] = _td.get("hot_theme_score_delta")
        kwargs["trend_flow_score_delta"] = _td.get("trend_flow_score_delta")
        kwargs["liquidity_execution_score_delta"] = _td.get("liquidity_execution_score_delta")
        kwargs["risk_control_score_delta"] = _td.get("risk_control_score_delta")
        kwargs["pass_stage1_change"] = _td.get("pass_stage1_change", "")
        kwargs["score_accelerating"] = _td.get("score_accelerating")
        kwargs["score_decelerating"] = _td.get("score_decelerating")

        # ── 5. 显式覆写：context_scores dict 来源字段 ──────────
        _cs = d.context_scores
        kwargs["ht8_score"] = _cs.get("ht8_score")
        kwargs["ht8_confirmation_level"] = _cs.get("ht8_confirmation_level", "")
        kwargs["ht9_score"] = _cs.get("ht9_score")
        kwargs["ht9_breadth_ratio"] = _cs.get("ht9_breadth_ratio")
        kwargs["ht9_sector_name"] = _cs.get("ht9_sector_name", "")
        kwargs["ht10_score"] = _cs.get("ht10_score")
        kwargs["ht10_position_type"] = _cs.get("ht10_position_type", "")
        kwargs["ht10_confidence"] = _cs.get("ht10_confidence", "")

        return cls(**kwargs)


def _calc_ma_bullish_count(d: "HotStockDetail") -> Optional[int]:
    """计算均线多头排列条件数（0/1/2/3）.

    条件：close>ma5, ma5>ma10, ma10>ma20
    """
    close = d.latest_price
    ma5 = d.ma5
    ma10 = d.ma10
    ma20 = d.ma20
    if close is None or ma5 is None or ma10 is None:
        return None
    count = 0
    if close > ma5:
        count += 1
    if ma5 > ma10:
        count += 1
    if ma20 is not None and ma10 > ma20:
        count += 1
    return count


# ════════════════════════════════════════════════════════
# 3. RejectedRecord – stage1_hot_rejected.csv 的一行
# ════════════════════════════════════════════════════════
@dataclass
class RejectedRecord:
    """被淘汰股票的记录行.

    reject_stage 枚举值：
      code_parse / validation / hard_filter / data_coverage / pipeline_error
    """

    code: str
    name: str = ""
    input_order: int = 0
    reject_stage: str = ""
    reject_reason: str = ""
    reject_detail: str = ""
    warnings: str = ""


# ════════════════════════════════════════════════════════
# 4. RunMetadata – stage1_hot_metadata.json
# ════════════════════════════════════════════════════════
@dataclass
class RunMetadata:
    """一次运行的元数据，写入 stage1_hot_metadata.json.

    Session 7 新增字段（向后兼容，均有默认值）：
      input_stock_codes         原始输入代码列表
      scoring_pool_size         参与横截面评分的股票数
      average_data_coverage     通过硬筛股票的平均数据覆盖率
      cache_hit_rate            缓存命中率（暂不统计，为 None）
      pass_stage1_thresholds    本次运行使用的 pass_stage1 各轴阈值

    Session 9 P2-4 字段名对齐：
      input_count → input_pool_size（保留 input_count 别名向后兼容）
      新增 rejected_before_scoring_count
      新增 axis_weights dict
    """

    run_date: str = ""
    trade_date_used: str = ""                   # 实际使用的交易日（Session 3 新增）
    generated_at: str = ""
    version: str = "0.1.0"

    # ── 输入摘要 ──────────────────────────────────────────
    input_pool_size: int = 0                    # P2-4: 原始输入代码数（原 input_count）
    input_stock_codes: List[str] = field(default_factory=list)  # S7 新增
    valid_input_count: int = 0
    invalid_input_count: int = 0

    # ── 各阶段过滤计数 ───────────────────────────────
    validation_passed: int = 0                  # = 通过 spot 校验的股票数
    validation_rejected: int = 0
    hard_filter_passed: int = 0                 # = scoring_pool_size
    hard_filter_rejected: int = 0
    rejected_before_scoring_count: int = 0      # P2-4 新增：validation_rejected + hard_filter_rejected
    data_coverage_passed: int = 0
    data_coverage_rejected: int = 0
    pass_stage1_count: int = 0
    fail_stage1_count: int = 0

    # ── 评分池信息（S7 新增）─────────────────────────────
    scoring_pool_size: int = 0                  # 参与横截面评分的股票数
    average_data_coverage: Optional[float] = None   # 通过硬筛股票平均数据覆盖率
    cache_hit_rate: Optional[float] = None      # 缓存命中率（暂不统计，保留占位）

    # ── 运行配置 ──────────────────────────────────────
    min_data_coverage: float = 0.75
    min_price: float = 3.0
    min_amount_avg_5d: float = 200_000_000.0
    min_float_market_cap: float = 1_500_000_000.0
    min_trading_days: int = 20
    include_beijing: bool = False
    enable_concept_heat_module: bool = False
    enable_lhb_module: bool = True
    enable_unlock_risk_module: bool = False
    max_workers: int = 3

    # ── pass_stage1 阈值（S7 新增）─────────────────────────
    pass_stage1_thresholds: Dict[str, float] = field(default_factory=dict)

    # ── 四轴权重（P2-4 新增）──────────────────────────────
    axis_weights: Dict[str, float] = field(default_factory=dict)

    elapsed_seconds: float = 0.0

    modules_enabled: Dict[str, bool] = field(default_factory=dict)
    used_baseline_pool: bool = False              # Session 12: 是否使用了基准 pool
    baseline_pool_stock_count: Optional[int] = None  # Session 12: 基准 pool 中的股票数
    trend_compare_enabled: bool = False             # Session 14: 是否启用了时序对比
    trend_compare_prev_run_date: str = ""            # Session 14: 上次运行日期
    global_warnings: List[str] = field(default_factory=list)
    output_files: Dict[str, str] = field(default_factory=dict)

    # ── 向后兼容属性 ────────────────────────────────────────
    @property
    def input_count(self) -> int:
        """向后兼容别名 → input_pool_size."""
        return self.input_pool_size
