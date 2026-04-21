"""HotScreenerConfig – 短线热点筛选全局配置 dataclass.

与 a_share_screener.ScreenerConfig 严格隔离，不共享任何状态。
"""

from __future__ import annotations

import datetime as dt
import os
from dataclasses import dataclass, field
from typing import List


@dataclass
class HotScreenerConfig:
    """第一阶段短线热点筛选运行配置.

    所有参数在 CLI 层解析后注入，运行期间只读。
    """

    # ── 必填 ────────────────────────────────────────────
    tushare_token: str
    run_date: dt.date
    stock_codes: List[str]          # 标准化后的 6 位纯数字代码列表
    output_dir: str

    # ── 基础筛选阈值（硬筛 H1-H9 对应参数）────────────────
    min_data_coverage: float = 0.75
    min_price: float = 3.0                      # H4: 最低价格（元）
    min_amount_avg_5d: float = 200_000_000.0    # H6: 5日均成交额下限（元）
    amount_tolerance_pct: float = 5.0             # H6: 成交额容差（%，允许边缘股通过）Session 11 新增
    min_float_market_cap: float = 1_500_000_000.0  # H7: 流通市值下限（元）
    min_trading_days: int = 20                  # H3: 最少已上市交易日数（去噪）

    # ── pass_stage1 评分阈值（Session 7 新增）─────────────
    # 评分尺度均为 0~1，下方注释中的 0~100 换算仅供人读参考
    # 全部条件必须同时满足（AND 关系）
    min_total_score: float = 0.68               # total_score >= 0.68（≈68分）
    min_hot_theme_score: float = 0.65           # hot_theme_score >= 0.65（≈65分）
    min_trend_flow_score: float = 0.60          # trend_flow_score >= 0.60（≈60分）
    min_liquidity_execution_score: float = 0.55 # liquidity_execution_score >= 0.55（≈55分）
    min_risk_control_score: float = 0.40        # risk_control_score >= 0.40（≈40分）
    # NOTE: 以上阈值均可通过 CLI 参数覆盖，详见 cli.py

    # ── 四轴权重（total_score 加权公式）──────────────────
    # Session 10: total_score = 0.40*hot_theme + 0.30*trend_flow + 0.20*liquidity + 0.10*risk_control
    # 仅对有数据的轴参与加权（与 AxisScore.compute() 口径一致）
    # 比例 40:30:20:10，总和=100（S10: 原 35:30:20:15，把 5% 从 RC 转移到 HT）
    axis_weight_hot_theme: float = 40.0
    axis_weight_trend_flow: float = 30.0
    axis_weight_liquidity_execution: float = 20.0
    axis_weight_risk_control: float = 10.0

    # ── 交易所控制 ──────────────────────────────────────
    include_beijing: bool = False               # 是否保留北交所
    include_finance: bool = False               # 是否保留金融行业（Session 10: 默认排除）

    # ── 模块开关（供后续 session 扩展）─────────────────
    enable_concept_heat_module: bool = False    # 概念热度模块（需 6000 积分/600元年，自动检测权限）
    enable_lhb_module: bool = True              # 龙虎榜模块（后续 session）
    enable_unlock_risk_module: bool = False     # 解禁风险模块（后续 session）
    enable_sector_rotation: bool = False        # 板块轮动信号（需 6000 积分/600元年，独立输出 sector_heat.csv）

    # ── preset 模式（Session 10）──────────────────────────
    # "default": 使用上方默认阈值
    # "relaxed":  自动降低各轴阈值，扩大候选集，适合初筛。
    preset: str = "default"  # default | relaxed

    # ── 基准 pool（Session 12 P0-2）──────────────────────
    # 大规模运行后保存的横截面基准数据，用于单只查询时补充百分位分母
    baseline_pool_path: str = ""                  # 基准 pool JSON 文件路径（空=自动检测 cache_dir）
    save_baseline_pool: bool = False              # 运行完成后是否保存当前 pool 为基准
    min_baseline_pool_size: int = 30              # scoring_pool < 此值时触发基准 pool 合并（#2: 从5提升到30避免小样本百分位失真）

    # ── 时序连续性 / 趋势加速信号（Session 14 P2-6）──────
    prev_run_dir: str = ""                        # 上次运行的输出目录（空=跳过时序对比）
    prev_run_date: str = ""                       # 指定上次运行日期（空=自动检测最新文件）

    # ── 批量运行（重构 #6）────────────────────────
    batch_size: int = 0                           # 0=不分批，>0 时每批处理此数量股票
    resume: bool = False                          # 断点续跑（跳过已完成批次）

    # ── 运行时派生（由 main 填充）───────────────────────
    cache_dir: str = ""
    log_level: str = "INFO"
    max_workers: int = 3                        # 并发工作线程数（1=串行）

    def __post_init__(self) -> None:
        # 缓存目录：放在用户家目录下，避免绑定 output_dir
        # 与 a_share_screener 使用不同路径，严格隔离
        if not self.cache_dir:
            self.cache_dir = os.path.join(
                os.path.expanduser("~"), ".a_share_hot_screener", "cache"
            )

    # ── 便捷属性 ─────────────────────────────────────────

    @property
    def run_date_str(self) -> str:
        """返回 YYYY-MM-DD 格式的 run_date 字符串."""
        return self.run_date.isoformat()

    def apply_preset(self) -> None:
        """应用 preset 预设覆盖阈值（在 __post_init__ 或 CLI 构建完 config 后调用）.

        relaxed 预设：降低各轴通过阈值，扩大候选集，适合初筛。
        """
        if self.preset == "relaxed":
            self.min_total_score = 0.50
            self.min_hot_theme_score = 0.40
            self.min_trend_flow_score = 0.40
            self.min_liquidity_execution_score = 0.35
            self.min_risk_control_score = 0.20
