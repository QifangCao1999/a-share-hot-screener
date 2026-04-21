"""CLI 入口 – 解析命令行参数，构建 HotScreenerConfig，执行 pipeline.

用法:
    python3 -m a_share_hot_screener \\
        --tushare-token "xxx" \\
        --run-date 2026-04-18 \\
        --stock-codes "600519,000858,300750" \\
        --output-dir ./output

或:
    python3 -m a_share_hot_screener \\
        --tushare-token "xxx" \\
        --run-date today \\
        --stock-codes "600519" "000858" \\
        --output-dir ./output \\
        --min-price 3.0 \\
        --min-amount-avg-5d 200000000 \\
        --enable-lhb-module \\
        --max-workers 3
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from typing import List, Optional


def build_parser() -> argparse.ArgumentParser:
    """构建 CLI ArgumentParser."""
    parser = argparse.ArgumentParser(
        prog="a_share_hot_screener",
        description="A股短线热点第一阶段客观筛选脚本",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── 必填参数 ──────────────────────────────────────────
    parser.add_argument(
        "--tushare-token",
        default="",
        help="Tushare Pro API Token（也可通过 TUSHARE_TOKEN 环境变量 或 .env 文件提供）",
    )
    parser.add_argument(
        "--run-date",
        required=True,
        help="运行日期，格式 YYYY-MM-DD 或 today",
    )
    parser.add_argument(
        "--stock-codes",
        required=True,
        nargs="+",
        help="输入股票代码，支持逗号分隔字符串或多个参数",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="输出目录路径",
    )

    # ── 筛选阈值 ──────────────────────────────────────────
    parser.add_argument(
        "--min-data-coverage",
        type=float,
        default=0.75,
        help="最低数据覆盖率（0~1）",
    )
    parser.add_argument(
        "--min-price",
        type=float,
        default=3.0,
        help="最低股价（元）",
    )
    parser.add_argument(
        "--min-amount-avg-5d",
        type=float,
        default=200_000_000.0,
        help="5日均成交额下限（元）",
    )
    parser.add_argument(
        "--min-float-market-cap",
        type=float,
        default=1_500_000_000.0,
        help="流通市值下限（元）",
    )
    parser.add_argument(
        "--min-trading-days",
        type=int,
        default=20,
        help="最少已上市交易日数",
    )
    parser.add_argument(
        "--amount-tolerance-pct",
        type=float,
        default=5.0,
        help="H6 成交额容差（%%，默认 5%%），成交额容差百分比",
    )

    # ── pass_stage1 评分阈値（Session 7 新增）─────────────────
    # 注：尺度 0~1，help 文内显示的 0~100 为人读等价信息
    parser.add_argument(
        "--min-total-score",
        type=float,
        default=0.68,
        help="total_score 阈値（0~1，人读等价≈68分）",
    )
    parser.add_argument(
        "--min-hot-theme-score",
        type=float,
        default=0.65,
        help="hot_theme_score 阈値（0~1，人读等价≈65分）",
    )
    parser.add_argument(
        "--min-trend-flow-score",
        type=float,
        default=0.60,
        help="trend_flow_score 阈値（0~1，人读等价≈60分）",
    )
    parser.add_argument(
        "--min-liquidity-execution-score",
        type=float,
        default=0.55,
        help="liquidity_execution_score 阈値（0~1，人读等价≈55分）",
    )
    parser.add_argument(
        "--min-risk-control-score",
        type=float,
        default=0.40,
        help="risk_control_score 阈値（0~1，人读等价≈40分）",
    )

    # ── 交易所控制 ──────────────────────────────────────
    parser.add_argument(
        "--include-beijing",
        action="store_true",
        default=False,
        help="是否包含北交所（默认排除）",
    )
    parser.add_argument(
        "--include-finance",
        action="store_true",
        default=False,
        help="是否包含金融行业（银行/证券/保险等，默认排除）（Session 10 新增）",
    )

    # ── 模块开关 ──────────────────────────────────────────
    parser.add_argument(
        "--enable-concept-heat-module",
        action="store_true",
        default=False,
        help="启用概念热度模块（默认关闭）",
    )
    parser.add_argument(
        "--enable-lhb-module",
        action="store_true",
        default=True,
        dest="enable_lhb_module",
        help="启用龙虎榜模块（默认开启）",
    )
    parser.add_argument(
        "--disable-lhb-module",
        action="store_false",
        dest="enable_lhb_module",
        help="关闭龙虎榜模块",
    )
    parser.add_argument(
        "--enable-unlock-risk-module",
        action="store_true",
        default=False,
        help="启用解禁风险模块（默认关闭）",
    )

    # ── 运行控制 ──────────────────────────────────────────
    parser.add_argument(
        "--max-workers",
        type=int,
        default=3,
        help="并发工作线程数（1=串行）",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="日志级别",
    )
    parser.add_argument(
        "--preset",
        default="default",
        choices=["default", "relaxed"],
        help="评分阈值预设：default=默认阈值；relaxed=宽松初筛（自动降低各轴阈值）（Session 10 新增）",
    )
    parser.add_argument(
        "--cache-dir",
        default="",
        help="缓存目录（默认 ~/.a_share_hot_screener/cache）",
    )

    # ── 基准 pool（Session 12 P0-2）──────────────────────
    parser.add_argument(
        "--baseline-pool-path",
        default="",
        help="基准 pool JSON 文件路径（空=自动检测 cache_dir 下的 baseline_pool.json）",
    )
    parser.add_argument(
        "--save-baseline-pool",
        action="store_true",
        default=False,
        help="运行完成后保存当前 scoring pool 为基准（需 scoring_pool >= 5 只）",
    )
    parser.add_argument(
        "--min-baseline-pool-size",
        type=int,
        default=5,
        help="scoring_pool 小于此值时触发基准 pool 合并（默认 5）",
    )

    # ── 时序连续性 / 趋势加速信号（Session 14 P2-6）──────
    parser.add_argument(
        "--prev-run-dir",
        default="",
        help="上次运行的输出目录（用于时序对比，空=跳过）",
    )
    parser.add_argument(
        "--prev-run-date",
        default="",
        help="指定上次运行日期 YYYY-MM-DD（空=自动检测目录中最新文件）",
    )

    # ── 批量运行（重构 #6）──────────────────────
    parser.add_argument(
        "--batch-size",
        type=int,
        default=0,
        help="每批股票数（0=不分批，建议 100）",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        default=False,
        help="断点续跑（跳过已完成批次）",
    )

    # ── 缓存维护 ──────────────────────────────
    parser.add_argument(
        "--purge-cache",
        action="store_true",
        default=False,
        help="清理旧版本缓存文件后退出（不执行筛选）",
    )

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    """CLI 主函数，返回退出码（0=成功，1=失败）."""
    # 延迟导入，避免循环依赖
    from dotenv import load_dotenv

    from a_share_hot_screener.config import HotScreenerConfig
    from a_share_hot_screener.date_utils import parse_run_date
    from a_share_hot_screener.logger import setup_logger
    from a_share_hot_screener.pipeline import Stage1HotPipeline
    from a_share_hot_screener.stock_codes import parse_stock_codes

    # 加载项目 .env（不覆盖已存在的环境变量）
    load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"), override=False)

    parser = build_parser()
    args = parser.parse_args(argv)

    # 初始化日志（在 config 构建前）
    logger = setup_logger("a_share_hot_screener", level=args.log_level)

    # purge-cache 子命令（不需要 run_date / stock_codes）
    if args.purge_cache:
        from a_share_hot_screener.cache import LocalCache
        cache_dir = args.cache_dir or os.path.join(
            os.path.expanduser("~"), ".a_share_hot_screener", "cache"
        )
        cache = LocalCache(cache_dir)
        removed = cache.purge_stale()
        logger.info("已清理 %d 个旧版本缓存文件 (cache_dir=%s)", removed, cache_dir)
        return 0

    # 解析 run_date
    try:
        run_date = parse_run_date(args.run_date)
    except ValueError as e:
        logger.error("run_date 解析失败: %s", e)
        return 1

    # 解析 stock_codes（支持 "600519,000858" 和 ["600519", "000858"] 两种形式）
    raw_codes = " ".join(args.stock_codes) if isinstance(args.stock_codes, list) else args.stock_codes
    valid_codes, invalid_codes = parse_stock_codes(raw_codes)
    if invalid_codes:
        logger.warning("以下代码格式无效，将被忽略: %s", invalid_codes)
    if not valid_codes:
        logger.error("没有有效的股票代码，退出")
        return 1

    # 构建 tushare_token（优先 CLI 参数，回退环境变量）
    tushare_token = args.tushare_token or os.environ.get("TUSHARE_TOKEN", "")
    if not tushare_token:
        logger.error("Tushare token 未提供，请通过 --tushare-token 或 TUSHARE_TOKEN 环境变量传入")
        return 1

    # 构建配置
    config = HotScreenerConfig(
        tushare_token=tushare_token,
        run_date=run_date,
        stock_codes=valid_codes,
        output_dir=args.output_dir,
        min_data_coverage=args.min_data_coverage,
        min_price=args.min_price,
        min_amount_avg_5d=args.min_amount_avg_5d,
        min_float_market_cap=args.min_float_market_cap,
        min_trading_days=args.min_trading_days,
        amount_tolerance_pct=args.amount_tolerance_pct,
        # pass_stage1 评分阈値（Session 7 新增）
        min_total_score=args.min_total_score,
        min_hot_theme_score=args.min_hot_theme_score,
        min_trend_flow_score=args.min_trend_flow_score,
        min_liquidity_execution_score=args.min_liquidity_execution_score,
        min_risk_control_score=args.min_risk_control_score,
        include_beijing=args.include_beijing,
        include_finance=args.include_finance,
        preset=args.preset,
        enable_concept_heat_module=args.enable_concept_heat_module,
        enable_lhb_module=args.enable_lhb_module,
        enable_unlock_risk_module=args.enable_unlock_risk_module,
        max_workers=args.max_workers,
        log_level=args.log_level,
        cache_dir=args.cache_dir,
        baseline_pool_path=args.baseline_pool_path,
        save_baseline_pool=args.save_baseline_pool,
        min_baseline_pool_size=args.min_baseline_pool_size,
        prev_run_dir=args.prev_run_dir,
        prev_run_date=args.prev_run_date,
        batch_size=args.batch_size,
        resume=args.resume,
    )

    # 应用 preset（可能覆盖部分阈值）
    config.apply_preset()
    if config.preset != "default":
        logger.info("preset=%s 已应用，阈值已调整", config.preset)

    logger.info(
        "配置: run_date=%s, codes=%d只, output_dir=%s, max_workers=%d, preset=%s",
        config.run_date_str,
        len(valid_codes),
        config.output_dir,
        config.max_workers,
        config.preset,
    )

    # 执行 pipeline（分批或直接）
    try:
        if config.batch_size > 0 and len(valid_codes) > config.batch_size:
            from a_share_hot_screener.batch_runner import run_batched
            metadata = run_batched(config)
        else:
            pipeline = Stage1HotPipeline(config)
            metadata = pipeline.run()
        logger.info(
            "运行完成: pass_stage1=%d, rejected=%d, elapsed=%.1fs",
            metadata.pass_stage1_count,
            metadata.validation_rejected + metadata.hard_filter_rejected,
            metadata.elapsed_seconds,
        )
        return 0
    except Exception as e:
        logger.error("Pipeline 运行失败: %s", e, exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
