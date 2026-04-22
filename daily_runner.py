#!/usr/bin/env python3
"""每日自动筛选运行器 — 收盘后自动执行筛选并推送 Discord 通知.

用法:
    python3 -m a_share_hot_screener.daily_runner           # 使用默认 universe
    python3 -m a_share_hot_screener.daily_runner --dry-run  # 只检查，不实际运行
    python3 -m a_share_hot_screener.daily_runner --date 2026-04-21  # 指定日期

环境变量（或 .env）:
    TUSHARE_TOKEN        — Tushare Pro API Token
    DISCORD_BOT_TOKEN    — Discord Bot Token
    DISCORD_CHANNEL_ID   — Discord 频道 ID（优先）
    DISCORD_USER_ID      — Discord 用户 ID（DM fallback）
"""

from __future__ import annotations

import csv
import datetime
import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ── 项目路径 ──────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = PROJECT_ROOT.parent
UNIVERSE_DIR = PROJECT_ROOT / "universe"
OUTPUT_BASE = WORKSPACE_ROOT / "screener_output"
ENV_FILE = PROJECT_ROOT / ".env"

# ── 日志 ──────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("daily_runner")


def load_env() -> None:
    """从 .env 文件加载环境变量（简易实现，无需 python-dotenv）."""
    if not ENV_FILE.exists():
        return
    with open(ENV_FILE) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value


def load_universe() -> List[str]:
    """加载股票池（full_universe.txt）."""
    universe_file = UNIVERSE_DIR / "full_universe.txt"
    if not universe_file.exists():
        raise FileNotFoundError(f"Universe 文件不存在: {universe_file}")
    codes = []
    with open(universe_file) as f:
        for line in f:
            code = line.strip()
            if code and not code.startswith("#"):
                codes.append(code)
    logger.info("加载 universe: %d 只股票", len(codes))
    return codes


def is_trading_day(date_str: str) -> bool:
    """检查指定日期是否为 A 股交易日.

    简易判断：排除周末。中国节假日由 Tushare trade_cal 处理，
    筛选脚本运行时如果不是交易日，数据拉取会自动 fallback 到最近交易日。
    """
    dt = datetime.datetime.strptime(date_str, "%Y-%m-%d")
    if dt.weekday() >= 5:  # 周六(5)或周日(6)
        logger.info("%s 是周末，跳过", date_str)
        return False
    return True


def get_run_date(override: Optional[str] = None) -> str:
    """获取运行日期.

    逻辑：
    - 如果有 override，直接使用
    - 否则取当前 UTC+8（北京时间）的日期
    - 如果北京时间已经过了 15:00 就用当天，否则用前一天
    """
    if override:
        return override

    # 计算北京时间
    utc_now = datetime.datetime.now(datetime.timezone.utc)
    cst = datetime.timezone(datetime.timedelta(hours=8))
    bj_now = utc_now.astimezone(cst)

    # 如果还没收盘（15:00 之前），理论上不该运行；用前一天
    if bj_now.hour < 15:
        bj_date = bj_now.date() - datetime.timedelta(days=1)
        logger.warning("北京时间 %s 尚未收盘，使用前一天 %s", bj_now.strftime("%H:%M"), bj_date)
    else:
        bj_date = bj_now.date()

    return bj_date.strftime("%Y-%m-%d")


def run_screener(
    stock_codes: List[str],
    run_date: str,
    output_dir: str,
    batch_size: int = 100,
    tushare_token: str = "",
) -> Tuple[int, float]:
    """执行筛选脚本，返回 (exit_code, elapsed_seconds)."""
    token = tushare_token or os.getenv("TUSHARE_TOKEN", "")
    if not token:
        raise ValueError("TUSHARE_TOKEN 未设置")

    codes_str = ",".join(stock_codes)
    cmd = [
        sys.executable, "-m", "a_share_hot_screener",
        "--tushare-token", token,
        "--run-date", run_date,
        "--stock-codes", codes_str,
        "--output-dir", output_dir,
        "--batch-size", str(batch_size),
        "--resume",
        "--enable-lhb-module",
    ]

    logger.info("启动筛选: %d 只股票, run_date=%s, output=%s", len(stock_codes), run_date, output_dir)
    start = time.time()

    result = subprocess.run(
        cmd,
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        timeout=7200,  # 2 小时超时
    )

    elapsed = time.time() - start
    logger.info("筛选完成: exit_code=%d, elapsed=%.1fs", result.returncode, elapsed)

    if result.returncode != 0:
        logger.error("STDOUT:\n%s", result.stdout[-2000:] if result.stdout else "(empty)")
        logger.error("STDERR:\n%s", result.stderr[-2000:] if result.stderr else "(empty)")

    return result.returncode, elapsed


def parse_summary_csv(output_dir: str, run_date: str) -> Tuple[List[Dict], int, int]:
    """解析 summary CSV，返回 (passed_stocks, total_input, passed_hard_filter).

    先找合并文件（batch_runner 的输出），再 fallback 到单批文件。
    """
    summary_file = Path(output_dir) / f"{run_date}_stage1_hot_summary.csv"
    if not summary_file.exists():
        logger.error("Summary 文件不存在: %s", summary_file)
        return [], 0, 0

    passed = []
    total = 0
    hard_filter_passed = 0

    with open(summary_file, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            total += 1
            if row.get("passed_hard_filter", "").lower() == "true":
                hard_filter_passed += 1
            if row.get("pass_stage1", "").lower() == "true":
                passed.append({
                    "code": row.get("code", ""),
                    "name": row.get("name", ""),
                    "industry": row.get("industry", ""),
                    "total_score": float(row.get("total_score", 0) or 0),
                    "hot_theme_score": float(row.get("hot_theme_score", 0) or 0),
                    "trend_flow_score": float(row.get("trend_flow_score", 0) or 0),
                    "liquidity_execution_score": float(row.get("liquidity_execution_score", 0) or 0),
                    "risk_control_score": float(row.get("risk_control_score", 0) or 0),
                    "return_5d": row.get("return_5d", ""),
                    "return_10d": row.get("return_10d", ""),
                    "concept_names_str": row.get("concept_names_str", ""),
                })

    logger.info("解析结果: total=%d, hard_filter_passed=%d, stage1_passed=%d", total, hard_filter_passed, len(passed))
    return passed, total, hard_filter_passed


def send_discord_notification(
    run_date: str,
    passed_stocks: List[Dict],
    total_input: int,
    passed_hard_filter: int,
    elapsed_seconds: float,
    summary_csv_path: Optional[str] = None,
    error_msg: Optional[str] = None,
) -> bool:
    """发送筛选结果到 Discord."""
    from a_share_hot_screener.discord_notifier import DiscordNotifier, format_screener_results

    bot_token = os.getenv("DISCORD_BOT_TOKEN", "")
    channel_id = os.getenv("DISCORD_CHANNEL_ID", "")
    user_id = os.getenv("DISCORD_USER_ID", "")

    if not bot_token:
        logger.error("DISCORD_BOT_TOKEN 未设置，跳过通知")
        return False

    notifier = DiscordNotifier(
        bot_token=bot_token,
        channel_id=channel_id,
        user_id=user_id,
    )

    # 错误情况
    if error_msg:
        return notifier.send_message(
            content=f"❌ **筛选失败** — {run_date}\n```\n{error_msg[:1500]}\n```"
        )

    # 正常结果
    embeds = format_screener_results(
        run_date=run_date,
        passed_stocks=passed_stocks,
        total_input=total_input,
        passed_hard_filter=passed_hard_filter,
        elapsed_seconds=elapsed_seconds,
    )

    success = notifier.send_message(embeds=embeds)

    # 上传 CSV 文件
    if success and summary_csv_path and Path(summary_csv_path).exists():
        notifier.send_file(
            summary_csv_path,
            content=f"📎 完整结果 CSV — {run_date}",
        )

    return success


def main() -> int:
    """主入口."""
    import argparse

    parser = argparse.ArgumentParser(description="每日自动筛选运行器")
    parser.add_argument("--date", default=None, help="指定运行日期 (YYYY-MM-DD)")
    parser.add_argument("--dry-run", action="store_true", help="只检查，不实际运行")
    parser.add_argument("--skip-weekend-check", action="store_true", help="跳过周末检查")
    parser.add_argument("--batch-size", type=int, default=100, help="每批股票数")
    parser.add_argument("--no-discord", action="store_true", help="跳过 Discord 通知")
    args = parser.parse_args()

    # 加载环境变量
    load_env()

    # 确定运行日期
    run_date = get_run_date(args.date)
    logger.info("运行日期: %s", run_date)

    # 周末检查
    if not args.skip_weekend_check and not is_trading_day(run_date):
        logger.info("非交易日，退出")
        return 0

    # 加载股票池
    try:
        stock_codes = load_universe()
    except FileNotFoundError as e:
        logger.error(str(e))
        return 1

    if not stock_codes:
        logger.error("股票池为空")
        return 1

    # 输出目录
    output_dir = str(OUTPUT_BASE / f"daily_{run_date.replace('-', '')}")
    os.makedirs(output_dir, exist_ok=True)

    if args.dry_run:
        logger.info("[DRY RUN] 将运行: %d 只股票, date=%s, output=%s", len(stock_codes), run_date, output_dir)
        return 0

    # 运行筛选
    exit_code, elapsed = run_screener(
        stock_codes=stock_codes,
        run_date=run_date,
        output_dir=output_dir,
        batch_size=args.batch_size,
    )

    if exit_code != 0:
        logger.error("筛选运行失败 (exit_code=%d)", exit_code)
        if not args.no_discord:
            send_discord_notification(
                run_date=run_date,
                passed_stocks=[],
                total_input=len(stock_codes),
                passed_hard_filter=0,
                elapsed_seconds=elapsed,
                error_msg=f"筛选脚本退出码 {exit_code}，请检查日志",
            )
        return exit_code

    # 解析结果
    passed_stocks, total_input, passed_hard_filter = parse_summary_csv(output_dir, run_date)
    summary_csv = str(Path(output_dir) / f"{run_date}_stage1_hot_summary.csv")

    logger.info(
        "===== 日报 =====\n"
        "日期: %s\n"
        "输入: %d 只\n"
        "通过硬筛: %d 只\n"
        "通过 Stage1: %d 只\n"
        "耗时: %.1f 分钟",
        run_date, total_input, passed_hard_filter, len(passed_stocks), elapsed / 60,
    )

    # Discord 通知
    if not args.no_discord:
        ok = send_discord_notification(
            run_date=run_date,
            passed_stocks=passed_stocks,
            total_input=total_input,
            passed_hard_filter=passed_hard_filter,
            elapsed_seconds=elapsed,
            summary_csv_path=summary_csv,
        )
        if ok:
            logger.info("Discord 通知已发送")
        else:
            logger.warning("Discord 通知发送失败")

    return 0


if __name__ == "__main__":
    sys.exit(main())
