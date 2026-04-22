#!/usr/bin/env python3
"""每日自动筛选运行器 v2 — 全流程集成.

流程:
  Step 0: 环境 + 交易日解析 (17:30 CST 数据就绪策略)
  Step 1: Universe 动态构建 (底仓 + 涨停池 + 龙虎榜 + 成交额Top + 热门板块)
  Step 2: Stage 1 筛选 (硬筛 + 四轴评分 + pass_stage1)
  Step 3: Setup Timing (if enabled, experimental)
  Step 4: Discord 推送 (tradeable/watch_only 分开展示 + CSV 附件)
  Step 5: 归档

用法:
    python3 -m a_share_hot_screener.daily_runner
    python3 -m a_share_hot_screener.daily_runner --date 2026-04-22
    python3 -m a_share_hot_screener.daily_runner --allow-partial-current-day
    python3 -m a_share_hot_screener.daily_runner --dry-run

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
from dataclasses import dataclass, field
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

# ── 常量 ──────────────────────────────────────────────────
CST = datetime.timezone(datetime.timedelta(hours=8))
DEFAULT_DATA_READY_HOUR_CST = 17
DEFAULT_DATA_READY_MINUTE_CST = 30
MARKET_CLOSE_HOUR_CST = 15
MARKET_CLOSE_MINUTE_CST = 0


# ════════════════════════════════════════════════════════
# 数据模型
# ════════════════════════════════════════════════════════

@dataclass
class TradeDate:
    """交易日解析结果."""
    trade_date_used: str            # YYYY-MM-DD
    run_datetime: str               # ISO 8601
    data_ready_policy: str          # "default_17:30" / "partial_allowed" / "explicit"
    partial_data_risk: bool = False


@dataclass
class DailyRunResult:
    """每日运行结果汇总."""
    run_date: str
    trade_date_used: str
    data_ready_policy: str
    partial_data_risk: bool = False
    universe_count: int = 0
    universe_static_count: int = 0
    universe_dynamic_count: int = 0
    total_input: int = 0
    passed_hard_filter: int = 0
    tradeable_count: int = 0
    watch_only_count: int = 0
    elapsed_seconds: float = 0.0
    output_dir: str = ""
    tradeable_stocks: List[Dict] = field(default_factory=list)
    watch_only_stocks: List[Dict] = field(default_factory=list)
    setup_timing_results: List[Dict] = field(default_factory=list)
    market_regime: str = "neutral"
    error_msg: str = ""


# ════════════════════════════════════════════════════════
# 环境 & 工具
# ════════════════════════════════════════════════════════

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


def is_trading_day(date_str: str) -> bool:
    """简易交易日检查（排除周末，节假日由 Tushare 处理）."""
    dt = datetime.datetime.strptime(date_str, "%Y-%m-%d")
    if dt.weekday() >= 5:
        logger.info("%s 是周末，跳过", date_str)
        return False
    return True


# ════════════════════════════════════════════════════════
# Step 0: 交易日解析 (v2.1 — 17:30 CST 策略)
# ════════════════════════════════════════════════════════

def resolve_complete_trade_date(
    override_date: Optional[str] = None,
    allow_partial: bool = False,
) -> TradeDate:
    """解析最近完整交易日.

    规则 (v2.1):
    - 交易日 data_ready_time(17:30 CST) 后 → 当天
    - 交易日 15:00~17:30 CST:
      - allow_partial=True  → 当天（标记 partial_data_risk=True）
      - allow_partial=False → 上一个交易日（默认，更安全）
    - 交易日盘中(<15:00 CST) → 上一个交易日
    - 非交易日(周末) → 最近的完整交易日（向前回溯）
    """
    utc_now = datetime.datetime.now(datetime.timezone.utc)
    bj_now = utc_now.astimezone(CST)
    run_datetime = bj_now.isoformat()

    if override_date:
        return TradeDate(
            trade_date_used=override_date,
            run_datetime=run_datetime,
            data_ready_policy="explicit",
            partial_data_risk=False,
        )

    bj_date = bj_now.date()
    bj_hour = bj_now.hour
    bj_minute = bj_now.minute
    bj_time_minutes = bj_hour * 60 + bj_minute

    data_ready_minutes = DEFAULT_DATA_READY_HOUR_CST * 60 + DEFAULT_DATA_READY_MINUTE_CST
    market_close_minutes = MARKET_CLOSE_HOUR_CST * 60 + MARKET_CLOSE_MINUTE_CST

    is_weekday = bj_date.weekday() < 5

    if is_weekday and bj_time_minutes >= data_ready_minutes:
        # 17:30 CST 之后 → 当天数据完整
        return TradeDate(
            trade_date_used=bj_date.isoformat(),
            run_datetime=run_datetime,
            data_ready_policy="default_17:30",
            partial_data_risk=False,
        )
    elif is_weekday and bj_time_minutes >= market_close_minutes:
        # 15:00~17:30 CST
        if allow_partial:
            logger.warning(
                "北京时间 %s（收盘后但数据未完全就绪），使用当天数据（partial_data_risk）",
                bj_now.strftime("%H:%M"),
            )
            return TradeDate(
                trade_date_used=bj_date.isoformat(),
                run_datetime=run_datetime,
                data_ready_policy="partial_allowed",
                partial_data_risk=True,
            )
        else:
            # 默认安全策略：用前一个交易日
            prev = _find_prev_weekday(bj_date)
            logger.info(
                "北京时间 %s，数据尚未完全就绪(17:30)，使用前一交易日 %s",
                bj_now.strftime("%H:%M"), prev,
            )
            return TradeDate(
                trade_date_used=prev.isoformat(),
                run_datetime=run_datetime,
                data_ready_policy="default_17:30",
                partial_data_risk=False,
            )
    else:
        # 盘中或非交易日 → 回溯到最近交易日
        if is_weekday and bj_time_minutes < market_close_minutes:
            # 盘中 → 用前一天
            target = _find_prev_weekday(bj_date)
        else:
            # 周末 → 回溯到周五
            target = _find_prev_weekday(bj_date)
        logger.info("非交易完成时段，使用 %s", target)
        return TradeDate(
            trade_date_used=target.isoformat(),
            run_datetime=run_datetime,
            data_ready_policy="default_17:30",
            partial_data_risk=False,
        )


def _find_prev_weekday(date: datetime.date) -> datetime.date:
    """找到 date 之前的最近一个工作日."""
    d = date - datetime.timedelta(days=1)
    while d.weekday() >= 5:
        d -= datetime.timedelta(days=1)
    return d


# ════════════════════════════════════════════════════════
# Step 1: Universe
# ════════════════════════════════════════════════════════

def load_universe_static() -> List[str]:
    """加载静态股票池 (fallback)."""
    universe_file = UNIVERSE_DIR / "full_universe.txt"
    if not universe_file.exists():
        raise FileNotFoundError(f"Universe 文件不存在: {universe_file}")
    codes = []
    with open(universe_file) as f:
        for line in f:
            code = line.strip()
            if code and not code.startswith("#"):
                codes.append(code)
    return codes


# ════════════════════════════════════════════════════════
# Step 2: 运行筛选
# ════════════════════════════════════════════════════════

def run_screener(
    stock_codes: List[str],
    run_date: str,
    output_dir: str,
    batch_size: int = 100,
    tushare_token: str = "",
    enable_setup_timing: bool = False,
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

    if enable_setup_timing:
        cmd.append("--enable-setup-timing")

    logger.info("启动筛选: %d 只股票, run_date=%s, output=%s", len(stock_codes), run_date, output_dir)
    start = time.time()

    result = subprocess.run(
        cmd,
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        timeout=7200,
    )

    elapsed = time.time() - start
    logger.info("筛选完成: exit_code=%d, elapsed=%.1fs", result.returncode, elapsed)

    if result.returncode != 0:
        logger.error("STDOUT:\n%s", result.stdout[-2000:] if result.stdout else "(empty)")
        logger.error("STDERR:\n%s", result.stderr[-2000:] if result.stderr else "(empty)")

    return result.returncode, elapsed


# ════════════════════════════════════════════════════════
# Step 3: 解析结果
# ════════════════════════════════════════════════════════

def parse_summary_csv(output_dir: str, run_date: str) -> Tuple[List[Dict], List[Dict], int, int]:
    """解析 summary CSV，返回 (tradeable, watch_only, total_input, passed_hard_filter).

    v2: 分别提取 tradeable 和 watch_only 候选。
    """
    summary_file = Path(output_dir) / f"{run_date}_stage1_hot_summary.csv"
    if not summary_file.exists():
        logger.error("Summary 文件不存在: %s", summary_file)
        return [], [], 0, 0

    tradeable = []
    watch_only = []
    total = 0
    hard_filter_passed = 0

    with open(summary_file, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            total += 1
            if row.get("passed_hard_filter", "").lower() == "true":
                hard_filter_passed += 1

            stock_data = {
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
                "candidate_pool_type": row.get("candidate_pool_type", ""),
                "candidate_pool_reason": row.get("candidate_pool_reason", ""),
                # v2 新增
                "timing_score": row.get("timing_score", ""),
                "timing_action": row.get("timing_action", ""),
            }

            if row.get("pass_stage1", "").lower() == "true":
                tradeable.append(stock_data)
            elif row.get("pass_stage1_watch", "").lower() == "true":
                watch_only.append(stock_data)

    logger.info(
        "解析结果: total=%d, hard_filter=%d, tradeable=%d, watch_only=%d",
        total, hard_filter_passed, len(tradeable), len(watch_only),
    )
    return tradeable, watch_only, total, hard_filter_passed


def parse_setup_timing_csv(output_dir: str, run_date: str) -> List[Dict]:
    """解析 setup_timing CSV."""
    csv_path = Path(output_dir) / f"{run_date}_setup_timing.csv"
    if not csv_path.exists():
        return []

    results = []
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            results.append({
                "code": row.get("code", ""),
                "name": row.get("name", ""),
                "timing_score": float(row.get("timing_score", 0) or 0),
                "action": row.get("action", ""),
                "support_zone_low": row.get("support_zone_low", ""),
                "support_zone_high": row.get("support_zone_high", ""),
                "invalidation_level": row.get("invalidation_level", ""),
                "resistance_1": row.get("resistance_1", ""),
                "ref_reward_risk": row.get("ref_reward_risk", ""),
                "level_confidence": row.get("level_confidence", ""),
                "support_basis": row.get("support_basis", ""),
                "reason": row.get("reason", ""),
            })
    logger.info("解析 setup_timing: %d 只", len(results))
    return results


# ════════════════════════════════════════════════════════
# Step 4: Discord 推送 (v2 — 分开展示)
# ════════════════════════════════════════════════════════

def format_overview_embed(result: DailyRunResult) -> Dict:
    """消息 1: 概览 Embed."""
    elapsed_min = result.elapsed_seconds / 60
    dynamic_count = result.universe_count - result.universe_static_count

    lines = [
        f"📊 **Universe**: {result.universe_count} 只"
        + (f" (底仓{result.universe_static_count} + 动态{dynamic_count})" if dynamic_count > 0 else ""),
        f"🔍 **通过硬筛**: {result.passed_hard_filter} 只",
        f"✅ **Tradeable**: **{result.tradeable_count}** 只"
        + (f" | 👁️ **Watch**: {result.watch_only_count} 只" if result.watch_only_count > 0 else ""),
        f"⏱️ **耗时**: {elapsed_min:.1f} 分钟",
        f"📋 **数据策略**: {result.data_ready_policy}"
        + (f" | ⚠️ partial_risk" if result.partial_data_risk else ""),
    ]

    return {
        "title": f"🔥 A股热点筛选日报 — {result.run_date}",
        "description": "\n".join(lines),
        "color": 0xFF6B35 if result.tradeable_count > 0 else 0x808080,
        "footer": {"text": "A-Share Hot Screener v2 | Stage 1"},
    }


def format_tradeable_embed(stocks: List[Dict]) -> List[Dict]:
    """消息 2A: Tradeable 候选."""
    if not stocks:
        return []

    sorted_stocks = sorted(stocks, key=lambda s: s.get("total_score", 0), reverse=True)
    lines = []
    for i, s in enumerate(sorted_stocks[:25], 1):
        code = s.get("code", "?")
        name = s.get("name", "?")
        score = s.get("total_score", 0)
        ht = s.get("hot_theme_score", 0)
        tf = s.get("trend_flow_score", 0)
        le = s.get("liquidity_execution_score", 0)
        rc = s.get("risk_control_score", 0)
        industry = s.get("industry", "?")
        concepts = s.get("concept_names_str", "")

        line = (
            f"**{i}. {name}** (`{code}`) — ⭐ {score:.4f}\n"
            f"　　HT={ht:.2f} TF={tf:.2f} LE={le:.2f} RC={rc:.2f} | {industry}"
        )
        if concepts:
            line += f"\n　　📎 {concepts}"

        # timing 信息
        timing_action = s.get("timing_action", "")
        timing_score = s.get("timing_score", "")
        if timing_action and timing_score:
            action_emoji = {"setup_ready": "🟢", "watch": "🟡", "wait": "⏳", "avoid_chase": "⛔"}.get(timing_action, "")
            line += f"\n　　{action_emoji} 时机: {timing_action} ({timing_score})"

        lines.append(line)

    embeds = []
    chunk = []
    chunk_len = 0
    for line in lines:
        if chunk_len + len(line) + 1 > 3800:
            embeds.append({
                "description": "\n".join(chunk),
                "color": 0x00C853,
            })
            chunk = []
            chunk_len = 0
        chunk.append(line)
        chunk_len += len(line) + 1

    if chunk:
        embeds.append({
            "title": f"📈 Tradeable 候选 ({len(sorted_stocks)}只, 按总分排序)",
            "description": "\n".join(chunk),
            "color": 0x00C853,
        })

    return embeds


def format_watch_only_embed(stocks: List[Dict]) -> Optional[Dict]:
    """消息 2B: Watch-Only 候选."""
    if not stocks:
        return None

    sorted_stocks = sorted(stocks, key=lambda s: s.get("total_score", 0), reverse=True)
    lines = []
    for i, s in enumerate(sorted_stocks[:10], 1):
        code = s.get("code", "?")
        name = s.get("name", "?")
        score = s.get("total_score", 0)
        reason = s.get("candidate_pool_reason", "")

        line = f"**{i}. {name}** (`{code}`) — ⭐ {score:.4f}"
        if reason:
            line += f" | ⚠️ {reason}"
        lines.append(line)

    return {
        "title": f"👁️ Watch-Only 高辨识度 ({len(sorted_stocks)}只, 仅观察不追高)",
        "description": "\n".join(lines),
        "color": 0xFFA000,
    }


def format_setup_timing_embed(timing_results: List[Dict]) -> Optional[Dict]:
    """消息 3: 观察时机评估 (experimental)."""
    if not timing_results:
        return None

    groups = {
        "setup_ready": [],
        "watch": [],
        "wait": [],
        "avoid_chase": [],
    }
    for r in timing_results:
        action = r.get("action", "wait")
        groups.setdefault(action, []).append(r)

    lines = []

    # setup_ready
    for r in sorted(groups.get("setup_ready", []), key=lambda x: -x.get("timing_score", 0)):
        code = r.get("code", "")
        name = r.get("name", "")
        score = r.get("timing_score", 0)
        support = f"{r.get('support_zone_low', '')}~{r.get('support_zone_high', '')}"
        conf = r.get("level_confidence", "")
        basis = r.get("support_basis", "")
        inv = r.get("invalidation_level", "")
        rrr = r.get("ref_reward_risk", "")
        reason = r.get("reason", "")
        lines.append(
            f"🟢 **{name}**(`{code}`) {score:.0f}分"
            f" | 支撑{support}({basis}) [置信:{conf}]"
            f" | 失效{inv} | 盈亏比{rrr}"
            f"\n　　→ {reason}"
        )

    # watch
    for r in sorted(groups.get("watch", []), key=lambda x: -x.get("timing_score", 0)):
        name = r.get("name", "")
        code = r.get("code", "")
        score = r.get("timing_score", 0)
        reason = r.get("reason", "")
        lines.append(f"🟡 **{name}**(`{code}`) {score:.0f}分 | {reason}")

    # wait / avoid_chase 简化
    wait_names = [f"{r.get('name', '')}({r.get('timing_score', 0):.0f})" for r in groups.get("wait", [])]
    if wait_names:
        lines.append(f"⏳ 等待: {', '.join(wait_names[:10])}")

    avoid_names = [f"{r.get('name', '')}" for r in groups.get("avoid_chase", [])]
    if avoid_names:
        lines.append(f"⛔ 避免追高: {', '.join(avoid_names[:10])}")

    if not lines:
        return None

    desc = "\n".join(lines)
    if len(desc) > 4000:
        desc = desc[:3990] + "\n..."

    return {
        "title": "⏰ 观察时机评估 🧪 (3~5天 / 低吸回踩)",
        "description": desc,
        "color": 0x2196F3,
    }


def send_discord_v2(result: DailyRunResult) -> bool:
    """发送 v2 格式的 Discord 通知."""
    from a_share_hot_screener.discord_notifier import DiscordNotifier

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
    if result.error_msg:
        return notifier.send_message(
            content=f"❌ **筛选失败** — {result.run_date}\n```\n{result.error_msg[:1500]}\n```"
        )

    # 消息 1: 概览
    embeds = [format_overview_embed(result)]

    # 消息 2A: Tradeable
    tradeable_embeds = format_tradeable_embed(result.tradeable_stocks)
    embeds.extend(tradeable_embeds)

    # 消息 2B: Watch-Only
    watch_embed = format_watch_only_embed(result.watch_only_stocks)
    if watch_embed:
        embeds.append(watch_embed)

    # 消息 3: Setup Timing (experimental)
    timing_embed = format_setup_timing_embed(result.setup_timing_results)
    if timing_embed:
        embeds.append(timing_embed)

    # Discord 限制: 每条消息最多 10 个 embed
    success = True
    for i in range(0, len(embeds), 10):
        batch = embeds[i:i + 10]
        ok = notifier.send_message(embeds=batch)
        if not ok:
            success = False

    # CSV 附件 — 失败不阻塞
    csv_files = [
        (Path(result.output_dir) / f"{result.run_date}_stage1_hot_summary.csv",
         f"📎 完整结果 CSV — {result.run_date}"),
    ]
    timing_csv = Path(result.output_dir) / f"{result.run_date}_setup_timing.csv"
    if timing_csv.exists():
        csv_files.append((timing_csv, f"⏰ 观察时机 CSV — {result.run_date}"))

    for csv_path, caption in csv_files:
        if csv_path.exists():
            try:
                notifier.send_file(str(csv_path), content=caption)
            except Exception as e:
                logger.warning("CSV 附件上传失败 (%s): %s", csv_path.name, e)

    return success


# ════════════════════════════════════════════════════════
# 主入口
# ════════════════════════════════════════════════════════

def main() -> int:
    """主入口."""
    import argparse

    parser = argparse.ArgumentParser(description="每日自动筛选运行器 v2")
    parser.add_argument("--date", default=None, help="指定运行日期 (YYYY-MM-DD)")
    parser.add_argument("--dry-run", action="store_true", help="只检查，不实际运行")
    parser.add_argument("--skip-weekend-check", action="store_true", help="跳过周末检查")
    parser.add_argument("--batch-size", type=int, default=100, help="每批股票数")
    parser.add_argument("--no-discord", action="store_true", help="跳过 Discord 通知")
    parser.add_argument("--allow-partial-current-day", action="store_true",
                        help="允许使用当天部分数据(15:00~17:30 CST)")
    parser.add_argument("--enable-setup-timing", action="store_true",
                        help="启用观察时机评估 (experimental)")
    parser.add_argument("--use-universe-builder", action="store_true",
                        help="使用动态 Universe Builder (需要更多 Tushare API)")
    args = parser.parse_args()

    # 加载环境变量
    load_env()

    # Step 0: 交易日解析
    trade_date = resolve_complete_trade_date(
        override_date=args.date,
        allow_partial=args.allow_partial_current_day,
    )
    logger.info(
        "交易日解析: trade_date_used=%s, policy=%s, partial_risk=%s",
        trade_date.trade_date_used, trade_date.data_ready_policy, trade_date.partial_data_risk,
    )

    run_date = trade_date.trade_date_used

    # 周末检查
    if not args.skip_weekend_check and not is_trading_day(run_date):
        logger.info("非交易日，退出")
        return 0

    # Step 1: Universe
    universe_result = None
    if args.use_universe_builder:
        try:
            from a_share_hot_screener.cache import LocalCache
            from a_share_hot_screener.clients.tushare_client import TushareClient
            from a_share_hot_screener.trade_calendar import TradeCalendar
            from a_share_hot_screener.universe_builder import UniverseBuilder

            token = os.getenv("TUSHARE_TOKEN", "")
            cache = LocalCache(os.path.join(os.path.expanduser("~"), ".a_share_hot_screener", "cache"))
            client = TushareClient(token=token, cache=cache)
            trade_cal = TradeCalendar(
                datetime.date.fromisoformat(run_date), cache=cache
            )
            trade_cal.load(tushare_client=client)
            trade_cal.resolve()

            builder = UniverseBuilder(
                tushare_client=client,
                trade_calendar=trade_cal,
                universe_dir=str(UNIVERSE_DIR),
            )
            universe_result = builder.build(run_date.replace("-", ""))
            stock_codes = universe_result.codes
            logger.info(
                "Universe Builder: %d 只 (底仓%d + 涨停池%d + 龙虎榜%d + 成交额Top%d + 热门板块%d)",
                universe_result.total_count,
                universe_result.static_count,
                universe_result.zt_pool_count,
                universe_result.lhb_count,
                universe_result.amount_top_count,
                universe_result.ths_hot_count,
            )
        except Exception as e:
            logger.warning("Universe Builder 失败，fallback 到静态 universe: %s", e)
            stock_codes = load_universe_static()
    else:
        stock_codes = load_universe_static()
        logger.info("使用静态 universe: %d 只", len(stock_codes))

    if not stock_codes:
        logger.error("股票池为空")
        return 1

    # 输出目录
    output_dir = str(OUTPUT_BASE / f"daily_{run_date.replace('-', '')}")
    os.makedirs(output_dir, exist_ok=True)

    if args.dry_run:
        logger.info(
            "[DRY RUN] 将运行: %d 只, date=%s, policy=%s, setup_timing=%s, output=%s",
            len(stock_codes), run_date, trade_date.data_ready_policy,
            args.enable_setup_timing, output_dir,
        )
        return 0

    # Step 2: 运行筛选
    exit_code, elapsed = run_screener(
        stock_codes=stock_codes,
        run_date=run_date,
        output_dir=output_dir,
        batch_size=args.batch_size,
        enable_setup_timing=args.enable_setup_timing,
    )

    # 构建结果
    result = DailyRunResult(
        run_date=run_date,
        trade_date_used=trade_date.trade_date_used,
        data_ready_policy=trade_date.data_ready_policy,
        partial_data_risk=trade_date.partial_data_risk,
        elapsed_seconds=elapsed,
        output_dir=output_dir,
        universe_count=len(stock_codes),
        universe_static_count=universe_result.static_count if universe_result else len(stock_codes),
    )

    if exit_code != 0:
        result.error_msg = f"筛选脚本退出码 {exit_code}，请检查日志"
        logger.error("筛选运行失败 (exit_code=%d)", exit_code)
        if not args.no_discord:
            send_discord_v2(result)
        return exit_code

    # Step 3: 解析结果
    tradeable, watch_only, total_input, passed_hard_filter = parse_summary_csv(output_dir, run_date)
    result.total_input = total_input
    result.passed_hard_filter = passed_hard_filter
    result.tradeable_count = len(tradeable)
    result.watch_only_count = len(watch_only)
    result.tradeable_stocks = tradeable
    result.watch_only_stocks = watch_only

    # Setup timing
    if args.enable_setup_timing:
        timing_results = parse_setup_timing_csv(output_dir, run_date)
        result.setup_timing_results = timing_results

    logger.info(
        "===== 日报 v2 =====\n"
        "日期: %s (policy=%s)\n"
        "Universe: %d 只\n"
        "通过硬筛: %d 只\n"
        "Tradeable: %d 只\n"
        "Watch-Only: %d 只\n"
        "Setup Timing: %d 只\n"
        "耗时: %.1f 分钟",
        run_date, trade_date.data_ready_policy,
        len(stock_codes),
        passed_hard_filter,
        len(tradeable),
        len(watch_only),
        len(result.setup_timing_results),
        elapsed / 60,
    )

    # Step 4: Discord
    if not args.no_discord:
        ok = send_discord_v2(result)
        if ok:
            logger.info("Discord 通知已发送")
        else:
            logger.warning("Discord 通知发送失败")

    return 0


if __name__ == "__main__":
    sys.exit(main())
