"""时序连续性 / 趋势加速信号模块（Session 14 P2-6）.

功能：
  加载上次运行的 summary CSV，与本次运行结果比较，生成 delta 信号。
  用于发现评分变化趋势、新进/退出 pass_stage1 的股票，辅助第二阶段决策。

设计原则：
  1. 纯只读：不修改上次运行文件，不影响当前评分逻辑
  2. 可选模块：prev_run_dir 为空时跳过，不影响现有流程
  3. 容错：上次运行文件缺失/格式错误时降级为 warnings，不中断
  4. delta 信号只做客观计算，不做主观判断（遵守第一阶段约束）
"""

from __future__ import annotations

import csv
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger("a_share_hot_screener.trend_compare")


# ════════════════════════════════════════════════════════
# 数据结构
# ════════════════════════════════════════════════════════

@dataclass
class TrendDelta:
    """单只股票与上次运行的时序比较结果.

    所有 delta 字段为 (本次 - 上次)，正值=改善/升高，负值=恶化/降低。
    状态变化字段为枚举字符串。
    """

    prev_run_date: str = ""                          # 上次运行的 trade_date_used

    # ── 评分 delta（本次 - 上次）──────────────────────
    total_score_delta: Optional[float] = None        # total_score 变化
    hot_theme_score_delta: Optional[float] = None
    trend_flow_score_delta: Optional[float] = None
    liquidity_execution_score_delta: Optional[float] = None
    risk_control_score_delta: Optional[float] = None

    # ── 上次评分快照（便于溯源）──────────────────────
    prev_total_score: Optional[float] = None
    prev_hot_theme_score: Optional[float] = None
    prev_trend_flow_score: Optional[float] = None
    prev_liquidity_execution_score: Optional[float] = None
    prev_risk_control_score: Optional[float] = None

    # ── pass_stage1 状态变化 ───────────────────────────
    # "new_pass"  : 上次未通过/不存在 → 本次通过
    # "lost_pass" : 上次通过 → 本次未通过
    # "keep_pass" : 连续通过
    # "keep_fail" : 连续未通过
    # "new_entry" : 上次运行中不存在此股票（首次入选）
    pass_stage1_change: str = ""

    # ── hard_filter 状态变化 ──────────────────────────
    # "new_pass"  : 上次未通过/不存在 → 本次通过
    # "lost_pass" : 上次通过 → 本次未通过
    # "keep_pass" / "keep_fail" / "new_entry"
    hard_filter_change: str = ""

    # ── blocked_by 变化 ──────────────────────────────
    prev_blocked_by: List[str] = field(default_factory=list)   # 上次的 blocked_by
    newly_unblocked: List[str] = field(default_factory=list)   # 本次不再 blocked 的轴
    newly_blocked: List[str] = field(default_factory=list)     # 本次新增 blocked 的轴

    # ── 关键指标 delta ──────────────────────────────────
    return_5d_delta: Optional[float] = None          # return_5d 变化
    return_10d_delta: Optional[float] = None         # return_10d 变化
    amount_avg_5d_delta: Optional[float] = None      # amount_avg_5d 变化

    # ── 趋势加速信号（bool flags）─────────────────────
    score_accelerating: Optional[bool] = None        # total_score 上升且连续通过
    score_decelerating: Optional[bool] = None        # total_score 下降且连续通过

    def to_dict(self) -> Dict[str, Any]:
        """序列化为可写入 JSON/CSV 的 dict."""
        return {
            "prev_run_date": self.prev_run_date,
            "total_score_delta": _round_opt(self.total_score_delta),
            "hot_theme_score_delta": _round_opt(self.hot_theme_score_delta),
            "trend_flow_score_delta": _round_opt(self.trend_flow_score_delta),
            "liquidity_execution_score_delta": _round_opt(self.liquidity_execution_score_delta),
            "risk_control_score_delta": _round_opt(self.risk_control_score_delta),
            "prev_total_score": _round_opt(self.prev_total_score),
            "prev_hot_theme_score": _round_opt(self.prev_hot_theme_score),
            "prev_trend_flow_score": _round_opt(self.prev_trend_flow_score),
            "prev_liquidity_execution_score": _round_opt(self.prev_liquidity_execution_score),
            "prev_risk_control_score": _round_opt(self.prev_risk_control_score),
            "pass_stage1_change": self.pass_stage1_change,
            "hard_filter_change": self.hard_filter_change,
            "prev_blocked_by": self.prev_blocked_by,
            "newly_unblocked": self.newly_unblocked,
            "newly_blocked": self.newly_blocked,
            "return_5d_delta": _round_opt(self.return_5d_delta),
            "return_10d_delta": _round_opt(self.return_10d_delta),
            "amount_avg_5d_delta": _round_opt(self.amount_avg_5d_delta),
            "score_accelerating": self.score_accelerating,
            "score_decelerating": self.score_decelerating,
        }


# ════════════════════════════════════════════════════════
# 上次运行数据加载
# ════════════════════════════════════════════════════════

@dataclass
class PrevRunSnapshot:
    """上次运行的精简快照（从 summary CSV 加载）."""
    trade_date_used: str = ""
    stocks: Dict[str, Dict[str, Any]] = field(default_factory=dict)  # code → row dict


def load_prev_run(prev_run_dir: str, prev_run_date: str = "") -> Optional[PrevRunSnapshot]:
    """从上次运行的输出目录加载 summary CSV.

    搜索策略：
      1. 如果 prev_run_date 非空，直接找 {prev_run_date}_stage1_hot_summary.csv
      2. 否则扫描目录找最新的 *_stage1_hot_summary.csv

    Returns:
        PrevRunSnapshot 或 None（目录不存在/无有效文件）
    """
    if not prev_run_dir or not os.path.isdir(prev_run_dir):
        return None

    summary_path = ""

    if prev_run_date:
        candidate = os.path.join(prev_run_dir, f"{prev_run_date}_stage1_hot_summary.csv")
        if os.path.isfile(candidate):
            summary_path = candidate

    if not summary_path:
        # 扫描目录找最新的 summary 文件
        candidates = []
        for fname in os.listdir(prev_run_dir):
            if fname.endswith("_stage1_hot_summary.csv"):
                candidates.append(fname)
        if not candidates:
            logger.warning("[trend_compare] 未在 %s 找到 summary CSV", prev_run_dir)
            return None
        # 按文件名降序（日期前缀自然排序）
        candidates.sort(reverse=True)
        summary_path = os.path.join(prev_run_dir, candidates[0])

    logger.info("[trend_compare] 加载上次运行: %s", summary_path)
    return _parse_summary_csv(summary_path)


def _parse_summary_csv(path: str) -> Optional[PrevRunSnapshot]:
    """解析 summary CSV 为 PrevRunSnapshot."""
    try:
        with open(path, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
    except Exception as e:
        logger.warning("[trend_compare] 读取 summary CSV 失败(%s): %s", path, e)
        return None

    if not rows:
        logger.warning("[trend_compare] summary CSV 为空: %s", path)
        return None

    # 从文件名提取 trade_date_used
    fname = os.path.basename(path)
    date_prefix = fname.split("_stage1_hot_summary")[0]  # e.g. "2026-04-17"

    snapshot = PrevRunSnapshot(trade_date_used=date_prefix)
    for row in rows:
        code = row.get("code", "").strip()
        if not code:
            continue
        # 解析关键字段为正确类型
        parsed = {
            "code": code,
            "name": row.get("name", ""),
            "pass_stage1": _parse_bool(row.get("pass_stage1", "")),
            "passed_hard_filter": _parse_bool(row.get("passed_hard_filter", "")),
            "total_score": _parse_float(row.get("total_score", "")),
            "hot_theme_score": _parse_float(row.get("hot_theme_score", "")),
            "trend_flow_score": _parse_float(row.get("trend_flow_score", "")),
            "liquidity_execution_score": _parse_float(row.get("liquidity_execution_score", "")),
            "risk_control_score": _parse_float(row.get("risk_control_score", "")),
            "blocked_by": _parse_list(row.get("blocked_by", "")),
            "return_5d": _parse_float(row.get("return_5d", "")),
            "return_10d": _parse_float(row.get("return_10d", "")),
            "amount_avg_5d": _parse_float(row.get("amount_avg_5d", "")),
        }
        snapshot.stocks[code] = parsed

    logger.info(
        "[trend_compare] 已加载 %d 只股票快照 (trade_date=%s)",
        len(snapshot.stocks), snapshot.trade_date_used,
    )
    return snapshot


# ════════════════════════════════════════════════════════
# Delta 计算
# ════════════════════════════════════════════════════════

def compute_trend_delta(
    code: str,
    current_total_score: Optional[float],
    current_hot_theme_score: Optional[float],
    current_trend_flow_score: Optional[float],
    current_liquidity_execution_score: Optional[float],
    current_risk_control_score: Optional[float],
    current_pass_stage1: bool,
    current_passed_hard_filter: bool,
    current_blocked_by: List[str],
    current_return_5d: Optional[float],
    current_return_10d: Optional[float],
    current_amount_avg_5d: Optional[float],
    prev_snapshot: PrevRunSnapshot,
) -> TrendDelta:
    """计算单只股票与上次运行的 delta.

    Args:
        code: 6位纯数字股票代码
        current_*: 本次运行的各项指标值
        prev_snapshot: 上次运行的快照数据

    Returns:
        TrendDelta 对象
    """
    delta = TrendDelta(prev_run_date=prev_snapshot.trade_date_used)

    prev = prev_snapshot.stocks.get(code)
    if prev is None:
        # 上次运行中不存在此股票
        delta.pass_stage1_change = "new_entry"
        delta.hard_filter_change = "new_entry"
        return delta

    # ── 评分 delta ────────────────────────────────────
    prev_total = prev.get("total_score")
    prev_ht = prev.get("hot_theme_score")
    prev_tf = prev.get("trend_flow_score")
    prev_le = prev.get("liquidity_execution_score")
    prev_rc = prev.get("risk_control_score")

    delta.prev_total_score = prev_total
    delta.prev_hot_theme_score = prev_ht
    delta.prev_trend_flow_score = prev_tf
    delta.prev_liquidity_execution_score = prev_le
    delta.prev_risk_control_score = prev_rc

    delta.total_score_delta = _safe_delta(current_total_score, prev_total)
    delta.hot_theme_score_delta = _safe_delta(current_hot_theme_score, prev_ht)
    delta.trend_flow_score_delta = _safe_delta(current_trend_flow_score, prev_tf)
    delta.liquidity_execution_score_delta = _safe_delta(current_liquidity_execution_score, prev_le)
    delta.risk_control_score_delta = _safe_delta(current_risk_control_score, prev_rc)

    # ── pass_stage1 状态变化 ──────────────────────────
    prev_pass = prev.get("pass_stage1", False)
    if prev_pass and current_pass_stage1:
        delta.pass_stage1_change = "keep_pass"
    elif not prev_pass and current_pass_stage1:
        delta.pass_stage1_change = "new_pass"
    elif prev_pass and not current_pass_stage1:
        delta.pass_stage1_change = "lost_pass"
    else:
        delta.pass_stage1_change = "keep_fail"

    # ── hard_filter 状态变化 ─────────────────────────
    prev_hf = prev.get("passed_hard_filter", False)
    if prev_hf and current_passed_hard_filter:
        delta.hard_filter_change = "keep_pass"
    elif not prev_hf and current_passed_hard_filter:
        delta.hard_filter_change = "new_pass"
    elif prev_hf and not current_passed_hard_filter:
        delta.hard_filter_change = "lost_pass"
    else:
        delta.hard_filter_change = "keep_fail"

    # ── blocked_by 变化 ──────────────────────────────
    prev_blocked = set(prev.get("blocked_by", []))
    curr_blocked = set(current_blocked_by)
    delta.prev_blocked_by = sorted(prev_blocked)
    delta.newly_unblocked = sorted(prev_blocked - curr_blocked)
    delta.newly_blocked = sorted(curr_blocked - prev_blocked)

    # ── 关键指标 delta ──────────────────────────────────
    delta.return_5d_delta = _safe_delta(current_return_5d, prev.get("return_5d"))
    delta.return_10d_delta = _safe_delta(current_return_10d, prev.get("return_10d"))
    delta.amount_avg_5d_delta = _safe_delta(current_amount_avg_5d, prev.get("amount_avg_5d"))

    # ── 趋势加速信号 ──────────────────────────────────
    if delta.pass_stage1_change == "keep_pass" and delta.total_score_delta is not None:
        delta.score_accelerating = delta.total_score_delta > 0.01  # 阈值 1%
        delta.score_decelerating = delta.total_score_delta < -0.01

    return delta


# ════════════════════════════════════════════════════════
# 批量计算入口
# ════════════════════════════════════════════════════════

def compute_all_deltas(
    details: "List[HotStockDetail]",  # type: ignore[name-defined]
    prev_snapshot: PrevRunSnapshot,
) -> Dict[str, TrendDelta]:
    """为所有股票计算 trend delta.

    Args:
        details: 本次运行的全量 HotStockDetail 列表
        prev_snapshot: 上次运行快照

    Returns:
        {code: TrendDelta} 映射
    """
    deltas: Dict[str, TrendDelta] = {}
    for d in details:
        delta = compute_trend_delta(
            code=d.code,
            current_total_score=d.total_score,
            current_hot_theme_score=d.hot_theme_score,
            current_trend_flow_score=d.trend_flow_score,
            current_liquidity_execution_score=d.liquidity_execution_score,
            current_risk_control_score=d.risk_control_score,
            current_pass_stage1=d.pass_stage1,
            current_passed_hard_filter=d.passed_hard_filter,
            current_blocked_by=d.blocked_by,
            current_return_5d=d.return_5d,
            current_return_10d=d.return_10d,
            current_amount_avg_5d=d.amount_avg_5d,
            prev_snapshot=prev_snapshot,
        )
        deltas[d.code] = delta
    return deltas


# ════════════════════════════════════════════════════════
# 工具函数
# ════════════════════════════════════════════════════════

def _safe_delta(
    current: Optional[float],
    prev: Optional[float],
) -> Optional[float]:
    """计算安全差值，任一方 None 返回 None."""
    if current is None or prev is None:
        return None
    return round(current - prev, 4)


def _parse_float(s: str) -> Optional[float]:
    """解析 CSV 字符串为 float，空/无效返回 None."""
    s = s.strip()
    if not s or s.lower() in ("none", "nan", ""):
        return None
    try:
        v = float(s)
        return v if v == v else None  # NaN check
    except (ValueError, TypeError):
        return None


def _parse_bool(s: str) -> bool:
    """解析 CSV 字符串为 bool."""
    return s.strip().lower() in ("true", "1", "yes")


def _parse_list(s: str) -> List[str]:
    """解析 CSV 中的 blocked_by 列表字符串.

    支持格式: "['a', 'b']" / "a,b" / "[a,b]" / 空字符串
    """
    s = s.strip()
    if not s or s in ("[]", "['']", '[""]'):
        return []
    # 去掉外层括号
    if s.startswith("[") and s.endswith("]"):
        s = s[1:-1]
    # 去掉引号并按逗号分割
    items = []
    for part in s.split(","):
        part = part.strip().strip("'\"")
        if part:
            items.append(part)
    return items


def _round_opt(v: Optional[float], ndigits: int = 4) -> Optional[float]:
    """对 Optional[float] 做 round."""
    if v is None:
        return None
    return round(v, ndigits)
