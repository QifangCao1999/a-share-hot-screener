"""评估框架 — Phase 1.

模块:
  label_generator  — 生成未来收益标签 (T+1, MFE, MAE, etc.)
  harness          — 核心评估引擎 (分组、指标计算、消融实验)
  report           — 评估报告生成 (文本 + CSV)
"""

from a_share_hot_screener.evaluation.label_generator import LabelGenerator
from a_share_hot_screener.evaluation.harness import EvaluationHarness
from a_share_hot_screener.evaluation.report import EvaluationReport

__all__ = ["LabelGenerator", "EvaluationHarness", "EvaluationReport"]
