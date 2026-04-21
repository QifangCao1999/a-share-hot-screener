"""A股短线第一阶段客观筛选脚本.

版本: 0.2.0
目标: 输入 stock_codes，输出短线热点股票的第一阶段结构化筛选结果。
数据源: Tushare Pro API（200元/年档，5000积分）

核心原则：
- 只做结构化、客观、可脚本化、可复现的筛选与打分
- 严格 as-of / point-in-time 口径
- 字段无法稳定获得时返回 None + warnings，不编造
"""

__version__ = "0.2.0"
__project__ = "a_share_hot_screener"
