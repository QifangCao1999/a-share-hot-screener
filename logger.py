"""通用日志 + warnings 收集器.

与 a_share_screener.logger 同设计模式，独立实现，使用不同 logger name。
"""

from __future__ import annotations

import logging
import sys
import threading
from dataclasses import dataclass, field
from typing import List


def setup_logger(name: str = "a_share_hot_screener", level: str = "INFO") -> logging.Logger:
    """创建统一格式的 logger.

    多次调用同 name 不会重复添加 handler。
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    logger.addHandler(handler)
    return logger


@dataclass
class WarningsCollector:
    """按股票收集 warnings，供最终输出使用.

    线程安全：add/add_global 均持锁操作。

    用法::

        wc = WarningsCollector()
        wc.add("600519", "price 字段缺失，已跳过量价特征计算")
        wc.add_global("Tushare top_list 接口本次请求失败，龙虎榜数据全部缺失")
        wc.get("600519")   # -> ["price 字段缺失..."]
        wc.all_warnings()  # -> {"600519": [...], ...}
    """

    _store: dict = field(default_factory=dict)
    _global: List[str] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def add(self, code: str, msg: str) -> None:
        """为某只股票添加一条 warning（线程安全）."""
        with self._lock:
            self._store.setdefault(code, []).append(msg)

    def add_global(self, msg: str) -> None:
        """添加全局级别的 warning（不绑定某只股票，线程安全）."""
        with self._lock:
            self._global.append(msg)

    def get(self, code: str) -> List[str]:
        """获取某只股票的 warnings 列表（线程安全读取拷贝）."""
        with self._lock:
            return list(self._store.get(code, []))

    def all_warnings(self) -> dict:
        """返回全部 per-stock warnings dict: {code: [msg, ...]}."""
        with self._lock:
            return {k: list(v) for k, v in self._store.items()}

    def global_warnings(self) -> List[str]:
        """返回全局 warnings 列表."""
        with self._lock:
            return list(self._global)

    def total_count(self) -> int:
        """总 warning 数（含全局）."""
        with self._lock:
            return sum(len(v) for v in self._store.values()) + len(self._global)
