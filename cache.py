"""本地缓存层 – 基于文件的 JSON 缓存，主要给 Tushare 和部分 Tushare 表使用.

缓存目录：~/.a_share_hot_screener/cache/
严格隔离：不与 a_share_screener 共享缓存目录或缓存键。

版本化：
  CACHE_SCHEMA_VERSION 嵌入缓存键哈希中。
  升级版本号会令旧缓存自动失效（哈希不同 → miss）。
  调用 purge_stale() 可清理与当前版本不匹配的孤立文件。
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from pathlib import Path
from typing import Any, Optional


# 全局缓存 schema 版本号。
# 递增此值即可令所有旧缓存自动失效（键哈希包含版本号）。
CACHE_SCHEMA_VERSION = "v2"


class LocalCache:
    """简单的基于文件的 JSON 缓存.

    缓存键 = version + namespace + key → md5 → 文件路径
    每个条目带 TTL（秒），过期自动失效。
    版本号变更时旧键哈希不同，自动 miss。

    TTL 参考值（在 clients 层设置）：
      Tushare daily 价格:      1 天  (86400)
      Tushare symbol list:   7 天  (604800)
      Tushare stock_basic: 1 天  (86400)
      Tushare daily_basic:     4 小时 (14400)
      Tushare 龙虎榜:       1 天  (86400)
    """

    def __init__(
        self,
        cache_dir: str,
        default_ttl: int = 86400,
        schema_version: str = CACHE_SCHEMA_VERSION,
    ) -> None:
        """
        Args:
            cache_dir: 缓存根目录
            default_ttl: 默认过期秒数（24h）
            schema_version: schema 版本号（影响键哈希，升级即失效旧缓存）
        """
        self.cache_dir = Path(cache_dir)
        self.default_ttl = default_ttl
        self._schema_version = schema_version
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        # Session 10 P3-5 / S13 线程安全化: 缓存命中率统计计数器
        self._hit_count: int = 0
        self._miss_count: int = 0
        self._stats_lock = threading.Lock()

    # ── 公共接口 ─────────────────────────────────────────

    def get(self, namespace: str, key: str, ttl: Optional[int] = None) -> Optional[Any]:
        """读取缓存，若不存在或过期返回 None."""
        path = self._path(namespace, key)
        if not path.exists():
            with self._stats_lock:
                self._miss_count += 1
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            with self._stats_lock:
                self._miss_count += 1
            return None
        expire_at = data.get("expire_at", 0)
        if time.time() > expire_at:
            path.unlink(missing_ok=True)
            with self._stats_lock:
                self._miss_count += 1
            return None
        with self._stats_lock:
            self._hit_count += 1
        return data.get("value")

    def get_hit_rate(self) -> Optional[float]:
        """返回缓存命中率（0.0~1.0），若从未访问则返回 None."""
        with self._stats_lock:
            total = self._hit_count + self._miss_count
            if total == 0:
                return None
            return round(self._hit_count / total, 4)

    def put(self, namespace: str, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """写入缓存."""
        effective_ttl = ttl if ttl is not None else self.default_ttl
        path = self._path(namespace, key)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "_schema_version": self._schema_version,
            "key": key,
            "namespace": namespace,
            "expire_at": time.time() + effective_ttl,
            "value": value,
        }
        path.write_text(
            json.dumps(payload, ensure_ascii=False, default=str),
            encoding="utf-8",
        )

    def invalidate(self, namespace: str, key: str) -> bool:
        """删除指定缓存条目，返回是否确实删除了."""
        path = self._path(namespace, key)
        if path.exists():
            path.unlink()
            return True
        return False

    def clear_namespace(self, namespace: str) -> int:
        """清除某个 namespace 下的所有缓存，返回删除数量."""
        ns_dir = self.cache_dir / namespace
        if not ns_dir.exists():
            return 0
        count = 0
        for f in ns_dir.rglob("*.json"):
            f.unlink()
            count += 1
        return count

    def cache_dir_size_mb(self) -> float:
        """返回缓存目录总大小（MB）."""
        total = 0
        for f in self.cache_dir.rglob("*"):
            if f.is_file():
                total += f.stat().st_size
        return total / (1024 * 1024)

    # ── 内部 ─────────────────────────────────────────────

    def purge_stale(self) -> int:
        """删除与当前 schema_version 不匹配的缓存文件.

        原理：遍历全部 .json 缓存文件，读取内部 `_schema_version` 字段，
        不匹配或缺失的视为旧版本并删除。

        Returns:
            删除的文件数量
        """
        removed = 0
        for f in self.cache_dir.rglob("*.json"):
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
                if data.get("_schema_version") != self._schema_version:
                    f.unlink()
                    removed += 1
            except Exception:
                # 损坏的文件也一并清理
                try:
                    f.unlink()
                    removed += 1
                except OSError:
                    pass
        return removed

    # ── 内部 ─────────────────────────────────────────────

    def _path(self, namespace: str, key: str) -> Path:
        """缓存文件路径: cache_dir / namespace / <md5(version:key)>.json."""
        versioned = f"{self._schema_version}:{key}"
        h = hashlib.md5(versioned.encode()).hexdigest()
        return self.cache_dir / namespace / f"{h}.json"
