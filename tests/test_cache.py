"""test_cache.py – 测试 LocalCache 基本功能 + 版本化."""

import time
import tempfile
import os
import pytest

from a_share_hot_screener.cache import CACHE_SCHEMA_VERSION, LocalCache


@pytest.fixture
def tmp_cache(tmp_path):
    return LocalCache(str(tmp_path / "cache"))


class TestLocalCache:
    def test_put_and_get(self, tmp_cache):
        tmp_cache.put("ns", "key1", {"val": 42})
        result = tmp_cache.get("ns", "key1")
        assert result == {"val": 42}

    def test_miss_returns_none(self, tmp_cache):
        assert tmp_cache.get("ns", "nonexistent") is None

    def test_ttl_expiry(self, tmp_cache):
        tmp_cache.put("ns", "key_ttl", "expire_soon", ttl=1)
        assert tmp_cache.get("ns", "key_ttl") is not None
        time.sleep(1.1)
        assert tmp_cache.get("ns", "key_ttl") is None

    def test_invalidate(self, tmp_cache):
        tmp_cache.put("ns", "k", "v")
        removed = tmp_cache.invalidate("ns", "k")
        assert removed is True
        assert tmp_cache.get("ns", "k") is None

    def test_invalidate_nonexistent(self, tmp_cache):
        removed = tmp_cache.invalidate("ns", "no_such_key")
        assert removed is False

    def test_clear_namespace(self, tmp_cache):
        tmp_cache.put("ns2", "a", 1)
        tmp_cache.put("ns2", "b", 2)
        tmp_cache.put("ns3", "c", 3)
        count = tmp_cache.clear_namespace("ns2")
        assert count == 2
        assert tmp_cache.get("ns3", "c") == 3

    def test_complex_value(self, tmp_cache):
        data = [{"date": "2026-04-18", "close": 1800.5}]
        tmp_cache.put("prices", "600519", data)
        result = tmp_cache.get("prices", "600519")
        assert result == data

    def test_overwrite(self, tmp_cache):
        tmp_cache.put("ns", "k", "v1")
        tmp_cache.put("ns", "k", "v2")
        assert tmp_cache.get("ns", "k") == "v2"

    def test_cache_dir_size_mb(self, tmp_cache):
        tmp_cache.put("ns", "k", "hello")
        size = tmp_cache.cache_dir_size_mb()
        assert size >= 0


class TestCacheVersioning:
    """#5 缓存版本化测试."""

    def test_version_constant_exists(self):
        assert isinstance(CACHE_SCHEMA_VERSION, str)
        assert len(CACHE_SCHEMA_VERSION) > 0

    def test_different_version_miss(self, tmp_path):
        """不同 schema_version 的缓存互不可见."""
        c_v1 = LocalCache(str(tmp_path / "cache"), schema_version="v1")
        c_v2 = LocalCache(str(tmp_path / "cache"), schema_version="v2")

        c_v1.put("ns", "key", "value_v1")
        # v2 版本看不到 v1 的数据
        assert c_v2.get("ns", "key") is None
        # v1 还能看到
        assert c_v1.get("ns", "key") == "value_v1"

    def test_same_version_hit(self, tmp_path):
        """相同 schema_version 正常命中."""
        c1 = LocalCache(str(tmp_path / "cache"), schema_version="v2")
        c2 = LocalCache(str(tmp_path / "cache"), schema_version="v2")

        c1.put("ns", "key", {"x": 1})
        assert c2.get("ns", "key") == {"x": 1}

    def test_put_stores_version(self, tmp_path):
        """写入时应包含 _schema_version."""
        import json
        c = LocalCache(str(tmp_path / "cache"), schema_version="v99")
        c.put("ns", "k", 42)
        # 找到写入的 JSON 文件
        json_files = list((tmp_path / "cache" / "ns").glob("*.json"))
        assert len(json_files) == 1
        data = json.loads(json_files[0].read_text())
        assert data["_schema_version"] == "v99"

    def test_purge_stale(self, tmp_path):
        """清理旧版本缓存文件."""
        c_old = LocalCache(str(tmp_path / "cache"), schema_version="v1")
        c_old.put("ns", "a", 1)
        c_old.put("ns", "b", 2)

        c_new = LocalCache(str(tmp_path / "cache"), schema_version="v2")
        c_new.put("ns", "c", 3)

        # 缓存目录中应有 3 个文件
        json_files = list((tmp_path / "cache").rglob("*.json"))
        assert len(json_files) == 3

        # purge_stale 应删除 v1 的 2 个
        removed = c_new.purge_stale()
        assert removed == 2

        # v2 的还在
        assert c_new.get("ns", "c") == 3
        # 共剩 1 个文件
        json_files = list((tmp_path / "cache").rglob("*.json"))
        assert len(json_files) == 1


class TestCacheMaintenance:
    """P8 缓存自动维护测试."""

    def test_purge_expired(self, tmp_path):
        """清理已过期文件."""
        c = LocalCache(str(tmp_path / "cache"))
        c.put("ns", "fresh", "alive", ttl=3600)
        c.put("ns", "stale", "dead", ttl=1)
        time.sleep(1.1)

        removed = c.purge_expired()
        assert removed == 1
        # fresh 还在
        assert c.get("ns", "fresh") == "alive"
        # stale 已被清理
        json_files = list((tmp_path / "cache").rglob("*.json"))
        assert len(json_files) == 1

    def test_purge_expired_all_fresh(self, tmp_path):
        """所有缓存都未过期时不删除."""
        c = LocalCache(str(tmp_path / "cache"))
        c.put("ns", "a", 1, ttl=3600)
        c.put("ns", "b", 2, ttl=3600)

        removed = c.purge_expired()
        assert removed == 0

    def test_stats_empty(self, tmp_path):
        """空缓存统计."""
        c = LocalCache(str(tmp_path / "cache"))
        st = c.stats()
        assert st["file_count"] == 0
        assert st["size_mb"] == 0.0
        assert st["expired_count"] == 0
        assert st["schema_version"] == CACHE_SCHEMA_VERSION
        assert st["namespaces"] == {}

    def test_stats_with_data(self, tmp_path):
        """有数据时的统计."""
        c = LocalCache(str(tmp_path / "cache"))
        c.put("prices", "600519", {"close": 1800}, ttl=3600)
        c.put("prices", "000858", {"close": 150}, ttl=3600)
        c.put("basic", "all", [1, 2, 3], ttl=3600)

        st = c.stats()
        assert st["file_count"] == 3
        assert st["size_mb"] >= 0  # 小文件四舍五入后可能为 0.0
        assert st["expired_count"] == 0
        assert "prices" in st["namespaces"]
        assert st["namespaces"]["prices"]["file_count"] == 2
        assert "basic" in st["namespaces"]
        assert st["namespaces"]["basic"]["file_count"] == 1

    def test_stats_expired_count(self, tmp_path):
        """统计已过期文件数量."""
        c = LocalCache(str(tmp_path / "cache"))
        c.put("ns", "fresh", "alive", ttl=3600)
        c.put("ns", "stale", "dead", ttl=1)
        time.sleep(1.1)

        st = c.stats()
        assert st["file_count"] == 2
        assert st["expired_count"] == 1
