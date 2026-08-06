"""离线测试：检索 HTTP 磁盘缓存层（P3-4）。

不发起任何真实网络请求；通过手工构造 Response 与各方法自测落盘/命中/过期。
"""

from __future__ import annotations

import time

import requests

from src.config import Settings
from src.ingest import base


def _fake_resp(status: int = 200, content: str = "{}", url: str = "http://x/y") -> requests.Response:
    r = requests.Response()
    r.status_code = status
    r.reason = "OK" if status < 400 else "ERR"
    r.url = url
    r.encoding = "utf-8"
    r._content = content.encode("utf-8")
    return r


def _settings(monkeypatch, tmp_path, *, enabled: bool = True, ttl: float = 7.0):
    monkeypatch.setenv("CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("HTTP_CACHE_ENABLED", "true" if enabled else "false")
    monkeypatch.setenv("HTTP_CACHE_TTL_DAYS", str(ttl))
    s = Settings()
    # 隔离全局单例，避免污染其它测试
    monkeypatch.setattr(base, "get_settings", lambda: s)
    return s


def test_cache_put_then_get_roundtrip(monkeypatch, tmp_path):
    _settings(monkeypatch, tmp_path)
    resp = _fake_resp(content='{"hello": 1}')
    base._http_cache_put("http://x/y", {"q": 1}, "openalex", resp)
    got = base._http_cache_get("http://x/y", {"q": 1}, "openalex")
    assert got is not None
    assert got.status_code == 200
    assert got.json() == {"hello": 1}


def test_cache_miss_when_disabled(monkeypatch, tmp_path):
    _settings(monkeypatch, tmp_path, enabled=False)
    assert base._http_cache_get("http://x/y", None, "openalex") is None


def test_cache_ttl_expiry(monkeypatch, tmp_path):
    _settings(monkeypatch, tmp_path, ttl=1e-7)  # ~0.0086s，远小于下面的 0.1s 睡眠
    resp = _fake_resp(content="data")
    base._http_cache_put("http://x/y", None, "openalex", resp)
    time.sleep(0.1)
    assert base._http_cache_get("http://x/y", None, "openalex") is None


def test_cache_skips_error_responses(monkeypatch, tmp_path):
    _settings(monkeypatch, tmp_path)
    base._http_cache_put("http://x/err", None, "openalex", _fake_resp(status=500, content="oops"))
    assert base._http_cache_get("http://x/err", None, "openalex") is None


def test_http_get_uses_cache_without_network(monkeypatch, tmp_path):
    _settings(monkeypatch, tmp_path)
    resp = _fake_resp(content='{"cached": true}', url="http://api/foo")
    base._http_cache_put("http://api/foo", None, "openalex", resp)

    # 若真发起网络请求则直接失败，证明走的是磁盘缓存
    def boom(*a, **k):
        raise AssertionError("不应发起网络请求")

    monkeypatch.setattr(base, "get_session", boom)
    got = base.http_get("http://api/foo", source="openalex")
    assert got is not None
    assert got.json() == {"cached": True}
