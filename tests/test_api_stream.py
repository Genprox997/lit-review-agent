"""离线测试：流式 API（SSE）与 Web UI 入口。

需要 fastapi；缺省跳过（不影响离线套件）。"""
from __future__ import annotations

import importlib.util

import pytest

fastapi_available = importlib.util.find_spec("fastapi") is not None
pytestmark = pytest.mark.skipif(not fastapi_available, reason="未安装 fastapi（pip install -e '.[api]'）")


import src.cluster.theme_cluster as TC


@pytest.fixture(autouse=True)
def no_embedding_download(monkeypatch):
    """离线测试强制 TF-IDF 降级，避免下载 sentence-transformers 模型。"""
    monkeypatch.setattr(TC, "_EMBEDDER_TRIED", True)
    monkeypatch.setattr(TC, "_EMBEDDER", None)


@pytest.fixture
def client(monkeypatch, tmp_path):
    monkeypatch.setenv("LLM_PROVIDER", "stub")
    monkeypatch.setenv("ENABLE_FULLTEXT", "false")
    monkeypatch.setenv("RELEVANCE_GATE", "0")
    monkeypatch.setenv("CONTACT_EMAIL", "test@example.com")
    monkeypatch.setenv("OUTPUT_DIR", str(tmp_path / "out"))
    monkeypatch.setenv("CACHE_DIR", str(tmp_path / "cache"))

    # 拦截网络检索，返回两篇假文献，保证全程离线。
    # 注意：retriever 通过 `src.agent.nodes` 模块级引用调用 multi_source_search，
    # 因此必须同时 patch `nodes` 与 `tools` 两个命名空间（二者各持一份引用），
    # 否则图仍会打真实 OpenAlex/arXiv API 而触网挂起。
    from src.ingest.base import make_paper
    from src.agent import nodes as N
    from src.agent import tools as T

    def fake_search(queries, settings=None, sources=None):
        return [
            make_paper(paper_id="arxiv:1", title="Vision method A",
                       abstract="vision recognition cnn", year=2020, citation_count=50,
                       source="arxiv", pdf_url="https://x/1"),
            make_paper(paper_id="doi:10.1/b", title="Vision method B",
                       abstract="vision transformer", year=2021, citation_count=80,
                       source="openalex", doi="10.1/b"),
        ]

    monkeypatch.setattr(N, "multi_source_search", fake_search)
    monkeypatch.setattr(T, "multi_source_search", fake_search)

    from src.api import create_app

    from fastapi.testclient import TestClient

    return TestClient(create_app())


def test_webui_index(client):
    resp = client.get("/")
    assert resp.status_code == 200
    assert "lit-review-agent" in resp.text
    assert "/review/stream" in resp.text


def test_healthz(client):
    assert client.get("/healthz").json() == {"status": "ok"}


def test_review_stream_emits_events(client):
    stages = []
    with client.stream("POST", "/review/stream", json={
        "topic": "vision transformers", "provider": "stub", "target": 20,
    }) as resp:
        assert resp.status_code == 200
        buf = ""
        for line in resp.iter_lines():
            if line.startswith("event:"):
                stages.append(line[6:].strip())
            # 收集到 done 即可结束解析
            if "event: done" in line or "event: error" in line:
                break
    assert "progress" in stages
    assert "done" in stages
