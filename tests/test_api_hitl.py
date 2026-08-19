"""离线测试：Web UI 接 HITL 反馈（方向 A'）。

覆盖：
- Web UI 页面含 HITL 开关与 /review/resume 入口；
- POST /review/stream 带 with_human=true 会在成稿后推送 interrupted 事件，
  且回传 thread_id 与草稿全文（report）；
- POST /review/resume 空意见 → 通过并定稿（done，不再挂起）；
- POST /review/resume 带意见 → 触发针对性改写回环，再次挂起（interrupted），
  再空意见 → 定稿（done）；
- POST /review/resume 非挂起 thread → 400。

全程 stub LLM + 假检索，不触网；HITL 依赖 SQLite 检查点（已安装）。"""
from __future__ import annotations

import importlib.util
import json

import pytest

fastapi_available = importlib.util.find_spec("fastapi") is not None
pytestmark = pytest.mark.skipif(not fastapi_available, reason="未安装 fastapi（pip install -e '.[api]'）")


import src.cluster.theme_cluster as TC
from src.ingest.base import make_paper
from src.agent import nodes as N
from src.agent import tools as T


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
    monkeypatch.setenv("TARGET_PAPER_COUNT", "2")
    monkeypatch.setenv("OUTPUT_DIR", str(tmp_path / "out"))
    monkeypatch.setenv("CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("CHECKPOINT_PATH", str(tmp_path / "ckpt.sqlite"))

    # 彻底拦截网络：假检索 + 关闭引用数补拉与全文下载（retriever 经 nodes 命名空间调用）
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
    monkeypatch.setattr(N, "enrich_citations", lambda papers, limit=25: 0)
    monkeypatch.setattr(N, "enrich_topn_fulltext", lambda papers, settings=None: 0)

    from src.api import create_app
    from fastapi.testclient import TestClient

    return TestClient(create_app())


def _collect(resp):
    """从 SSE 响应收集 (stage, payload) 列表。"""
    events = []
    stage = None
    data = None
    for line in resp.iter_lines():
        if line is None:
            continue
        if line.startswith("event:"):
            stage = line[6:].strip()
        elif line.startswith("data:"):
            data = line[5:].strip()
            if stage is not None and data is not None:
                try:
                    payload = json.loads(data)
                except Exception:
                    payload = data
                events.append((stage, payload))
                stage, data = None, None
    return events


def _tid_of(events):
    for s, p in events:
        if s == "interrupted":
            return p.get("thread_id")
    return None


def test_webui_index_has_hitl(client):
    resp = client.get("/")
    assert resp.status_code == 200
    assert "/review/resume" in resp.text
    assert "with_human" in resp.text
    assert "启用人工审核" in resp.text


def test_stream_human_emits_interrupted(client):
    with client.stream("POST", "/review/stream", json={
        "topic": "vision transformers", "provider": "stub", "target": 2, "with_human": True,
    }) as resp:
        assert resp.status_code == 200
        events = _collect(resp)

    stages = [s for s, _ in events]
    assert "progress" in stages
    assert "interrupted" in stages

    inter = [p for s, p in events if s == "interrupted"][0]
    assert inter["thread_id"], "interrupted 事件应回传 thread_id 供续跑"
    assert inter["report"], "interrupted 事件应回传草稿全文"
    assert ("## 摘要" in inter["report"]) or ("参考文献" in inter["report"]) or ("附录" in inter["report"])
    assert inter["paper_count"] >= 1
    assert inter["section_count"] >= 1


def test_resume_approve_finalizes(client):
    with client.stream("POST", "/review/stream", json={
        "topic": "vision transformers", "provider": "stub", "target": 2, "with_human": True,
    }) as resp:
        first = _collect(resp)
    tid = _tid_of(first)
    assert tid, "首轮应回传 thread_id"

    with client.stream("POST", "/review/resume", json={"thread_id": tid, "feedback": ""}) as resp:
        second = _collect(resp)

    stages2 = [s for s, _ in second]
    assert "done" in stages2, "空意见应直接定稿"
    assert "interrupted" not in stages2
    done = [p for s, p in second if s == "done"][0]
    assert done["artifacts"], "定稿应产出产物路径"


def test_resume_with_feedback_triggers_rewrite_then_approve(client):
    with client.stream("POST", "/review/stream", json={
        "topic": "vision transformers", "provider": "stub", "target": 2, "with_human": True,
    }) as resp:
        first = _collect(resp)
    tid = _tid_of(first)
    assert tid

    # 带意见续跑：应进入针对性改写回环并再次挂起
    with client.stream("POST", "/review/resume", json={
        "thread_id": tid, "feedback": "请在第一个主题小节补充与对比方法的实验分析",
    }) as resp:
        second = _collect(resp)
    stages2 = [s for s, _ in second]
    assert "interrupted" in stages2, "带意见的续跑应再次挂起等待下一轮审核"
    tid2 = _tid_of(second)
    assert tid2 == tid, "改写后应保持同一 thread_id"

    # 通过并定稿
    with client.stream("POST", "/review/resume", json={"thread_id": tid, "feedback": ""}) as resp:
        third = _collect(resp)
    stages3 = [s for s, _ in third]
    assert "done" in stages3
    assert "interrupted" not in stages3


def test_resume_bad_thread_returns_400(client):
    resp = client.post("/review/resume", json={"thread_id": "does-not-exist", "feedback": ""})
    assert resp.status_code == 400


def test_stream_done_includes_citation_graph(client):
    """方向 E'：done 事件应回传 citation_graph（nodes/edges），供 Web UI 渲染。"""
    with client.stream("POST", "/review/stream", json={
        "topic": "vision transformers", "provider": "stub", "target": 2, "with_human": False,
    }) as resp:
        assert resp.status_code == 200
        events = _collect(resp)

    stages = [s for s, _ in events]
    assert "done" in stages
    done = [p for s, p in events if s == "done"][0]
    cg = done.get("citation_graph") or {}
    assert cg.get("nodes"), "done 事件应携带引用网络节点"
    assert isinstance(cg.get("edges"), list), "done 事件应携带引用网络边"
    assert cg.get("stats") and "node_count" in cg["stats"]
    # 假检索文献无 referenced_works -> 边为空但节点仍在（安全降级）
    assert len(cg["nodes"]) == 2
    assert cg["edges"] == []


def test_webui_index_has_graph_panel(client):
    """方向 E'：Web UI 含引用网络面板与渲染入口。"""
    resp = client.get("/")
    assert resp.status_code == 200
    assert 'id="graph"' in resp.text
    assert "renderGraph" in resp.text
    assert "引用网络" in resp.text
    assert "hubOnly" in resp.text
