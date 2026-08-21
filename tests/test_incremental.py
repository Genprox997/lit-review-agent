"""方向 B'：增量更新已有综述。全程离线（stub LLM + 假检索）。

验证：增量模式能载上一版文献池、沿用历史引用编号、只拉新文献、
保留未变小节正文，并在成稿头渲染增量说明、写出 meta 侧车。
"""

from __future__ import annotations

import os

import pytest

from src.agent.graph import build_graph
from src.agent.state import initial_state
from src.config import get_settings
from src.ingest.base import make_paper


@pytest.fixture(autouse=True)
def stub_env(monkeypatch, tmp_path):
    monkeypatch.setenv("LLM_PROVIDER", "stub")
    monkeypatch.setenv("ENABLE_FULLTEXT", "false")
    monkeypatch.setenv("RELEVANCE_GATE", "0")
    monkeypatch.setenv("CONTACT_EMAIL", "test@example.com")
    monkeypatch.setenv("OUTPUT_DIR", str(tmp_path / "out"))
    monkeypatch.setenv("CACHE_DIR", str(tmp_path / "cache"))
    settings = get_settings(refresh=True)
    settings.output_dir = tmp_path / "out"
    settings.cache_dir = tmp_path / "cache"
    yield settings
    get_settings(refresh=True)


@pytest.fixture(autouse=True)
def no_embedding_download(monkeypatch):
    import src.cluster.theme_cluster as TC

    monkeypatch.setattr(TC, "_EMBEDDER_TRIED", True)
    monkeypatch.setattr(TC, "_EMBEDDER", None)


def _old_pool():
    out = []
    for i in range(12):
        out.append(make_paper(
            paper_id=f"arxiv:vision.{i}", title=f"Vision method {i}",
            abstract="convolutional neural network image recognition",
            year=2019 + i % 5, citation_count=50 - i, source="arxiv",
            authors=[f"A{i}"], pdf_url=f"https://arxiv.org/pdf/v{i}",
            matched_queries=["computer vision"],
        ))
    for i in range(12):
        out.append(make_paper(
            paper_id=f"doi:10.1/nlp.{i}", doi=f"10.1/nlp.{i}",
            title=f"Language model {i}",
            abstract="transformer language model pretraining",
            year=2020 + i % 5, citation_count=80 - i, source="openalex",
            authors=[f"W{i}"], venue="ACL",
            matched_queries=["language model"],
        ))
    return out


def _new_papers():
    return [
        make_paper(paper_id="arxiv:new1", title="New vision transformer 2024",
                   abstract="novel vision transformer 2024", year=2024,
                   citation_count=5, source="arxiv", authors=["N1"],
                   matched_queries=["computer vision"]),
        make_paper(paper_id="doi:10.1/new2", title="New multimodal 2024",
                   abstract="novel multimodal 2024", year=2024,
                   citation_count=5, source="openalex", authors=["N2"], venue="CVPR",
                   doi="10.1/new2", matched_queries=["computer vision"]),
    ]


# --------------------------------------------------------------------------
# 节点级：incremental_plan 保留高重叠小节、标记低重叠小节重写
# --------------------------------------------------------------------------
def test_incremental_plan_keeps_matching_clusters(stub_env):
    from src.agent import nodes as N

    prev_clusters = [
        {"cluster_id": 0, "label": "视觉识别", "keywords": ["vision"],
         "paper_ids": ["p1", "p2", "p3", "p4"], "size": 4},
        {"cluster_id": 1, "label": "语言模型", "keywords": ["nlp"],
         "paper_ids": ["q1", "q2", "q3"], "size": 3},
    ]
    prev_sections = {"视觉识别": "旧视觉正文。", "语言模型": "旧语言正文。"}

    state = initial_state("t", incremental=True)
    state["papers"] = [{"paper_id": x} for x in ["p1", "p2", "p3", "p4", "n1", "n2"]]
    state["previous_pids"] = ["p1", "p2", "p3", "p4", "q1", "q2", "q3"]
    state["previous_sections"] = prev_sections
    state["previous_clusters"] = prev_clusters
    state["clusters"] = [
        {"cluster_id": 0, "label": "视觉识别（新）", "keywords": ["vision"],
         "paper_ids": ["p1", "p2", "p3", "p4"], "size": 4},
        {"cluster_id": 1, "label": "全新子领域", "keywords": ["x"],
         "paper_ids": ["n1", "n2"], "size": 2},
    ]

    out = N.incremental_plan(state)
    assert "视觉识别（新）" in out["incremental_keep"]
    assert "全新子领域" not in out["incremental_keep"]
    assert out["sections"].get("视觉识别（新）") == "旧视觉正文。"
    note = out["incremental_note"]
    assert note["new"] == 2                       # n1, n2 不在上一版
    assert note["kept"] == 1
    assert note["rewritten"] == 1


def test_incremental_plan_noop_when_not_incremental(stub_env):
    from src.agent import nodes as N

    state = initial_state("t", incremental=False)
    state["clusters"] = [{"cluster_id": 0, "label": "x", "paper_ids": ["p1"]}]
    assert N.incremental_plan(state) == {}, "非增量模式应直接放行"


# --------------------------------------------------------------------------
# 端到端：第二轮复用第一版、沿用编号、渲染增量说明
# --------------------------------------------------------------------------
def test_incremental_run_adds_new_papers_and_notes(stub_env, tmp_path, monkeypatch):
    from src.agent import nodes as N

    stub_env.target_paper_count = 24   # = 旧池规模，避免内环重复检索
    stub_env.max_retrieval_rounds = 1

    # 两轮共用同一假检索：第 1 轮（正常生成）只返回旧池；
    # 第 2 轮（增量）返回旧池 + 2 篇新论文。
    # 注意：用「第几轮」而非「调用次数」区分——方向 H' 的伪相关反馈扩词
    # 会在每轮首排后额外触发一次检索，不能依赖调用次数。
    run_flag = {"second": False}

    def fake_search(queries, settings=None):
        return _old_pool() + (_new_papers() if run_flag["second"] else [])

    monkeypatch.setattr(N, "multi_source_search", fake_search)
    monkeypatch.setattr(N, "enrich_citations", lambda p, limit=25: 0)
    monkeypatch.setattr(N, "enrich_topn_fulltext", lambda p, settings=None: 0)

    # 第一轮：正常生成，产出上一版成稿
    g1 = build_graph()
    r1 = g1.invoke(initial_state("inc topic"),
                   {"configurable": {"thread_id": "inc1"}, "recursion_limit": 60})
    base = r1["artifacts"]["report"]
    r1cmap = r1["citation_map"]

    # 第二轮：增量模式，检索返回旧池 + 2 篇新论文
    run_flag["second"] = True
    g2 = build_graph()
    r2 = g2.invoke(
        initial_state("inc topic", incremental=True, since_date="2023-01-01", base_path=base),
        {"configurable": {"thread_id": "inc2"}, "recursion_limit": 60},
    )

    # 新论文进入文献池
    pids2 = {p["paper_id"] for p in r2["papers"]}
    assert "arxiv:new1" in pids2 and "doi:10.1/new2" in pids2

    # 历史引用编号跨版本保持一致（方向 A 稳定 + 方向 B' 沿用）
    for pid in ("arxiv:vision.0", "doi:10.1/nlp.0"):
        assert r2["citation_map"].get(pid) == r1cmap.get(pid), "历史编号应跨版本保持一致"

    # 增量说明写入状态，成稿头含「增量更新」
    note = r2.get("incremental_note") or {}
    assert note.get("new") == 2
    assert "增量更新" in r2["report"]

    # meta 侧车存在（供下一版沿用）
    assert r2["artifacts"].get("meta") and os.path.exists(r2["artifacts"]["meta"])


def test_incremental_no_base_degrades_gracefully(stub_env, tmp_path, monkeypatch):
    """增量模式但无 base_path 时，安全降级为正常生成（不报错、产出成稿）。"""
    from src.agent import nodes as N

    stub_env.target_paper_count = 10
    stub_env.max_retrieval_rounds = 1

    def fake_search(queries, settings=None):
        return _old_pool()

    monkeypatch.setattr(N, "multi_source_search", fake_search)
    monkeypatch.setattr(N, "enrich_citations", lambda p, limit=25: 0)
    monkeypatch.setattr(N, "enrich_topn_fulltext", lambda p, settings=None: 0)

    g = build_graph()
    final = g.invoke(
        initial_state("topic x", incremental=True, since_date="2023-01-01", base_path=None),
        {"configurable": {"thread_id": "inc-nobase"}, "recursion_limit": 60},
    )
    assert final["report"]
    assert final["sections"]
