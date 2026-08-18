"""方向 D'：引用网络分析（枢纽/桥接论文 + 共引空白）单测与端到端验证。

全程离线：引用关系由测试构造，不触网；算法为纯 Python 确定性实现。
"""

from __future__ import annotations

import pytest

from src.agent import citation_graph as CG
from src.agent.graph import build_graph
from src.agent.state import initial_state
from src.config import get_settings
from src.ingest.base import make_paper


@pytest.fixture(autouse=True)
def stub_env(monkeypatch, tmp_path):
    """强制 stub LLM + 关闭全文下载 + 输出到临时目录。"""
    monkeypatch.setenv("LLM_PROVIDER", "stub")
    monkeypatch.setenv("ENABLE_FULLTEXT", "false")
    monkeypatch.setenv("RELEVANCE_GATE", "0")   # 离线桩测试关闭相关性闸门
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
    """离线测试强制 TF-IDF 降级，避免下载 sentence-transformers 模型。"""
    import src.cluster.theme_cluster as TC

    monkeypatch.setattr(TC, "_EMBEDDER_TRIED", True)
    monkeypatch.setattr(TC, "_EMBEDDER", None)


def _gp(pid, refs, title="t", year=2020, citation_count=10):
    return make_paper(
        paper_id=pid, openalex_id=pid, title=title, year=year,
        citation_count=citation_count, source="openalex", abstract="x",
        referenced_works=refs, matched_queries=["q"],
    )


def _graph_papers():
    """已知小引用图：C 被 A/B/F 引用（最高枢纽），A 被 D/E 引用。"""
    return [
        _gp("W1", ["https://openalex.org/W2", "https://openalex.org/W3"], title="A cites B C"),
        _gp("W2", ["https://openalex.org/W3"], title="B cites C"),
        _gp("W3", [], title="C root"),
        _gp("W4", ["https://openalex.org/W1"], title="D cites A"),
        _gp("W5", ["https://openalex.org/W1"], title="E cites A"),
        _gp("W6", ["https://openalex.org/W3"], title="F cites C"),
    ]


# --------------------------------------------------------------------------
# 构图
# --------------------------------------------------------------------------
def test_build_graph_strips_url_and_links():
    papers = _graph_papers()
    adj, _ = CG.build_graph(papers)
    assert adj["W1"] == {"W2", "W3"}
    assert adj["W3"] == set()          # 入度最高、无出边
    assert adj["W4"] == {"W1"}
    # 只保留池内边
    assert all(t in adj for s in adj for t in adj[s])


# --------------------------------------------------------------------------
# 枢纽度（PageRank）/ 桥接度（betweenness）
# --------------------------------------------------------------------------
def test_score_centrality_hub_order():
    papers = _graph_papers()
    CG.score_centrality(papers)
    by_title = {p["title"]: p["hub_score"] for p in papers}
    # C 被 A/B/F 引用，应为最高枢纽
    assert by_title["C root"] == max(by_title.values())
    assert by_title["C root"] > by_title["A cites B C"]  # C > A


def test_betweenness_leaves_zero():
    papers = _graph_papers()
    CG.score_centrality(papers)
    by_title = {p["title"]: p["bridge_score"] for p in papers}
    # 叶子节点（D/E/F 度=1）不应是桥接点
    assert by_title["D cites A"] == 0.0
    assert by_title["E cites A"] == 0.0
    assert by_title["F cites C"] == 0.0
    # 内部节点才是桥
    assert max(by_title.values()) > 0.0


def test_score_centrality_no_refs_is_safe():
    papers = [_gp("X1", [], title="no refs"), _gp("X2", [], title="no refs2")]
    CG.score_centrality(papers)
    assert all(p["hub_score"] == 0.0 and p["bridge_score"] == 0.0 for p in papers)


# --------------------------------------------------------------------------
# 共引空白
# --------------------------------------------------------------------------
def test_cocitation_gaps_finds_uncovered_subarea():
    # X/Y 共享 2 篇参考文献，构成未被现有主题簇覆盖的共引子群
    papers = [
        _gp("W20", ["W9", "W10"], title="X study"),
        _gp("W21", ["W9", "W10"], title="Y study"),
        _gp("W22", ["W9"], title="Z alone"),
    ]
    # 现有主题簇只覆盖 Z，X/Y 未覆盖
    clusters = [{"cluster_id": 0, "label": "covered", "paper_ids": ["W22"]}]
    gaps = CG.cocitation_gaps(papers, clusters)
    assert gaps, "应识别出未覆盖的共引子群"
    assert any("X study" in g or "Y study" in g for g in gaps)


def test_cocitation_gaps_ignores_fully_covered():
    papers = [
        _gp("W20", ["W9", "W10"], title="X study"),
        _gp("W21", ["W9", "W10"], title="Y study"),
    ]
    clusters = [{"cluster_id": 0, "label": "all", "paper_ids": ["W20", "W21"]}]
    gaps = CG.cocitation_gaps(papers, clusters)
    assert gaps == []  # 已被现有主题簇完全覆盖，不应报空白


# --------------------------------------------------------------------------
# 节点级：ranker 写入 citation_analysis
# --------------------------------------------------------------------------
def test_ranker_populates_citation_analysis(stub_env, monkeypatch):
    monkeypatch.setattr("src.agent.nodes.enrich_citations", lambda papers, limit=25: 0)
    monkeypatch.setattr("src.agent.nodes.enrich_topn_fulltext", lambda papers, settings=None: 0)
    from src.agent import nodes as N

    st = initial_state("t")
    st["papers"] = _graph_papers()
    st["queries"] = ["q"]
    out = N.ranker(st)
    analysis = out["citation_analysis"]
    assert analysis["available"] is True
    assert analysis["top_hub"]
    assert analysis["top_hub"][0]["title"] == "C root"   # 最高枢纽
    # hub_score 已写回论文
    assert any(p.get("hub_score", 0) > 0 for p in out["papers"])


# --------------------------------------------------------------------------
# 端到端：成稿附录 A.8 出现
# --------------------------------------------------------------------------
def test_full_graph_citation_appendix(stub_env, tmp_path, monkeypatch):
    from src.agent import nodes as N

    stub_env.target_paper_count = 6
    stub_env.max_retrieval_rounds = 1

    monkeypatch.setattr(N, "multi_source_search", lambda queries, settings=None: _graph_papers())
    monkeypatch.setattr(N, "enrich_citations", lambda papers, limit=25: 0)
    monkeypatch.setattr(N, "enrich_topn_fulltext", lambda papers, settings=None: 0)

    graph = build_graph()
    final = graph.invoke(
        initial_state("citation network topic"),
        {"configurable": {"thread_id": "cite-e2e"}, "recursion_limit": 60},
    )
    report = final["report"]
    assert "A.8" in report, "成稿应含引用网络分析附录 A.8"
    assert "引用网络分析" in report
    assert "枢纽度" in report
    # 最高枢纽论文应进入 top_hub
    top = (final.get("citation_analysis") or {}).get("top_hub") or []
    assert top and top[0]["title"] == "C root"
