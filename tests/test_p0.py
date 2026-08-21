"""P0 质量改进的单元测试：相关性闸门、全文落地、元数据清洗。

全部离线，不触发网络检索（必要时用 monkeypatch 替换下载/LLM）。
"""

import json
from pathlib import Path

import pytest

from src.agent.tools import (
    apply_relevance_gate,
    compute_relevance,
    enrich_topn_fulltext,
    high_hub_dropped,
    rank_papers,
)
from src.config import Settings, get_settings
from src.ingest.base import normalize_doi
from src.report.bibtex import _effective_venue, _journal_from_doi, build_bibtex, build_reference_list


# ---------------------------------------------------------------------------
# 相关性闸门 + 排序
# ---------------------------------------------------------------------------
def _paper(pid, title, abstract="", cite=0, year=2020, queries=("dm",), pdf_url=None, doi=None):
    return {
        "paper_id": pid, "title": title, "abstract": abstract,
        "year": year, "citation_count": cite, "source": "arxiv",
        "venue": "", "doi": doi, "pdf_url": pdf_url, "url": "",
        "authors": [], "fulltext": None, "score": 0.0, "matched_queries": list(queries),
        "relevance": 0.0, "has_fulltext": False, "fulltext_chars": 0,
    }


def test_compute_relevance_raw_range_and_storage():
    papers = [
        _paper("p1", "Adaptive optics with a deformable mirror for wavefront correction",
               "We use a deformable mirror to correct atmospheric turbulence in adaptive optics systems."),
        _paper("p2", "The PASCAL Visual Object Classes (VOC) Challenge",
               "A benchmark dataset for object detection and recognition in natural images."),
    ]
    scores = compute_relevance(papers, "deformable mirror adaptive optics", ["dm"])
    assert len(scores) == 2
    assert all(0.0 <= s <= 1.0 for s in scores)
    assert scores[0] > scores[1]               # 主题相关 > 跑题
    assert papers[0]["relevance"] == pytest.approx(scores[0], abs=1e-3)  # 存的是 round 到 4 位的值


def test_apply_relevance_gate_drops_offtopic():
    papers = [
        _paper("a", "on topic deformable mirror", cite=10),
        _paper("b", "totally unrelated cat videos", cite=99999),
        _paper("c", "adaptive optics wavefront", cite=5),
    ]
    scores = [0.6, 0.03, 0.4]
    kept, dropped = apply_relevance_gate(papers, scores, threshold=0.1, min_keep=1)
    kept_ids = {p["paper_id"] for p in kept}
    assert kept_ids == {"a", "c"}
    # apply_relevance_gate 返回 (保留论文, 被剔除标题列表)
    assert isinstance(dropped, list) and all(isinstance(t, str) for t in dropped)
    assert dropped == ["totally unrelated cat videos"]


def test_apply_relevance_gate_adaptive_floor():
    # 全部低于阈值，但应保底保留 min_keep 篇
    papers = [_paper(f"p{i}", f"paper {i}") for i in range(5)]
    scores = [0.02] * 5
    kept, dropped = apply_relevance_gate(papers, scores, threshold=0.5, min_keep=2)
    assert len(kept) == 2
    assert len(dropped) == 3


def test_rank_papers_relevance_dominates_citations():
    # 高被引但跑题的论文不应压过相关性高的论文（P0-1 核心诉求）
    on = _paper("on", "deformable mirror control", cite=5, year=2021)
    off = _paper("off", "generic deep learning survey", cite=100000, year=2022)
    papers = [on, off]
    relevance = [0.6, 0.1]
    ranked = rank_papers(papers, "deformable mirror", ["dm"], relevance=relevance)
    assert ranked[0]["paper_id"] == "on"


# ---------------------------------------------------------------------------
# 全文落地（P0-2）
# ---------------------------------------------------------------------------
def test_enrich_topn_prefers_oa_available(monkeypatch):
    settings = get_settings()
    settings.enable_fulltext = True
    settings.top_n_fulltext = 2

    oa = _paper("oa1", "has oa pdf", pdf_url="https://arxiv.org/pdf/1234")
    paywalled = _paper("pay1", "high cite no oa", cite=5000)  # 无 pdf_url / doi
    oa_late = _paper("oa2", "also oa", pdf_url="https://arxiv.org/pdf/5678")

    # 模拟下载成功（不真连网）
    called = {}

    def fake_fetch(targets):
        called["targets"] = [t["paper_id"] for t in targets]
        for t in targets:
            t["fulltext"] = "X" * 600
            t["has_fulltext"] = True
            t["fulltext_chars"] = 600
        return len(targets)

    monkeypatch.setattr("src.agent.tools.fetch_fulltexts", fake_fetch)
    n = enrich_topn_fulltext([paywalled, oa, oa_late], settings)
    # 优先取有 OA 的 oa1 / oa2，而非付费墙的 pay1
    assert n == 2
    assert set(called["targets"]) == {"oa1", "oa2"}


# ---------------------------------------------------------------------------
# 元数据清洗（P0-3）
# ---------------------------------------------------------------------------
def test_normalize_doi_invalid_dropped():
    assert normalize_doi("not a doi") is None
    assert normalize_doi("10.1364/ao.49.00g148 ") == "10.1364/ao.49.00g148"  # 形态合法
    assert normalize_doi("https://doi.org/10.1038/s41586-020-2649-2") == "10.1038/s41586-020-2649-2"
    # 缺 10. 前缀或空白 → 丢弃
    assert normalize_doi("doi:something invalid") is None


def test_journal_from_doi_mapping():
    assert _journal_from_doi("10.1364/AO.49.00123") == "Applied Optics"
    assert _journal_from_doi("10.1364/OE.1.00001") == "Optics Express"
    assert _journal_from_doi("10.1109/5.123") == "IEEE"
    assert _journal_from_doi("10.1038/nature123") == "Nature"
    assert _journal_from_doi("10.9999/xyz") is None


def test_effective_venue_fixes_arxiv_preprint_with_journal_doi():
    # 关键场景：检索层把 arXiv 预印本与正式期刊 DOI 并存
    paper = {"venue": "arXiv preprint", "doi": "10.1364/AO.49.00123", "source": "arxiv",
             "paper_id": "arxiv:123"}
    assert _effective_venue(paper) == "Applied Optics"

    # 正常 venue 不被覆盖
    paper2 = {"venue": "Optics Express", "doi": "10.1364/OE.1.1", "source": "openalex",
              "paper_id": "doi:x"}
    assert _effective_venue(paper2) == "Optics Express"


def test_bibtex_entry_type_uses_effective_venue():
    paper = {
        "paper_id": "arxiv:2301.00001", "title": "Wavefront correction via deformable mirror",
        "authors": ["Jane Doe"], "year": 2023, "venue": "arXiv preprint",
        "doi": "10.1364/AO.49.00123", "url": "https://arxiv.org/abs/2301.00001",
        "source": "arxiv", "citation_count": 0,
    }
    bib, mapping = build_bibtex([paper])
    assert "@article{" in bib                       # 不再是 misc（预印本）
    assert "journal = {Applied Optics}" in bib
    assert "doi = {10.1364/AO.49.00123}" in bib

    ref = build_reference_list([paper], {"arxiv:2301.00001": 1}, mapping)
    assert "Applied Optics" in ref
    assert "arXiv preprint" not in ref


# ---------------------------------------------------------------------------
# 回归：相关性闸门高枢纽告警（dropped 是标题字符串，不能当 dict 用）
# ---------------------------------------------------------------------------
def test_high_hub_dropped_ignores_title_strings():
    # apply_relevance_gate 返回的 dropped 是「标题字符串」列表；
    # 计算高枢纽告警时必须从 papers 反查论文对象，不能对字符串调 .get。
    papers = [
        {"paper_id": "keep1", "title": "on topic deformable mirror", "hub_score": 0.9},
        {"paper_id": "drop1", "title": "off topic but pivotal survey", "hub_score": 0.95},
        {"paper_id": "drop2", "title": "another off topic", "hub_score": 0.2},
    ]
    scores = [0.6, 0.03, 0.02]  # drop1 / drop2 被闸门剔除
    kept, dropped = apply_relevance_gate(papers, scores, threshold=0.1, min_keep=1)
    kept_ids = {p["paper_id"] for p in kept}
    # dropped 是字符串标题，绝不能当 dict 用
    assert dropped == ["off topic but pivotal survey", "another off topic"]
    # 修复后：用 kept_ids 反查，drop1(枢纽 0.95) 应被标出，drop2(0.2) 不标
    result = high_hub_dropped(papers, kept_ids)
    assert result == [{"title": "off topic but pivotal survey", "hub": 0.95}]
    # 断言实现没有对字符串调 .get（旧实现会抛 AttributeError）


def test_ranker_no_crash_when_dropped_titles_present(monkeypatch):
    # 回归：ranker 在 dropped（标题字符串列表）非空时，曾因对字符串调 .get 抛出
    # AttributeError: 'str' object has no attribute 'get'。这里离线复现并断言不再崩溃。
    from src.agent import nodes as N
    from src.agent.state import initial_state

    # 全部以确定性桩替换网络/重计算依赖，保持离线
    monkeypatch.setattr(N, "enrich_citations", lambda papers, limit=30: 0)
    monkeypatch.setattr(N, "enrich_topn_fulltext", lambda papers, settings=None: 0)
    monkeypatch.setattr(N, "compute_relevance", lambda papers, topic, queries: [0.6, 0.02])
    monkeypatch.setattr(N.CG, "score_centrality", lambda papers: _assign_hub(papers))

    papers = [
        {"paper_id": "keep1", "title": "on topic deformable mirror", "matched_queries": ["dm"],
         "referenced_works": [], "abstract": "dm"},
        {"paper_id": "drop1", "title": "off topic but pivotal survey", "matched_queries": ["dm"],
         "referenced_works": [], "abstract": "cats"},
    ]
    st = initial_state("deformable mirror adaptive optics", "")
    st["papers"] = papers
    st["queries"] = ["dm"]

    out = N.ranker(st)  # 旧实现此处会抛 AttributeError

    assert isinstance(out.get("papers"), list)
    analysis = out.get("citation_analysis") or {}
    dh = analysis.get("dropped_high_hub") or []
    assert any(d["title"] == "off topic but pivotal survey" for d in dh)


def _assign_hub(papers):
    for p in papers:
        p["hub_score"] = 0.95 if p["paper_id"] == "drop1" else 0.1
        p["bridge_score"] = 0.0
