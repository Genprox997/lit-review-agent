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
