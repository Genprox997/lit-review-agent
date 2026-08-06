"""聚类与成稿输出单测。"""

from __future__ import annotations

import numpy as np
import pytest

import src.cluster.theme_cluster as TC
from src.cluster.theme_cluster import choose_k, cluster_papers, embed_texts
from src.ingest.base import make_paper
from src.report.bibtex import (
    build_bibtex,
    build_reference_list,
    format_authors_inline,
    make_cite_key,
)


@pytest.fixture(autouse=True)
def no_embedding_download(monkeypatch):
    """离线测试强制 TF-IDF 降级，避免下载 sentence-transformers 模型。

    embedding 路径由下面的 fake embedder 单测专门覆盖，不依赖网络。
    """
    monkeypatch.setattr(TC, "_EMBEDDER_TRIED", True)
    monkeypatch.setattr(TC, "_EMBEDDER", None)


def _paper(pid: str, title: str, abstract: str, **kw):
    return make_paper(paper_id=pid, title=title, abstract=abstract, **kw)


# --------------------------------------------------------------------------
# 聚类
# --------------------------------------------------------------------------
def test_choose_k_bounds():
    assert choose_k(2) == 1
    assert choose_k(50) <= 8
    assert choose_k(50) >= 2
    assert choose_k(100, requested=3) == 3
    assert choose_k(4, requested=99) == 3  # 不能超过 n-1


def test_embed_texts_shape():
    vecs = embed_texts(["deep learning for vision", "graph neural network on molecules"])
    assert vecs.shape[0] == 2
    assert vecs.shape[1] >= 1


# --------------------------------------------------------------------------
# embedding 路径（用 fake embedder 验证代码分支，不下载真实模型）
# --------------------------------------------------------------------------
def _fake_embedder():
    """返回一个确定性的假 embedder：第 i 篇向量在维度 i%2 上置 1。"""
    class FakeEmbedder:
        def encode(self, texts, **kw):
            out = []
            for i in range(len(texts)):
                v = np.zeros(4, dtype=np.float32)
                v[0 if i < len(texts) // 2 else 1] = 1.0
                out.append(v)
            return np.array(out, dtype=np.float32)

    return FakeEmbedder()


def test_embed_texts_prefers_sentence_transformer(monkeypatch):
    """embed_texts 在 embedder 可用时应优先走语义向量（维度由 embedder 决定）。"""
    monkeypatch.setattr(TC, "_EMBEDDER_TRIED", True)
    monkeypatch.setattr(TC, "_EMBEDDER", _fake_embedder())
    vecs = embed_texts(["a", "b", "c"])
    assert vecs.shape == (3, 4)          # 用了 fake 的 4 维，而非 TF-IDF 的 8000 维
    assert vecs.dtype == np.float32


def test_cluster_uses_embedder_when_available(monkeypatch):
    """embedder 可用时，聚类应基于语义向量把两套主题清晰分开。"""
    monkeypatch.setattr(TC, "_EMBEDDER_TRIED", True)
    monkeypatch.setattr(TC, "_EMBEDDER", _fake_embedder())
    vision = [_paper(f"v{i}", f"Vision paper {i}", "vision image cnn pixels")
               for i in range(6)]
    nlp = [_paper(f"n{i}", f"NLP paper {i}", "language text transformer tokens")
            for i in range(6)]
    clusters = cluster_papers(vision + nlp, n_clusters=2)
    assert len(clusters) == 2
    v_ids = {p["paper_id"] for p in vision}
    groups = [set(c.paper_ids) for c in clusters]
    assert any(g == v_ids for g in groups)   # 一个簇恰好是全部 vision


def test_cluster_separates_distinct_topics():
    """两组语义明显不同的文献应被分到不同簇。"""
    vision = [
        _paper(f"v{i}", f"Convolutional network for image classification {i}",
               "convolutional neural network image classification vision imagenet pixels")
        for i in range(6)
    ]
    nlp = [
        _paper(f"n{i}", f"Transformer language model pretraining {i}",
               "transformer language model pretraining text tokens corpus translation")
        for i in range(6)
    ]
    clusters = cluster_papers(vision + nlp, n_clusters=2)
    assert len(clusters) == 2

    groups = [set(c.paper_ids) for c in clusters]
    v_ids, n_ids = {p["paper_id"] for p in vision}, {p["paper_id"] for p in nlp}
    # 每个簇应基本由单一主题构成
    assert any(g <= v_ids or g <= n_ids for g in groups)
    assert sum(len(g) for g in groups) == 12


def test_cluster_tiny_pool():
    clusters = cluster_papers([_paper("a", "One", "x"), _paper("b", "Two", "y")])
    assert len(clusters) == 1
    assert len(clusters[0].paper_ids) == 2


def test_cluster_empty():
    assert cluster_papers([]) == []


def test_cluster_ids_are_renumbered_by_size():
    papers = [
        _paper(f"a{i}", f"Alpha topic study {i}", "alpha alpha alpha beta") for i in range(8)
    ] + [
        _paper(f"z{i}", f"Zeta unrelated survey {i}", "zeta zeta zeta gamma") for i in range(3)
    ]
    clusters = cluster_papers(papers, n_clusters=2)
    assert clusters[0].cluster_id == 0
    assert clusters[0].size >= clusters[1].size  # 0 号必须是最大簇


# --------------------------------------------------------------------------
# BibTeX
# --------------------------------------------------------------------------
def test_cite_key_unique_and_readable():
    used: set = set()
    p1 = make_paper(paper_id="1", title="The Attention Mechanism Explained",
                    authors=["Ashish Vaswani"], year=2017)
    p2 = make_paper(paper_id="2", title="The Attention Mechanism Explained",
                    authors=["Ashish Vaswani"], year=2017)
    k1, k2 = make_cite_key(p1, used), make_cite_key(p2, used)
    assert k1 == "vaswani2017attention"   # 跳过停用词 "The"
    assert k2 != k1                        # 冲突自动加后缀


def test_bibtex_entry_types():
    arxiv_p = make_paper(paper_id="arxiv:2301.00001", title="Preprint Work",
                         authors=["A B"], year=2023, source="arxiv")
    conf_p = make_paper(paper_id="doi:10.1/y", title="Conference Work",
                        authors=["C D"], year=2022, venue="Proceedings of ICML",
                        source="openalex", doi="10.1/y")
    bib, mapping = build_bibtex([arxiv_p, conf_p])

    assert "@misc{" in bib and "archivePrefix = {arXiv}" in bib
    assert "@inproceedings{" in bib and "booktitle = {Proceedings of ICML}" in bib
    assert set(mapping) == {"arxiv:2301.00001", "doi:10.1/y"}


def test_bibtex_escapes_special_chars():
    p = make_paper(paper_id="x", title="Cost & Performance of 50% Models",
                   authors=["E F"], year=2024)
    bib, _ = build_bibtex([p])
    assert r"\&" in bib and r"\%" in bib


def test_reference_list_ordered_by_citation_number():
    papers = [
        make_paper(paper_id="a", title="First Paper", authors=["X Y"], year=2020),
        make_paper(paper_id="b", title="Second Paper", authors=["Z W"], year=2021),
    ]
    refs = build_reference_list(papers, {"b": 1, "a": 2})
    assert refs.index("**[1]**") < refs.index("**[2]**")
    assert "Second Paper" in refs.split("**[2]**")[0]  # [1] 是 Second Paper


def test_reference_list_skips_uncited():
    papers = [make_paper(paper_id="a", title="Cited", authors=["X"], year=2020),
              make_paper(paper_id="b", title="Uncited", authors=["Y"], year=2020)]
    refs = build_reference_list(papers, {"a": 1})
    assert "Cited" in refs and "Uncited" not in refs


def test_format_authors_inline():
    assert format_authors_inline(["A", "B"]) == "A, B"
    assert format_authors_inline(["A", "B", "C", "D"]) == "A, B, C et al."
    assert format_authors_inline([]) == "Anon."
