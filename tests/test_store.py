"""离线测试：跨主题持久化文献池（SQLite）。"""

from __future__ import annotations

from pathlib import Path

from src.ingest.base import make_paper
from src.ingest.store import PaperStore, _paper_key


def _mk(doi, title, **kw):
    p = make_paper(doi=doi, title=title, year=2021, citation_count=5, abstract=f"abstract about {title}")
    p.update(kw)
    return p


def test_paper_key_priority():
    assert _paper_key(make_paper(doi="10.1/abc", title="X")) == "doi:10.1/abc"
    assert _paper_key(make_paper(paper_id="arxiv:2301.0001", title="Y")) == "arxiv:2301.0001"
    assert _paper_key(make_paper(title="Some Long Enough Title Here")) == "title:somelongenoughtitlehere"


def test_upsert_and_get(tmp_path: Path):
    store = PaperStore(tmp_path / "papers.sqlite")
    p = _mk("10.1145/1", "Neural retrieval methods")
    n = store.upsert([p])
    assert n == 1
    got = store.get(make_paper(doi="10.1145/1", title="Neural retrieval methods"))
    assert got is not None
    assert got["citation_count"] == 5


def test_hydrate_backfills_missing_fields(tmp_path: Path):
    store = PaperStore(tmp_path / "papers.sqlite")
    # 先存一份带完整摘要/引用的
    full = _mk("10.1145/2", "Graph neural networks survey", abstract="long cached abstract", citation_count=99)
    store.upsert([full])

    # 新检索到的同一篇，但缺摘要/引用
    fresh = make_paper(doi="10.1145/2", title="Graph neural networks survey")
    hydrated = store.hydrate([fresh])
    assert hydrated[0]["abstract"] == "long cached abstract"
    assert hydrated[0]["citation_count"] == 99


def test_recall_returns_similar(tmp_path: Path):
    store = PaperStore(tmp_path / "papers.sqlite")
    store.upsert([
        _mk("10.1/a", "diffusion models for image generation", abstract="denoising score matching"),
        _mk("10.1/b", "cooking recipes dataset", abstract="food images"),
    ])
    got = store.recall("diffusion probabilistic models", top_k=1)
    assert got and "diffusion" in got[0]["title"].lower()


def test_upsert_is_idempotent(tmp_path: Path):
    store = PaperStore(tmp_path / "papers.sqlite")
    store.upsert([_mk("10.1/c", "topic A")])
    store.upsert([_mk("10.1/c", "topic A", citation_count=10)])
    # 主键相同，应只有一条；重新 get 应为最新值
    assert store.get(make_paper(doi="10.1/c", title="topic A"))["citation_count"] == 10
