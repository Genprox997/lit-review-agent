"""ingest 层单测：解析、去重、排序、限流。"""

from __future__ import annotations

import time

import pytest

from src.agent.tools import dedup_papers, rank_papers
from src.config import get_settings
from src.ingest.base import (
    RateLimiter,
    make_paper,
    normalize_arxiv_id,
    normalize_doi,
    normalize_title_key,
)
from src.ingest.openalex import _sanitize_filter_value, reconstruct_abstract
from src.ingest.pdf_parser import clean_pdf_text, condense_fulltext


# --------------------------------------------------------------------------
# 标识符规范化
# --------------------------------------------------------------------------
def test_normalize_doi():
    assert normalize_doi("https://doi.org/10.1000/ABC") == "10.1000/abc"
    assert normalize_doi("doi:10.1000/abc") == "10.1000/abc"
    assert normalize_doi(None) is None
    assert normalize_doi("") is None


def test_normalize_arxiv_id():
    assert normalize_arxiv_id("2301.00001v3") == "2301.00001"
    assert normalize_arxiv_id("2301.00001") == "2301.00001"


def test_normalize_title_key():
    a = normalize_title_key("Attention Is All You Need!")
    b = normalize_title_key("attention is all-you-need")
    assert a == b == "attentionisallyouneed"


# --------------------------------------------------------------------------
# OpenAlex 摘要还原
# --------------------------------------------------------------------------
def test_reconstruct_abstract():
    inverted = {"Deep": [0], "learning": [1, 4], "is": [2], "great": [3]}
    assert reconstruct_abstract(inverted) == "Deep learning is great learning"


def test_reconstruct_abstract_empty():
    assert reconstruct_abstract(None) == ""
    assert reconstruct_abstract({}) == ""


def test_sanitize_filter_value():
    """标题里的 , : | 会破坏 OpenAlex filter 语法，必须剔除。"""
    out = _sanitize_filter_value("PINNs for Turbines: Trends, Advances | Challenges (2024)")
    for bad in ",:|()":
        assert bad not in out
    assert "PINNs for Turbines" in out


# --------------------------------------------------------------------------
# 去重与合并
# --------------------------------------------------------------------------
def test_dedup_by_doi_merges_fields():
    """同一篇论文在 arXiv（有 PDF）与 OpenAlex（有引用数）各出现一次，应合并成一条。"""
    p1 = make_paper(
        paper_id="arxiv:2301.00001", title="A Great Paper", doi="10.1/x",
        pdf_url="https://arxiv.org/pdf/2301.00001", abstract="short",
        source="arxiv", matched_queries=["q1"],
    )
    p2 = make_paper(
        paper_id="doi:10.1/x", title="A Great Paper", doi="10.1/x",
        citation_count=120, abstract="a much longer abstract with detail",
        venue="ICML", source="openalex", matched_queries=["q2"],
    )
    out = dedup_papers([p1, p2])
    assert len(out) == 1
    merged = out[0]
    assert merged["citation_count"] == 120           # 取 OpenAlex 的引用数
    assert merged["pdf_url"].endswith("2301.00001")  # 保留 arXiv 的 PDF 直链
    assert merged["abstract"] == "a much longer abstract with detail"
    assert merged["venue"] == "ICML"
    assert set(merged["matched_queries"]) == {"q1", "q2"}
    assert "arxiv" in merged["source"] and "openalex" in merged["source"]


def test_dedup_by_title_without_doi():
    p1 = make_paper(paper_id="s2:aaa", title="Learning to Rank with Transformers", source="s2")
    p2 = make_paper(paper_id="s2:bbb", title="learning to rank with transformers", source="s2")
    assert len(dedup_papers([p1, p2])) == 1


def test_dedup_keeps_distinct_papers():
    p1 = make_paper(paper_id="arxiv:1", title="Completely Different Topic One", source="arxiv")
    p2 = make_paper(paper_id="arxiv:2", title="Another Unrelated Subject Two", source="arxiv")
    assert len(dedup_papers([p1, p2])) == 2


def test_dedup_drops_untitled():
    assert dedup_papers([make_paper(paper_id="x", title="")]) == []


# --------------------------------------------------------------------------
# 排序
# --------------------------------------------------------------------------
def test_rank_prefers_relevant_and_cited():
    topic = "graph neural networks"
    papers = [
        make_paper(paper_id="a", title="A Survey on Graph Neural Networks",
                   abstract="graph neural networks message passing", year=2021,
                   citation_count=5000, matched_queries=["graph neural networks"]),
        make_paper(paper_id="b", title="Baking Sourdough Bread",
                   abstract="fermentation and flour hydration", year=2021,
                   citation_count=2, matched_queries=[]),
    ]
    ranked = rank_papers(papers, topic, [topic])
    assert ranked[0]["paper_id"] == "a"
    assert ranked[0]["score"] > ranked[1]["score"]


def test_rank_min_year_filter(monkeypatch):
    settings = get_settings(refresh=True)
    settings.min_year = 2020
    papers = [
        make_paper(paper_id="old", title="Old Work On Topic", year=2010, abstract="topic"),
        make_paper(paper_id="new", title="New Work On Topic", year=2023, abstract="topic"),
    ]
    ranked = rank_papers(papers, "topic", ["topic"], settings)
    assert [p["paper_id"] for p in ranked] == ["new"]


def test_rank_empty():
    assert rank_papers([], "topic", ["topic"]) == []


# --------------------------------------------------------------------------
# PDF 文本处理
# --------------------------------------------------------------------------
def test_clean_pdf_text_strips_references():
    body = "Introduction " + "x" * 500
    text = f"{body}\n\nReferences\n[1] Someone et al."
    out = clean_pdf_text(text)
    assert "Someone et al." not in out
    assert "Introduction" in out


def test_condense_fulltext_keeps_head_and_tail():
    text = "H" * 8000 + "T" * 5000
    out = condense_fulltext(text, head_chars=100, tail_chars=50)
    assert out.startswith("H" * 100)
    assert out.endswith("T" * 50)
    assert "省略" in out


def test_condense_short_text_untouched():
    assert condense_fulltext("short") == "short"


# --------------------------------------------------------------------------
# 限流
# --------------------------------------------------------------------------
def test_rate_limiter_enforces_interval():
    limiter = RateLimiter(0.25, "test")
    limiter.wait()
    start = time.monotonic()
    limiter.wait()
    assert time.monotonic() - start >= 0.2


# --------------------------------------------------------------------------
# 联网测试（默认跳过）
# --------------------------------------------------------------------------
@pytest.mark.network
def test_live_arxiv():
    from src.ingest.arxiv_client import search_arxiv

    papers = search_arxiv("physics-informed neural networks", max_results=5)
    assert len(papers) > 0
    assert all(p["paper_id"].startswith("arxiv:") for p in papers)
    assert all(p["pdf_url"] for p in papers)


@pytest.mark.network
def test_live_openalex():
    from src.ingest.openalex import search_openalex

    papers = search_openalex("physics-informed neural networks", per_page=5)
    assert len(papers) > 0
    assert any(p["citation_count"] > 0 for p in papers)


# --------------------------------------------------------------------------
# PubMed / Crossref 离线解析（monkeypatch http_get 返回假响应）
# --------------------------------------------------------------------------
class _FakeResp:
    def __init__(self, *, json=None, content=b"", text=""):
        self._json = json
        self.content = content
        self.text = text

    def json(self):
        if self._json is None:
            raise ValueError("no json")
        return self._json


_PUBMED_XML = b"""<PubmedArticleSet>
  <PubmedArticle>
    <MedlineCitation Status="MEDLINE">
      <PMID Version="1">1</PMID>
      <Article>
        <ArticleTitle>Adaptive Optics for Deformable Mirrors</ArticleTitle>
        <Abstract><AbstractText>We present a method for wavefront control.</AbstractText></Abstract>
        <AuthorList><Author><LastName>Smith</LastName><ForeName>John A</ForeName></Author></AuthorList>
        <Journal><Title>Optics Express</Title>
          <JournalIssue><PubDate><Year>2021</Year></PubDate></JournalIssue></Journal>
        <ELocationID EIdType="doi" ValidYN="Y">10.1364/OE.2021.0001</ELocationID>
      </Article>
    </MedlineCitation>
  </PubmedArticle>
  <PubmedArticle>
    <MedlineCitation Status="MEDLINE">
      <PMID Version="1">2</PMID>
      <Article>
        <ArticleTitle>Another Study on Mirrors</ArticleTitle>
        <Abstract><AbstractText>Another abstract text.</AbstractText></Abstract>
        <AuthorList><Author><LastName>Doe</LastName><Initials>J</Initials></Author></AuthorList>
        <Journal><Title>Applied Optics</Title>
          <JournalIssue><PubDate><Year>2022</Year></PubDate></JournalIssue></Journal>
      </Article>
    </MedlineCitation>
  </PubmedArticle>
</PubmedArticleSet>"""


def _fake_pubmed_get(url, *, source, params=None, timeout=25, retries=2, stream=False):
    if "esearch.fcgi" in url:
        return _FakeResp(json={"esearchresult": {"idlist": ["1", "2"]}})
    if "efetch.fcgi" in url:
        return _FakeResp(content=_PUBMED_XML)
    if "idconv" in url:
        return _FakeResp(json={"records": [
            {"pmid": "1", "pmcid": "PMC111", "oa": "1"},
            {"pmid": "2"},
        ]})
    return None


def test_search_pubmed_parses_and_oa(monkeypatch):
    import src.ingest.pubmed as pubmed_mod

    monkeypatch.setattr(pubmed_mod, "http_get", _fake_pubmed_get)
    papers = pubmed_mod.search_pubmed("deformable mirror", max_results=10)
    assert len(papers) == 2

    by_id = {p["paper_id"]: p for p in papers}
    p1 = by_id["doi:10.1364/oe.2021.0001"]
    assert p1["title"] == "Adaptive Optics for Deformable Mirrors"
    assert p1["year"] == 2021
    assert p1["venue"] == "Optics Express"
    assert p1["authors"] == ["John A Smith"]
    assert p1["pdf_url"] == "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC111/pdf/"
    assert p1["url"] == "https://pubmed.ncbi.nlm.nih.gov/1/"

    p2 = by_id["pmid:2"]
    assert p2["year"] == 2022
    assert p2["pdf_url"] is None  # 无 PMC OA，pdf_url 留空


def test_search_one_pubmed_dispatched(monkeypatch):
    from src.agent.tools import _search_one
    from src.config import get_settings
    import src.ingest.pubmed as pubmed_mod

    monkeypatch.setattr(pubmed_mod, "http_get", _fake_pubmed_get)
    out = _search_one("pubmed", "deformable mirror", get_settings(refresh=True))
    assert len(out) == 2
    assert all(p["source"] == "pubmed" for p in out)


_CROSSREF_ITEM = {
    "DOI": "10.1145/123.456",
    "title": ["A Crossref Paper Title"],
    "author": [{"given": "Jane", "family": "Roe"}, {"given": "Bob", "family": "Lee"}],
    "issued": {"date-parts": [[2020, 5]]},
    "abstract": "<jats:p>We study <jats:bold>things</jats:bold>.</jats:p>",
    "container-title": ["Journal of Stuff"],
    "is-referenced-by-count": 42,
    "URL": "https://doi.org/10.1145/123.456",
    "type": "journal-article",
}


def _fake_crossref_get(url, *, source, params=None, timeout=25, retries=2, stream=False):
    return _FakeResp(json={"message": {"items": [_CROSSREF_ITEM]}})


def test_search_crossref_parses(monkeypatch):
    import src.ingest.crossref as crossref_mod

    monkeypatch.setattr(crossref_mod, "http_get", _fake_crossref_get)
    papers = crossref_mod.search_crossref("crossref paper", max_results=10)
    assert len(papers) == 1
    p = papers[0]
    assert p["paper_id"] == "doi:10.1145/123.456"
    assert p["title"] == "A Crossref Paper Title"
    assert p["authors"] == ["Jane Roe", "Bob Lee"]
    assert p["year"] == 2020
    assert p["citation_count"] == 42
    assert p["abstract"] == "We study things."  # JATS 标签被剥除
    assert p["pdf_url"] is None                  # 出版社页，靠 Unpaywall 兜底
    assert p["source"] == "crossref"


def test_search_one_crossref_dispatched(monkeypatch):
    from src.agent.tools import _search_one
    from src.config import get_settings
    import src.ingest.crossref as crossref_mod

    monkeypatch.setattr(crossref_mod, "http_get", _fake_crossref_get)
    out = _search_one("crossref", "crossref paper", get_settings(refresh=True))
    assert len(out) == 1
    assert out[0]["source"] == "crossref"


# --------------------------------------------------------------------------
# LLM 429 退避重试（离线）
# --------------------------------------------------------------------------
def test_llm_429_backoff_retries(monkeypatch):
    from src.agent import llm as llm_mod
    from src.config import Settings

    # 用局部 Settings 实例，避免污染全局单例影响其它测试
    settings = Settings()
    settings.llm_provider = "deepseek"
    settings.llm_max_retries = 1  # 允许 1 次重试
    monkeypatch.setattr(llm_mod.time, "sleep", lambda *a, **k: None)  # 加速测试

    class _FakeRateLimit(Exception):
        status_code = 429

    class _FakeResp:
        content = "generated text"

    calls = {"n": 0}

    class _FakeModel:
        def invoke(self, messages):
            calls["n"] += 1
            if calls["n"] == 1:
                raise _FakeRateLimit("rate limited")
            return _FakeResp()

    monkeypatch.setattr(llm_mod, "_get_chat_model", lambda s, j: _FakeModel())
    out = llm_mod.chat("section_writer", "sys", "user", settings=settings)
    assert out == "generated text"
    assert calls["n"] == 2  # 第 1 次 429，第 2 次成功


def test_llm_429_eventually_raises(monkeypatch):
    from src.agent import llm as llm_mod
    from src.config import Settings

    settings = Settings()
    settings.llm_provider = "deepseek"
    settings.llm_max_retries = 1
    monkeypatch.setattr(llm_mod.time, "sleep", lambda *a, **k: None)

    class _FakeRateLimit(Exception):
        status_code = 429

    class _FakeModel:
        def invoke(self, messages):
            raise _FakeRateLimit("always limited")

    monkeypatch.setattr(llm_mod, "_get_chat_model", lambda s, j: _FakeModel())
    with pytest.raises(Exception):
        llm_mod.chat("section_writer", "sys", "user", settings=settings)

