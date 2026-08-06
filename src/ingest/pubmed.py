"""PubMed / PMC 检索客户端（NCBI E-utilities）。

- 端点（无需 key，靠 `tool`+`email` 识别善意机器人）：
  - esearch ：`eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi`
  - efetch  ：`eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi`（摘要 XML）
  - idconv  ：`ncbi.nlm.nih.gov/pmc/utils/idconv/v1.0/`（PMID → PMCID + OA 标志）
- 官方建议无 key 限 3 次/秒（由 `LIMITERS["pubmed"]` 保证）。
- OA 全文：命中 PMC 且 `oa=true` 时拼 `ncbi.nlm.nih.gov/pmc/articles/PMC{id}/pdf/`。
- 引用数 PubMed 接口不直接给，留 0（可由 OpenAlex 按 DOI 后续补全）。
"""

from __future__ import annotations

import logging
import xml.etree.ElementTree as ET
from typing import Dict, List, Optional

from src.config import get_settings
from src.ingest.base import (
    Paper,
    clean_text,
    http_get,
    make_paper,
    normalize_doi,
)

logger = logging.getLogger(__name__)

ESEARCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
EFETCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
IDCONV = "https://www.ncbi.nlm.nih.gov/pmc/utils/idconv/v1.0/"


def _ncbi_params(extra: Dict[str, str]) -> Dict[str, str]:
    settings = get_settings()
    return {
        "tool": "lit-review-agent",
        "email": settings.contact_email,
        **extra,
    }


def _esearch(query: str, max_results: int, min_year: int) -> List[str]:
    params = _ncbi_params({
        "db": "pubmed",
        "term": query,
        "retmode": "json",
        "retmax": str(max(1, min(max_results, 200))),
    })
    if min_year and min_year > 0:
        params["datetype"] = "pdat"
        params["mindate"] = str(min_year)
        params["maxdate"] = "3000"
    resp = http_get(ESEARCH, source="pubmed", params=params)
    if resp is None:
        return []
    try:
        data = resp.json()
        return [str(pmid) for pmid in data.get("esearchresult", {}).get("idlist", [])]
    except (ValueError, KeyError) as exc:
        logger.warning("PubMed esearch 解析失败: %s", exc)
        return []


def _efetch(pmids: List[str]) -> Dict[str, dict]:
    """批量拉摘要 XML，返回 {pmid: 解析出的元数据字段}。"""
    if not pmids:
        return {}
    params = _ncbi_params({
        "db": "pubmed",
        "id": ",".join(pmids),
        "retmode": "xml",
        "rettype": "abstract",
    })
    resp = http_get(EFETCH, source="pubmed", params=params)
    if resp is None:
        return {}
    try:
        root = ET.fromstring(resp.content)
    except ET.ParseError as exc:
        logger.warning("PubMed efetch XML 解析失败: %s", exc)
        return {}

    out: Dict[str, dict] = {}
    for art in root.findall(".//PubmedArticle"):
        try:
            meta = _parse_article(art)
            if meta.get("pmid"):
                out[meta["pmid"]] = meta
        except Exception as exc:  # 单篇失败不影响整体
            logger.debug("跳过一篇 PubMed 解析: %s", exc)
    return out


def _text(el: Optional[ET.Element]) -> str:
    if el is None:
        return ""
    return clean_text("".join(el.itertext()))


def _parse_article(art: ET.Element) -> dict:
    cit = art.find("MedlineCitation")
    pmid = clean_text((cit.findtext("PMID") if cit is not None else "") or "")
    article = cit.find("Article") if cit is not None else None
    if article is None:
        return {"pmid": pmid}

    title = _text(article.find("ArticleTitle"))

    # 摘要（可能分多段，含标签）
    abstract_el = article.find("Abstract")
    abstract = ""
    if abstract_el is not None:
        parts = [_text(p) for p in abstract_el.findall("AbstractText")]
        abstract = " ".join(p for p in parts if p)

    # 作者
    authors: List[str] = []
    author_list = article.find("AuthorList")
    if author_list is not None:
        for a in author_list.findall("Author"):
            last = clean_text(a.findtext("LastName"))
            fore = clean_text(a.findtext("ForeName") or a.findtext("Initials"))
            name = " ".join(x for x in (fore, last) if x)  # 名 + 姓，与 Crossref/arXiv 一致
            if name:
                authors.append(name)

    # 期刊 / 年份
    journal_el = article.find("Journal")
    venue = clean_text(journal_el.findtext("Title")) if journal_el is not None else ""
    year = _extract_year(article, journal_el)

    # DOI
    doi = None
    for loc in article.findall(".//ELocationID"):
        if (loc.get("EIdType") == "doi") and (loc.get("ValidYN", "Y") != "N"):
            doi = normalize_doi(loc.text)
            if doi:
                break

    return {
        "pmid": pmid,
        "title": title,
        "abstract": abstract,
        "authors": authors,
        "venue": venue,
        "year": year,
        "doi": doi,
    }


def _extract_year(article: ET.Element, journal_el: Optional[ET.Element]) -> Optional[int]:
    # 优先 Article/ArticleDate（部分记录有）
    for node in article.findall("ArticleDate"):
        y = node.findtext("Year")
        if y and y.isdigit():
            return int(y)
    # 退而求其次 Journal/JournalIssue/PubDate
    if journal_el is not None:
        pubdate = journal_el.find("JournalIssue/PubDate")
        if pubdate is not None:
            y = pubdate.findtext("Year")
            if y and y.isdigit():
                return int(y)
            md = pubdate.findtext("MedlineDate") or ""
            digits = "".join(ch for ch in md[:4] if ch.isdigit())
            if len(digits) == 4:
                return int(digits)
    return None


def _pmc_oa_map(pmids: List[str]) -> Dict[str, str]:
    """PMID → PMC OA PDF 直链（仅 open access 的 PMC 文章）。"""
    if not pmids:
        return {}
    params = _ncbi_params({
        "ids": ",".join(pmids),
        "format": "json",
        "tool": "lit-review-agent",
    })
    resp = http_get(IDCONV, source="pubmed", params=params, retries=1)
    if resp is None:
        return {}
    try:
        records = resp.json().get("records", [])
    except ValueError:
        return {}

    out: Dict[str, str] = {}
    for rec in records:
        pmid = str(rec.get("pmid", ""))
        if not pmid or not rec.get("oa"):
            continue
        pmcid = rec.get("pmcid")
        if pmcid:
            out[pmid] = f"https://www.ncbi.nlm.nih.gov/pmc/articles/{pmcid}/pdf/"
    return out


def search_pubmed(query: str, max_results: int = 25, min_year: int = 0) -> List[Paper]:
    """按关键词检索 PubMed/PMC，返回统一 Paper 列表。"""
    pmids = _esearch(query, max_results, min_year)
    if not pmids:
        logger.info("PubMed 检索 %r → 0 篇", query)
        return []

    metas = _efetch(pmids)
    oa_map = _pmc_oa_map(list(metas.keys()))

    papers: List[Paper] = []
    for pmid, meta in metas.items():
        doi = meta.get("doi")
        paper_id = f"doi:{doi}" if doi else f"pmid:{pmid}"
        pdf_url = oa_map.get(pmid)
        papers.append(make_paper(
            paper_id=paper_id,
            title=meta["title"],
            authors=meta.get("authors") or [],
            year=meta.get("year"),
            venue=meta.get("venue") or "PubMed",
            citation_count=0,
            abstract=meta.get("abstract") or "",
            url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
            pdf_url=pdf_url,
            doi=doi,
            source="pubmed",
            matched_queries=[query],
        ))
    logger.info("PubMed 检索 %r → %d 篇（其中 %d 篇有 PMC OA 全文）",
                query, len(papers), len(oa_map))
    return papers
