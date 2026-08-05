"""arXiv 检索客户端。

- 端点：`http://export.arxiv.org/api/query`（Atom XML）
- 无需 API key；官方要求两次请求间隔 >= 3 秒（由 `LIMITERS["arxiv"]` 保证）
- PDF 直链：`https://arxiv.org/pdf/{id}`，全部为作者自存档 OA 副本
"""

from __future__ import annotations

import logging
import xml.etree.ElementTree as ET
from typing import List

from src.ingest.base import (
    Paper,
    clean_text,
    http_get,
    make_paper,
    normalize_arxiv_id,
)

logger = logging.getLogger(__name__)

ARXIV_ENDPOINT = "http://export.arxiv.org/api/query"
NS = {"a": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}


def _build_search_query(query: str) -> str:
    """把自然语言检索式转成 arXiv 语法。

    多词时用 AND 连接各词的 all: 字段，比整串 all:"..." 召回更稳。
    已含 arXiv 字段前缀（ti:/abs:/cat: 等）时原样透传。
    """
    q = query.strip()
    if any(q.startswith(p) for p in ("ti:", "abs:", "au:", "cat:", "all:")):
        return q
    tokens = [t for t in q.replace('"', " ").split() if t]
    if len(tokens) <= 1:
        return f"all:{q}"
    return " AND ".join(f'all:"{t}"' if "-" in t else f"all:{t}" for t in tokens)


def search_arxiv(query: str, max_results: int = 25, sort_by: str = "relevance") -> List[Paper]:
    """按关键词检索 arXiv，返回统一 Paper 列表。"""
    params = {
        "search_query": _build_search_query(query),
        "start": 0,
        "max_results": max(1, min(max_results, 100)),
        "sortBy": sort_by,          # relevance | lastUpdatedDate | submittedDate
        "sortOrder": "descending",
    }
    resp = http_get(ARXIV_ENDPOINT, source="arxiv", params=params)
    if resp is None:
        return []

    try:
        root = ET.fromstring(resp.content)
    except ET.ParseError as exc:
        logger.warning("arXiv XML 解析失败: %s", exc)
        return []

    papers: List[Paper] = []
    for entry in root.findall("a:entry", NS):
        try:
            papers.append(_parse_entry(entry, query))
        except Exception as exc:  # 单条解析失败不影响整体
            logger.debug("跳过一条 arXiv entry: %s", exc)
    logger.info("arXiv 检索 %r → %d 篇", query, len(papers))
    return papers


def _parse_entry(entry: ET.Element, query: str) -> Paper:
    raw_id = entry.findtext("a:id", default="", namespaces=NS)
    aid = normalize_arxiv_id(raw_id.split("/abs/")[-1])

    published = entry.findtext("a:published", default="", namespaces=NS)
    year = int(published[:4]) if published[:4].isdigit() else None

    authors = [
        clean_text(a.findtext("a:name", default="", namespaces=NS))
        for a in entry.findall("a:author", NS)
    ]

    doi = entry.findtext("arxiv:doi", default=None, namespaces=NS)
    journal_ref = entry.findtext("arxiv:journal_ref", default="", namespaces=NS)

    # 优先取 <link title="pdf">，否则用 id 拼直链
    pdf_url = f"https://arxiv.org/pdf/{aid}"
    for link in entry.findall("a:link", NS):
        if link.get("title") == "pdf" and link.get("href"):
            pdf_url = link.get("href")
            break

    return make_paper(
        paper_id=f"arxiv:{aid}",
        title=entry.findtext("a:title", default="", namespaces=NS),
        authors=[a for a in authors if a],
        year=year,
        venue=clean_text(journal_ref) or "arXiv preprint",
        citation_count=0,  # arXiv API 不提供引用数，后续由 OpenAlex 按 DOI/标题补全
        abstract=entry.findtext("a:summary", default="", namespaces=NS),
        url=f"https://arxiv.org/abs/{aid}",
        pdf_url=pdf_url,
        doi=doi,
        source="arxiv",
        matched_queries=[query],
    )
