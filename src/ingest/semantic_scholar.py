"""Semantic Scholar 检索客户端（可选源）。

- 端点：`https://api.semanticscholar.org/graph/v1/paper/search`
- 无 key 时限流较紧（约 100 次 / 5min），配置 `SEMANTIC_SCHOLAR_API_KEY` 可显著提配额
- 提供 `citationCount` 与 `openAccessPdf.url`
"""

from __future__ import annotations

import logging
from typing import List

from src.config import get_settings
from src.ingest.base import Paper, http_get, make_paper, get_session

logger = logging.getLogger(__name__)

S2_ENDPOINT = "https://api.semanticscholar.org/graph/v1/paper/search"

S2_FIELDS = ",".join(
    [
        "paperId",
        "externalIds",
        "title",
        "abstract",
        "year",
        "venue",
        "citationCount",
        "openAccessPdf",
        "url",
        "authors",
    ]
)


def search_semantic_scholar(query: str, limit: int = 25, min_year: int = 0) -> List[Paper]:
    settings = get_settings()
    params = {"query": query, "limit": max(1, min(limit, 100)), "fields": S2_FIELDS}
    if min_year:
        params["year"] = f"{min_year}-"

    # API key 通过请求头传递
    session = get_session()
    old_key = session.headers.pop("x-api-key", None)
    if settings.s2_api_key:
        session.headers["x-api-key"] = settings.s2_api_key

    try:
        resp = http_get(S2_ENDPOINT, source="semantic_scholar", params=params)
    finally:
        session.headers.pop("x-api-key", None)
        if old_key:
            session.headers["x-api-key"] = old_key

    if resp is None:
        return []
    try:
        data = resp.json()
    except ValueError:
        return []

    papers: List[Paper] = []
    for item in data.get("data", []) or []:
        try:
            papers.append(_parse_item(item, query))
        except Exception as exc:
            logger.debug("跳过一条 S2 结果: %s", exc)
    logger.info("Semantic Scholar 检索 %r → %d 篇", query, len(papers))
    return papers


def _parse_item(item: dict, query: str) -> Paper:
    ext = item.get("externalIds") or {}
    doi = ext.get("DOI")
    arxiv_id = ext.get("ArXiv")

    if doi:
        paper_id = f"doi:{doi.lower()}"
    elif arxiv_id:
        paper_id = f"arxiv:{arxiv_id}"
    else:
        paper_id = f"s2:{item.get('paperId')}"

    oa_pdf = (item.get("openAccessPdf") or {}).get("url")
    if not oa_pdf and arxiv_id:
        oa_pdf = f"https://arxiv.org/pdf/{arxiv_id}"

    return make_paper(
        paper_id=paper_id,
        title=item.get("title") or "",
        authors=[a.get("name", "") for a in (item.get("authors") or [])[:20]],
        year=item.get("year"),
        venue=item.get("venue") or "",
        citation_count=item.get("citationCount") or 0,
        abstract=item.get("abstract") or "",
        url=item.get("url") or "",
        pdf_url=oa_pdf,
        doi=doi,
        source="semantic_scholar",
        matched_queries=[query],
    )
