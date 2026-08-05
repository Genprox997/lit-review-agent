"""OpenAlex 检索客户端。

- 端点：`https://api.openalex.org/works`
- 免 key、额度大；URL 带 `mailto=` 进 polite pool
- 提供 `cited_by_count`（引用数排序的主要信号）与 `open_access.oa_url`（OA 全文）
- 摘要以「倒排索引」形式返回，需要还原成正文
"""

from __future__ import annotations

import logging
import re
from typing import Dict, List, Optional

from src.config import get_settings
from src.ingest.base import Paper, clean_text, http_get, make_paper, normalize_doi

logger = logging.getLogger(__name__)

OPENALEX_ENDPOINT = "https://api.openalex.org/works"

SELECT_FIELDS = ",".join(
    [
        "id",
        "doi",
        "title",
        "display_name",
        "publication_year",
        "cited_by_count",
        "abstract_inverted_index",
        "open_access",
        "best_oa_location",
        "primary_location",
        "authorships",
        "type",
    ]
)


def reconstruct_abstract(inverted_index: Optional[Dict[str, List[int]]]) -> str:
    """把 OpenAlex 的 abstract_inverted_index 还原为摘要正文。

    结构形如 {"Deep": [0], "learning": [1, 7], ...}，值是该词出现的位置列表。
    """
    if not inverted_index:
        return ""
    positions: List[tuple] = []
    for word, idxs in inverted_index.items():
        for i in idxs:
            positions.append((i, word))
    if not positions:
        return ""
    positions.sort(key=lambda x: x[0])
    return clean_text(" ".join(word for _, word in positions))


def search_openalex(
    query: str,
    per_page: int = 25,
    min_year: int = 0,
    oa_only: bool = False,
) -> List[Paper]:
    """按关键词检索 OpenAlex。"""
    settings = get_settings()
    filters = ["type:article|preprint|book-chapter"]
    if min_year:
        filters.append(f"publication_year:>{min_year - 1}")
    if oa_only:
        filters.append("is_oa:true")

    params = {
        "search": query,
        "per-page": max(1, min(per_page, 200)),
        "select": SELECT_FIELDS,
        "filter": ",".join(filters),
        "mailto": settings.contact_email,
    }
    resp = http_get(OPENALEX_ENDPOINT, source="openalex", params=params)
    if resp is None:
        return []

    try:
        data = resp.json()
    except ValueError as exc:
        logger.warning("OpenAlex 响应非 JSON: %s", exc)
        return []

    papers: List[Paper] = []
    for work in data.get("results", []):
        try:
            papers.append(_parse_work(work, query))
        except Exception as exc:
            logger.debug("跳过一条 OpenAlex work: %s", exc)
    logger.info("OpenAlex 检索 %r → %d 篇", query, len(papers))
    return papers


def _parse_work(work: dict, query: str) -> Paper:
    doi = normalize_doi(work.get("doi"))
    oa_id = (work.get("id") or "").rsplit("/", 1)[-1]

    # paper_id 优先用 DOI（跨源去重的最强键），否则退回 OpenAlex ID
    paper_id = f"doi:{doi}" if doi else f"openalex:{oa_id}"

    oa = work.get("open_access") or {}
    best_oa = work.get("best_oa_location") or {}
    pdf_url = best_oa.get("pdf_url") or oa.get("oa_url")

    primary = work.get("primary_location") or {}
    source_obj = primary.get("source") or {}
    venue = source_obj.get("display_name") or ""

    authors = [
        clean_text((a.get("author") or {}).get("display_name"))
        for a in (work.get("authorships") or [])[:20]
    ]

    landing = primary.get("landing_page_url") or (
        f"https://doi.org/{doi}" if doi else work.get("id") or ""
    )

    return make_paper(
        paper_id=paper_id,
        title=work.get("display_name") or work.get("title") or "",
        authors=[a for a in authors if a],
        year=work.get("publication_year"),
        venue=venue,
        citation_count=work.get("cited_by_count", 0),
        abstract=reconstruct_abstract(work.get("abstract_inverted_index")),
        url=landing,
        pdf_url=pdf_url,
        doi=doi,
        source="openalex",
        matched_queries=[query],
    )


def _sanitize_filter_value(text: str) -> str:
    """清洗 OpenAlex filter 取值。

    filter 语法用 `,` 分隔条件、`:` 分隔字段与值，标题里的这些字符会导致 400。
    括号、竖线（OR 运算符）同样需要剔除。
    """
    cleaned = re.sub(r"[,:|()\[\]{}<>\"'&+]", " ", text or "")
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned[:110].rstrip()


def enrich_citations(papers: List[Paper], limit: int = 25) -> int:
    """用 OpenAlex 为缺引用数的文献（主要是 arXiv 来源）补全 citation_count。

    优先按 DOI 精确匹配，无 DOI 时按标题检索并校验首条结果的标题相似度。
    返回成功补全的条数。
    """
    targets = [p for p in papers if not p.get("citation_count") and p.get("title")][:limit]
    if not targets:
        return 0

    settings = get_settings()
    filled = 0
    for paper in targets:
        params = {"select": "display_name,cited_by_count,doi", "mailto": settings.contact_email}
        if paper.get("doi"):
            params["filter"] = f"doi:{paper['doi']}"
        else:
            safe_title = _sanitize_filter_value(paper["title"])
            if len(safe_title) < 8:
                continue
            params["filter"] = f"title.search:{safe_title}"
        params["per-page"] = 1

        resp = http_get(OPENALEX_ENDPOINT, source="openalex", params=params, retries=0)
        if resp is None:
            continue
        try:
            results = resp.json().get("results", [])
        except ValueError:
            continue
        if not results:
            continue

        hit = results[0]
        # 无 DOI 时做一次标题校验，避免张冠李戴
        if not paper.get("doi"):
            from src.ingest.base import normalize_title_key

            if normalize_title_key(hit.get("display_name", ""))[:60] != \
               normalize_title_key(paper["title"])[:60]:
                continue
        paper["citation_count"] = hit.get("cited_by_count", 0)
        if not paper.get("doi") and hit.get("doi"):
            paper["doi"] = normalize_doi(hit["doi"])
        filled += 1

    logger.info("OpenAlex 引用数补全：%d/%d 篇", filled, len(targets))
    return filled
