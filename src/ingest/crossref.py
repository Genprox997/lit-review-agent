"""Crossref 检索客户端（DOI 元数据 + 引用数）。

- 端点：`https://api.crossref.org/works`（免 key，带 `mailto` 进 polite pool）
- 主要价值：补全 DOI 规范元数据、`is-referenced-by-count`（被引数）。
  出版社页多为付费，故 `pdf_url` 留空，由下游 Unpaywall 按 DOI 兜底找 OA 副本。
- 限流：带 mailto 后约 30~50 次/秒，这里保守用 0.1s（由 `LIMITERS["crossref"]`）。
"""

from __future__ import annotations

import logging
import re
from typing import List, Optional

from src.ingest.base import (
    Paper,
    clean_text,
    http_get,
    make_paper,
    normalize_doi,
)

logger = logging.getLogger(__name__)

CROSSREF_ENDPOINT = "https://api.crossref.org/works"

# Crossref 摘要常为 JATS XML，含 <jats:p> / <mml:math> 等标签
_TAG_RE = re.compile(r"<[^>]+>")


def _strip_jats(text: Optional[str]) -> str:
    if not text:
        return ""
    cleaned = _TAG_RE.sub(" ", text)
    # 去掉标签后可能残留的「标点前空格」，如 "things ." -> "things."
    cleaned = re.sub(r"\s+([.,;:!?])", r"\1", cleaned)
    return clean_text(cleaned)


def _first(val):
    if isinstance(val, list):
        return val[0] if val else None
    return val


def _authors_from_item(item: dict) -> List[str]:
    out: List[str] = []
    for a in item.get("author", []) or []:
        given = clean_text(a.get("given"))
        family = clean_text(a.get("family"))
        name = " ".join(x for x in (given, family) if x)
        if name:
            out.append(name)
    return out


def _year_from_item(item: dict) -> Optional[int]:
    issued = item.get("issued") or {}
    parts = issued.get("date-parts") or []
    if parts and parts[0] and parts[0][0]:
        try:
            return int(parts[0][0])
        except (TypeError, ValueError):
            return None
    # 退化：从 published-print / created 取
    for key in ("published-print", "published", "created"):
        dp = (item.get(key) or {}).get("date-parts") or []
        if dp and dp[0] and dp[0][0]:
            try:
                return int(dp[0][0])
            except (TypeError, ValueError):
                continue
    return None


def search_crossref(query: str, max_results: int = 25, min_year: int = 0) -> List[Paper]:
    """按关键词检索 Crossref，返回统一 Paper 列表。"""
    from src.config import get_settings

    settings = get_settings()
    params = {
        "query": query,
        "rows": str(max(1, min(max_results, 100))),
        "select": (
            "DOI,title,author,year,issued,abstract,container-title,"
            "is-referenced-by-count,URL,type"
        ),
        "mailto": settings.contact_email,
    }
    if min_year and min_year > 0:
        params["filter"] = f"from-pub-date:{min_year}-01-01"

    resp = http_get(CROSSREF_ENDPOINT, source="crossref", params=params)
    if resp is None:
        return []

    try:
        items = resp.json().get("message", {}).get("items", [])
    except ValueError:
        return []

    papers: List[Paper] = []
    for item in items:
        try:
            papers.append(_parse_item(item, query))
        except Exception as exc:  # 单条失败不影响整体
            logger.debug("跳过一条 Crossref item: %s", exc)
    logger.info("Crossref 检索 %r → %d 篇", query, len(papers))
    return papers


def _parse_item(item: dict, query: str) -> Paper:
    doi = normalize_doi(_first(item.get("DOI")))
    title = clean_text(_first(item.get("title")))
    venue = clean_text(_first(item.get("container-title"))) or "Crossref"
    year = _year_from_item(item)
    abstract = _strip_jats(_first(item.get("abstract")))
    cites = int(item.get("is-referenced-by-count") or 0)
    url = item.get("URL") or (f"https://doi.org/{doi}" if doi else "")

    return make_paper(
        paper_id=f"doi:{doi}" if doi else f"crossref:{title[:40]}",
        title=title,
        authors=_authors_from_item(item),
        year=year,
        venue=venue,
        citation_count=cites,
        abstract=abstract,
        url=url,
        pdf_url=None,  # 出版社页多为付费，OA 副本交给 Unpaywall 兜底
        doi=doi,
        source="crossref",
        matched_queries=[query],
    )
