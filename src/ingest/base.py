"""ingest 层公共设施：Paper 结构、限流器、带礼貌 UA 的 HTTP 会话。"""

from __future__ import annotations

import logging
import re
import threading
import time
from typing import Any, Dict, List, Optional, TypedDict

import requests

from src.config import get_settings

logger = logging.getLogger(__name__)


class Paper(TypedDict, total=False):
    """跨数据源统一的文献结构。"""

    paper_id: str          # 规范化主键，如 arxiv:2301.00001 / doi:10.1145/xxx / openalex:W123
    title: str
    authors: List[str]
    year: Optional[int]
    venue: str
    citation_count: int
    abstract: str
    url: str               # 落地页
    pdf_url: Optional[str] # OA 全文直链
    doi: Optional[str]
    source: str            # arxiv / openalex / semantic_scholar
    fulltext: Optional[str]
    score: float           # Ranker 打分
    matched_queries: List[str]


# --------------------------------------------------------------------------
# 限流：各站点独立计时，保证「两次请求间隔 >= interval」
# --------------------------------------------------------------------------
class RateLimiter:
    def __init__(self, interval: float, name: str = ""):
        self.interval = interval
        self.name = name
        self._last = 0.0
        self._lock = threading.Lock()

    def wait(self) -> None:
        with self._lock:
            gap = time.monotonic() - self._last
            if gap < self.interval:
                sleep_for = self.interval - gap
                logger.debug("[%s] 限流等待 %.2fs", self.name, sleep_for)
                time.sleep(sleep_for)
            self._last = time.monotonic()


# 各源官方建议的最小间隔
LIMITERS: Dict[str, RateLimiter] = {
    "arxiv": RateLimiter(3.0, "arxiv"),            # arXiv 明确要求 >= 3s
    "openalex": RateLimiter(0.12, "openalex"),     # polite pool 约 10 req/s
    "semantic_scholar": RateLimiter(3.2, "semantic_scholar"),  # 无 key 100 次 / 5min
    "unpaywall": RateLimiter(0.15, "unpaywall"),
    "pdf": RateLimiter(1.0, "pdf"),
}


# --------------------------------------------------------------------------
# HTTP
# --------------------------------------------------------------------------
_session: Optional[requests.Session] = None


def get_session() -> requests.Session:
    global _session
    if _session is None:
        s = requests.Session()
        s.headers.update({"User-Agent": get_settings().user_agent})
        _session = s
    return _session


def http_get(
    url: str,
    *,
    source: str,
    params: Optional[Dict[str, Any]] = None,
    timeout: int = 25,
    retries: int = 2,
    stream: bool = False,
) -> Optional[requests.Response]:
    """带限流 + 指数退避重试的 GET。失败返回 None 而不抛异常（单源失败不应中断全局）。"""
    limiter = LIMITERS.get(source)
    for attempt in range(retries + 1):
        if limiter:
            limiter.wait()
        try:
            resp = get_session().get(url, params=params, timeout=timeout, stream=stream)
            if resp.status_code == 429:
                backoff = 2 ** attempt * 3
                logger.warning("[%s] 429 限流，退避 %ds", source, backoff)
                time.sleep(backoff)
                continue
            resp.raise_for_status()
            return resp
        except requests.RequestException as exc:
            if attempt >= retries:
                logger.warning("[%s] 请求失败 %s: %s", source, url, exc)
                return None
            time.sleep(1.5 * (attempt + 1))
    return None


# --------------------------------------------------------------------------
# 文本与标识符规范化
# --------------------------------------------------------------------------
_WS = re.compile(r"\s+")


def clean_text(text: Optional[str]) -> str:
    if not text:
        return ""
    return _WS.sub(" ", str(text)).strip()


def normalize_doi(doi: Optional[str]) -> Optional[str]:
    """去掉 https://doi.org/ 前缀并小写化。"""
    if not doi:
        return None
    d = str(doi).strip().lower()
    for prefix in ("https://doi.org/", "http://doi.org/", "doi:"):
        if d.startswith(prefix):
            d = d[len(prefix):]
    return d or None


def normalize_title_key(title: str) -> str:
    """用于去重的标题指纹：仅保留小写字母数字。"""
    return re.sub(r"[^a-z0-9]", "", (title or "").lower())


def normalize_arxiv_id(raw: str) -> str:
    """2301.00001v3 -> 2301.00001"""
    return re.sub(r"v\d+$", "", (raw or "").strip())


def make_paper(**kwargs) -> Paper:
    """构造带默认值的 Paper。"""
    paper: Paper = {
        "paper_id": kwargs.get("paper_id", ""),
        "title": clean_text(kwargs.get("title")),
        "authors": kwargs.get("authors") or [],
        "year": kwargs.get("year"),
        "venue": clean_text(kwargs.get("venue")),
        "citation_count": int(kwargs.get("citation_count") or 0),
        "abstract": clean_text(kwargs.get("abstract")),
        "url": kwargs.get("url") or "",
        "pdf_url": kwargs.get("pdf_url"),
        "doi": normalize_doi(kwargs.get("doi")),
        "source": kwargs.get("source", ""),
        "fulltext": None,
        "score": 0.0,
        "matched_queries": kwargs.get("matched_queries") or [],
    }
    return paper
