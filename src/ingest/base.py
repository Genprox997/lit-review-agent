"""ingest 层公共设施：Paper 结构、限流器、带礼貌 UA 的 HTTP 会话。"""

from __future__ import annotations

import hashlib
import json as _json
import logging
import re
import threading
import time
from pathlib import Path
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
    relevance: float       # 与主题的相关性（P0-1 闸门）
    has_fulltext: bool     # 是否成功获取 OA 全文（P0-2）
    fulltext_chars: int    # 全文字符数
    fulltext_struct: Optional[Dict[str, Any]]  # PDF 深度解析结构化结果（方向 J'）
    openalex_id: str       # OpenAlex 作品 ID（用于构建引用图，方向 D'）
    referenced_works: List[str]  # 引用作品的 OpenAlex ID 列表（方向 D' 引用网络分析）


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
    "pubmed": RateLimiter(0.34, "pubmed"),        # NCBI 无 key 限 3 次/秒
    "crossref": RateLimiter(0.1, "crossref"),      # 带 mailto 约 30~50 次/秒，保守
    "unpaywall": RateLimiter(0.15, "unpaywall"),
    "pdf": RateLimiter(1.0, "pdf"),
}


# --------------------------------------------------------------------------
# HTTP（含磁盘缓存层，P3-4）
# --------------------------------------------------------------------------
_session: Optional[requests.Session] = None


def get_session() -> requests.Session:
    global _session
    if _session is None:
        s = requests.Session()
        s.headers.update({"User-Agent": get_settings().user_agent})
        _session = s
    return _session


# --- 磁盘缓存：把成功的 GET 响应（2xx）落盘，下次同请求直接复用，省配额、提速 ---
def _http_cache_path(url: str, params: Optional[Dict[str, Any]], source: str) -> Path:
    key = _json.dumps({"u": url, "p": params or {}, "s": source}, sort_keys=True)
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:32]
    settings = get_settings()
    return settings.cache_dir / "http" / f"{source}_{digest}.json"


def _http_cache_get(url: str, params: Optional[Dict[str, Any]], source: str) -> Optional[requests.Response]:
    """命中且未过期则返回重建的 Response，否则 None。"""
    settings = get_settings()
    if not settings.http_cache_enabled:
        return None
    path = _http_cache_path(url, params, source)
    if not path.exists():
        return None
    try:
        rec = _json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001 - 缓存损坏直接忽略
        return None
    ttl = max(0.0, settings.http_cache_ttl_days) * 86400.0
    if ttl and (time.time() - rec.get("ts", 0)) > ttl:
        return None
    if rec.get("status", 200) >= 400:
        return None  # 不回放失败响应

    resp = requests.Response()
    resp.status_code = rec.get("status", 200)
    resp.reason = rec.get("reason", "OK")
    resp.url = rec.get("url", url)
    resp.encoding = rec.get("encoding", "utf-8")
    resp._content = (rec.get("content") or "").encode("utf-8")
    try:
        resp.headers.update(rec.get("headers", {}))
    except Exception:  # noqa: BLE001
        pass
    return resp


def _http_cache_put(url: str, params: Optional[Dict[str, Any]], source: str,
                   resp: requests.Response) -> None:
    settings = get_settings()
    if not settings.http_cache_enabled or resp is None:
        return
    if resp.status_code >= 400:  # 不缓存失败响应
        return
    try:
        path = _http_cache_path(url, params, source)
        path.parent.mkdir(parents=True, exist_ok=True)
        rec = {
            "ts": time.time(),
            "status": resp.status_code,
            "reason": resp.reason,
            "url": resp.url,
            "encoding": resp.encoding or "utf-8",
            "headers": {k: v for k, v in resp.headers.items()},
            "content": resp.text,
        }
        # 临时写 + 原子替换，避免并发半截文件
        tmp = path.with_suffix(".tmp")
        tmp.write_text(_json.dumps(rec, ensure_ascii=False), encoding="utf-8")
        tmp.replace(path)
    except Exception:  # noqa: BLE001 - 缓存写失败不应影响主流程
        logger.debug("[%s] HTTP 缓存写入失败，忽略: %s", source, url)


def http_get(
    url: str,
    *,
    source: str,
    params: Optional[Dict[str, Any]] = None,
    timeout: int = 25,
    retries: int = 2,
    stream: bool = False,
) -> Optional[requests.Response]:
    """带限流 + 指数退避重试的 GET。失败返回 None 而不抛异常（单源失败不应中断全局）。

    命中磁盘缓存（且 2xx、未过期）时直接复用，跳过网络；流式请求（PDF 下载）不缓存。
    """
    if not stream:
        cached = _http_cache_get(url, params, source)
        if cached is not None:
            logger.debug("[%s] HTTP 缓存命中: %s", source, url)
            return cached

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
            if not stream:
                _http_cache_put(url, params, source, resp)
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

# 合法 DOI 形态：10.<注册码>/<非空非空白后缀>。
# 注册码长度在真实 DOI 中多为 4~9 位，但 1 位亦属合法语法，故不过度收紧，
# 仅拦截明显非法的（缺 10. 前缀、含空白、后缀为空）。
_DOI_RE = re.compile(r"^10\.\d{1,9}/[^\s]+$")


def clean_text(text: Optional[str]) -> str:
    if not text:
        return ""
    return _WS.sub(" ", str(text)).strip()


def normalize_doi(doi: Optional[str]) -> Optional[str]:
    """规范化并校验 DOI。

    - 去掉 https://doi.org/、doi: 等前缀，小写化；
    - 用正则校验 `10.<注册码>/<后缀>` 形态，明显非法的直接丢弃（返回 None），
      避免把畸形 DOI 写进 BibTeX / 参考文献表（P0-3）。
    """
    if not doi:
        return None
    d = str(doi).strip().lower()
    for prefix in ("https://doi.org/", "http://doi.org/", "doi.org/", "doi:"):
        if d.startswith(prefix):
            d = d[len(prefix):]
    d = d.strip().rstrip(".").strip()
    if not _DOI_RE.match(d):
        logger.debug("丢弃非法 DOI: %r", doi)
        return None
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
        "relevance": 0.0,
        "has_fulltext": False,
        "fulltext_chars": 0,
        "fulltext_struct": kwargs.get("fulltext_struct"),
        "openalex_id": kwargs.get("openalex_id", ""),
        "referenced_works": kwargs.get("referenced_works") or [],
    }
    return paper
