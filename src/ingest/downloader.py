"""全文下载层：解析 OA 直链 → 下载 PDF → 解析文本。

合规原则（对应设计方案 6.4）：
- 只下载 OA / 作者自存档副本，不抓出版社付费 PDF；
- 落盘时在 sidecar 记录来源 URL 与 license；
- 下载走独立限流器，且带联系方式 UA。
"""

from __future__ import annotations

import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional

from src.config import get_settings
from src.ingest.base import LIMITERS, Paper, get_session
from src.ingest.pdf_parser import condense_fulltext, deep_parse_pdf, parse_pdf
from src.ingest.unpaywall import find_oa_pdf

logger = logging.getLogger(__name__)

# 明确的付费墙域名，直接跳过（避免误抓）
_PAYWALL_HOSTS = (
    "sciencedirect.com",
    "springer.com/article",
    "onlinelibrary.wiley.com/doi/abs",
    "ieeexplore.ieee.org/document",
    "dl.acm.org/doi/abs",
)


def resolve_fulltext(paper: Paper) -> tuple[Optional[str], Optional[str]]:
    """返回 (可下载的 PDF URL, license)。拿不到返回 (None, None) → 退回摘要模式。"""
    pdf_url = paper.get("pdf_url")
    if pdf_url and not _is_paywalled(pdf_url):
        return pdf_url, "source-provided"

    doi = paper.get("doi")
    if doi:
        url, lic = find_oa_pdf(doi)
        if url and not _is_paywalled(url):
            return url, lic or "unpaywall"
    return None, None


def _is_paywalled(url: str) -> bool:
    low = (url or "").lower()
    return any(host in low for host in _PAYWALL_HOSTS)


def oa_available(paper: Paper) -> bool:
    """该文献是否有可能获取 OA 全文：已有直链，或可用 DOI 经 Unpaywall 解析。"""
    if paper.get("pdf_url") and not _is_paywalled(paper["pdf_url"]):
        return True
    return bool(paper.get("doi"))


def _safe_filename(paper_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "_", paper_id)[:120] + ".pdf"


def download_pdf(paper: Paper, save_dir: Path) -> Optional[Path]:
    """下载单篇 PDF 到 save_dir，已存在则复用缓存。"""
    pdf_url, license_ = resolve_fulltext(paper)
    if not pdf_url:
        return None

    save_dir.mkdir(parents=True, exist_ok=True)
    path = save_dir / _safe_filename(paper["paper_id"])
    if path.exists() and path.stat().st_size > 2048:
        logger.debug("命中 PDF 缓存: %s", path.name)
        return path

    LIMITERS["pdf"].wait()
    try:
        with get_session().get(pdf_url, stream=True, timeout=45, allow_redirects=True) as resp:
            resp.raise_for_status()
            ctype = resp.headers.get("Content-Type", "").lower()
            if "pdf" not in ctype and not pdf_url.lower().endswith(".pdf"):
                logger.debug("非 PDF 响应(%s)，跳过 %s", ctype, paper["paper_id"])
                return None
            with open(path, "wb") as f:
                total = 0
                for chunk in resp.iter_content(8192):
                    f.write(chunk)
                    total += len(chunk)
                    if total > 40 * 1024 * 1024:  # 单篇上限 40MB
                        logger.warning("PDF 过大，截断下载: %s", paper["paper_id"])
                        break
    except Exception as exc:
        logger.debug("PDF 下载失败 %s: %s", paper["paper_id"], exc)
        path.unlink(missing_ok=True)
        return None

    # sidecar：记录来源与许可，满足合规要求
    meta = {
        "paper_id": paper["paper_id"],
        "title": paper.get("title"),
        "source_url": pdf_url,
        "license": license_,
        "origin": paper.get("source"),
    }
    path.with_suffix(".json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return path


def fetch_fulltexts(papers: List[Paper], max_workers: int = 3) -> int:
    """并发下载并解析 Top-N 文献全文，结果写回 `paper["fulltext"]`。

    返回成功获取全文的篇数。
    """
    settings = get_settings()
    if not settings.enable_fulltext or not papers:
        return 0

    save_dir = settings.cache_dir / "pdf"
    ok = 0

    def _work(paper: Paper) -> bool:
        path = download_pdf(paper, save_dir)
        if path is None:
            return False
        text = parse_pdf(path)
        if not text or len(text) < 500:
            return False
        paper["fulltext"] = condense_fulltext(text)
        paper["has_fulltext"] = True
        paper["fulltext_chars"] = len(paper["fulltext"])
        if settings.enable_pdf_deep_parse:
            try:
                struct = deep_parse_pdf(path, max_pages=settings.pdf_max_pages)
                if struct:
                    paper["fulltext_struct"] = struct
            except Exception as exc:  # noqa: BLE001 - 深度解析失败不应影响主流程
                logger.debug("PDF 深度解析失败 %s: %s", paper["paper_id"], exc)
        return True

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_work, p): p for p in papers}
        for fut in as_completed(futures):
            try:
                if fut.result():
                    ok += 1
            except Exception as exc:
                logger.debug("全文任务异常: %s", exc)

    logger.info("全文获取成功 %d/%d 篇（其余走摘要模式）", ok, len(papers))
    return ok
