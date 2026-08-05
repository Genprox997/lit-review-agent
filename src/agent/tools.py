"""Agent 工具层：多源检索、去重、排序、全文获取。

这些函数既被 graph 节点直接调用（确定性流水线，避免 LLM 乱调工具），
也通过 `AGENT_TOOLS` 暴露为 LangChain Tool，供需要自由工具调用的场景使用。
"""

from __future__ import annotations

import logging
import math
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Sequence

import numpy as np

from src.config import Settings, get_settings
from src.ingest.arxiv_client import search_arxiv
from src.ingest.base import Paper, normalize_title_key
from src.ingest.downloader import fetch_fulltexts
from src.ingest.openalex import enrich_citations, search_openalex
from src.ingest.semantic_scholar import search_semantic_scholar

logger = logging.getLogger(__name__)


# ==========================================================================
# 多源检索
# ==========================================================================
def _search_one(source: str, query: str, settings: Settings) -> List[Paper]:
    limit = settings.max_results_per_query
    try:
        if source == "arxiv":
            return search_arxiv(query, max_results=limit)
        if source == "openalex":
            return search_openalex(query, per_page=limit, min_year=settings.min_year)
        if source == "semantic_scholar":
            return search_semantic_scholar(query, limit=limit, min_year=settings.min_year)
    except Exception as exc:
        logger.warning("[%s] 检索 %r 异常: %s", source, query, exc)
    return []


def multi_source_search(
    queries: Sequence[str],
    settings: Optional[Settings] = None,
    sources: Optional[Sequence[str]] = None,
) -> List[Paper]:
    """并发打多源学术 API，合并候选文献（未去重）。

    并发粒度按「源」隔离：同源内部串行以尊重限流，不同源之间并行。
    """
    settings = settings or get_settings()
    sources = list(sources or settings.enabled_sources)
    if not queries or not sources:
        return []

    def _run_source(source: str) -> List[Paper]:
        out: List[Paper] = []
        for q in queries:
            out.extend(_search_one(source, q, settings))
        return out

    results: List[Paper] = []
    with ThreadPoolExecutor(max_workers=max(1, len(sources))) as pool:
        futures = {pool.submit(_run_source, s): s for s in sources}
        for fut in as_completed(futures):
            src = futures[fut]
            try:
                got = fut.result()
                results.extend(got)
                logger.info("源 %s 累计返回 %d 条", src, len(got))
            except Exception as exc:
                logger.warning("源 %s 整体失败: %s", src, exc)

    logger.info("多源检索合计 %d 条（去重前），检索式 %d 条，源 %s",
                len(results), len(queries), sources)
    return results


# ==========================================================================
# 去重
# ==========================================================================
def _merge_into(base: Paper, other: Paper) -> None:
    """把 other 的信息补进 base（保留信息更全的一方）。"""
    if len(other.get("abstract") or "") > len(base.get("abstract") or ""):
        base["abstract"] = other["abstract"]
    if (other.get("citation_count") or 0) > (base.get("citation_count") or 0):
        base["citation_count"] = other["citation_count"]
    if not base.get("pdf_url") and other.get("pdf_url"):
        base["pdf_url"] = other["pdf_url"]
    if not base.get("doi") and other.get("doi"):
        base["doi"] = other["doi"]
    if not base.get("venue") and other.get("venue"):
        base["venue"] = other["venue"]
    if not base.get("year") and other.get("year"):
        base["year"] = other["year"]
    if len(other.get("authors") or []) > len(base.get("authors") or []):
        base["authors"] = other["authors"]
    merged_q = list(dict.fromkeys(
        (base.get("matched_queries") or []) + (other.get("matched_queries") or [])
    ))
    base["matched_queries"] = merged_q
    srcs = set((base.get("source") or "").split("+")) | {other.get("source", "")}
    base["source"] = "+".join(sorted(s for s in srcs if s))


def dedup_papers(papers: Sequence[Paper]) -> List[Paper]:
    """三级去重：DOI → arXiv ID → 标题指纹。

    同一篇论文常同时出现在 arXiv（有全文）与 OpenAlex（有引用数），
    合并后能同时拿到 PDF 直链和被引量。
    """
    by_key: Dict[str, Paper] = {}
    alias: Dict[str, str] = {}   # 各种标识 -> 主键

    def keys_of(p: Paper) -> List[str]:
        ks = []
        if p.get("doi"):
            ks.append(f"doi:{p['doi']}")
        pid = p.get("paper_id", "")
        if pid.startswith("arxiv:"):
            ks.append(pid)
        tk = normalize_title_key(p.get("title", ""))
        if len(tk) >= 12:
            ks.append(f"title:{tk}")
        if not ks and pid:
            ks.append(pid)
        return ks

    for paper in papers:
        if not paper.get("title"):
            continue
        ks = keys_of(paper)
        hit = next((alias[k] for k in ks if k in alias), None)
        if hit is None:
            primary = ks[0]
            by_key[primary] = dict(paper)  # type: ignore[assignment]
            for k in ks:
                alias[k] = primary
        else:
            _merge_into(by_key[hit], paper)
            for k in ks:
                alias.setdefault(k, hit)

    out = list(by_key.values())
    logger.info("去重：%d → %d 篇", len(papers), len(out))
    return out


# ==========================================================================
# 相关性 + 闸门 + 排序
# ==========================================================================
def compute_relevance(
    papers: Sequence[Paper], topic: str, queries: Sequence[str]
) -> List[float]:
    """用 TF-IDF 余弦相似度算文献与主题的相关性，结果写回 `paper["relevance"]`。

    返回**原始**相似度（0~1，负值已裁剪为 0）。不做「除以最大值」归一化，
    否则最相关论文恒等于 1.0，会让真正偏离主题但相对最接近的论文也拿到高分。
    原始分才能支撑相关性闸门（P0-1）按绝对阈值过滤。
    """
    if not papers:
        return []
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
    except ImportError:  # pragma: no cover
        return [0.5] * len(papers)

    query_doc = " ".join([topic] * 3 + list(queries))
    docs = [f"{p.get('title','')} {p.get('title','')} {p.get('abstract','')}" for p in papers]
    try:
        vec = TfidfVectorizer(stop_words="english", max_features=20000, ngram_range=(1, 2))
        matrix = vec.fit_transform(docs + [query_doc])
        sims = cosine_similarity(matrix[-1], matrix[:-1]).ravel()
    except Exception as exc:
        logger.debug("相关性计算失败: %s", exc)
        return [0.5] * len(papers)

    sims = np.clip(sims, 0.0, 1.0)
    out = [float(s) for s in sims]
    for i, p in enumerate(papers):
        p["relevance"] = round(out[i], 4)
    return out


def apply_relevance_gate(
    papers: Sequence[Paper],
    scores: Sequence[float],
    threshold: float,
    min_keep: int = 20,
) -> tuple[List[Paper], List[str]]:
    """硬闸门：剔除相关性低于阈值的论文，避免「高被引但主题跑偏」的文献混入候选池。

    自适应保底：若剔除过多导致文献池塌缩（少于 min_keep），则按相关性降序
    保底保留前 min_keep 篇，保证下游聚类 / 撰写仍有足够素材。
    返回 (保留的论文, 被剔除标题列表)。
    """
    if threshold <= 0 or not papers:
        return list(papers), []

    kept: List[Paper] = []
    kept_ids: set = set()
    dropped: List[Paper] = []
    for p, s in zip(papers, scores):
        if s >= threshold:
            kept.append(p)
            kept_ids.add(p["paper_id"])
        else:
            dropped.append(p)

    if len(kept) < min_keep and len(papers) > min_keep:
        order = sorted(range(len(papers)), key=lambda i: scores[i], reverse=True)
        for i in order:
            if len(kept) >= min_keep:
                break
            if papers[i]["paper_id"] not in kept_ids:
                kept.append(papers[i])
                kept_ids.add(papers[i]["paper_id"])
        dropped = [p for p in papers if p["paper_id"] not in kept_ids]
        logger.warning("相关性闸门触发自适应保底：保留至少 %d 篇（阈值 %.2f）", min_keep, threshold)

    return kept, [p.get("title", "") for p in dropped]


def rank_papers(
    papers: List[Paper],
    topic: str,
    queries: Sequence[str],
    settings: Optional[Settings] = None,
    relevance: Optional[Sequence[float]] = None,
) -> List[Paper]:
    """综合相关性 + 引用数 + 新颖度 + 检索式覆盖度排序，结果写回 `paper["score"]`。

    相关性权重占主导（默认 0.55），引用数退居次要（默认 0.20），
    不再让高被引但跑题的论文绑架排序（P0-1）。
    `relevance` 可传入已算好的原始分数（与 papers 对齐），避免重复计算。
    """
    settings = settings or get_settings()
    if not papers:
        return []

    if settings.min_year:
        papers = [p for p in papers if (p.get("year") or 0) >= settings.min_year]
        if not papers:
            return []

    if relevance is None:
        relevance = compute_relevance(papers, topic, queries)
    # 排序阶段对「幸存」论文做组内归一，保证 top 幸存者 rel 分量为 1.0
    rel_max = max(relevance) if relevance else 1.0

    cites = [p.get("citation_count") or 0 for p in papers]
    max_log_cite = math.log1p(max(cites)) if max(cites) > 0 else 1.0

    years = [p.get("year") or 0 for p in papers if p.get("year")]
    y_min, y_max = (min(years), max(years)) if years else (0, 0)
    y_span = max(1, y_max - y_min)

    n_queries = max(1, len(set(queries)))

    w = (
        settings.relevance_weight,
        settings.citation_weight,
        settings.recency_weight,
        settings.coverage_weight,
    )

    for i, paper in enumerate(papers):
        cite_score = math.log1p(paper.get("citation_count") or 0) / max_log_cite
        year = paper.get("year") or y_min
        recency = (year - y_min) / y_span if years else 0.5
        coverage = min(1.0, len(set(paper.get("matched_queries") or [])) / n_queries)
        rel_norm = (relevance[i] / rel_max) if rel_max > 0 else 0.0

        paper["score"] = round(
            w[0] * rel_norm + w[1] * cite_score + w[2] * recency + w[3] * coverage, 4
        )

    papers.sort(key=lambda p: p.get("score", 0.0), reverse=True)
    logger.info("排序完成，Top3: %s",
                [(p["title"][:40], p["score"]) for p in papers[:3]])
    return papers


def year_histogram(papers: Sequence[Paper]) -> Dict[int, int]:
    hist: Dict[int, int] = defaultdict(int)
    for p in papers:
        if p.get("year"):
            hist[int(p["year"])] += 1
    return dict(sorted(hist.items()))


def year_range(papers: Sequence[Paper]) -> str:
    years = [p["year"] for p in papers if p.get("year")]
    return f"{min(years)}–{max(years)}" if years else "未知"


# ==========================================================================
# 全文
# ==========================================================================
def enrich_topn_fulltext(papers: List[Paper], settings: Optional[Settings] = None) -> int:
    """仅对 Top-N 文献拉 OA 全文，其余保持摘要模式（对应设计方案 6.1）。

    优先选取「有可下载 OA 全文」的论文（pdf_url 直链或可用 DOI 经 Unpaywall 解析），
    否则旧逻辑按总分取 Top-N 时，高分文献常是付费期刊论文（无 OA 副本），
    导致「全文解析 0 篇」——这正是 P0-2 要修掉的问题。
    """
    settings = settings or get_settings()
    if not settings.enable_fulltext or settings.top_n_fulltext <= 0:
        return 0

    from src.ingest.downloader import oa_available

    oa = [p for p in papers if oa_available(p)]
    rest = [p for p in papers if p not in oa]
    ordered = oa + rest
    targets = [p for p in ordered[: settings.top_n_fulltext] if not p.get("fulltext")]
    return fetch_fulltexts(targets)


# ==========================================================================
# LangChain Tool 封装（供自由工具调用场景使用）
# ==========================================================================
def _build_langchain_tools():
    try:
        from langchain_core.tools import tool
    except ImportError:  # pragma: no cover
        return []

    @tool
    def search_academic_papers(query: str, max_results: int = 20) -> str:
        """检索学术文献。输入英文关键词短语，返回标题/年份/引用数/摘要片段列表。"""
        settings = get_settings()
        papers = dedup_papers(multi_source_search([query], settings))
        papers = rank_papers(papers, query, [query], settings)[:max_results]
        lines = [
            f"[{p['paper_id']}] {p['title']} ({p.get('year')}, 被引 {p.get('citation_count', 0)})\n"
            f"  {(p.get('abstract') or '')[:300]}"
            for p in papers
        ]
        return "\n".join(lines) or "未检索到结果。"

    @tool
    def fetch_paper_fulltext(paper_id: str, pdf_url: str) -> str:
        """下载并解析指定论文的 OA 全文 PDF，返回文本节选。无 OA 副本时返回提示。"""
        paper: Paper = {"paper_id": paper_id, "pdf_url": pdf_url, "title": paper_id}  # type: ignore
        ok = fetch_fulltexts([paper])
        return paper.get("fulltext") or "" if ok else "该文献无可用的开放获取全文，请使用摘要。"

    return [search_academic_papers, fetch_paper_fulltext]


AGENT_TOOLS = _build_langchain_tools()
