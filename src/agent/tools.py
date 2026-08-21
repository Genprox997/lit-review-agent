"""Agent 工具层：多源检索、去重、排序、全文获取。

这些函数既被 graph 节点直接调用（确定性流水线，避免 LLM 乱调工具），
也通过 `AGENT_TOOLS` 暴露为 LangChain Tool，供需要自由工具调用的场景使用。
"""

from __future__ import annotations

import logging
import math
import re
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from src.config import Settings, get_settings
from src.ingest.arxiv_client import search_arxiv
from src.ingest.base import Paper, normalize_title_key
from src.ingest.crossref import search_crossref
from src.ingest.downloader import fetch_fulltexts
from src.ingest.openalex import enrich_citations, search_openalex
from src.ingest.pubmed import search_pubmed
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
        if source == "pubmed":
            return search_pubmed(query, max_results=limit, min_year=settings.min_year)
        if source == "crossref":
            return search_crossref(query, max_results=limit, min_year=settings.min_year)
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
        if not isinstance(paper, dict):
            # 防御性过滤：检索/合并偶发混入非 dict 元素时跳过而非崩掉整条流水线（方向 G'）
            logger.warning("dedup_papers 跳过非 dict 元素：%r", paper)
            continue
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


def high_hub_dropped(
    papers: Sequence[Paper],
    kept_ids: set,
    hub_threshold: float = 0.6,
    top_k: int = 10,
) -> List[Dict[str, Any]]:
    """从原始文献池反查被相关性闸门剔除、但枢纽度仍高的论文，用于运行告警。

    `apply_relevance_gate` 返回的 ``dropped`` 仅是标题字符串列表（不含 ``hub_score``），
    因此这里用 ``kept_ids`` 反查 ``papers`` 中未保留的论文对象再过滤，避免对字符串调 ``.get``
    （真实数据下 ``dropped`` 含字符串标题，直接 ``p.get`` 会抛 ``AttributeError``）。
    """
    out: List[Dict[str, Any]] = []
    for p in papers:
        if p.get("paper_id") in kept_ids:
            continue
        if (p.get("hub_score") or 0.0) >= hub_threshold:
            out.append({"title": p.get("title", ""), "hub": round(p.get("hub_score", 0.0), 4)})
    return out[:top_k]


def rank_papers(
    papers: List[Paper],
    topic: str,
    queries: Sequence[str],
    settings: Optional[Settings] = None,
    relevance: Optional[Sequence[float]] = None,
) -> List[Paper]:
    """综合相关性 + 引用数 + 新颖度 + 检索式覆盖度 + 引用枢纽度排序，写回 `paper["score"]`。

    相关性权重占主导（默认 0.55），引用数退居次要（默认 0.20），
    不再让高被引但跑题的论文绑架排序（P0-1）。方向 D' 新增的 `hub_score`
    （引用网络 PageRank 枢纽度，默认 0.10）让「被大量论文引用的必引文献」权重提升，
    缓解关键流派漏检；未提供引用关系数据时 hub_score 恒为 0，等价于旧行为。
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

    hubs = [float(p.get("hub_score") or 0.0) for p in papers]
    max_hub = max(hubs) if hubs else 1.0

    w = (
        settings.relevance_weight,
        settings.citation_weight,
        settings.recency_weight,
        settings.coverage_weight,
        settings.citation_hub_weight,
    )

    for i, paper in enumerate(papers):
        cite_score = math.log1p(paper.get("citation_count") or 0) / max_log_cite
        year = paper.get("year") or y_min
        recency = (year - y_min) / y_span if years else 0.5
        coverage = min(1.0, len(set(paper.get("matched_queries") or [])) / n_queries)
        rel_norm = (relevance[i] / rel_max) if rel_max > 0 else 0.0
        hub_norm = (hubs[i] / max_hub) if max_hub > 0 else 0.0

        paper["score"] = round(
            w[0] * rel_norm + w[1] * cite_score + w[2] * recency
            + w[3] * coverage + w[4] * hub_norm, 4
        )

    papers.sort(key=lambda p: p.get("score", 0.0), reverse=True)
    logger.info("排序完成，Top3: %s",
                [(p["title"][:40], p["score"]) for p in papers[:3]])
    return papers


# ==========================================================================
# 检索式自动扩词（方向 H'）：伪相关反馈（PRF）
# ==========================================================================
# 学术英文停用词（高频低区分度），扩词时丢弃
_PRF_STOP = set(
    """
a an the of for and or to in on at by with from as is are be been being was were
this that these those we our their his her its their they them it he she you your
i me my us using used use uses based via can may also more most less such which who
whom whose than then thus there here between among within without over under above
below into onto upon during before after while if else when where why how what all
any each both either neither not no nor so too very just only own same other another
paper papers study studies method methods methodology approach approaches result
results proposed propose present presents presentation show shows shown demonstrate
demonstrates analysis analyses model models modeling data dataset datasets set sets
experiment experiments experimental system systems algorithm algorithms framework
novel new recent research work works using use via using
""".split()
)

_PRF_TOKEN_RE = re.compile(r"[a-z][a-z0-9\-]{2,}")


def _prf_doc_text(paper: Paper) -> str:
    """拼装一篇文献的可挖掘文本（标题权重高于摘要）。"""
    title = (paper.get("title") or "").lower()
    abstract = (paper.get("abstract") or "").lower()
    return f"{title} {title} {abstract}"


def _prf_tokens(text: str) -> List[str]:
    return [
        t for t in _PRF_TOKEN_RE.findall(text)
        if t not in _PRF_STOP and len(t) >= 3 and not t.isdigit()
    ]


def _prf_bigrams(tokens: List[str]) -> List[str]:
    return [f"{tokens[i]} {tokens[i + 1]}" for i in range(len(tokens) - 1)]


def auto_expand_queries(
    papers: Sequence[Paper],
    topic: str = "",
    existing_queries: Optional[Sequence[str]] = None,
    settings: Optional[Settings] = None,
    top_k: Optional[int] = None,
    max_queries: Optional[int] = None,
) -> List[str]:
    """伪相关反馈（PRF）自动扩词：从 Top-K 篇相关文献的标题+摘要中挖掘
    高区分度检索词与词组，生成补充检索式以扩充召回。

    - 纯函数，无 LLM、无 sklearn 依赖，离线可确定性单测；
    - 打分 = 词在 Top 集中的频度 × IDF（全池文档频次越低越有区分度）；
    - 优先抽取词组（bigram，如 "deformable mirror"），其次单字检索词；
    - 与已有检索式 / 主题做去重，避免重复检索。

    Returns:
        新生成的检索式列表（已去重、按区分度降序截断到 max_queries）。
    """
    settings = settings or get_settings()
    if not settings.enable_auto_expand or not papers:
        return []

    top_k = top_k or settings.auto_expand_top_k
    max_queries = max_queries or settings.max_auto_queries
    if max_queries <= 0 or top_k <= 0:
        return []

    existing = {str(q).lower().strip() for q in (existing_queries or [])}
    existing.add((topic or "").lower().strip())

    # 选 Top-K：优先相关性，其次 score，再次被引
    ranked = sorted(
        papers,
        key=lambda p: (
            p.get("relevance") or 0,
            p.get("score") or 0,
            p.get("citation_count") or 0,
        ),
        reverse=True,
    )
    top = ranked[:top_k]

    # 全池用于 IDF
    all_docs = [_prf_doc_text(p) for p in papers]
    top_docs = [_prf_doc_text(p) for p in top]
    n_all = max(1, len(all_docs))
    n_top = max(1, len(top_docs))

    # 词频（全池 / Top 集）
    tf_all: Dict[str, int] = defaultdict(int)
    tf_top: Dict[str, int] = defaultdict(int)
    df_all: Dict[str, int] = defaultdict(int)
    df_top: Dict[str, int] = defaultdict(int)
    for d in all_docs:
        toks = set(_prf_tokens(d))
        for t in toks:
            tf_all[t] += 1
            df_all[t] += 1
    for d in top_docs:
        toks = set(_prf_tokens(d))
        for t in toks:
            tf_top[t] += 1
            df_top[t] += 1

    # 词组频（Top 集内）
    bg_top: Dict[str, int] = defaultdict(int)
    bg_df_top: Dict[str, int] = defaultdict(int)
    for d in top_docs:
        toks = _prf_tokens(d)
        for bg in _prf_bigrams(toks):
            bg_top[bg] += 1
        for bg in set(_prf_bigrams(toks)):
            bg_df_top[bg] += 1

    def _idf(term: str) -> float:
        df = df_all.get(term, 0) + 1
        return math.log((n_all + 1) / df) + 1.0

    # 候选词组：在至少 2 篇 Top 文献出现，按 频度×IDF 排序
    scored_bg: List[tuple] = []
    for bg, c in bg_top.items():
        if bg_df_top.get(bg, 0) < 2:
            continue
        if any(w in _PRF_STOP for w in bg.split()):
            continue
        scored_bg.append((bg, c * _idf(bg.split()[0]) * _idf(bg.split()[1])))
    scored_bg.sort(key=lambda x: x[1], reverse=True)

    # 候选单词：在 Top 集出现，按 频度×IDF 排序
    scored_uni: List[tuple] = []
    for t, c in tf_top.items():
        if df_top.get(t, 0) < 1:
            continue
        scored_uni.append((t, c * _idf(t)))
    scored_uni.sort(key=lambda x: x[1], reverse=True)

    # 组装检索式：优先词组，再补未覆盖的高区分度单词
    chosen: List[str] = []
    covered_stems: set = set()

    def _redundant(q: str) -> bool:
        ql = q.strip().lower()
        if not ql or ql in existing:
            return True
        for ex in existing:
            if ql in ex or ex in ql:
                return True
        for ch in chosen:
            if ql in ch or ch in ql:
                return True
        return False

    for bg, _ in scored_bg:
        if len(chosen) >= max_queries:
            break
        if _redundant(bg):
            continue
        chosen.append(bg)
        for w in bg.split():
            covered_stems.add(w)

    for t, _ in scored_uni:
        if len(chosen) >= max_queries:
            break
        if t in covered_stems:
            continue
        if _redundant(t):
            continue
        chosen.append(t)

    return chosen


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
# ==========================================================================
# 持久化文献池（跨主题复用）
# ==========================================================================
def recall_from_store(
    queries: Sequence[str], settings: Optional[Settings] = None, top_k: int = 15
) -> List[Paper]:
    """从本地文献池召回与检索式语义相近的历史文献，跨主题复用、省配额。"""
    from src.ingest import store as store_mod

    store = store_mod.get_paper_store()
    if not store:
        return []
    out: List[Paper] = []
    for q in queries:
        out.extend(store.recall(q, top_k=top_k))
    # 按规范化主键去重；防御性跳过缺 paper_id 的残次条目
    seen: set = set()
    uniq: List[Paper] = []
    for p in out:
        if not p.get("paper_id"):
            continue
        k = store_mod._paper_key(p)
        if k and k not in seen:
            seen.add(k)
            uniq.append(p)
    return uniq


def hydrate_from_store(papers: Sequence[Paper]) -> List[Paper]:
    """用本地缓存回填缺失字段（引用数 / 摘要 / DOI 等）。"""
    from src.ingest import store as store_mod

    store = store_mod.get_paper_store()
    return store.hydrate(papers) if store else list(papers)


def save_to_store(papers: Sequence[Paper]) -> None:
    """把本轮文献池写回本地文献池。"""
    from src.ingest import store as store_mod

    store = store_mod.get_paper_store()
    if store:
        store.upsert(papers)


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
