"""LangGraph 节点实现。

每个节点接收完整 AgentState，返回**局部更新字典**（LangGraph 会按 reducer 合并）。
节点内部只做一件事，路由判断交给 graph.py 的条件边。
"""

from __future__ import annotations

import json
import logging
import re
import sys
from datetime import datetime
from typing import Dict, List, Sequence

from langgraph.types import interrupt

from src.agent import prompts as P
from src.agent.llm import chat, chat_json, chat_json_many, chat_many
from src.agent.state import AgentState, Evidence
from src.agent.tools import (
    apply_relevance_gate,
    compute_relevance,
    dedup_papers,
    enrich_topn_fulltext,
    hydrate_from_store,
    multi_source_search,
    rank_papers,
    recall_from_store,
    save_to_store,
    year_histogram,
    year_range,
)
from src.cluster.theme_cluster import cluster_papers
from src.config import get_settings
from src.ingest.openalex import enrich_citations
from src.report.bibtex import build_bibtex, build_reference_list

logger = logging.getLogger(__name__)

EXTRACT_BATCH_SIZE = 5          # 每次 LLM 调用抽取的论文数
MAX_EXTRACT_PAPERS = 60         # 抽取上限，控制 token 成本
MAX_EVIDENCE_PER_SECTION = 15   # 单小节最多喂给 writer 的证据条数
POOL_HARD_CAP = 120             # 文献池硬上限


def _log(msg: str) -> str:
    stamp = datetime.now().strftime("%H:%M:%S")
    line = f"[{stamp}] {msg}"
    logger.info(msg)
    return line


def _lang_hint() -> str:
    return P.LANG_HINT.get(get_settings().report_language, P.LANG_HINT["zh"])


# ==========================================================================
# 1. QueryExpander
# ==========================================================================
def query_expander(state: AgentState) -> dict:
    """把研究主题扩展成多组检索式；Critic 打回时只生成补缺口的新检索式。"""
    topic = state["topic"]
    old_queries = state.get("queries") or []
    critic = state.get("critic") or {}
    is_refine = bool(old_queries) and critic.get("verdict") == "need_more"

    if is_refine:
        missing = critic.get("missing_topics") or []
        hinted = critic.get("extra_queries") or []
        user = P.QUERY_EXPANDER_REFINE_USER.format(
            topic=topic,
            old_queries="\n".join(f"- {q}" for q in old_queries),
            missing="\n".join(f"- {m}" for m in missing) or "- （未指明，请自行判断薄弱面）",
        )
        data = chat_json("query_expander", P.QUERY_EXPANDER_SYSTEM, user, default={})
        new_queries = list(data.get("queries") or []) + hinted
    else:
        constraints = (state.get("constraints") or "").strip()
        block = f"额外约束：{constraints}\n\n" if constraints else ""
        user = P.QUERY_EXPANDER_USER.format(topic=topic, constraints_block=block)
        data = chat_json("query_expander", P.QUERY_EXPANDER_SYSTEM, user, default={})
        new_queries = list(data.get("queries") or [])

    # 清洗 + 去重（大小写不敏感）
    seen = {q.lower().strip() for q in old_queries}
    pending: List[str] = []
    for q in new_queries:
        q = re.sub(r"\s+", " ", str(q)).strip().strip('"')
        if q and q.lower() not in seen and 2 <= len(q) <= 120:
            seen.add(q.lower())
            pending.append(q)

    if not pending and not old_queries:  # LLM 失败兜底
        pending = [topic, f"{topic} survey", f"{topic} review"]

    return {
        "queries": old_queries + pending,
        "pending_queries": pending,
        "logs": [_log(f"QueryExpander: {'补充' if is_refine else '生成'} {len(pending)} 条检索式 → {pending}")],
    }


# ==========================================================================
# 2. Retriever
# ==========================================================================
def retriever(state: AgentState) -> dict:
    """并发打多源学术 API 并与已有文献池合并去重。"""
    settings = get_settings()
    round_no = state.get("retrieval_round", 0)
    pending = state.get("pending_queries") or state.get("queries") or []

    # 数量不足触发的重检索：换用全量检索式并放大每条的返回上限
    if not pending:
        pending = state.get("queries") or [state["topic"]]

    # 跨主题复用：先从本地文献池召回与检索式相近的历史文献，再打网络
    seed = recall_from_store(pending, settings)
    if seed:
        logger.info("Retriever: 从本地文献池召回 %d 条历史文献", len(seed))

    original_limit = settings.max_results_per_query
    if round_no > 0:
        settings.max_results_per_query = min(100, original_limit * (round_no + 1))

    try:
        fresh = multi_source_search(pending, settings)
    finally:
        settings.max_results_per_query = original_limit

    merged = dedup_papers(list(state.get("papers") or []) + seed + fresh)

    # 用本地缓存回填引用数/摘要/DOI，并写回本轮文献池（跨运行复用）
    merged = hydrate_from_store(merged)
    save_to_store(merged)

    return {
        "papers": merged,
        "pending_queries": [],
        "retrieval_round": round_no + 1,
        "logs": [
            _log(
                f"Retriever(第 {round_no + 1} 轮): 新检索 {len(fresh)} 条"
                + (f"（本地召回 {len(seed)} 条）" if seed else "")
                + f"，合并去重后文献池 {len(merged)} 篇"
            )
        ],
    }


# ==========================================================================
# 3. Ranker
# ==========================================================================
def ranker(state: AgentState) -> dict:
    """去重后排序 + 相关性闸门 + 补引用数 + 对 Top-N 拉全文。"""
    settings = get_settings()
    papers = list(state.get("papers") or [])
    queries = state.get("queries") or []

    # arXiv 源没有引用数，用 OpenAlex 补齐，让排序信号完整
    if "openalex" in settings.enabled_sources:
        enrich_citations(papers, limit=30)

    # --- 相关性闸门（P0-1）---
    relevance = compute_relevance(papers, state["topic"], queries)
    kept, dropped = apply_relevance_gate(
        papers, relevance, settings.relevance_gate, settings.min_pool_after_gate
    )
    gate_stats = {
        "total": len(papers),
        "kept": len(kept),
        "dropped": len(dropped),
        "threshold": settings.relevance_gate,
        "dropped_titles": dropped[:25],
    }

    # 排序用闸门幸存者；relevance 已对齐 kept 顺序，避免重复计算
    score_map = {p["paper_id"]: relevance[i] for i, p in enumerate(papers)}
    kept_relevance = [score_map[p["paper_id"]] for p in kept]
    ranked = rank_papers(kept, state["topic"], queries, settings, relevance=kept_relevance)[
        :POOL_HARD_CAP
    ]
    n_full = enrich_topn_fulltext(ranked, settings)

    top_preview = " | ".join(f"{p['title'][:38]}({p.get('year')})" for p in ranked[:3])
    return {
        "papers": ranked,
        "relevance_gate": gate_stats,
        "logs": [
            _log(
                f"Ranker: 相关性闸门移除 {gate_stats['dropped']} 篇偏离主题"
                f"（阈值 {gate_stats['threshold']}），保留 {gate_stats['kept']} 篇入池；"
                f"年份 {year_range(ranked)}，全文解析 {n_full} 篇；Top3: {top_preview}"
            )
        ],
    }


# ==========================================================================
# 4. Extractor
# ==========================================================================
def _paper_block(paper: dict, idx: int) -> str:
    body = paper.get("fulltext") or paper.get("abstract") or ""
    body = body[:5000] if paper.get("fulltext") else body[:1800]
    mode = "全文节选" if paper.get("fulltext") else "摘要"
    return (
        f"### 论文 {idx}\n"
        f"paper_id: {paper['paper_id']}\n"
        f"标题: {paper.get('title')}\n"
        f"年份: {paper.get('year')} | 被引: {paper.get('citation_count', 0)}\n"
        f"{mode}: {body}\n"
    )


def extractor(state: AgentState) -> dict:
    """分批从摘要/全文抽取「方法、结论、数据集、指标」，绑定 paper_id。"""
    papers = state.get("papers") or []
    done_ids = {e["paper_id"] for e in (state.get("evidence") or [])}
    targets = [p for p in papers[:MAX_EXTRACT_PAPERS] if p["paper_id"] not in done_ids]

    if not targets:
        return {"logs": [_log("Extractor: 无新增文献，跳过")]}

    # 把每个 batch 打包成一次独立的 LLM 调用，整体并发执行（P3-4）
    batches: List[List[dict]] = []
    items: List[Dict[str, Any]] = []
    for start in range(0, len(targets), EXTRACT_BATCH_SIZE):
        batch = targets[start : start + EXTRACT_BATCH_SIZE]
        block = "\n".join(_paper_block(p, i + 1) for i, p in enumerate(batch))
        user = P.EXTRACTOR_USER.format(count=len(batch), papers_block=block)
        items.append({"node": "extractor", "system": P.EXTRACTOR_SYSTEM, "user": user})
        batches.append(batch)

    results = chat_json_many(items, default={})

    new_evidence: List[Evidence] = []
    for batch, data in zip(batches, results):
        got = {}
        for item in (data.get("evidence") or []):
            pid = str(item.get("paper_id", "")).strip()
            if pid:
                got[pid] = item

        for paper in batch:
            item = got.get(paper["paper_id"], {})
            claim = (item.get("claim") or "").strip()
            if not claim:  # 抽取失败时用摘要首句兜底，保证每篇都有可用证据
                claim = (paper.get("abstract") or paper.get("title") or "")[:200]
            new_evidence.append(
                {
                    "paper_id": paper["paper_id"],
                    "claim": claim,
                    "method": (item.get("method") or "").strip(),
                    "dataset": (item.get("dataset") or "").strip(),
                    "metric": (item.get("metric") or "").strip(),
                    "section": "",
                }
            )

    return {
        "evidence": new_evidence,
        "logs": [_log(f"Extractor: 抽取 {len(new_evidence)} 篇文献的结构化证据")],
    }


# ==========================================================================
# 5. Clusterer
# ==========================================================================
def clusterer(state: AgentState) -> dict:
    """embedding + KMeans 分主题簇，再由 LLM 给每簇起小节标题，并分配引用编号。"""
    settings = get_settings()
    papers = state.get("papers") or []
    covered = {e["paper_id"] for e in (state.get("evidence") or [])}
    pool = [p for p in papers if p["paper_id"] in covered] or papers

    clusters = cluster_papers(pool, n_clusters=settings.n_clusters)
    if not clusters:
        return {"clusters": [], "citation_map": {}, "logs": [_log("Clusterer: 文献池为空")]}

    # --- LLM 命名 ---
    by_id = {p["paper_id"]: p for p in pool}
    blocks = []
    for c in clusters:
        titles = [by_id[pid]["title"] for pid in c.paper_ids[:8] if pid in by_id]
        blocks.append(
            f"簇 {c.cluster_id}（{c.size} 篇）\n"
            f"关键词: {', '.join(c.keywords[:8]) or '（无）'}\n"
            + "\n".join(f"  - {t[:110]}" for t in titles)
        )
    user = P.CLUSTER_NAMER_USER.format(topic=state["topic"], clusters_block="\n\n".join(blocks))
    data = chat_json(
        "cluster_namer",
        P.CLUSTER_NAMER_SYSTEM.format(lang_hint=_lang_hint()),
        user,
        default={},
    )
    labels = {str(k): str(v) for k, v in (data.get("labels") or {}).items()}

    used_labels = set()
    for c in clusters:
        raw = labels.get(str(c.cluster_id), "").strip()
        if not raw or raw in used_labels:
            raw = f"主题 {c.cluster_id + 1}：{', '.join(c.keywords[:3]) or '其他相关研究'}"
        used_labels.add(raw)
        c.label = raw

    # --- 引用编号：按簇顺序、簇内按得分降序 ---
    citation_map: Dict[str, int] = {}
    counter = 1
    for c in clusters:
        ordered = sorted(
            (pid for pid in c.paper_ids if pid in by_id),
            key=lambda pid: by_id[pid].get("score", 0.0),
            reverse=True,
        )
        c.paper_ids = ordered
        for pid in ordered:
            if pid not in citation_map:
                citation_map[pid] = counter
                counter += 1

    return {
        "clusters": [c.to_dict() for c in clusters],
        "citation_map": citation_map,
        "logs": [
            _log(
                "Clusterer: "
                + " / ".join(f"{c.label}({c.size})" for c in clusters)
            )
        ],
    }


# ==========================================================================
# 6. SectionWriter
# ==========================================================================
def _evidence_block(
    paper_ids: Sequence[str],
    by_id: Dict[str, dict],
    ev_by_id: Dict[str, Evidence],
    citation_map: Dict[str, int],
) -> str:
    lines = []
    for pid in paper_ids[:MAX_EVIDENCE_PER_SECTION]:
        paper = by_id.get(pid)
        if not paper:
            continue
        num = citation_map.get(pid)
        ev = ev_by_id.get(pid, {})
        parts = [
            f"[{num}] {paper.get('title')} ({paper.get('year')}, 被引 {paper.get('citation_count', 0)})",
            f"    结论: {ev.get('claim') or (paper.get('abstract') or '')[:180]}",
        ]
        if ev.get("method"):
            parts.append(f"    方法: {ev['method']}")
        if ev.get("dataset"):
            parts.append(f"    数据: {ev['dataset']}")
        if ev.get("metric"):
            parts.append(f"    指标: {ev['metric']}")
        lines.append("\n".join(parts))
    return "\n\n".join(lines)


def section_writer(state: AgentState) -> dict:
    """为每个主题簇生成带内联引用的综述段落。"""
    clusters = state.get("clusters") or []
    papers = state.get("papers") or []
    by_id = {p["paper_id"]: p for p in papers}
    ev_by_id = {e["paper_id"]: e for e in (state.get("evidence") or [])}
    citation_map = state.get("citation_map") or {}

    sections: Dict[str, str] = {}
    specs: List[tuple] = []  # (cluster_label, allowed_set, llm_item)
    for cluster in clusters:
        pids = cluster["paper_ids"]
        allowed = [citation_map[pid] for pid in pids[:MAX_EVIDENCE_PER_SECTION] if pid in citation_map]
        if not allowed:
            continue
        user = P.SECTION_WRITER_USER.format(
            topic=state["topic"],
            section_title=cluster["label"],
            keywords=", ".join(cluster.get("keywords") or []) or "（无）",
            allowed_citations=", ".join(f"[{n}]" for n in allowed),
            evidence_block=_evidence_block(pids, by_id, ev_by_id, citation_map),
        )
        specs.append((
            cluster["label"],
            set(allowed),
            {
                "node": "section_writer",
                "system": P.SECTION_WRITER_SYSTEM.format(lang_hint=_lang_hint()),
                "user": user,
            },
        ))

    # 各小节互相独立，整体并发生成（P3-4）
    texts = chat_many([s[2] for s in specs]) if specs else []
    for (label, allowed_set, _), text in zip(specs, texts):
        text = (text or "").strip()
        if text:
            sections[label] = _strip_invalid_citations(text, allowed_set)

    return {
        "sections": sections,
        "logs": [_log(f"SectionWriter: 生成 {len(sections)} 个小节，共 {sum(len(v) for v in sections.values())} 字")],
    }


# ==========================================================================
# 6.5 GroundClaims —— Claim 级证据锚定（P3-2）
# ==========================================================================
def ground_claims(state: AgentState) -> dict:
    """把每个小节中的核心论断拆成 (text, paper_ids, confidence) 三元组。

    让综述从「段落 + [n] 引用」升级到「逐条论断可审计」：每条 claim 都绑定支撑它的
    具体论文与证据强度，便于读者判断可信度，也方便下游做引用防幻觉二次校验。
    """
    sections = state.get("sections") or {}
    citation_map = state.get("citation_map") or {}
    ev_by_id = {e["paper_id"]: e for e in (state.get("evidence") or [])}
    by_id = {p["paper_id"]: p for p in (state.get("papers") or [])}
    num_to_pid = {n: pid for pid, n in citation_map.items()}

    all_grounded: List[Dict[str, Any]] = []
    specs: List[tuple] = []  # (title, llm_item)
    for title, body in sections.items():
        used_nums = sorted({int(m) for m in re.findall(r"\[(\d{1,3})\]", body)})
        allowed_pids = [num_to_pid[n] for n in used_nums if n in num_to_pid]
        if not allowed_pids:
            continue
        ev_lines = []
        for pid in allowed_pids:
            ev = ev_by_id.get(pid, {})
            num = citation_map.get(pid)
            head = f"[{num}] {by_id.get(pid, {}).get('title', '')}"
            claim = (ev.get("claim") or "").strip()
            ev_lines.append(f"{head}: {claim[:200]}" if claim else head)
        user = P.GROUND_CLAIMS_USER.format(
            section_title=title,
            evidence_block="\n".join(ev_lines) or "（无）",
        )
        specs.append((
            title,
            {
                "node": "ground_claims",
                "system": P.GROUND_CLAIMS_SYSTEM.format(lang_hint=_lang_hint()),
                "user": user,
            },
        ))

    # 各小节互相独立，整体并发执行（P3-4）
    results = chat_json_many([s[1] for s in specs], default={}) if specs else []
    for (title, _), data in zip(specs, results):
        data = data or {}
        claims: List[Dict[str, Any]] = []
        for c in (data.get("claims") or []):
            text = str(c.get("text", "")).strip()
            conf = str(c.get("confidence", "medium")).lower()
            if conf not in ("high", "medium", "low"):
                conf = "medium"
            valid: List[str] = []
            for raw in (c.get("paper_ids") or []):
                pid = str(raw).strip().lstrip("#")
                if pid in citation_map:
                    valid.append(pid)
                elif pid.isdigit() and int(pid) in num_to_pid:
                    valid.append(num_to_pid[int(pid)])
            valid = list(dict.fromkeys(valid))
            if text and valid:
                claims.append({"text": text, "paper_ids": valid, "confidence": conf})
        if claims:
            all_grounded.append({"section": title, "claims": claims})

    total = sum(len(g["claims"]) for g in all_grounded)
    return {
        "grounded_claims": all_grounded,
        "logs": [
            _log(
                f"GroundClaims: 为 {len(all_grounded)} 个小节标注 {total} 条 claim 级证据锚定"
            )
        ],
    }


def _strip_invalid_citations(text: str, allowed: set) -> str:
    """删除模型编造的、不在候选列表中的引用编号，保证引用可追溯。"""

    def repl(m: re.Match) -> str:
        num = int(m.group(1))
        return m.group(0) if num in allowed else ""

    return re.sub(r"\[(\d{1,3})\]", repl, text)


# ==========================================================================
# 7. Critic
# ==========================================================================
def critic(state: AgentState) -> dict:
    """评审覆盖度与矛盾处理，决定补文献还是放行。"""
    settings = get_settings()
    sections = state.get("sections") or {}
    papers = state.get("papers") or []
    round_no = state.get("critic_round", 0)

    if not sections:
        return {
            "critic": {"verdict": "pass", "coverage_score": 0, "comments": "无小节内容，直接放行"},
            "critic_round": round_no + 1,
            "logs": [_log("Critic: 无小节内容，跳过评审")],
        }

    block = "\n\n".join(f"### {title}\n{body}" for title, body in sections.items())
    user = P.CRITIC_USER.format(
        topic=state["topic"],
        paper_count=len(papers),
        year_range=year_range(papers),
        critic_round=round_no + 1,
        max_rounds=settings.max_critic_rounds,
        sections_block=block[:16000],
    )
    report = chat_json("critic", P.CRITIC_SYSTEM, user, default={}) or {}

    verdict = str(report.get("verdict", "pass")).lower()
    report["verdict"] = "need_more" if verdict.startswith("need") else "pass"
    try:
        report["coverage_score"] = int(report.get("coverage_score", 7))
    except (TypeError, ValueError):
        report["coverage_score"] = 7

    return {
        "critic": report,
        "critic_round": round_no + 1,
        "logs": [
            _log(
                f"Critic(第 {round_no + 1} 轮): {report['verdict']}，"
                f"覆盖度 {report['coverage_score']}/10，"
                f"缺口 {report.get('missing_topics') or '无'}"
            )
        ],
    }


# ==========================================================================
# 8. GapAnalyzer
# ==========================================================================
def gap_analyzer(state: AgentState) -> dict:
    """识别研究空白与技术演进趋势。"""
    sections = state.get("sections") or {}
    papers = state.get("papers") or []
    clusters = state.get("clusters") or []

    hist = year_histogram(papers)
    hist_str = ", ".join(f"{y}:{n}" for y, n in list(hist.items())[-12:]) or "未知"
    cluster_summary = "; ".join(f"{c['label']}({c['size']}篇)" for c in clusters) or "无"
    block = "\n\n".join(f"### {t}\n{b}" for t, b in sections.items())

    user = P.GAP_ANALYZER_USER.format(
        topic=state["topic"],
        year_hist=hist_str,
        cluster_summary=cluster_summary,
        critic_comments=(state.get("critic") or {}).get("comments", "（无）"),
        sections_block=block[:16000],
    )
    data = chat_json(
        "gap_analyzer",
        P.GAP_ANALYZER_SYSTEM.format(lang_hint=_lang_hint()),
        user,
        default={},
    ) or {}

    gaps = [str(g).strip() for g in (data.get("gaps") or []) if str(g).strip()]
    trends = [str(t).strip() for t in (data.get("trends") or []) if str(t).strip()]
    return {
        "gaps": gaps,
        "trends": trends,
        "logs": [_log(f"GapAnalyzer: {len(gaps)} 条研究空白，{len(trends)} 条趋势")],
    }


# ==========================================================================
# 9. Synthesizer
# ==========================================================================
def _slugify(text: str) -> str:
    s = re.sub(r"[^\w\u4e00-\u9fff]+", "_", text).strip("_")
    return s[:60] or "review"


def synthesizer(state: AgentState) -> dict:
    """汇总成稿：摘要 + 引言 + 各主题小节 + 空白趋势 + 结论 + 参考文献 + BibTeX。"""
    settings = get_settings()
    settings.ensure_dirs()

    topic = state["topic"]
    sections = state.get("sections") or {}
    papers = state.get("papers") or []
    citation_map = state.get("citation_map") or {}
    gaps = state.get("gaps") or []
    trends = state.get("trends") or []
    lang_hint = _lang_hint()

    cited_papers = [p for p in papers if p["paper_id"] in citation_map]
    allowed = sorted(citation_map.values())
    allowed_str = ", ".join(f"[{n}]" for n in allowed[:80])
    section_titles = "\n".join(f"- {t}" for t in sections) or "- （无）"

    # --- 摘要 + 引言 ---
    digest = "\n\n".join(f"### {t}\n{b[:400]}" for t, b in sections.items())
    head_raw = chat(
        "synthesizer",
        P.SYNTH_ABSTRACT_SYSTEM.format(lang_hint=lang_hint),
        P.SYNTH_ABSTRACT_USER.format(
            topic=topic,
            paper_count=len(cited_papers),
            year_range=year_range(cited_papers),
            section_titles=section_titles,
            allowed_citations=allowed_str,
            sections_digest=digest[:12000],
        ),
    )
    abstract, introduction = _split_abstract_intro(head_raw)

    # --- 结论 ---
    conclusion = chat(
        "synthesizer",
        P.SYNTH_CONCLUSION_SYSTEM.format(lang_hint=lang_hint),
        P.SYNTH_CONCLUSION_USER.format(
            topic=topic,
            section_titles=section_titles,
            gaps_block="\n".join(f"- {g}" for g in gaps) or "- （无）",
            trends_block="\n".join(f"- {t}" for t in trends) or "- （无）",
            allowed_citations=allowed_str,
        ),
    ).strip()

    allowed_set = set(allowed)
    abstract = _strip_invalid_citations(abstract, allowed_set)
    introduction = _strip_invalid_citations(introduction, allowed_set)
    conclusion = _strip_invalid_citations(conclusion, allowed_set)

    # --- 组装 Markdown ---
    bibtex, cite_keys = build_bibtex(cited_papers)
    refs = build_reference_list(cited_papers, citation_map, cite_keys)
    report = _assemble_markdown(
        state, abstract, introduction, sections, gaps, trends, conclusion, refs
    )

    # --- 落盘 ---
    slug = _slugify(topic)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = settings.output_dir
    paths = {
        "report": out / f"{slug}_{stamp}_review.md",
        "bibtex": out / f"{slug}_{stamp}_references.bib",
        "papers": out / f"{slug}_{stamp}_papers.json",
    }
    paths["report"].write_text(report, encoding="utf-8")
    paths["bibtex"].write_text(bibtex, encoding="utf-8")
    paths["papers"].write_text(
        json.dumps(
            [
                {
                    **{k: v for k, v in p.items() if k != "fulltext"},
                    "has_fulltext": bool(p.get("has_fulltext")),
                    "fulltext_chars": int(p.get("fulltext_chars") or 0),
                    "citation_index": citation_map.get(p["paper_id"]),
                }
                for p in papers
            ],
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    return {
        "report": report,
        "bibtex": bibtex,
        "artifacts": {k: str(v) for k, v in paths.items()},
        "logs": [
            _log(
                f"Synthesizer: 成稿 {len(report)} 字，引用 {len(cited_papers)} 篇 → {paths['report'].name}"
            )
        ],
    }


def _split_abstract_intro(raw: str) -> tuple[str, str]:
    """从「## 摘要 / ## 引言」格式的输出中拆分两段。"""
    raw = (raw or "").strip()
    m = re.split(r"\n#+\s*(?:引言|Introduction)\s*\n", raw, maxsplit=1, flags=re.IGNORECASE)
    if len(m) == 2:
        abstract = re.sub(r"^#+\s*(?:摘要|Abstract)\s*\n", "", m[0].strip(), flags=re.IGNORECASE)
        return abstract.strip(), m[1].strip()
    return raw, ""


def _assemble_markdown(
    state: AgentState,
    abstract: str,
    introduction: str,
    sections: Dict[str, str],
    gaps: List[str],
    trends: List[str],
    conclusion: str,
    references: str,
) -> str:
    settings = get_settings()
    papers = state.get("papers") or []
    citation_map = state.get("citation_map") or {}
    critic_report = state.get("critic") or {}

    parts: List[str] = []
    cited_papers_all = [p for p in papers if p["paper_id"] in citation_map]
    n_full = sum(1 for p in cited_papers_all if p.get("has_fulltext"))
    if n_full:
        fulltext_note = f"已获取 {n_full} 篇 OA 全文（其余基于摘要）"
    else:
        fulltext_note = "基于摘要撰写（未获取到可用 OA 全文）"
    parts.append(f"# {state['topic']} —— 文献综述\n")
    parts.append(
        "> 本文由 **lit-review-agent** 自动生成。\n>\n"
        f"> 生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M')} | "
        f"检索源：{', '.join(settings.enabled_sources)} | "
        f"候选文献 {len(papers)} 篇，引用 {len(citation_map)} 篇 | "
        f"年份跨度 {year_range(cited_papers_all)}\n>\n"
        "> 所有论断均带内联引用 `[n]`，编号对应文末参考文献表；"
        f"{fulltext_note}。\n"
    )

    parts.append("## 摘要\n\n" + (abstract or "（生成失败）"))

    # 章节号连续自增，缺失引言时不留空号
    counter = iter(range(1, 100))
    if introduction:
        parts.append(f"## {next(counter)}. 引言\n\n{introduction}")

    for title, body in sections.items():
        parts.append(f"## {next(counter)}. {title}\n\n{body}")

    if trends or gaps:
        parts.append(f"## {next(counter)}. 研究空白与演进趋势")
        if trends:
            parts.append("### 技术演进趋势\n\n" + "\n".join(f"{i}. {t}" for i, t in enumerate(trends, 1)))
        if gaps:
            parts.append("### 研究空白\n\n" + "\n".join(f"{i}. {g}" for i, g in enumerate(gaps, 1)))

    if conclusion:
        parts.append(f"## {next(counter)}. 结论与展望\n\n{conclusion}")

    parts.append(f"## {next(counter)}. 参考文献\n\n{references or '（无）'}")

    # 附录：让整个生成过程可复核
    appendix = [
        f"## 附录 A：生成过程\n",
        "### A.1 检索式\n",
        "\n".join(f"{i}. `{q}`" for i, q in enumerate(state.get("queries") or [], 1)) or "（无）",
        "\n### A.2 主题簇\n",
        "\n".join(
            f"- **{c['label']}**（{c['size']} 篇）关键词：{', '.join(c.get('keywords') or []) or '—'}"
            for c in (state.get("clusters") or [])
        ) or "（无）",
        "\n### A.3 评审结论\n",
        f"- 判定：`{critic_report.get('verdict', 'n/a')}`，覆盖度 {critic_report.get('coverage_score', 'n/a')}/10",
        f"- 评语：{critic_report.get('comments', '（无）')}",
    ]
    if critic_report.get("contradictions"):
        appendix.append("- 已识别的结论冲突：\n" + "\n".join(
            f"  - {c}" for c in critic_report["contradictions"]))

    # A.5 相关性闸门（P0-1）：让检索质量可审计
    gate = state.get("relevance_gate") or {}
    if gate:
        appendix.append("\n### A.5 相关性闸门\n")
        appendix.append(
            f"- 阈值 `{gate.get('threshold')}`：候选 {gate.get('total')} 篇 → "
            f"移除 **{gate.get('dropped')}** 篇偏离主题，保留 {gate.get('kept')} 篇入池"
        )
        dropped_titles = gate.get("dropped_titles") or []
        if dropped_titles:
            appendix.append("移除的论文（主题相关性不足）：\n" + "\n".join(
                f"  - {t}" for t in dropped_titles[:15]) or "（无）")

    appendix.append("\n### A.4 执行轨迹\n")
    appendix.append("```\n" + "\n".join(state.get("logs") or []) + "\n```")

    # A.6 Claim 级证据锚定（P3-2）：让每条核心论断可追溯、可评强度
    grounded = state.get("grounded_claims") or []
    if grounded:
        appendix.append("\n### A.6 Claim 级证据锚定\n")
        badge = {"high": "● 强", "medium": "◐ 中", "low": "○ 弱"}
        lines = []
        for g in grounded:
            lines.append(f"**{g.get('section', '')}**")
            for c in g.get("claims", []):
                nums = " ".join(
                    f"[{state.get('citation_map', {}).get(pid)}]"
                    for pid in c.get("paper_ids", [])
                    if pid in state.get("citation_map", {})
                )
                conf = c.get("confidence", "medium")
                lines.append(f"- {badge.get(conf, '◐ 中')} {c.get('text', '')} {nums}")
        appendix.append("\n".join(lines) or "（无）")

    parts.append("\n".join(appendix))

    return "\n\n".join(parts) + "\n"


# ==========================================================================
# 10. Human（可选人工审核挂起点）
# ==========================================================================
def human_review(state: AgentState) -> dict:
    """人工审核节点（Human-in-the-loop）。

    图运行到此处时调用 `interrupt()` 挂起：把草稿路径交给调用方，等待其用
    `Command(resume=feedback)` 续跑。feedback 为空或近似 'approve' 视为通过；
    否则把人工意见写入日志留痕（当前版本不自动重生成，留待后续回环增强）。
    """
    artifacts = state.get("artifacts") or {}
    report_path = artifacts.get("report")
    payload = {
        "kind": "human_review",
        "message": "综述草稿已生成，请审核后定稿。回复 'approve' 通过，或给出修改意见。",
        "report_path": report_path,
    }
    decision = interrupt(payload)
    feedback = (decision or "").strip()
    if feedback and feedback.lower() not in ("approve", "ok", "yes", "通过", "y", "true"):
        return {
            "human_feedback": feedback,
            "logs": [_log(f"Human: 收到审核意见 → {feedback[:200]}")],
        }
    return {"logs": [_log("Human: 审核通过，定稿")]}


def skip_human_review(state: AgentState) -> dict:
    """未启用 --human 时的占位节点：直接放行，不挂起（避免无谓中断）。"""
    return {"logs": [_log("Human: 未启用人工审核，跳过")]}
