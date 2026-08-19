"""质量评估仪表盘（方向 F'）。

把综述已有的质量信号聚合成一份**结构化、可序列化**的质量报告，供成稿侧车
（``*_quality_report.json``）与 Web UI 仪表盘渲染：

- 方向 B 的 faithfulness（引用-论断一致性，LLM-as-Judge）
- 方向 D'/E' 的引用网络指标（枢纽度、高枢纽论文被闸门剔除、引用边、研究空白）
- P3-2 的 claim 级证据锚定（置信度分布）
- 主题簇覆盖均衡度、时效（近 5 年占比）、小节引用标注完整度

纯函数、零外部依赖；任一信号缺失即安全降级（``available=False``，不计入总分）。
"""

from __future__ import annotations

import datetime
import re
import statistics
from typing import Any, Dict, List, Optional

_CITATION_RE = re.compile(r"\[\d{1,3}\]")


def _this_year() -> int:
    return datetime.datetime.now().year


def _clamp(x: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return max(lo, min(hi, x))


def _mean(xs: List[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def compute_quality_report(
    papers: Optional[List[Dict[str, Any]]] = None,
    clusters: Optional[List[Dict[str, Any]]] = None,
    sections: Optional[Dict[str, str]] = None,
    faithfulness: Optional[Dict[str, Any]] = None,
    citation_analysis: Optional[Dict[str, Any]] = None,
    citation_graph: Optional[Dict[str, Any]] = None,
    grounded_claims: Optional[List[Dict[str, Any]]] = None,
    gaps: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """计算综述质量报告。

    Returns:
        .. code-block:: text

            {
              "overall": int,            # 0-100 加权总分
              "grade": str,              # A / B / C / D
              "dimensions": [            # 有序维度列表
                {"key","name","score"(0-100|None),"weight","available","note"}
              ],
              "weaknesses": [str...],    # 低于阈值的改进建议
              "highlights": [str...],    # 正向亮点
              "stats": {paper_count, section_count, cluster_count,
                        gap_count, edge_count, claim_count},
            }
    """
    papers = papers or []
    clusters = clusters or []
    sections = sections or {}
    faithfulness = faithfulness or {}
    citation_analysis = citation_analysis or {}
    citation_graph = citation_graph or {}
    grounded_claims = grounded_claims or []
    gaps = gaps or []

    dims: List[Dict[str, Any]] = []
    weaknesses: List[str] = []
    highlights: List[str] = []

    # ---- 1. 引用-论断一致性（faithfulness，方向 B）----
    f_score: Optional[float] = None
    if faithfulness.get("skipped"):
        f_available = False
        f_note = "faithfulness 校验已跳过（关闭或无 claim 可校验）"
    elif "score" in faithfulness:
        f_available = True
        f_score = _clamp(round((faithfulness.get("score") or 0.0) * 100))
        checked = faithfulness.get("checked", 0)
        flagged = faithfulness.get("flagged") or []
        f_note = f"校验 {checked} 条 claim，{len(flagged)} 条疑似无充分支撑"
        if f_score < 90:
            weaknesses.append(
                f"引用-论断一致性 {f_score}%：{len(flagged)} 条论断疑似无充分支撑，"
                f"建议补充引用或修订表述"
            )
        elif f_score >= 95:
            highlights.append("引用-论断一致性高，自检闭环可信")
    else:
        f_available = False
        f_note = "faithfulness 未运行"
    dims.append({"key": "faithfulness", "name": "引用-论断一致性", "score": f_score,
                 "weight": 0.25, "available": f_available, "note": f_note})

    # ---- 2. 引用覆盖（小节引用标注完整度）----
    total_sec = len(sections)
    sec_cited = sum(1 for body in sections.values() if _CITATION_RE.search(body or ""))
    cov = (sec_cited / total_sec) if total_sec else 0.0
    cov_score = _clamp(round(cov * 100))
    cov_note = f"{sec_cited}/{total_sec} 个小节含引用标注"
    dims.append({"key": "citation_coverage", "name": "引用覆盖", "score": cov_score,
                 "weight": 0.20, "available": total_sec > 0, "note": cov_note})
    if total_sec > 0 and cov_score < 80:
        weaknesses.append(
            f"引用覆盖 {cov_score}%：{total_sec - sec_cited} 个小节缺少引用标注，建议补充支撑文献"
        )

    # ---- 3. 引用网络枢纽度（方向 D'/E'）----
    dropped = citation_analysis.get("dropped_high_hub") or []
    gstats = citation_graph.get("stats") or {}
    edge_count = gstats.get("edge_count", 0)
    if edge_count <= 0:
        net_score = 60
        net_note = "文献池无引用边，无法评估枢纽度"
    else:
        net_score = 100
        net_note = f"引用网络含 {edge_count} 条边，枢纽论文均保留"
        if dropped:
            pen = min(60, 30 * len(dropped))
            net_score = _clamp(net_score - pen)
            titles = "、".join(
                (d.get("title", d) if isinstance(d, dict) else str(d)) for d in dropped[:3]
            )
            net_note = f"{len(dropped)} 篇高枢纽论文被相关性闸门剔除（如 {titles}）"
            weaknesses.append(
                f"高枢纽论文被剔除：{len(dropped)} 篇必引级工作（{titles} 等）可能被遗漏，"
                f"建议放宽闸门或手动纳入"
            )
    dims.append({"key": "network_hub", "name": "引用网络枢纽度", "score": net_score,
                 "weight": 0.15, "available": True, "note": net_note})

    # ---- 4. 主题覆盖均衡（簇规模均衡 + 研究空白）----
    sizes = [int(c.get("size") or len(c.get("paper_ids") or [])) for c in clusters]
    if sizes:
        mean_sz = _mean(sizes)
        if mean_sz > 0 and len(sizes) > 1:
            cv = (statistics.pstdev(sizes) / mean_sz) if mean_sz else 0.0
            balance = _clamp(1 - cv, 0, 1)
        else:
            balance = 1.0
        topic_score = _clamp(round(balance * 100))
        topic_note = f"{len(clusters)} 个主题簇，规模均衡度 {round(balance * 100)}%"
        if gaps:
            topic_score = _clamp(round(topic_score * 0.85))
            topic_note += f"；识别出 {len(gaps)} 个研究空白"
            weaknesses.append(
                f"存在 {len(gaps)} 个研究空白，建议针对性补检相关方向文献"
            )
        if balance < 0.6 and len(clusters) > 1:
            weaknesses.append("主题簇规模失衡，建议补充小规模方向文献或合并过细簇")
    else:
        topic_score = 0
        topic_note = "无主题簇"
    dims.append({"key": "topic_balance", "name": "主题覆盖均衡", "score": topic_score,
                 "weight": 0.20, "available": bool(sizes), "note": topic_note})

    # ---- 5. 时效（近 5 年占比）----
    years = [int(p.get("year")) for p in papers if p.get("year")]
    if years:
        yr = _this_year()
        recent = sum(1 for y in years if y >= yr - 5)
        recency = recent / len(years)
        rec_score = _clamp(round(recency * 100))
        rec_note = f"近 5 年文献 {recent}/{len(years)} 篇"
        if rec_score < 50:
            weaknesses.append(f"时效偏弱：近 5 年文献仅占 {rec_score}%，建议补充最新进展")
    else:
        rec_score = 0
        rec_note = "无年份信息"
    dims.append({"key": "recency", "name": "时效（近 5 年占比）", "score": rec_score,
                 "weight": 0.10, "available": bool(years), "note": rec_note})

    # ---- 6. 证据锚定强度（P3-2 claim 置信度分布）----
    total_claims = sum(len(g.get("claims") or []) for g in grounded_claims)
    if total_claims:
        strength_sum = 0.0
        for g in grounded_claims:
            for c in (g.get("claims") or []):
                conf = str(c.get("confidence", "medium")).lower()
                strength_sum += {"high": 1.0, "medium": 0.6, "low": 0.2}.get(conf, 0.6)
        strength = strength_sum / total_claims
        gr_score = _clamp(round(strength * 100))
        gr_note = f"{total_claims} 条 claim 级证据锚定，平均强度 {round(strength * 100)}%"
        if gr_score < 70:
            weaknesses.append("部分论断证据强度弱（low 置信占比高），建议增强抽取或补充支撑证据")
    else:
        gr_score = 0
        gr_note = "无 claim 级证据锚定"
    dims.append({"key": "grounding", "name": "证据锚定强度", "score": gr_score,
                 "weight": 0.10, "available": total_claims > 0, "note": gr_note})

    # ---- 总分（按可用维度加权）----
    avail = [d for d in dims if d["available"] and d["score"] is not None]
    if avail:
        wsum = sum(d["weight"] for d in avail)
        overall = round(sum(d["score"] * d["weight"] for d in avail) / wsum)
    else:
        overall = 0
    grade = "A" if overall >= 85 else "B" if overall >= 70 else "C" if overall >= 55 else "D"

    return {
        "overall": overall,
        "grade": grade,
        "dimensions": dims,
        "weaknesses": weaknesses,
        "highlights": highlights,
        "stats": {
            "paper_count": len(papers),
            "section_count": total_sec,
            "cluster_count": len(clusters),
            "gap_count": len(gaps),
            "edge_count": edge_count,
            "claim_count": total_claims,
        },
    }
