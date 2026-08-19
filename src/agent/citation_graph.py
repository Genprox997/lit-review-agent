"""引用网络分析（方向 D'）。

用各论文的引用关系（OpenAlex `referenced_works`）在本地文献池内构建有向引用图，
计算两类结构信号，缓解「关键流派 / 必引文献漏检」：

- ``hub_score``（PageRank）：被大量高质量论文引用的「必引候选」，枢纽度高；
- ``bridge_score``（betweenness）：跨子领域被频繁作为桥接的论文，连接不同研究方向。

同时为 GapAnalyzer 提供共引分析：把共享大量参考文献的论文聚成潜在子领域，
若某共引子群未被现有主题簇覆盖，则标为研究空白候选。

全部为纯 Python 确定性算法（不依赖 networkx），离线可测、可复现。
"""

from __future__ import annotations

import logging
from typing import Dict, List, Sequence

logger = logging.getLogger(__name__)


def _node_id(paper: dict) -> str:
    """引用图的节点键：优先 OpenAlex ID，否则退回 paper_id。"""
    return paper.get("openalex_id") or paper.get("paper_id", "")


def _strip_oa(ref: str) -> str:
    """把 OpenAlex 引用 ID（可能带完整 URL）规范为作品 ID。"""
    return ref.rsplit("/", 1)[-1] if "/" in ref else ref


def build_graph(papers: Sequence[dict]):
    """构建有向引用图。

    返回 ``(adj, pid_to_oa)``：
    - ``adj``：paper_id -> {被其引用的、且也在池内的 paper_id 集合}；
    - ``pid_to_oa``：paper_id -> openalex_id（便于回溯）。
    只保留「池中互相引用」的边，避免把引用图扩散到池外未知论文。
    """
    oa_to_pid: Dict[str, str] = {}
    pid_to_oa: Dict[str, str] = {}
    for p in papers:
        oa = _node_id(p)
        pid = p["paper_id"]
        if oa:
            oa_to_pid[oa] = pid
        pid_to_oa[pid] = oa

    adj: Dict[str, set] = {p["paper_id"]: set() for p in papers}
    for p in papers:
        src = p["paper_id"]
        for ref in (p.get("referenced_works") or []):
            ref_oa = _strip_oa(ref)
            tgt = oa_to_pid.get(ref) or oa_to_pid.get(ref_oa)
            if tgt and tgt != src:
                adj[src].add(tgt)
    return adj, pid_to_oa


def pagerank(adj: Dict[str, set], damping: float = 0.85, iters: int = 60) -> Dict[str, float]:
    """有向图 PageRank（幂迭代），返回 node -> 分数。"""
    nodes = list(adj.keys())
    n = len(nodes)
    if n == 0:
        return {}
    rank = {x: 1.0 / n for x in nodes}
    for _ in range(iters):
        new = {x: (1 - damping) / n for x in nodes}
        for x in nodes:
            links = adj[x]
            if links:
                share = damping * rank[x] / len(links)
                for y in links:
                    new[y] += share
            else:
                # 悬挂节点（无出边）把得分均匀散给所有节点
                share = damping * rank[x] / n
                for y in nodes:
                    new[y] += share
        rank = new
    return rank


def betweenness(adj: Dict[str, set]) -> Dict[str, float]:
    """无向图中介中心性（Brandes 算法），返回 node -> 分数。"""
    nodes = list(adj.keys())
    # 无向邻接
    und: Dict[str, set] = {x: set() for x in nodes}
    for x in nodes:
        for y in adj[x]:
            und[x].add(y)
            und[y].add(x)
    C = {x: 0.0 for x in nodes}
    if not nodes:
        return C

    for s in nodes:
        stack: List[str] = []
        pred = {x: [] for x in nodes}
        sigma = {x: 0.0 for x in nodes}
        dist = {x: -1 for x in nodes}
        sigma[s] = 1.0
        dist[s] = 0
        queue = [s]
        while queue:
            v = queue.pop(0)
            stack.append(v)
            for w in und[v]:
                if dist[w] < 0:
                    dist[w] = dist[v] + 1
                    queue.append(w)
                if dist[w] == dist[v] + 1:
                    sigma[w] += sigma[v]
                    pred[w].append(v)
        delta = {x: 0.0 for x in nodes}
        while stack:
            w = stack.pop()
            for v in pred[w]:
                delta[v] += (sigma[v] / sigma[w]) * (1 + delta[w])
            if w != s:
                C[w] += delta[w]
    # 无向图每条最短路被两个方向各计一次，故除以 2
    for x in C:
        C[x] /= 2.0
    return C


def score_centrality(papers: Sequence[dict]) -> None:
    """就地为每篇论文写入 ``hub_score``（PageRank）与 ``bridge_score``（betweenness）。

    若引用图无有效边（论文未提供 ``referenced_works`` / 不在池内），
    则所有分数置 0，等价于「无引用网络数据」的安全降级。
    """
    adj, _ = build_graph(papers)
    n_edges = sum(len(v) for v in adj.values())
    if n_edges == 0:
        for p in papers:
            p["hub_score"] = 0.0
            p["bridge_score"] = 0.0
        return
    pr = pagerank(adj)
    bt = betweenness(adj)
    max_pr = max(pr.values()) or 1.0
    max_bt = max(bt.values()) or 1.0
    for p in papers:
        pid = p["paper_id"]
        p["hub_score"] = round(pr.get(pid, 0.0) / max_pr, 4)
        p["bridge_score"] = round(bt.get(pid, 0.0) / max_bt, 4)


def cocitation_gaps(
    papers: Sequence[dict],
    clusters: Sequence[dict],
    min_cocite: int = 2,
) -> List[str]:
    """用共引强度识别「未被现有主题簇覆盖」的潜在子领域。

    - 把共享 >= ``min_cocite`` 篇参考文献的论文连成共引边，做并查集分群；
    - 若某共引子群有 >=2 篇论文不在任何现有主题簇中，视为研究空白候选；
    - 返回自然语言形式的 gap 描述（含代表性论文标题）。
    """
    pid_to_title = {p["paper_id"]: p.get("title", "") for p in papers}

    # 共引：统计每对论文共同引用的参考文献数量
    by_ref: Dict[str, List[str]] = {}
    for p in papers:
        for ref in (p.get("referenced_works") or []):
            by_ref.setdefault(_strip_oa(ref), []).append(p["paper_id"])
    cocite: Dict[tuple, int] = {}
    for ref, pids in by_ref.items():
        for i in range(len(pids)):
            for j in range(i + 1, len(pids)):
                a, b = pids[i], pids[j]
                if a == b:
                    continue
                key = (a, b) if a < b else (b, a)
                cocite[key] = cocite.get(key, 0) + 1

    # 并查集：强共引（>= min_cocite）聚成同一子群
    parent = {p["paper_id"]: p["paper_id"] for p in papers}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for (a, b), w in cocite.items():
        if w >= min_cocite:
            union(a, b)

    comps: Dict[str, List[str]] = {}
    for p in papers:
        root = find(p["paper_id"])
        comps.setdefault(root, []).append(p["paper_id"])

    # 现有主题簇已覆盖的论文集合
    covered: set = set()
    for c in clusters:
        covered.update(c.get("paper_ids") or [])

    gaps: List[str] = []
    for members in comps.values():
        if len(members) < 2:
            continue
        if set(members) <= covered:
            continue  # 子群已被现有主题簇完全覆盖
        uncovered = [m for m in members if m not in covered]
        if len(uncovered) >= 2:
            reps = "、".join(pid_to_title.get(m, m) for m in uncovered[:3])
            gaps.append(
                f"共引子群（{len(uncovered)} 篇）未被现有主题覆盖，可能构成研究空白：{reps}"
                + (" 等" if len(uncovered) > 3 else "")
            )
    return gaps


def _cluster_label_map(clusters: Sequence[dict]) -> Dict[str, str]:
    """paper_id -> 所属主题簇 label（用于可视化着色）。"""
    m: Dict[str, str] = {}
    for c in (clusters or []):
        label = c.get("label") or ""
        for pid in (c.get("paper_ids") or []):
            m[pid] = label
    return m


def export_graph(
    papers: Sequence[dict],
    clusters: Sequence[dict] = None,
    gaps: Sequence[str] = None,
) -> Dict[str, Any]:
    """序列化引用网络，供 Web UI 渲染交互式力导向图（方向 E'）。

    返回结构（全部可 JSON 化，确定性）：
    - ``nodes``：[{id, label, year, citations, hub, bridge, cluster}]；
    - ``edges``：[[src_id, dst_id], ...]，仅含池中互相引用的边（与 ``build_graph`` 一致）；
    - ``gaps``：研究空白描述（由 ``gap_analyzer`` 计算后透传）；
    - ``stats``：节点/边数、hub/bridge Top5。

    若论文未提供 ``referenced_works``（无引用边），``edges`` 为空、hub/bridge 均为 0，
    等价于「无引用网络数据」的安全降级。
    """
    # 确保 hub/bridge 已计算（自身完备，不依赖调用方先跑过 score_centrality）
    score_centrality(papers)
    adj, _ = build_graph(papers)
    cmap = _cluster_label_map(clusters)

    nodes = []
    for p in papers:
        pid = p["paper_id"]
        nodes.append(
            {
                "id": pid,
                "label": (p.get("title") or "").strip(),
                "year": int(p.get("year") or 0),
                "citations": int(p.get("citation_count") or 0),
                "hub": round(float(p.get("hub_score") or 0.0), 4),
                "bridge": round(float(p.get("bridge_score") or 0.0), 4),
                "cluster": cmap.get(pid, ""),
            }
        )

    edges: List[List[str]] = []
    for src, dsts in adj.items():
        for d in dsts:
            edges.append([src, d])

    by_hub = sorted(nodes, key=lambda n: n["hub"], reverse=True)[:5]
    by_bridge = sorted(nodes, key=lambda n: n["bridge"], reverse=True)[:5]
    stats = {
        "node_count": len(nodes),
        "edge_count": len(edges),
        "top_hub": [
            {"id": n["id"], "label": n["label"], "hub": n["hub"]}
            for n in by_hub
            if n["hub"] > 0
        ],
        "top_bridge": [
            {"id": n["id"], "label": n["label"], "bridge": n["bridge"]}
            for n in by_bridge
            if n["bridge"] > 0
        ],
    }

    return {
        "nodes": nodes,
        "edges": edges,
        "gaps": list(gaps or []),
        "stats": stats,
    }
