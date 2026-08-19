"""LangGraph 状态定义。

设计原则：
- 所有节点间传递的中间结果都进 State，配合 checkpointer 实现断点续跑；
- `evidence` 用累加 reducer，Extractor 分批产出时自动合并；
- `messages` 保留 LangGraph 消息通道，方便接 Human-in-the-loop 与 LangSmith 观测。
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, Dict, List, Optional, TypedDict

from langgraph.graph.message import add_messages


class Paper(TypedDict, total=False):
    """文献元数据（与 src.ingest.base.Paper 同构）。"""

    paper_id: str
    title: str
    authors: List[str]
    year: Optional[int]
    venue: str
    citation_count: int
    abstract: str
    url: str
    pdf_url: Optional[str]
    doi: Optional[str]
    source: str
    fulltext: Optional[str]
    score: float
    matched_queries: List[str]
    relevance: float                  # 与主题的相关性（P0-1 闸门使用，0-1）
    has_fulltext: bool                # 是否成功获取 OA 全文（P0-2）
    fulltext_chars: int               # 全文字节数（持久化用）
    openalex_id: str                  # OpenAlex 作品 ID（方向 D' 引用网络分析）
    referenced_works: List[str]       # 引用作品 OpenAlex ID 列表（方向 D'）


class Evidence(TypedDict, total=False):
    """从单篇文献抽取出的可追溯证据。"""

    paper_id: str
    claim: str        # 核心结论 / 贡献
    method: str       # 方法或技术路线
    dataset: str      # 使用的数据集
    metric: str       # 报告的指标与数值
    section: str      # 归属主题（聚类后回填）


class CriticReport(TypedDict, total=False):
    verdict: str                  # pass | need_more
    coverage_score: int           # 0-10
    missing_topics: List[str]
    contradictions: List[str]
    extra_queries: List[str]
    comments: str


class AgentState(TypedDict, total=False):
    """综述 Agent 的全局状态。"""

    # ---- 输入 ----
    topic: str                                  # 用户给定的研究主题
    constraints: str                            # 额外约束（年份范围、必须覆盖的方法等）

    # ---- 过程产物 ----
    messages: Annotated[list, add_messages]
    queries: List[str]                          # 累计的全部检索式
    pending_queries: List[str]                  # 本轮待执行的检索式（Retriever 消费）
    papers: List[Paper]                         # 去重排序后的文献池
    evidence: Annotated[List[Evidence], operator.add]
    clusters: List[Dict[str, Any]]              # [{cluster_id,label,keywords,paper_ids,size}]
    sections: Dict[str, str]                    # 主题标题 -> 带引用的段落（有序）
    citation_map: Dict[str, int]                # paper_id -> 引用编号 [n]
    relevance_gate: Dict[str, Any]              # P0-1：闸门统计 {total,kept,dropped,threshold,dropped_titles}
    critic: CriticReport
    gaps: List[str]
    trends: List[str]
    grounded_claims: Annotated[List[Dict[str, Any]], operator.add]  # Claim 级证据锚定（P3-2）
    faithfulness: Dict[str, Any]                # 引用-论断一致性校验（faithfulness，方向 B）
    citation_analysis: Dict[str, Any]           # 引用网络分析（方向 D'）：枢纽/桥接论文、告警
    citation_graph: Dict[str, Any]              # 引用网络可视化数据（方向 E'）：nodes/edges/gaps/stats
    quality_report: Dict[str, Any]              # 质量评估仪表盘（方向 F'）：各维度得分、总分、薄弱项

    # ---- 增量更新（方向 B'）----
    incremental: bool                         # 是否增量更新模式
    since_date: Optional[str]                 # 仅拉取该日期之后的新文献（YYYY-MM-DD）
    base_path: Optional[str]                  # 上一版成稿路径（沿用编号 + 保留旧小节）
    previous_loaded: bool                     # retriever 是否已加载上一版（防重复加载）
    previous_pids: List[str]                  # 上一版文献 paper_id 集合
    previous_sections: Dict[str, str]         # 上一版各小节正文
    previous_clusters: List[Dict[str, Any]]   # 上一版主题簇（按论文重叠匹配用）
    incremental_keep: List[str]               # 本版需保留旧正文的小节标题
    incremental_note: Dict[str, Any]          # 增量更新说明（成稿头/附录渲染）

    # ---- 输出 ----
    report: str                                 # 最终 Markdown 成稿
    bibtex: str                                 # BibTeX 内容
    artifacts: Dict[str, str]                   # 产物文件路径

    # ---- Human-in-the-loop 闭环改写（方向 A）----
    human_feedback: str                         # 人工审核意见（非空表示需要 targeted rewrite）
    rewrite_targets: List[Dict[str, Any]]       # 解析后的改写动作列表
    human_round: int                            # 已完成的 targeted 改写轮次

    # ---- 控制 ----
    retrieval_round: int                        # 内环：检索轮次
    critic_round: int                           # 外环：评审打回轮次
    logs: Annotated[List[str], operator.add]    # 人类可读的执行轨迹


def initial_state(
    topic: str,
    constraints: str = "",
    incremental: bool = False,
    since_date: Optional[str] = None,
    base_path: Optional[str] = None,
) -> AgentState:
    return {
        "topic": topic,
        "constraints": constraints,
        "messages": [],
        "queries": [],
        "pending_queries": [],
        "papers": [],
        "evidence": [],
        "clusters": [],
        "sections": {},
        "citation_map": {},
        "relevance_gate": {},
        "critic": {},
        "gaps": [],
        "trends": [],
        "grounded_claims": [],
        "faithfulness": {},
        "citation_analysis": {},
        "citation_graph": {},
        "quality_report": {},
        "report": "",
        "bibtex": "",
        "artifacts": {},
        "human_feedback": "",
        "rewrite_targets": [],
        "human_round": 0,
        "retrieval_round": 0,
        "critic_round": 0,
        "logs": [],
        "incremental": incremental,
        "since_date": since_date,
        "base_path": base_path,
        "previous_loaded": False,
        "previous_pids": [],
        "previous_sections": {},
        "previous_clusters": [],
        "incremental_keep": [],
        "incremental_note": {},
    }
