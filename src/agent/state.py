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
    critic: CriticReport
    gaps: List[str]
    trends: List[str]

    # ---- 输出 ----
    report: str                                 # 最终 Markdown 成稿
    bibtex: str                                 # BibTeX 内容
    artifacts: Dict[str, str]                   # 产物文件路径

    # ---- 控制 ----
    retrieval_round: int                        # 内环：检索轮次
    critic_round: int                           # 外环：评审打回轮次
    logs: Annotated[List[str], operator.add]    # 人类可读的执行轨迹


def initial_state(topic: str, constraints: str = "") -> AgentState:
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
        "critic": {},
        "gaps": [],
        "trends": [],
        "report": "",
        "bibtex": "",
        "artifacts": {},
        "retrieval_round": 0,
        "critic_round": 0,
        "logs": [],
    }
