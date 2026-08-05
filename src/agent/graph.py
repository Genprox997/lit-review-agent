"""组装 LangGraph 状态机。

拓扑（对应设计方案 §2）：

    START → QueryExpander → Retriever ⟲(数量不足) → Ranker → Extractor
          → Clusterer → SectionWriter → Critic ⟲(覆盖不足→QueryExpander)
          → GapAnalyzer → Synthesizer → Human(可选) → END

双环路：
- 内环 `Retriever ⟲`：文献池未达目标规模时放大检索量重试；
- 外环 `Critic → QueryExpander`：评审判定覆盖不足时，带着缺口生成新检索式补文献。
两个环都有硬性轮次上限，杜绝死循环。
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from langgraph.graph import END, START, StateGraph

from src.agent.nodes import (
    clusterer,
    critic,
    extractor,
    gap_analyzer,
    human_review,
    query_expander,
    ranker,
    retriever,
    section_writer,
    synthesizer,
)
from src.agent.state import AgentState, initial_state
from src.config import get_settings

logger = logging.getLogger(__name__)


# ==========================================================================
# 条件路由
# ==========================================================================
def route_after_retrieval(state: AgentState) -> str:
    """内环：文献池不足且未超轮次上限 → 回到 Retriever 放大检索。"""
    settings = get_settings()
    n = len(state.get("papers") or [])
    round_no = state.get("retrieval_round", 0)

    if n < settings.target_paper_count and round_no < settings.max_retrieval_rounds:
        logger.info("内环触发：文献 %d < 目标 %d，第 %d 轮重检索",
                    n, settings.target_paper_count, round_no + 1)
        return "retriever"
    if n == 0:
        logger.warning("检索不到任何文献，直接结束")
        return "__end__"
    return "ranker"


def route_after_critic(state: AgentState) -> str:
    """外环：评审判定覆盖不足且未超打回次数上限 → 回到 QueryExpander 补文献。

    `critic_round` 记录已完成的评审次数；`max_critic_rounds` 是允许的**打回次数**。
    因此 max=2 时，第 1、2 次评审可以打回，第 3 次评审必须放行。
    """
    settings = get_settings()
    report = state.get("critic") or {}
    round_no = state.get("critic_round", 0)

    if report.get("verdict") == "need_more" and round_no <= settings.max_critic_rounds:
        logger.info("外环触发：覆盖度 %s，第 %d 次打回补文献",
                    report.get("coverage_score"), round_no)
        return "query_expander"
    return "gap_analyzer"


# ==========================================================================
# 构图
# ==========================================================================
def build_graph(
    with_human: bool = False,
    checkpointer: Optional[Any] = None,
):
    """构建并编译综述 Agent。

    Args:
        with_human: 是否在 Synthesizer 之后挂起等待人工审核
        checkpointer: 自定义检查点后端；None 时按配置自动选择
    """
    builder = StateGraph(AgentState)

    builder.add_node("query_expander", query_expander)
    builder.add_node("retriever", retriever)
    builder.add_node("ranker", ranker)
    builder.add_node("extractor", extractor)
    builder.add_node("clusterer", clusterer)
    builder.add_node("section_writer", section_writer)
    builder.add_node("critic", critic)
    builder.add_node("gap_analyzer", gap_analyzer)
    builder.add_node("synthesizer", synthesizer)
    builder.add_node("human_review", human_review)

    builder.add_edge(START, "query_expander")
    builder.add_edge("query_expander", "retriever")

    # 内环
    builder.add_conditional_edges(
        "retriever",
        route_after_retrieval,
        {"retriever": "retriever", "ranker": "ranker", "__end__": END},
    )

    builder.add_edge("ranker", "extractor")
    builder.add_edge("extractor", "clusterer")
    builder.add_edge("clusterer", "section_writer")
    builder.add_edge("section_writer", "critic")

    # 外环
    builder.add_conditional_edges(
        "critic",
        route_after_critic,
        {"query_expander": "query_expander", "gap_analyzer": "gap_analyzer"},
    )

    builder.add_edge("gap_analyzer", "synthesizer")
    builder.add_edge("synthesizer", "human_review")
    builder.add_edge("human_review", END)

    if checkpointer is None:
        checkpointer = _default_checkpointer()

    compile_kwargs: Dict[str, Any] = {"checkpointer": checkpointer}
    if with_human:
        compile_kwargs["interrupt_before"] = ["human_review"]

    graph = builder.compile(**compile_kwargs)
    graph.name = "lit-review-agent"
    return graph


def _default_checkpointer():
    settings = get_settings()
    if settings.checkpoint_backend == "sqlite":
        try:
            import sqlite3

            from langgraph.checkpoint.sqlite import SqliteSaver

            settings.ensure_dirs()
            conn = sqlite3.connect(settings.checkpoint_path, check_same_thread=False)
            logger.info("使用 SQLite 检查点：%s", settings.checkpoint_path)
            return SqliteSaver(conn)
        except ImportError:
            logger.warning("未安装 langgraph-checkpoint-sqlite，回退到内存检查点")

    from langgraph.checkpoint.memory import InMemorySaver

    return InMemorySaver()


# ==========================================================================
# 便捷入口
# ==========================================================================
def run_review(
    topic: str,
    constraints: str = "",
    thread_id: str = "default",
    with_human: bool = False,
    stream: bool = True,
) -> AgentState:
    """跑一次完整综述流程，返回终态。"""
    graph = build_graph(with_human=with_human)
    config = {"configurable": {"thread_id": thread_id}, "recursion_limit": 80}
    state = initial_state(topic, constraints)

    if not stream:
        return graph.invoke(state, config)  # type: ignore[return-value]

    final: AgentState = state
    for chunk in graph.stream(state, config, stream_mode="values"):
        final = chunk  # type: ignore[assignment]
        logs = chunk.get("logs") or []
        if logs:
            print(logs[-1], flush=True)
    return final
