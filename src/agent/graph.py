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
import sys
from typing import Any, Callable, Dict, Optional

from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from src.agent.nodes import (
    clusterer,
    critic,
    extractor,
    gap_analyzer,
    ground_claims,
    faithfulness,
    human_review,
    incremental_plan,
    parse_human_feedback,
    query_expander,
    ranker,
    retriever,
    rewrite_sections,
    section_writer,
    skip_human_review,
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


def route_after_human(state: AgentState) -> str:
    """人工审核后的路由（方向 A）。

    - 收到非空 `human_feedback`（且非 approve 类词）且未超 `max_human_rounds`
      → 进入 targeted 改写回环；
    - 否则（通过 / 已达改写上限）→ 结束。
    """
    settings = get_settings()
    feedback = str(state.get("human_feedback") or "").strip().lower()
    approve_words = {"", "approve", "ok", "yes", "通过", "y", "true"}
    if feedback not in approve_words and state.get("human_round", 0) < settings.max_human_rounds:
        logger.info("HITL 回环：收到修改意见，第 %d 轮 targeted 改写", state.get("human_round", 0) + 1)
        return "parse_human_feedback"
    return END


# ==========================================================================
# 构图
# ==========================================================================
def build_graph(
    with_human: bool = False,
    checkpointer: Optional[Any] = None,
):
    """构建并编译综述 Agent。

    Args:
        with_human: 是否在 Synthesizer 之后挂起等待人工审核（需持久化检查点以支持跨进程续跑）
        checkpointer: 自定义检查点后端；None 时按配置自动选择
    """
    builder = StateGraph(AgentState)

    builder.add_node("query_expander", query_expander)
    builder.add_node("retriever", retriever)
    builder.add_node("ranker", ranker)
    builder.add_node("extractor", extractor)
    builder.add_node("clusterer", clusterer)
    builder.add_node("incremental_plan", incremental_plan)  # 方向 B'：增量更新规划
    builder.add_node("section_writer", section_writer)
    builder.add_node("ground_claims", ground_claims)
    builder.add_node("critic", critic)
    builder.add_node("gap_analyzer", gap_analyzer)
    builder.add_node("faithfulness", faithfulness)
    builder.add_node("synthesizer", synthesizer)
    # 仅在启用 --human 时挂起等待审核；否则用占位节点直接放行（避免无谓中断）
    builder.add_node("human_review", human_review if with_human else skip_human_review)
    # 方向 A：人工意见解析 + 针对性重写（仅在收到意见时由 route_after_human 触发）
    builder.add_node("parse_human_feedback", parse_human_feedback)
    builder.add_node("rewrite_sections", rewrite_sections)

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
    builder.add_edge("clusterer", "incremental_plan")
    builder.add_edge("incremental_plan", "section_writer")
    builder.add_edge("section_writer", "ground_claims")
    builder.add_edge("ground_claims", "faithfulness")  # 方向 B：引用-论断一致性校验
    builder.add_edge("faithfulness", "critic")

    # 外环
    builder.add_conditional_edges(
        "critic",
        route_after_critic,
        {"query_expander": "query_expander", "gap_analyzer": "gap_analyzer"},
    )

    builder.add_edge("gap_analyzer", "synthesizer")
    builder.add_edge("synthesizer", "human_review")
    builder.add_conditional_edges(
        "human_review",
        route_after_human,
        {"parse_human_feedback": "parse_human_feedback", END: END},
    )
    builder.add_edge("parse_human_feedback", "rewrite_sections")
    builder.add_edge("rewrite_sections", "ground_claims")

    if checkpointer is None:
        checkpointer = _default_checkpointer(with_human=with_human)

    # 人工审核的挂起由 human_review 节点内的 interrupt() 实现（可携带审核意见），
    # 因此这里不再使用 interrupt_before。
    graph = builder.compile(checkpointer=checkpointer)
    graph.name = "lit-review-agent"
    return graph


def _default_checkpointer(with_human: bool = False):
    """选择检查点后端。

    - with_human=True 或 CHECKPOINT_BACKEND=sqlite → SQLite（持久化，支持跨进程续跑）；
    - 否则 → 内存（零配置、一次性运行）。
    缺少 langgraph-checkpoint-sqlite 且要求 human 时直接报错，给出明确安装指引。
    """
    settings = get_settings()
    if with_human or settings.checkpoint_backend == "sqlite":
        try:
            import sqlite3

            from langgraph.checkpoint.sqlite import SqliteSaver

            settings.ensure_dirs()
            conn = sqlite3.connect(settings.checkpoint_path, check_same_thread=False)
            logger.info("使用 SQLite 检查点：%s", settings.checkpoint_path)
            return SqliteSaver(conn)
        except ImportError:
            if with_human:
                raise RuntimeError(
                    "Human-in-the-loop 需要持久化检查点，但未安装 langgraph-checkpoint-sqlite。"
                    "请运行 `pip install -e \".[persist]\"` 后重试。"
                )
            logger.warning(
                "CHECKPOINT_BACKEND=sqlite 但未安装 langgraph-checkpoint-sqlite，回退到内存检查点"
            )

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
    feedback: Optional[str] = None,
    stream: bool = True,
    on_progress: Optional[Callable[[str, Any], None]] = None,
    incremental: bool = False,
    since_date: Optional[str] = None,
    base_path: Optional[str] = None,
) -> AgentState:
    """跑一次完整综述流程，返回终态。

    Args:
        feedback: 非 None 时视为「续跑」——用 ``Command(resume=feedback)`` 从被
            ``human_review`` 挂起的检查点继续，而不是从头开始。配合 SQLite 检查点
            即可实现跨进程断点续跑。
        with_human: 是否在 Synthesizer 之后挂起等待人工审核（强制 SQLite 检查点）。
    """
    graph = build_graph(with_human=with_human)
    config = {"configurable": {"thread_id": thread_id}, "recursion_limit": 80}

    if feedback is not None:
        # 续跑：先确认该 thread 确实处于 human_review 挂起态，避免误操作全新线程
        snap = graph.get_state(config)
        if snap.next and "human_review" in snap.next:
            initial_input: Any = Command(resume=feedback)
        else:
            print(f"[提示] thread_id={thread_id} 未处于挂起态，作为全新运行处理。",
                  file=sys.stderr)
            initial_input = initial_state(
                topic, constraints, incremental=incremental,
                since_date=since_date, base_path=base_path,
            )
    else:
        initial_input = initial_state(
            topic, constraints, incremental=incremental,
            since_date=since_date, base_path=base_path,
        )

    if not stream:
        result = graph.invoke(initial_input, config)  # type: ignore[arg-type]
        if isinstance(result, dict) and "__interrupt__" in result:
            result = dict(result)
            result["interrupted"] = True
        return result  # type: ignore[return-value]

    interrupted = False
    final: AgentState = {}
    for chunk in graph.stream(initial_input, config, stream_mode="values"):  # type: ignore[arg-type]
        if "__interrupt__" in chunk:
            interrupted = True
            interrupt_tuple = chunk.get("__interrupt__") or ()
            payload = interrupt_tuple[0].value if interrupt_tuple else {}
            report_path = (payload or {}).get("report_path")
            print("\n" + "=" * 72)
            print("[人工审核] 综述草稿已生成并挂起，等待人工审核。")
            if report_path:
                print(f"  草稿文件 : {report_path}")
            print("  审核后定稿：")
            print(f"    python -m src.main --resume --thread-id {thread_id}"
                  + (" --feedback \"你的修改意见（可选）\"" if False else ""))
            print("=" * 72)
            if on_progress:
                on_progress("human_review", {"report_path": report_path})
            final = chunk  # type: ignore[assignment]
            break
        final = chunk  # type: ignore[assignment]
        logs = chunk.get("logs") or []
        if logs:
            print(logs[-1], flush=True)
            if on_progress:
                on_progress("progress", logs[-1])

    final = dict(final)  # type: ignore[arg-type]
    final["interrupted"] = interrupted
    if on_progress:
        if interrupted:
            on_progress("interrupted", {"report_path": (final.get("artifacts") or {}).get("report")})
        else:
            on_progress(
                "done",
                {
                    "paper_count": len(final.get("papers") or []),
                    "citation_count": len(final.get("citation_map") or {}),
                    "section_count": len(final.get("sections") or {}),
                    "artifacts": final.get("artifacts") or {},
                },
            )
    return final
