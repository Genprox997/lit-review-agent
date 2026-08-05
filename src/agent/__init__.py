"""综述 Agent：状态、节点、工具与状态机组装。"""

from src.agent.state import AgentState, Evidence, Paper, initial_state

__all__ = ["AgentState", "Evidence", "Paper", "initial_state", "build_graph", "run_review"]


def __getattr__(name):
    # 惰性导入，避免 `import src.agent` 时就拉起 langgraph 依赖链
    if name in {"build_graph", "run_review"}:
        from src.agent import graph as _graph

        return getattr(_graph, name)
    raise AttributeError(name)
