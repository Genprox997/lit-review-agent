"""长任务健壮性（方向 G'）：节点级错误隔离。

把每个节点包进 `node_guard`：节点抛异常时不再中断整条流水线，而是——
1. 记录一条结构化错误到 `state["run_errors"]`（带 reducer，跨节点累积）；
2. 打一条人类可读的降级日志；
3. 返回安全的空更新，让图继续往下走，最终仍产出最佳努力成稿。

唯一例外是 Human-in-the-loop 的 `interrupt()`：它靠抛出 `Interrupt` 把图挂起，
必须原样透传，否则会破坏人工审核的挂起/续跑语义。
"""

from __future__ import annotations

import logging
import time
from typing import Any, Callable, Dict

from langgraph.errors import GraphInterrupt

from src.agent.state import AgentState

logger = logging.getLogger(__name__)


def node_guard(name: str, fn: Callable[[AgentState], Dict[str, Any]]):
    """包裹一个节点函数，使其在异常时降级而非崩溃整条流水线。"""

    def guarded(state: AgentState) -> Dict[str, Any]:
        try:
            return fn(state)
        except GraphInterrupt:
            # HITL 挂起信号必须透传，绝不能吞掉
            raise
        except Exception as exc:  # noqa: BLE001 - 单节点失败不应拖垮整条长任务
            logger.exception("节点 %s 执行失败，降级继续", name)
            err = {
                "node": name,
                "error": str(exc)[:600],
                "kind": type(exc).__name__,
                "time": time.strftime("%H:%M:%S"),
            }
            return {
                "run_errors": [err],
                "logs": [f"[{name}] ⚠ 节点执行失败，已降级继续：{err['error']}"],
            }

    guarded.__name__ = f"guard({name})"
    guarded.__doc__ = fn.__doc__
    return guarded
