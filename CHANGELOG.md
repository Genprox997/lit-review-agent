# 改动日志 / Changelog

> 按改进方向逐条记录功能变更、涉及文件与验证方式。每条改动均在离线测试（stub LLM）下通过后再合并。

---

## 方向 A：Human-in-the-loop 闭环重生成 + 引用编号跨轮次稳定（2026-08-18）

**目标**：把 LangGraph 的 `interrupt` / `Command(resume)` 能力用透——人工审核意见不再只留痕，而是触发**针对性重写**；同时让引用编号在 Critic 打回重聚类后保持稳定，避免用户看到的 `[n]` 指向完全不同的论文。

**变更内容**
- **HITL 闭环重生成**：新增 `parse_human_feedback` 与 `rewrite_sections` 两个节点，并在 `human_review` 之后增加条件路由 `route_after_human`。
  - 续跑时若收到具体修改意见（非 approve），先由 LLM 把意见解析成针对现有小节的 `rewrite` / `add` 动作列表；
  - `rewrite_sections` **只重跑受影响的小节**（复用 `section_writer` 提示词并附加人工意见），而非整篇重生成，随后经 `ground_claims → critic → gap_analyzer → synthesizer` 重新定稿并再次挂起，形成可迭代的审核闭环。
  - 受 `max_human_rounds`（默认 2）上限约束，杜绝无限回环；通过时清除 `human_feedback` 防止误判。
- **引用编号稳定**：`clusterer` 现在保留上一轮已分配的 `citation_map`，仅对新增论文追加编号。Critic 外环打回重聚类后，旧论文的 `[n]` 编号保持不变。

**涉及文件**
- `src/agent/state.py`：新增 `human_feedback` / `rewrite_targets` / `human_round` 状态字段与初值。
- `src/agent/config.py`：新增 `max_human_rounds`（默认 2）。
- `src/agent/prompts.py`：新增 `PARSE_HUMAN_FEEDBACK_SYSTEM/USER`。
- `src/agent/llm.py`：stub 新增 `_parse_human_feedback` 桩。
- `src/agent/nodes.py`：`clusterer` 编号保留逻辑；新增 `parse_human_feedback` / `rewrite_sections` / `_best_cluster_match` / `_generate_section`；`human_review` 通过时清除 `human_feedback`。
- `src/agent/graph.py`：接入新节点与 `route_after_human` 条件边。
- `tests/test_graph.py`：新增稳定编号、改写路由、改写回环等 7 个测试。

**验证**：`pytest tests/test_graph.py` 全部通过（含 `test_human_targeted_rewrite_loop` 端到端改写回环、稳定编号单测）。完整离线套件 96 passed。
