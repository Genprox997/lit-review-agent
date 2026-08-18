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

---

## 方向 B：引用-论断一致性自动评测（faithfulness，LLM-as-Judge）（2026-08-18）

**目标**：把已有的 Claim 级证据锚定（`grounded_claims`）物化为一道**自动质检闸**——逐条校验正文论断是否真被其引用的论文支撑，让综述从「生成」进阶到「生成 + 自检」闭环，也是面试常备的 LLM-as-Judge 能力落地。

**变更内容**
- 新增 `faithfulness` 节点（插入 `ground_claims` 与 `critic` 之间）：把每条 claim 与其支撑论文证据交给 LLM 判定 `supported / partial / unsupported`，汇总一致性得分（1.0 = 全部有充分支撑）与疑似无支撑论断列表。
- `synthesizer` 成稿新增**附录 A.7**渲染校验结果（校验条数、得分、逐条告警），使自检过程可复核。
- 可用 `ENABLE_FAITHFULNESS=false` 关闭（默认开启），避免对成本敏感的批量任务额外开销。

**涉及文件**
- `src/agent/prompts.py`：新增 `FAITHFULNESS_SYSTEM/USER`。
- `src/agent/llm.py`：stub 新增 `_faithfulness` 桩（默认全部 supported）。
- `src/agent/nodes.py`：新增 `faithfulness` 节点；`synthesizer` 附录 A.7 渲染。
- `src/agent/graph.py`：接入 `faithfulness` 节点（`ground_claims → faithfulness → critic`）。
- `src/config.py`：新增 `enable_faithfulness`（默认 True）。
- `tests/test_ground.py`：新增 faithfulness 节点单测（含关闭 / 无 claim 跳过分支）。
- `tests/test_graph.py`：新增 `test_faithfulness_appears_in_report` 端到端校验。

**验证**：`pytest tests/test_ground.py tests/test_graph.py` 全部通过；完整离线套件 100 passed。

---

## 方向 C：多格式成稿输出（LaTeX + Word/.docx）（2026-08-18）

**目标**：在默认 Markdown + BibTeX 之外，支持直接产出**可编译 LaTeX（`.tex`）**与 **Word（`.docx`）**，覆盖学术投稿与日常汇报两类交付场景（你本人常用 .docx 双格式交付）。

**变更内容**
- 新增 `_assemble_latex`：与 `_assemble_markdown` 平行，把结构化成稿字段组装为完整 LaTeX 文档；正文 `[n]` 转为 `\cite{key}`（key 取自已写出的 `.bib`），配合同目录 `.bib` 即可 `pdflatex` 编译。
- 新增 `_write_docx`：用 `python-docx` 写出带标题层级的 `.docx`（惰性导入，未装 `[docx]` extra 时给出清晰安装提示）。
- `synthesizer` 按 `settings.output_format`（`md` / `latex` / `docx`，默认 `md`）决定是否追加产出对应文件，产物路径写入 `artifacts.report_latex` / `artifacts.report_docx`（Markdown 仍始终产出，作为规范底稿）。
- CLI 新增 `--format`、FastAPI `ReviewRequest` 新增 `format` 字段，均接回 `OUTPUT_FORMAT` 配置。
- `pyproject.toml` 新增 `[docx]` extra（含 `python-docx`）。

**涉及文件**
- `src/agent/nodes.py`：新增 `_assemble_latex` / `_write_docx`；`synthesizer` 格式分支。
- `src/agent/config.py`：新增 `output_format`（默认 `md`）。
- `src/main.py`：新增 `--format` 参数与配置接线。
- `src/api.py`：`ReviewRequest` 新增 `format` 字段与接线。
- `src/pyproject.toml`：新增 `docx` extra。
- `tests/test_graph.py`：新增 `test_latex_output_generated` / `test_docx_output_generated`。

**验证**：`pytest tests/test_graph.py` 全部通过（LaTeX 断言 `\begin{document}`/`\cite{`/`\bibliography{`；docx 断言产物存在且可打开）。完整离线套件 102 passed（含 LaTeX + docx 两测试）。
