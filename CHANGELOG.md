# 改动日志 / Changelog

> 按改进方向逐条记录功能变更、涉及文件与验证方式。每条改动均在离线测试（stub LLM）下通过后再合并。

---

## 方向 A'：Web UI 接 HITL 反馈（2026-08-19）

**目标**：把已落地的 HITL 闭环（方向 A 的 CLI `--human` 续跑）搬到浏览器——用户无需命令行，即可在网页里看草稿、提修改意见、多次迭代改稿后再一键定稿。

**变更内容**
- **API 层**
  - `ReviewRequest` 新增 `with_human`（启用人工审核）与 `thread_id`（HITL 续跑标识）字段；`/review/stream` 在 `with_human=True` 时强制 SQLite 检查点并以 SSE 推送 `interrupted` 事件（携带草稿全文 `report`、统计与 `thread_id`）。
  - 新增 `POST /review/resume` SSE 端点：以 `Command(resume=feedback)` 续跑被挂起的 thread；续跑前预校验 thread 确处于 `human_review` 挂起态（否则 400，避免误触发全新运行）。空意见视为通过定稿，非空意见触发方向 A 的针对性改写回环。
  - 抽取 `_sse_endpoint(runner)` 助手脚手，两个流式端点共用，避免重复。
- **Web UI（`GET /`）**
  - 新增「启用人工审核（HITL）」勾选框与草稿审核面板：挂起后展示草稿全文、文献/引用/小节统计与成稿文件路径；意见输入框 + 「提交修改意见并重生成」/「通过并定稿」两个按钮。
  - 提交意见经 `/review/resume` 续跑并复用同一 SSE 消费逻辑；支持多轮「看草稿 → 提意见 → 改稿 → 再审核」闭环，最终 `done` 事件展示产物路径。

**涉及文件**
- `src/api.py`：`ReviewRequest`/`ResumeRequest` 字段；`/review/stream` 支持 `with_human` + 动态 `thread_id`；新增 `/review/resume` 与 `_sse_endpoint`；模块 docstring 更新。
- `src/agent/graph.py`：`run_review` 的 `interrupted` 事件回传 `thread_id` / `report` / 统计。
- `src/api.py` 的 `_WEBUI_HTML`：HITL 开关 + 草稿审核面板 + 续跑逻辑。
- `tests/test_api_hitl.py`（新增）：Web UI 含 HITL 入口、interrupted 事件回传、approve 定稿、带意见改写后再定稿、坏 thread 400，共 5 个离线测试。

**验证**：`pytest tests/test_api_hitl.py` 5 passed；完整离线套件 119 passed / 2 skipped / 2 deselected（无回归）。

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

---

## 方向 D'：引用网络分析（必引/桥接论文 + 共引空白）（2026-08-18）

**目标**：用 OpenAlex 的引用关系（`referenced_works`）在本地文献池内构建有向引用图，算**枢纽度（PageRank）**与**桥接度（betweenness）**，缓解「高被引但主题跑偏的论文绑架排序」与「关键流派/必引文献漏检」两大痛点；并让 GapAnalyzer 用共引找未覆盖子领域。

**变更内容**
- **抓取引用图数据**：`openalex._parse_work` 现在提取 `openalex_id` 与 `referenced_works`（SELECT 字段加 `referenced_works`）；`base.make_paper` / `state.Paper` 新增 `openalex_id` / `referenced_works` 字段；`store._PERSIST_FIELDS` 与 `hydrate` 一并持久化，跨运行复用池不丢引用关系。
- **新增 `src/agent/citation_graph.py`**（纯 Python 确定性算法，不依赖 networkx）：
  - `build_graph`：以 `openalex_id` 为节点、只在池内互引的边构建有向图；
  - `score_centrality`：就地写回每篇 `hub_score`（PageRank）/ `bridge_score`（betweenness），无引用数据时全部置 0（安全降级）；
  - `cocitation_gaps`：把共享 >=2 篇参考文献的论文并查集聚成子领域，未被现有主题簇覆盖者标为研究空白候选。
- **排序信号**：`tools.rank_papers` 综合分新增 `citation_hub_weight`（默认 0.10）× `hub_score` 项，让必引文献权重提升；未提供引用数据时该项恒为 0，等价旧行为。配置项 `CITATION_HUB_WEIGHT`。
- **节点接线**：
  - `ranker`：在闸门前后调用 `score_centrality`，并产出 `citation_analysis`（含 top_hub / top_bridge / 被闸门剔除的高枢纽论文告警）；
  - `gap_analyzer`：调用 `cocitation_gaps` 把共引空白合并进 gaps（去重）；
  - `synthesizer`：成稿新增**附录 A.8「引用网络分析」**，列出枢纽度 Top / 桥接度 Top 与高枢纽误删告警，使关键文献识别可复核。

**涉及文件**
- `src/ingest/openalex.py`：`_parse_work` 抓取 `openalex_id`/`referenced_works`；SELECT 加 `referenced_works`。
- `src/ingest/base.py`：`Paper` 与 `make_paper` 新增 `openalex_id`/`referenced_works`。
- `src/ingest/store.py`：`_PERSIST_FIELDS` 与 `hydrate` 持久化这两个字段。
- `src/agent/citation_graph.py`：新增模块（build_graph / pagerank / betweenness / score_centrality / cocitation_gaps）。
- `src/agent/config.py`：新增 `citation_hub_weight`（默认 0.10）。
- `src/agent/state.py`：新增 `citation_analysis` 状态字段与初值；`Paper` 加引用图字段。
- `src/agent/tools.py`：`rank_papers` 综合分加入枢纽度信号。
- `src/agent/nodes.py`：`ranker` 调用 `score_centrality` 并产出 `citation_analysis`；`gap_analyzer` 接 `cocitation_gaps`；`synthesizer` 附录 A.8。
- `tests/test_citation_graph.py`：新增 8 个测试（构图边、PageRank 枢纽序、betweenness 叶子为 0、共引空白命中/忽略已覆盖、ranker 写入分析、端到端 A.8 附录）。

**验证**：`pytest tests/test_citation_graph.py` 全部通过；完整离线套件 110 passed（含 D' 8 个测试）。

---

## 方向 B'：增量更新已有综述（2026-08-18）

**目标**：让 lit-review-agent 能「接住」上一版成稿并做增量更新——载入历史文献池与引用编号、只新检索 `since_date` 之后的新论文、保留未变小节正文（省 LLM token）、沿用历史引用编号，并在成稿头渲染「新增/重写/保留」统计。直接对接你「定期更新某主题综述」的真实工作流。

**变更内容**
- **状态与配置**：`state.initial_state` 新增 `incremental / since_date / base_path` 入参；`AgentState` 新增 `incremental / since_date / base_path / previous_loaded / previous_pids / previous_sections / previous_clusters / incremental_keep / incremental_note` 字段（初值均为空/False/null）。`config` 新增 `incremental_default_days`（默认 180，作为未给 `since_date` 时的回看窗口）。
- **载入历史文献池（retriever）**：增量模式下从 `base_path` 的 `<stem>_papers.json` 还原上一版文献并并入检索 seed，同时把上一版 `citation_map`（由 `citation_index` 重建）并入本轮，使历史引用编号跨版本延续；不入池（`previous_loaded`）只做一次。
- **增量规划节点 `incremental_plan`（新增）**：把本版主题簇与上一版按论文重叠度匹配，重叠 ≥50% 的小节沿用上一版正文（`carry` 进 `sections`），其余标记重写；统计 `new`（不在上一版的论文数）/ `rewritten` / `kept` 写入 `incremental_note`。非增量模式直接放行、不改状态。
- **section_writer 沿用旧正文**：用 `incremental_keep` 预填 `sections` 并跳过这些小节的重新生成，避免对未变小节重复消耗 LLM。
- **成稿产物（synthesizer）**：写出 `<stem>_meta.json` 侧车（保存本版 `sections` + `clusters`），供下一版增量更新沿用；成稿头在增量模式下渲染「增量更新：新增 N 篇，重写 M 个小节，保留 K 个小节（沿用历史引用编号）」。
- **图接线 / CLI / API**：`graph` 在 `clusterer → section_writer` 间插入 `incremental_plan`；`run_review` 新增 `incremental/since_date/base_path` 透传；`main.py` 新增 `--incremental/--since/--base`；`api.ReviewRequest` 新增同名字段，两个端点均接回。
- **修复：retriever 日期过滤误伤常规运行**：原实现对 `since_date` 的过滤**无条件**执行——当 `since_date` 为空时回退到 `now - incremental_default_days`（≈2026），会把常规（非增量）运行里的历史文献全部过滤为空池导致图提前结束。已改为**仅增量模式**才按 `since_date` 过滤，常规运行不再受时间窗口影响。

**涉及文件**
- `src/agent/state.py`：`initial_state` 加增量入参；`AgentState` 加 9 个增量字段与初值。
- `src/agent/config.py`：新增 `incremental_default_days`。
- `src/agent/nodes.py`：新增 `_parse_since_year` / `_load_previous` / `incremental_plan`；`retriever` 载入历史池与编号、增量模式才做 since 过滤；`section_writer` 沿用 `incremental_keep`；`synthesizer` 写 `_meta.json` 侧车并渲染增量说明。
- `src/agent/graph.py`：插入 `incremental_plan` 节点；`run_review` 透传增量参数。
- `src/main.py`：新增 `--incremental/--since/--base`。
- `src/api.py`：`ReviewRequest` 新增增量字段并接回 `run_review`。
- `tests/test_incremental.py`：新增 4 个测试（节点级保留/跳过、端到端增量复用+编号延续+说明渲染、无 base 安全降级）。

**配套测试加固（验证中发现）**
- `tests/test_api_stream.py` 的 `client` fixture 只 patch 了 `src.agent.tools.multi_source_search`，但 `retriever` 通过 `src.agent.nodes` 模块级引用调用该函数，patch 实为无效——流式测试此前实际打真实 OpenAlex API，仅在 HTTP 磁盘缓存命中时「碰巧」通过，冷缓存即在 `base.py:http_get` 挂起。已改为同时 patch `nodes` 与 `tools` 两个命名空间的 `multi_source_search`，使该测试真正离线、确定可绿。

**验证**：`pytest tests/test_incremental.py` 4 个全过；完整离线套件 **114 passed / 2 skipped / 2 deselected**（含 B' 4 个测试；2 skipped 为无 key 的真 LLM 测试，2 deselected 为联网标记测试）。
