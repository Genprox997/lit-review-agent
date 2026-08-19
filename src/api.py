"""FastAPI 入口（设计文档 §3：CLI / FastAPI 入口）。

把 `run_review` 包成 HTTP 服务，便于把综述能力嵌进其它系统或做 Web 前端。

启动：
    pip install -e ".[api]"
    uvicorn src.api:app --host 0.0.0.0 --port 8000

接口：
    GET  /           自包含 Web UI（表单 + 实时进度 + 人工审核面板）
    GET  /healthz
    POST /review              一次性返回产物路径（非流式）
    POST /review/stream       SSE 流式返回执行进度，结束推送 done 事件
    POST /review/resume       HITL 续跑：把人工意见回填给被挂起的 thread 并继续
"""

from __future__ import annotations

import json as _json
import logging
import os
import queue
import threading
import uuid
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

# 顶层导入：未安装 fastapi 时整体降级，app=None 不报错
try:  # pragma: no cover - 取决于是否安装 fastapi
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import HTMLResponse, StreamingResponse
    from pydantic import BaseModel, Field
except Exception:  # pragma: no cover
    FastAPI = None  # type: ignore


class ReviewRequest(BaseModel):
    """综述生成请求体（模块顶层定义，FastAPI 才能正确识别为请求体）。"""
    topic: str = Field(..., description="研究主题（英文效果更好）", examples=["diffusion models for image super-resolution"])
    constraints: str = Field("", description="额外约束，如「只看 2020 年后的工作」")
    sources: Optional[str] = Field(None, description="检索源，逗号分隔：arxiv,openalex,semantic_scholar,pubmed,crossref")
    target: Optional[int] = Field(None, description="文献池目标规模")
    per_query: Optional[int] = Field(None, description="单检索式单源最大返回条数")
    min_year: Optional[int] = Field(None, description="只保留该年份及之后的文献")
    top_fulltext: Optional[int] = Field(None, description="下载全文的 Top-N 篇数")
    clusters: Optional[int] = Field(None, description="主题簇数量，0=自动")
    lang: Optional[str] = Field(None, description="正文语言：zh / en")
    provider: Optional[str] = Field(None, description="LLM 提供商：deepseek/openai/ollama/stub")
    format: Optional[str] = Field(None, description="成稿格式：md / latex / docx")
    incremental: bool = Field(False, description="增量更新模式：基于上一版成稿，只拉新文献、沿用编号")
    since_date: Optional[str] = Field(None, description="增量起始日期 YYYY-MM-DD")
    base_path: Optional[str] = Field(None, description="上一版成稿路径（.md），用于沿用编号与保留旧小节")
    with_human: bool = Field(False, description="启用人工审核（HITL）：成稿后挂起，等 Web UI 回填意见再定稿")
    thread_id: Optional[str] = Field(None, description="HITL 续跑用的 thread id；留空由服务端生成并在 interrupted 事件中返回")


class ResumeRequest(BaseModel):
    """HITL 续跑请求体：把人工意见回填给被挂起的 thread。"""
    thread_id: str = Field(..., description="人工审核挂起时由 /review/stream 的 interrupted 事件返回的 thread id")
    feedback: str = Field("", description="人工审核意见；空字符串表示通过并直接定稿")


def _sse_endpoint(runner: Callable[[Callable[[str, Any], None]], None]):
    """把 `runner(progress)` 的执行进度以 SSE 流式返回。

    `runner` 内部调用 ``run_review(stream=True, on_progress=progress)``；进度事件、
    挂起(``interrupted``)、完成(``done``)、错误(``error``) 都经同一队列回传，前端
    据此刷新日志或弹出人工审核表单。响应体为 ``text/event-stream``。
    """
    q: "queue.Queue" = queue.Queue()

    def progress(stage: str, payload: Any) -> None:
        q.put({"stage": stage, "payload": payload})

    def run() -> None:
        try:
            runner(progress)
        except Exception as exc:  # noqa: BLE001 - 转成 error 事件
            logger.exception("综述生成失败")
            q.put({"stage": "error", "payload": str(exc)})

    worker = threading.Thread(target=run, daemon=True)
    worker.start()

    def gen():
        while True:
            item = q.get()
            yield f"event: {item['stage']}\ndata: {_json.dumps(item['payload'], ensure_ascii=False)}\n\n"
            if item["stage"] in ("done", "error", "interrupted"):
                break
        # 确保工作线程在流式响应结束时退出，避免遗留守护线程拖累后续测试/进程
        worker.join(timeout=60)

    return StreamingResponse(gen(), media_type="text/event-stream")


def create_app():
    """惰性构造 FastAPI app；未安装 fastapi 时抛出清晰错误。"""
    if FastAPI is None:  # pragma: no cover
        raise RuntimeError(
            "FastAPI 入口未启用：请先 `pip install -e \".[api]\"` 安装 fastapi/uvicorn。"
        )

    app = FastAPI(
        title="lit-review-agent",
        version="0.1.0",
        description="给定研究主题，自动检索、聚类并撰写带引用的文献综述。",
    )

    def _apply_request(req: ReviewRequest) -> None:
        """把请求参数写回环境变量，复用 CLI 的同一套配置分发逻辑。"""
        overrides = {
            "ENABLED_SOURCES": req.sources,
            "TARGET_PAPER_COUNT": req.target,
            "MAX_RESULTS_PER_QUERY": req.per_query,
            "MIN_YEAR": req.min_year,
            "TOP_N_FULLTEXT": req.top_fulltext,
            "N_CLUSTERS": req.clusters,
            "REPORT_LANGUAGE": req.lang,
            "LLM_PROVIDER": req.provider,
            "OUTPUT_FORMAT": req.format,
        }
        for key, value in overrides.items():
            if value is not None:
                os.environ[key] = str(value)

    def _check_key() -> None:
        from src.config import get_settings

        settings = get_settings(refresh=True)
        if settings.llm_provider != "stub" and not settings.model_config().get("api_key"):
            raise HTTPException(
                status_code=400,
                detail=f"LLM_PROVIDER={settings.llm_provider} 但未配置 API key；"
                       f"或设置 provider=stub 离线试跑。",
            )

    @app.get("/healthz")
    def healthz() -> Dict[str, str]:
        return {"status": "ok"}

    @app.get("/", response_model=None)
    def index() -> HTMLResponse:
        return HTMLResponse(_WEBUI_HTML)

    @app.post("/review", response_model=None)
    def review(req: ReviewRequest) -> Dict[str, Any]:
        _apply_request(req)
        from src.agent.graph import run_review

        _check_key()
        try:
            final = run_review(
                topic=req.topic,
                constraints=req.constraints,
                thread_id=f"api-{abs(hash(req.topic)) % 10**9}",
                with_human=False,
                stream=False,
                incremental=req.incremental,
                since_date=req.since_date,
                base_path=req.base_path,
            )
        except Exception as exc:  # noqa: BLE001 - 转成 500 并返回原因
            logger.exception("综述生成失败")
            raise HTTPException(status_code=500, detail=f"生成失败：{exc}") from exc

        if final.get("interrupted"):
            return {"interrupted": True, "message": "已挂起，需人工审核（API 模式默认不启用）。"}
        artifacts: Dict[str, str] = final.get("artifacts") or {}
        if not artifacts:
            raise HTTPException(status_code=422, detail="未生成成稿，请检查检索是否为空或 LLM 是否失败。")
        return {
            "topic": req.topic,
            "paper_count": len(final.get("papers") or []),
            "citation_count": len(final.get("citation_map") or {}),
            "section_count": len(final.get("sections") or {}),
            "gaps": final.get("gaps") or [],
            "artifacts": artifacts,
        }

    @app.post("/review/stream", response_model=None)
    def review_stream(req: ReviewRequest):
        """SSE 流式接口：实时推送执行进度，结束后推送 done 事件与产物路径。

        当 ``with_human=True`` 时，图会在 Synthesizer 之后挂起（interrupt），
        本接口推送 ``interrupted`` 事件（携带草稿全文与 thread_id），前端据此
        弹出人工审核面板；随后由 ``POST /review/resume`` 回填意见续跑。
        """
        _apply_request(req)
        _check_key()

        from src.agent.graph import run_review

        if req.with_human:
            tid = req.thread_id or f"hitl-{uuid.uuid4().hex[:12]}"
            with_human = True
        else:
            tid = req.thread_id or f"api-{abs(hash(req.topic)) % 10**9}"
            with_human = False

        def runner(progress: Callable[[str, Any], None]) -> None:
            run_review(
                topic=req.topic,
                constraints=req.constraints,
                thread_id=tid,
                with_human=with_human,
                stream=True,
                on_progress=progress,
                incremental=req.incremental,
                since_date=req.since_date,
                base_path=req.base_path,
            )

        return _sse_endpoint(runner)

    @app.post("/review/resume", response_model=None)
    def review_resume(req: ResumeRequest):
        """HITL 续跑：把人工意见回填给被挂起的 thread 并继续生成。

        先校验该 thread 确实处于 ``human_review`` 挂起态（否则 400，避免误触发
        全新运行）；随后以 ``Command(resume=feedback)`` 续跑——空意见视为通过定稿，
        非空意见进入针对性改写回环。以 SSE 流回传进度/再次挂起/完成。
        """
        from src.agent.graph import build_graph, run_review

        # 预校验：thread 必须处于 human_review 挂起态
        graph = build_graph(with_human=True)
        config = {"configurable": {"thread_id": req.thread_id}, "recursion_limit": 80}
        snap = graph.get_state(config)
        if not snap.next or "human_review" not in snap.next:
            raise HTTPException(
                status_code=400,
                detail=f"thread_id={req.thread_id} 未处于人工审核挂起态，无法续跑。",
            )

        _check_key()

        def runner(progress: Callable[[str, Any], None]) -> None:
            run_review(
                topic="",  # 续跑沿用检查点中的主题，无需重新传
                with_human=True,
                feedback=req.feedback,
                thread_id=req.thread_id,
                stream=True,
                on_progress=progress,
            )

        return _sse_endpoint(runner)

    return app


# --------------------------------------------------------------------------
# 自包含 Web UI（无额外前端依赖，直接由 GET / 返回）
# --------------------------------------------------------------------------
_WEBUI_HTML = """<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>lit-review-agent</title>
<style>
  :root { color-scheme: light; }
  body { font-family: -apple-system, "Segoe UI", "Microsoft YaHei", sans-serif; max-width: 820px; margin: 32px auto; padding: 0 16px; color: #1f2933; background: #fafafa; }
  h1 { font-size: 22px; }
  label { display: block; margin: 12px 0 4px; font-weight: 600; }
  input, select { width: 100%; padding: 8px; box-sizing: border-box; border: 1px solid #cbd5e1; border-radius: 6px; font-size: 14px; }
  button { margin-top: 16px; padding: 10px 18px; border: 0; border-radius: 6px; background: #2563eb; color: #fff; font-size: 15px; cursor: pointer; }
  button:disabled { background: #94a3b8; cursor: not-allowed; }
  #log { margin-top: 18px; background: #0f172a; color: #e2e8f0; padding: 12px; border-radius: 6px; height: 320px; overflow: auto; white-space: pre-wrap; font-family: ui-monospace, Menlo, Consolas, monospace; font-size: 13px; }
  .done { color: #22c55e; font-weight: 700; }
  a { color: #2563eb; }
  .row { display: flex; align-items: center; gap: 8px; margin-top: 12px; }
  .row input[type=checkbox] { width: auto; }
  #draft { margin-top: 18px; border: 1px solid #cbd5e1; border-radius: 8px; padding: 12px; background: #fff; }
  #draft h3 { margin: 0 0 8px; font-size: 15px; }
  #draftMeta { color: #64748b; font-size: 12px; margin-bottom: 8px; }
  #draftText { max-height: 340px; overflow: auto; white-space: pre-wrap; word-break: break-word;
              background: #f1f5f9; border-radius: 6px; padding: 10px; font-size: 12px;
              font-family: ui-monospace, Menlo, Consolas, monospace; }
  #feedback { width: 100%; min-height: 64px; margin-top: 10px; padding: 8px; box-sizing: border-box;
              border: 1px solid #cbd5e1; border-radius: 6px; font-size: 14px; resize: vertical; }
  .btn-row { margin-top: 10px; display: flex; gap: 10px; }
  .btn-row button { margin-top: 0; }
  #approve { background: #16a34a; }
</style>
</head>
<body>
  <h1>📚 lit-review-agent</h1>
  <p>输入研究主题，自动检索、聚类并生成带引用的文献综述。进度实时滚动。</p>
  <label>研究主题（英文效果更好）</label>
  <input id="topic" value="diffusion models for image super-resolution">
  <label>额外约束（可空）</label>
  <input id="constraints" placeholder="如：只看 2020 年后的工作">
  <label>检索源</label>
  <input id="sources" value="arxiv,openalex">
  <label>LLM 提供商</label>
  <select id="provider">
    <option value="stub">stub（离线试跑）</option>
    <option value="deepseek" selected>deepseek</option>
    <option value="openai">openai</option>
    <option value="ollama">ollama</option>
  </select>
  <label>目标文献规模</label>
  <input id="target" type="number" value="40">
  <div class="row">
    <input type="checkbox" id="with_human">
    <label style="margin:0;font-weight:600;cursor:pointer" for="with_human">启用人工审核（HITL）：成稿后可在网页里看草稿、提意见再定稿</label>
  </div>
  <button id="go">开始生成</button>
  <div id="log"></div>

  <div id="draft" style="display:none">
    <h3>⏸ 草稿待审核</h3>
    <div id="draftMeta"></div>
    <pre id="draftText"></pre>
    <textarea id="feedback" placeholder="在此填写修改意见，例如「在第 1 个主题小节补充与对比方法的实验分析」；留空点「通过并定稿」即直接定稿。"></textarea>
    <div class="btn-row">
      <button id="submitFeedback">提交修改意见并重生成</button>
      <button id="approve">通过并定稿</button>
    </div>
  </div>

  <div id="graphPanel" style="display:none; margin-top:18px;">
    <h3>🕸 引用网络（点击节点看引用关系，拖拽可调整布局）</h3>
    <div class="row">
      <input type="checkbox" id="hubOnly">
      <label style="margin:0;cursor:pointer" for="hubOnly">只显示枢纽论文（hub ≥ 0.3）</label>
      <span id="graphStat" style="margin-left:auto;color:#64748b;font-size:12px;"></span>
    </div>
    <svg id="graph" width="100%" height="460" style="background:#fff;border:1px solid #cbd5e1;border-radius:8px;touch-action:none;"></svg>
    <div id="legend" style="display:flex;flex-wrap:wrap;gap:12px;margin-top:8px;font-size:12px;color:#475569;"></div>
    <div id="graphInfo" style="margin-top:8px;font-size:13px;color:#1f2933;min-height:48px;line-height:1.5;"></div>
  </div>

<script>
const log = document.getElementById("log");
let currentThreadId = null;

function append(text, cls) {
  const div = document.createElement("div");
  if (cls) div.className = cls;
  div.textContent = text;
  log.appendChild(div);
  log.scrollTop = log.scrollHeight;
}

function handleEvent(ev) {
  const stage = ev.stage;
  let payload;
  try { payload = JSON.parse(ev.data); } catch (e) { payload = ev.data; }

  if (stage === "progress") {
    append("• " + payload);
  } else if (stage === "human_review") {
    append("⏸ 草稿已生成并挂起，请在下方预览并提交审核意见。", "done");
  } else if (stage === "interrupted") {
    currentThreadId = payload.thread_id || null;
    showDraft(payload);
  } else if (stage === "done") {
    append("✅ 完成：" + (payload.paper_count||0) + " 篇文献，" + (payload.citation_count||0) + " 条引用", "done");
    const arts = payload.artifacts || {};
    for (const k in arts) append("  " + k + ": " + arts[k]);
    document.getElementById("draft").style.display = "none";
    if (payload.citation_graph && payload.citation_graph.nodes && payload.citation_graph.nodes.length) {
      renderGraph(payload.citation_graph);
    }
  } else if (stage === "error") {
    append("❌ 错误：" + payload, "done");
  } else {
    append("[" + stage + "] " + JSON.stringify(payload));
  }
}

function showDraft(payload) {
  const draft = document.getElementById("draft");
  document.getElementById("draftText").textContent = payload.report || "(无草稿内容)";
  document.getElementById("draftMeta").textContent =
    "文献 " + (payload.paper_count||0) + " 篇 / 引用 " + (payload.citation_count||0) + " 条 / 小节 " +
    (payload.section_count||0) + " 个" + (payload.report_path ? " ｜ 文件：" + payload.report_path : "");
  draft.style.display = "block";
  document.getElementById("feedback").value = "";
  append("⏸ 已挂起等待人工审核（thread_id=" + (currentThreadId||"") + "，可多次提意见后再定稿）", "done");
}

async function consumeStream(resp) {
  const reader = resp.body.getReader();
  const decoder = new TextDecoder();
  let buf = "";
  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buf += decoder.decode(value, { stream: true });
    let idx;
    while ((idx = buf.indexOf("\\n\\n")) !== -1) {
      const raw = buf.slice(0, idx); buf = buf.slice(idx + 2);
      const ev = {};
      raw.split("\\n").forEach(function(line) {
        if (line.startsWith("event:")) ev.stage = line.slice(6).trim();
        else if (line.startsWith("data:")) ev.data = line.slice(5).trim();
      });
      if (ev.stage && ev.data) handleEvent(ev);
    }
  }
}

function readForm() {
  return {
    topic: document.getElementById("topic").value,
    constraints: document.getElementById("constraints").value,
    sources: document.getElementById("sources").value,
    provider: document.getElementById("provider").value,
    target: Number(document.getElementById("target").value) || undefined,
    with_human: document.getElementById("with_human").checked,
  };
}

async function startRun() {
  const btn = document.getElementById("go");
  btn.disabled = true; log.innerHTML = "";
  document.getElementById("draft").style.display = "none";
  const body = readForm();
  append("▶ 已提交，等待流…" + (body.with_human ? "（启用人工审核）" : ""));
  try {
    const resp = await fetch("/review/stream", {
      method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(body),
    });
    if (!resp.ok) { append("❌ 服务端返回 " + resp.status + "：" + (await resp.text()), "done"); return; }
    await consumeStream(resp);
  } catch (e) {
    append("❌ 请求失败：" + e, "done");
  } finally { btn.disabled = false; }
}

async function resumeRun(feedback) {
  if (!currentThreadId) { append("❌ 当前没有可续跑的 thread_id", "done"); return; }
  const btn = document.getElementById("go");
  btn.disabled = true;
  append("⏳ 已提交审核意见，正在重生成…" + (feedback ? "" : "（通过并定稿）"));
  try {
    const resp = await fetch("/review/resume", {
      method: "POST", headers: {"Content-Type": "application/json"},
      body: JSON.stringify({ thread_id: currentThreadId, feedback: feedback }),
    });
    if (!resp.ok) { append("❌ 续跑失败 " + resp.status + "：" + (await resp.text()), "done"); return; }
    await consumeStream(resp);
  } catch (e) {
    append("❌ 续跑失败：" + e, "done");
  } finally { btn.disabled = false; }
}

// ===== 引用网络可视化（方向 E'）=====
let graphSim = null;       // {nodes, edges, idIndex, W, H, alpha}
let highlightNode = -1;    // 当前选中节点 index
let highlightSet = null;   // Set<index> 高亮集合（含邻居）
let hubOnlyOn = false;
let draggingNode = null;
let dragMoved = false;

const PALETTE = ["#2563eb","#dc2626","#16a34a","#d97706","#7c3aed","#0891b2","#db2777","#65a30d","#ea580c","#4f46e5","#0d9488","#ca8a04"];
const _clusterColor = {};
function colorFor(cluster) {
  if (!cluster) return "#94a3b8";
  if (_clusterColor[cluster] === undefined) {
    const keys = Object.keys(_clusterColor);
    _clusterColor[cluster] = PALETTE[keys.length % PALETTE.length];
  }
  return _clusterColor[cluster];
}
function esc(s){ return String(s).replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;"); }

function renderGraph(data) {
  const svg = document.getElementById("graph");
  document.getElementById("graphPanel").style.display = "block";
  const W = svg.clientWidth || 760, H = 460;
  const n = data.nodes.length;
  const R = Math.min(W, H) * 0.38;
  const nodes = data.nodes.map((d, i) => ({
    ...d, i,
    x: W/2 + R * Math.cos(2*Math.PI*i/n) + (Math.random()-0.5)*20,
    y: H/2 + R * Math.sin(2*Math.PI*i/n) + (Math.random()-0.5)*20,
    vx: 0, vy: 0, fx: null, fy: null,
  }));
  const idIndex = {}; nodes.forEach(nd => idIndex[nd.id] = nd.i);
  const edges = (data.edges || []).filter(e => idIndex[e[0]]!==undefined && idIndex[e[1]]!==undefined)
                              .map(e => [idIndex[e[0]], idIndex[e[1]]]);
  graphSim = {nodes, edges, idIndex, W, H, alpha: 1};

  const legend = document.getElementById("legend");
  const seen = {}; let lh = "";
  for (const nd of nodes) {
    if (nd.cluster && !seen[nd.cluster]) { seen[nd.cluster]=1;
      lh += '<span style="display:inline-flex;align-items:center;gap:4px;"><span style="width:11px;height:11px;border-radius:50%;background:'+colorFor(nd.cluster)+';display:inline-block"></span>'+esc(nd.cluster)+'</span>'; }
  }
  legend.innerHTML = lh;
  const gaps = data.gaps || [];
  document.getElementById("graphInfo").innerHTML = gaps.length
    ? "<b>⚠ 研究空白候选：</b><br>" + gaps.map(g=>"• "+esc(g)).join("<br>")
    : "提示：点击节点查看引用关系，拖拽可调整布局。";
  document.getElementById("graphStat").textContent = nodes.length + " 篇 / " + edges.length + " 条引用边";
  bindGraphEvents();
  runSim();
}

function runSim() {
  if (!graphSim) return;
  cancelAnimationFrame(runSim._raf);
  function step() {
    tick(graphSim); draw(graphSim);
    if (graphSim.alpha > 0.02 || draggingNode) runSim._raf = requestAnimationFrame(step);
  }
  runSim._raf = requestAnimationFrame(step);
}

function tick(s) {
  const repel = 1400;
  for (let i=0;i<s.nodes.length;i++){
    const a=s.nodes[i];
    for (let j=i+1;j<s.nodes.length;j++){
      const b=s.nodes[j];
      let dx=a.x-b.x, dy=a.y-b.y, d2=dx*dx+dy*dy; if(d2<1)d2=1;
      const d=Math.sqrt(d2), f=repel/d2, fx=f*dx/d, fy=f*dy/d;
      a.vx+=fx; a.vy+=fy; b.vx-=fx; b.vy-=fy;
    }
  }
  const L=58;
  for (const [u,v] of s.edges){
    const a=s.nodes[u], b=s.nodes[v];
    let dx=b.x-a.x, dy=b.y-a.y, d=Math.sqrt(dx*dx+dy*dy)||1;
    const f=(d-L)*0.05, fx=f*dx/d, fy=f*dy/d;
    a.vx+=fx; a.vy+=fy; b.vx-=fx; b.vy-=fy;
  }
  for (const nd of s.nodes){
    if (nd.fx!==null){ nd.x=nd.fx; nd.y=nd.fy; nd.vx=0; nd.vy=0; continue; }
    nd.vx += (s.W/2 - nd.x)*0.003; nd.vy += (s.H/2 - nd.y)*0.003;
    nd.vx*=0.86; nd.vy*=0.86;
    nd.x += nd.vx * s.alpha; nd.y += nd.vy * s.alpha;
    nd.x = Math.max(12, Math.min(s.W-12, nd.x));
    nd.y = Math.max(12, Math.min(s.H-12, nd.y));
  }
  s.alpha *= 0.97;
}

function draw(s) {
  const svg = document.getElementById("graph");
  let html = "";
  for (const [u,v] of s.edges){
    const a=s.nodes[u], b=s.nodes[v];
    const on = !highlightSet || (highlightSet.has(u) && highlightSet.has(v));
    html += '<line x1="'+a.x.toFixed(1)+'" y1="'+a.y.toFixed(1)+'" x2="'+b.x.toFixed(1)+'" y2="'+b.y.toFixed(1)+'" stroke="'+(on?'#94a3b8':'#eef2f7')+'" stroke-width="'+(on?1.3:0.5)+'"/>';
  }
  for (const nd of s.nodes){
    if (hubOnlyOn && nd.hub < 0.3) continue;
    const r = 4 + 12*Math.max(nd.hub,0);
    const op = (!highlightSet || highlightSet.has(nd.i)) ? 1 : 0.16;
    const sel = nd.i === highlightNode;
    html += '<circle data-i="'+nd.i+'" cx="'+nd.x.toFixed(1)+'" cy="'+nd.y.toFixed(1)+'" r="'+r.toFixed(1)+'" fill="'+colorFor(nd.cluster)+'" fill-opacity="'+op+'" stroke="'+(sel?'#0f172a':'#fff')+'" stroke-width="'+(sel?2.5:1)+'" style="cursor:pointer"/>';
    if (nd.hub>=0.35 || sel){
      const t = nd.label.length>26? nd.label.slice(0,25)+"…" : nd.label;
      html += '<text data-i="'+nd.i+'" x="'+(nd.x+r+3).toFixed(1)+'" y="'+(nd.y+3.5).toFixed(1)+'" font-size="10" fill="#334155" fill-opacity="'+op+'" style="pointer-events:none">'+esc(t)+'</text>';
    }
  }
  svg.setAttribute("viewBox", "0 0 " + s.W + " " + s.H);
  svg.innerHTML = html;
}

function bindGraphEvents() {
  const svg = document.getElementById("graph");
  if (svg._bound) return; svg._bound = 1;
  function svgPoint(evt){
    const r = svg.getBoundingClientRect();
    const vb = (svg.getAttribute("viewBox")||("0 0 "+svg.clientWidth+" 460")).split(" ");
    return { x: (evt.clientX - r.left) * (parseFloat(vb[2])/r.width),
             y: (evt.clientY - r.top)  * (parseFloat(vb[3])/r.height) };
  }
  function nodeAt(evt){
    const p = svgPoint(evt); let best=null, bd=1e9;
    for (const nd of graphSim.nodes){ if(hubOnlyOn && nd.hub<0.3) continue;
      const dx=nd.x-p.x, dy=nd.y-p.y, d=dx*dx+dy*dy, rr=(4+12*Math.max(nd.hub,0))+5;
      if(d<rr*rr && d<bd){bd=d;best=nd;} }
    return best;
  }
  svg.addEventListener("mousedown", (e)=>{ const nd=nodeAt(e); if(!nd) return; draggingNode=nd; dragMoved=false; nd.fx=nd.x; nd.fy=nd.y; });
  window.addEventListener("mousemove", (e)=>{ if(!draggingNode) return; dragMoved=true; const p=svgPoint(e); draggingNode.fx=p.x; draggingNode.fy=p.y; graphSim.alpha=Math.max(graphSim.alpha,0.3); });
  window.addEventListener("mouseup", ()=>{ if(!draggingNode) return; const nd=draggingNode; draggingNode=null; nd.fx=null; nd.fy=null; if(!dragMoved) selectNode(nd.i); });
  svg.addEventListener("click", (e)=>{ if (e.target.tagName === "svg") clearHighlight(); });
}

function selectNode(i){
  highlightNode = i;
  const s = graphSim; const nb = new Set([i]);
  for (const [u,v] of s.edges){ if(u===i) nb.add(v); if(v===i) nb.add(u); }
  highlightSet = nb; draw(s);
  const nd = s.nodes[i];
  const outE=[], inE=[];
  for (const [u,v] of s.edges){ if(u===i) outE.push(s.nodes[v].label); if(v===i) inE.push(s.nodes[u].label); }
  const info = document.getElementById("graphInfo");
  let html = "<b>"+esc(nd.label)+"</b><br>";
  html += "年份 "+(nd.year||"-")+" ｜ 被引 "+nd.citations+" ｜ 枢纽度 "+nd.hub+" ｜ 桥接度 "+nd.bridge+" ｜ 簇："+esc(nd.cluster||"-")+"<br>";
  if (outE.length) html += "<br><b>引用（池内 "+outE.length+"）：</b><br>" + outE.slice(0,8).map(t=>"• "+esc(t.length>40?t.slice(0,39)+"…":t)).join("<br>");
  if (inE.length) html += "<br><b>被引（池内 "+inE.length+"）：</b><br>" + inE.slice(0,8).map(t=>"• "+esc(t.length>40?t.slice(0,39)+"…":t)).join("<br>");
  info.innerHTML = html;
}
function clearHighlight(){ highlightNode=-1; highlightSet=null; if(graphSim) draw(graphSim); document.getElementById("graphInfo").innerHTML="提示：点击节点查看引用关系，拖拽可调整布局。"; }

document.getElementById("hubOnly").addEventListener("change", (e)=>{ hubOnlyOn=e.target.checked; if(graphSim) draw(graphSim); });

document.getElementById("go").addEventListener("click", startRun);
document.getElementById("submitFeedback").addEventListener("click", () => {
  resumeRun(document.getElementById("feedback").value);
});
document.getElementById("approve").addEventListener("click", () => resumeRun(""));
</script>
</body>
</html>"""


try:
    app = create_app() if FastAPI is not None else None
except RuntimeError as exc:  # pragma: no cover
    logger.warning("FastAPI app 未创建：%s", exc)
    app = None  # type: ignore[assignment]


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("src.api:app", host="0.0.0.0", port=8000, reload=False)
