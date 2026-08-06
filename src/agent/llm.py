"""统一 LLM 调用层。

对上层节点屏蔽 provider 差异，并提供：
1. `chat()`        —— 纯文本生成；
2. `chat_json()`   —— 强制 JSON 输出 + 容错解析（自动剥离代码围栏 / 截取首个 JSON 对象）；
3. `stub` 后端     —— 不发任何网络请求的桩模型，用于离线跑通流程与单测。

所有调用都带 `node` 参数，既方便日志定位，也是 stub 后端的分发依据。
"""

from __future__ import annotations

import json
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional

from src.config import Settings, get_settings

logger = logging.getLogger(__name__)

# 延迟导入 OpenAI 异常类型（langchain_openai 依赖 openai，但用桩模式下不应强依赖）。
try:  # pragma: no cover - 取决于是否安装 openai
    from openai import APIStatusError, RateLimitError as _OpenAIRateLimitError
except Exception:  # pragma: no cover
    _OpenAIRateLimitError = None  # type: ignore
    APIStatusError = None  # type: ignore


def _is_rate_limit_error(exc: Exception) -> bool:
    """判断异常是否为 429 限流（兼容 OpenAI / langchain 包装）。"""
    if _OpenAIRateLimitError is not None and isinstance(exc, _OpenAIRateLimitError):
        return True
    if APIStatusError is not None and isinstance(exc, APIStatusError):
        return getattr(exc, "status_code", None) == 429
    # langchain 可能抛自己的 RateLimitError；或任意异常的文本/状态码含 429
    if "ratelimit" in type(exc).__name__.lower():
        return True
    status = getattr(exc, "status_code", None)
    if status == 429:
        return True
    return "429" in str(exc) and ("rate" in str(exc).lower() or "limit" in str(exc).lower())

_JSON_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


# --------------------------------------------------------------------------
# JSON 容错解析
# --------------------------------------------------------------------------
def _extract_json_blob(text: str) -> str:
    """从可能夹带解释文字的模型输出中截取 JSON 主体。"""
    text = (text or "").strip()
    if not text:
        return ""

    fenced = _JSON_FENCE.search(text)
    if fenced:
        text = fenced.group(1).strip()

    # 截取首个平衡的 {...} 或 [...]
    for opener, closer in (("{", "}"), ("[", "]")):
        start = text.find(opener)
        if start == -1:
            continue
        depth, in_str, escape = 0, False, False
        for idx in range(start, len(text)):
            ch = text[idx]
            if in_str:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch == opener:
                depth += 1
            elif ch == closer:
                depth -= 1
                if depth == 0:
                    return text[start : idx + 1]
    return text


def parse_json(text: str, default: Any = None) -> Any:
    """尽最大努力把模型输出解析成 Python 对象，失败返回 default。"""
    blob = _extract_json_blob(text)
    if not blob:
        return default
    try:
        return json.loads(blob)
    except json.JSONDecodeError:
        # 常见破损：尾随逗号、单引号、中文引号
        repaired = re.sub(r",\s*([}\]])", r"\1", blob)
        repaired = repaired.replace("“", '"').replace("”", '"')
        try:
            return json.loads(repaired)
        except json.JSONDecodeError:
            logger.warning("JSON 解析失败，返回默认值。原始片段: %.200s", blob)
            return default


# --------------------------------------------------------------------------
# 桩模型：离线跑通全流程
# --------------------------------------------------------------------------
class StubBackend:
    """按节点名返回结构合法的假数据，不触发任何网络请求。"""

    def __call__(self, node: str, system: str, user: str, json_mode: bool) -> str:
        handler = getattr(self, f"_{node}", None)
        if handler is not None:
            return handler(user)
        return json.dumps({"result": f"[stub:{node}]"}, ensure_ascii=False) if json_mode \
            else f"[stub:{node}] 这是桩模型生成的占位文本。"

    # -- 各节点桩输出 --
    @staticmethod
    def _query_expander(user: str) -> str:
        topic = _guess_topic(user)
        return json.dumps(
            {
                "queries": [
                    topic,
                    f"{topic} survey",
                    f"{topic} benchmark evaluation",
                    f"{topic} deep learning",
                ]
            },
            ensure_ascii=False,
        )

    @staticmethod
    def _extractor(user: str) -> str:
        ids = re.findall(r"paper_id:\s*(\S+)", user)
        return json.dumps(
            {
                "evidence": [
                    {
                        "paper_id": pid,
                        "claim": "[stub] 该工作提出了一种改进方法并在公开数据集上取得提升。",
                        "method": "[stub] method",
                        "dataset": "[stub] dataset",
                        "metric": "[stub] metric",
                    }
                    for pid in ids
                ]
            },
            ensure_ascii=False,
        )

    @staticmethod
    def _cluster_namer(user: str) -> str:
        ids = sorted(set(re.findall(r"簇\s*(\d+)", user)))
        return json.dumps(
            {"labels": {cid: f"[stub] 主题 {cid}" for cid in ids} or {"0": "[stub] 主题 0"}},
            ensure_ascii=False,
        )

    @staticmethod
    def _section_writer(user: str) -> str:
        cites = re.findall(r"\[(\d+)\]", user)[:3] or ["1"]
        marks = "".join(f"[{c}]" for c in cites)
        return (
            f"[stub] 本主题下的研究主要沿两条路线展开{marks}。第一条路线关注表示学习的改进，"
            f"第二条路线强调评测协议的统一{marks}。整体上，现有工作在标准数据集上已趋于饱和。"
        )

    @staticmethod
    def _critic(user: str) -> str:
        return json.dumps(
            {
                "verdict": "pass",
                "coverage_score": 8,
                "missing_topics": [],
                "contradictions": [],
                "extra_queries": [],
                "comments": "[stub] 覆盖度可接受，放行。",
            },
            ensure_ascii=False,
        )

    @staticmethod
    def _gap_analyzer(user: str) -> str:
        return json.dumps(
            {
                "gaps": [
                    "[stub] 缺乏跨数据集的统一评测基准。",
                    "[stub] 长尾场景下的鲁棒性研究不足。",
                ],
                "trends": ["[stub] 从手工特征转向端到端表示学习。"],
            },
            ensure_ascii=False,
        )

    @staticmethod
    def _ground_claims(user: str) -> str:
        # 从 evidence_block 中抽取出现的引用编号，造两条假 claim
        nums = re.findall(r"\[(\d+)\]", user)[:2] or ["1", "2"]
        return json.dumps(
            {
                "claims": [
                    {
                        "text": "[stub] 该方向已形成相对成熟的表示学习路线。",
                        "paper_ids": [nums[0]],
                        "confidence": "high",
                    },
                    {
                        "text": "[stub] 近年方法在公开基准上取得了一致的性能提升。",
                        "paper_ids": nums,
                        "confidence": "medium",
                    },
                ]
            },
            ensure_ascii=False,
        )

    @staticmethod
    def _synthesizer(user: str) -> str:
        cite = (re.findall(r"\[(\d+)\]", user) or ["1"])[0]
        if "撰写摘要与引言" in user:
            return (
                "## 摘要\n"
                "[stub] 本综述系统梳理了该领域的代表性工作，归纳出若干主题脉络，"
                "并指出了现存的研究空白与未来方向。\n\n"
                "## 引言\n"
                f"[stub] 该问题近年来受到广泛关注[{cite}]。本文按技术路线组织，"
                "先综述各主题下的代表工作，再归纳研究空白与趋势。"
            )
        return (
            f"[stub] 综合来看，该领域已形成若干成熟路线[{cite}]，"
            "未来最值得投入的方向是统一评测与跨场景泛化。"
        )


def _guess_topic(user: str) -> str:
    m = re.search(r"研究主题[:：]\s*(.+)", user)
    if m:
        return m.group(1).strip().splitlines()[0]
    return "research topic"


# --------------------------------------------------------------------------
# 真实后端
# --------------------------------------------------------------------------
_client_cache: Dict[str, Any] = {}


def _get_chat_model(settings: Settings, json_mode: bool):
    """惰性构造 ChatOpenAI（DeepSeek / OpenAI / Ollama 均走 OpenAI 兼容协议）。"""
    cache_key = f"{settings.llm_provider}:{json_mode}"
    if cache_key in _client_cache:
        return _client_cache[cache_key]

    from langchain_openai import ChatOpenAI  # 延迟导入，stub 模式无需安装

    cfg = settings.model_config()
    kwargs: Dict[str, Any] = {
        "model": cfg["model"],
        "api_key": cfg["api_key"],
        "temperature": settings.llm_temperature,
        "timeout": settings.llm_timeout,
        "max_retries": settings.llm_max_retries,
    }
    if cfg.get("base_url"):
        kwargs["base_url"] = cfg["base_url"]
    # Ollama 的 OpenAI 兼容层不支持 response_format
    if json_mode and settings.llm_provider in {"deepseek", "openai"}:
        kwargs["model_kwargs"] = {"response_format": {"type": "json_object"}}

    model = ChatOpenAI(**kwargs)
    _client_cache[cache_key] = model
    return model


_stub = StubBackend()


def chat(
    node: str,
    system: str,
    user: str,
    *,
    json_mode: bool = False,
    settings: Optional[Settings] = None,
) -> str:
    """调用 LLM 并返回原始文本。"""
    settings = settings or get_settings()

    if settings.llm_provider == "stub":
        return _stub(node, system, user, json_mode)

    model = _get_chat_model(settings, json_mode)
    messages: List[tuple] = [("system", system), ("human", user)]
    logger.debug("[%s] 调用 LLM，user prompt %d 字符", node, len(user))

    max_retries = max(0, settings.llm_max_retries)
    last_exc: Optional[Exception] = None
    resp = None
    for attempt in range(max_retries + 1):
        try:
            resp = model.invoke(messages)
            break
        except Exception as exc:  # noqa: BLE001 - 需区分限流与其他错误
            if _is_rate_limit_error(exc):
                last_exc = exc
                backoff = 2 ** attempt * 5  # 5s, 10s, 20s ...
                logger.warning(
                    "[%s] LLM 触发 429 限流，%ds 后重试（第 %d/%d 次）",
                    node, backoff, attempt + 1, max_retries,
                )
                if attempt < max_retries:
                    time.sleep(backoff)
                    continue
            raise

    if resp is None:  # 全部重试耗尽仍未拿到响应
        assert last_exc is not None
        raise last_exc

    content = resp.content
    if isinstance(content, list):  # 部分 provider 返回分块内容
        content = "".join(
            part.get("text", "") if isinstance(part, dict) else str(part) for part in content
        )
    return content or ""


def chat_json(
    node: str,
    system: str,
    user: str,
    *,
    default: Any = None,
    settings: Optional[Settings] = None,
) -> Any:
    """调用 LLM 并解析为 JSON。解析失败时自动重试一次（追加更强的格式约束）。"""
    settings = settings or get_settings()
    # DeepSeek JSON 模式要求 prompt 中出现 "json" 字样
    system_json = system.rstrip() + "\n\n严格只输出一个合法的 JSON 对象，不要任何解释文字或代码围栏。"

    raw = chat(node, system_json, user, json_mode=True, settings=settings)
    parsed = parse_json(raw, default=None)
    if parsed is not None:
        return parsed

    logger.warning("[%s] 首次 JSON 解析失败，重试一次。", node)
    retry_user = user + "\n\n【重要】上一次输出不是合法 JSON。请重新输出，且只输出 JSON 对象本身。"
    raw = chat(node, system_json, retry_user, json_mode=True, settings=settings)
    parsed = parse_json(raw, default=default)
    return parsed if parsed is not None else default


# --------------------------------------------------------------------------
# 批量并发：把多个互相独立的 LLM 调用并行掉，显著缩短端到端时延
# （P3-4）。stub 模式直接串行，简单可预测；真实 provider 用线程池并发。
# --------------------------------------------------------------------------
def _run_many(
    items: List[Dict[str, Any]],
    json_mode: bool,
    *,
    settings: Optional[Settings] = None,
) -> List[str]:
    """并发执行多个独立调用，返回与 items 等长的结果文本列表。

    items 每个元素：{"node", "system", "user"}。顺序严格对应输入顺序，
    便于调用方按位置 zip 回原文。
    """
    settings = settings or get_settings()
    if not items:
        return []
    # stub / 单条：直接串行，避免无谓起线程（且保证离线测试可预测）
    if settings.llm_provider == "stub" or len(items) == 1:
        return [
            chat(it["node"], it["system"], it["user"], json_mode=json_mode, settings=settings)
            for it in items
        ]
    workers = min(max(1, settings.llm_max_workers), len(items))
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [
            ex.submit(
                chat, it["node"], it["system"], it["user"],
                json_mode=json_mode, settings=settings,
            )
            for it in items
        ]
        return [f.result() for f in futs]


def chat_many(
    items: List[Dict[str, Any]],
    *,
    settings: Optional[Settings] = None,
) -> List[str]:
    """并发的纯文本生成，见 `_run_many`。"""
    return _run_many(items, json_mode=False, settings=settings)


def chat_json_many(
    items: List[Dict[str, Any]],
    *,
    default: Any = None,
    settings: Optional[Settings] = None,
) -> List[Any]:
    """并发的 JSON 调用：每个 item 单独走 chat_json 的「失败重试一次」逻辑。

    返回与 items 等长的结果列表，单条失败返回 default（不抛，避免一个坏项拖垮整批）。
    """
    settings = settings or get_settings()
    if settings.llm_provider == "stub" or len(items) == 1:
        return [
            chat_json(it["node"], it["system"], it["user"], default=default, settings=settings)
            for it in items
        ]

    def _one(it: Dict[str, Any]) -> Any:
        try:
            return chat_json(it["node"], it["system"], it["user"], default=default, settings=settings)
        except Exception:  # noqa: BLE001 - 单条失败不影响整批
            logger.exception("[%s] 并发 JSON 调用失败，返回默认值", it.get("node"))
            return default

    workers = min(max(1, settings.llm_max_workers), len(items))
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(_one, it) for it in items]
        return [f.result() for f in futs]
