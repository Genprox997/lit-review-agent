"""全局配置：模型接入、检索源开关、路径与礼貌策略参数。

所有配置项均可通过环境变量或项目根目录的 `.env` 覆盖，见 `.env.example`。
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

try:  # python-dotenv 是可选依赖，缺失时退化为纯环境变量
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    def load_dotenv(*_args, **_kwargs):  # type: ignore
        return False

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")


# --------------------------------------------------------------------------
# 环境变量读取辅助
# --------------------------------------------------------------------------
def _env(key: str, default: str = "") -> str:
    return os.getenv(key, default).strip()


def _env_int(key: str, default: int) -> int:
    raw = _env(key)
    try:
        return int(raw) if raw else default
    except ValueError:
        return default


def _env_float(key: str, default: float) -> float:
    raw = _env(key)
    try:
        return float(raw) if raw else default
    except ValueError:
        return default


def _env_bool(key: str, default: bool) -> bool:
    raw = _env(key).lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "y", "on"}


def _env_list(key: str, default: List[str]) -> List[str]:
    raw = _env(key)
    if not raw:
        return list(default)
    return [item.strip() for item in raw.split(",") if item.strip()]


# --------------------------------------------------------------------------
# 配置对象
# --------------------------------------------------------------------------
@dataclass
class Settings:
    """运行时配置快照。"""

    # ---- LLM ----
    llm_provider: str = field(default_factory=lambda: _env("LLM_PROVIDER", "deepseek").lower())
    llm_temperature: float = field(default_factory=lambda: _env_float("LLM_TEMPERATURE", 0.3))
    llm_max_retries: int = field(default_factory=lambda: _env_int("LLM_MAX_RETRIES", 2))
    llm_timeout: int = field(default_factory=lambda: _env_int("LLM_TIMEOUT", 120))

    # ---- 学术 API 礼貌策略 ----
    contact_email: str = field(default_factory=lambda: _env("CONTACT_EMAIL", "you@example.com"))
    s2_api_key: str = field(default_factory=lambda: _env("SEMANTIC_SCHOLAR_API_KEY"))
    enabled_sources: List[str] = field(
        default_factory=lambda: _env_list("ENABLED_SOURCES", ["arxiv", "openalex"])
    )

    # ---- 检索与流程 ----
    max_results_per_query: int = field(default_factory=lambda: _env_int("MAX_RESULTS_PER_QUERY", 25))
    target_paper_count: int = field(default_factory=lambda: _env_int("TARGET_PAPER_COUNT", 40))
    max_retrieval_rounds: int = field(default_factory=lambda: _env_int("MAX_RETRIEVAL_ROUNDS", 3))
    max_critic_rounds: int = field(default_factory=lambda: _env_int("MAX_CRITIC_ROUNDS", 2))
    top_n_fulltext: int = field(default_factory=lambda: _env_int("TOP_N_FULLTEXT", 8))
    n_clusters: int = field(default_factory=lambda: _env_int("N_CLUSTERS", 0))
    min_year: int = field(default_factory=lambda: _env_int("MIN_YEAR", 0))

    # ---- 相关性闸门与排序（P0-1）----
    relevance_gate: float = field(
        default_factory=lambda: _env_float("RELEVANCE_GATE", 0.10)
    )
    relevance_weight: float = field(
        default_factory=lambda: _env_float("RELEVANCE_WEIGHT", 0.55)
    )
    citation_weight: float = field(
        default_factory=lambda: _env_float("CITATION_WEIGHT", 0.20)
    )
    recency_weight: float = field(
        default_factory=lambda: _env_float("RECENCY_WEIGHT", 0.15)
    )
    coverage_weight: float = field(
        default_factory=lambda: _env_float("COVERAGE_WEIGHT", 0.10)
    )
    min_pool_after_gate: int = field(
        default_factory=lambda: _env_int("MIN_POOL_AFTER_GATE", 20)
    )

    # ---- 输出 ----
    output_dir: Path = field(default_factory=lambda: PROJECT_ROOT / _env("OUTPUT_DIR", "output"))
    cache_dir: Path = field(default_factory=lambda: PROJECT_ROOT / _env("CACHE_DIR", ".cache"))
    report_language: str = field(default_factory=lambda: _env("REPORT_LANGUAGE", "zh").lower())
    enable_fulltext: bool = field(default_factory=lambda: _env_bool("ENABLE_FULLTEXT", True))

    # ---- 持久化 ----
    checkpoint_backend: str = field(default_factory=lambda: _env("CHECKPOINT_BACKEND", "memory").lower())
    checkpoint_path: str = field(default_factory=lambda: _env("CHECKPOINT_PATH", ".cache/checkpoints.sqlite"))

    # ---- 可观测（LangSmith，可选）----
    langsmith_api_key: str = field(default_factory=lambda: _env("LANGSMITH_API_KEY"))
    langsmith_project: str = field(default_factory=lambda: _env("LANGSMITH_PROJECT", "lit-review-agent"))
    langsmith_tracing: bool = field(
        default_factory=lambda: _env_bool("LANGSMITH_TRACING", _env_bool("LANGCHAIN_TRACING_V2", False))
    )

    # ------------------------------------------------------------------
    @property
    def user_agent(self) -> str:
        """学术 API 要求 UA 中带联系方式。"""
        return f"lit-review-agent/0.1 (mailto:{self.contact_email})"

    def model_config(self) -> dict:
        """按 provider 返回 OpenAI 兼容的模型参数。"""
        p = self.llm_provider
        if p == "deepseek":
            return {
                "model": _env("DEEPSEEK_MODEL", "deepseek-chat"),
                "api_key": _env("DEEPSEEK_API_KEY"),
                "base_url": _env("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
            }
        if p == "openai":
            return {
                "model": _env("OPENAI_MODEL", "gpt-4o-mini"),
                "api_key": _env("OPENAI_API_KEY"),
                "base_url": _env("OPENAI_BASE_URL", "https://api.openai.com/v1"),
            }
        if p == "ollama":
            return {
                "model": _env("OLLAMA_MODEL", "qwen2.5:7b"),
                "api_key": "ollama",  # Ollama 不校验，但 OpenAI SDK 要求非空
                "base_url": _env("OLLAMA_BASE_URL", "http://localhost:11434/v1"),
            }
        if p == "stub":
            return {"model": "stub", "api_key": "stub", "base_url": ""}
        raise ValueError(
            f"未知的 LLM_PROVIDER={p!r}，可选：deepseek / openai / ollama / stub"
        )

    def ensure_dirs(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        (self.cache_dir / "pdf").mkdir(parents=True, exist_ok=True)

    def apply_langsmith_env(self) -> bool:
        """把 LangSmith 配置透传为标准环境变量，使 langgraph 自动上报轨迹。

        返回是否启用了追踪。仅在提供 API key 时打开 `LANGCHAIN_TRACING_V2`，
        否则保持关闭以免无 key 时报错。
        """
        if not self.langsmith_api_key:
            return False
        if self.langsmith_tracing or os.getenv("LANGCHAIN_TRACING_V2") != "true":
            os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_API_KEY"] = self.langsmith_api_key
        if self.langsmith_project:
            os.environ["LANGCHAIN_PROJECT"] = self.langsmith_project
        logger = logging.getLogger(__name__)
        logger.info("LangSmith 追踪已开启（project=%s）", self.langsmith_project)
        return True

    def validate(self) -> List[str]:
        """返回配置告警列表（不抛异常，交由调用方决定是否阻断）。"""
        warns: List[str] = []
        if self.llm_provider != "stub" and not self.model_config().get("api_key"):
            warns.append(
                f"LLM_PROVIDER={self.llm_provider} 但未配置 API key，"
                f"请在 .env 中填写；或设置 LLM_PROVIDER=stub 离线试跑。"
            )
        if self.contact_email in {"", "you@example.com"}:
            warns.append(
                "CONTACT_EMAIL 仍是占位值。学术 API 靠 mailto 识别善意机器人，"
                "建议填真实邮箱以获得更高配额。"
            )
        unknown = set(self.enabled_sources) - {
            "arxiv", "openalex", "semantic_scholar", "pubmed", "crossref"
        }
        if unknown:
            warns.append(f"ENABLED_SOURCES 含未知源：{sorted(unknown)}")
        return warns


_settings: Settings | None = None


def get_settings(refresh: bool = False) -> Settings:
    """获取全局配置单例。"""
    global _settings
    if _settings is None or refresh:
        _settings = Settings()
        _settings.apply_langsmith_env()
    return _settings
