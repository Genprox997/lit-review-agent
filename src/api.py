"""FastAPI 入口（设计文档 §3：CLI / FastAPI 入口）。

把 `run_review` 包成 HTTP 服务，便于把综述能力嵌进其它系统或做 Web 前端。

启动：
    pip install -e ".[api]"
    uvicorn src.api:app --host 0.0.0.0 --port 8000

调用：
    POST /review  {"topic": "diffusion models for image super-resolution"}
    GET  /healthz
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def create_app():
    """惰性构造 FastAPI app；未安装 fastapi 时抛出清晰错误。"""
    try:
        from fastapi import FastAPI, HTTPException
        from pydantic import BaseModel, Field
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "FastAPI 入口未启用：请先 `pip install -e \".[api]\"` 安装 fastapi/uvicorn。"
        ) from exc

    app = FastAPI(
        title="lit-review-agent",
        version="0.1.0",
        description="给定研究主题，自动检索、聚类并撰写带引用的文献综述。",
    )

    class ReviewRequest(BaseModel):
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

    @app.get("/healthz")
    def healthz() -> Dict[str, str]:
        return {"status": "ok"}

    @app.post("/review")
    def review(req: ReviewRequest) -> Dict[str, Any]:
        # 把请求参数写回环境变量，复用 CLI 的同一套配置分发逻辑
        import os

        overrides = {
            "ENABLED_SOURCES": req.sources,
            "TARGET_PAPER_COUNT": req.target,
            "MAX_RESULTS_PER_QUERY": req.per_query,
            "MIN_YEAR": req.min_year,
            "TOP_N_FULLTEXT": req.top_fulltext,
            "N_CLUSTERS": req.clusters,
            "REPORT_LANGUAGE": req.lang,
            "LLM_PROVIDER": req.provider,
        }
        for key, value in overrides.items():
            if value is not None:
                os.environ[key] = str(value)

        from src.agent.graph import run_review
        from src.config import get_settings

        settings = get_settings(refresh=True)
        if settings.llm_provider != "stub" and not settings.model_config().get("api_key"):
            raise HTTPException(
                status_code=400,
                detail=f"LLM_PROVIDER={settings.llm_provider} 但未配置 API key；"
                       f"或设置 provider=stub 离线试跑。",
            )

        try:
            final = run_review(
                topic=req.topic,
                constraints=req.constraints,
                thread_id=f"api-{abs(hash(req.topic)) % 10**9}",
                with_human=False,
                stream=False,
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

    return app


try:
    app = create_app()
except RuntimeError as exc:  # pragma: no cover
    logger.warning("FastAPI app 未创建：%s", exc)
    app = None  # type: ignore[assignment]


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("src.api:app", host="0.0.0.0", port=8000, reload=False)
