"""真 LLM 端到端冒烟测试。

默认跳过：需要环境变量 `DEEPSEEK_API_KEY` 才运行（会真实消耗 token）。
本地用 `pytest -m "not network"` 跑离线套件时不会触及此文件——
本文件无 `network` marker，但被 `skipif` 拦截，故离线也安全。

运行：
    DEEPSEEK_API_KEY=sk-xxx pytest tests/test_real_llm.py
"""

from __future__ import annotations

import os

import pytest

pytestmark = pytest.mark.skipif(
    not (os.getenv("DEEPSEEK_API_KEY") and os.getenv("RUN_REAL_LLM_TESTS")),
    reason="需要 DEEPSEEK_API_KEY 且 RUN_REAL_LLM_TESTS=1 才运行真 LLM 测试（默认跳过，避免联网挂起/耗 token）",
)


def test_real_llm_chat_smoke():
    """确认真实 DeepSeek 能返回非空文本。"""
    from src.agent.llm import chat

    out = chat(
        "synthesizer",
        "严格只输出一个英文单词：ok",
        "请输出一个词。",
    )
    assert isinstance(out, str) and len(out.strip()) > 0


def test_real_review_end_to_end():
    """用真实 DeepSeek 跑一次完整综述（arxiv 单源、stub 外环），校验产物结构。"""
    from src.agent.graph import run_review

    final = run_review(
        topic="deformable mirror wavefront control",
        constraints="",
        thread_id="real-llm-smoke",
        with_human=False,
        stream=False,
    )
    assert not final.get("interrupted")
    artifacts = final.get("artifacts") or {}
    assert artifacts, "未生成成稿（检索可能为空或 LLM 调用失败）"
    assert len(final.get("papers") or []) > 0
