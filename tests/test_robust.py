"""长任务健壮性（方向 G'）测试：节点级错误隔离、LLM 瞬时错误重试、超时看门狗最佳努力定稿。

全程离线（stub LLM + 假检索），不触网。
"""

from __future__ import annotations

import os

import pytest

from src.agent.llm import _is_retryable_error
from src.agent.robust import node_guard
from src.agent.state import initial_state
from src.config import get_settings
from src.ingest.base import make_paper


# --------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def stub_env(monkeypatch, tmp_path):
    """强制 stub LLM + 关闭全文下载 + 输出到临时目录。"""
    monkeypatch.setenv("LLM_PROVIDER", "stub")
    monkeypatch.setenv("ENABLE_FULLTEXT", "false")
    monkeypatch.setenv("RELEVANCE_GATE", "0")
    monkeypatch.setenv("CONTACT_EMAIL", "test@example.com")
    monkeypatch.setenv("OUTPUT_DIR", str(tmp_path / "out"))
    monkeypatch.setenv("CACHE_DIR", str(tmp_path / "cache"))
    settings = get_settings(refresh=True)
    settings.output_dir = tmp_path / "out"
    settings.cache_dir = tmp_path / "cache"
    yield settings
    get_settings(refresh=True)


@pytest.fixture(autouse=True)
def no_embedding_download(monkeypatch):
    """离线测试强制 TF-IDF 降级，避免下载 sentence-transformers 模型。"""
    import src.cluster.theme_cluster as TC

    monkeypatch.setattr(TC, "_EMBEDDER_TRIED", True)
    monkeypatch.setattr(TC, "_EMBEDDER", None)


@pytest.fixture
def offline_retrieval(monkeypatch):
    """拦截所有网络调用。"""
    import src.agent.nodes as N

    def _fake_pool(n: int = 24):
        out = []
        for i in range(n):
            out.append(make_paper(
                paper_id=f"arxiv:p.{i}", title=f"Study on topic variant {i}",
                abstract="deep learning method benchmark evaluation improvement study",
                year=2019 + i % 6, citation_count=100 - i, source="arxiv",
                authors=[f"Author {i}"], pdf_url=f"https://arxiv.org/pdf/p{i}",
                matched_queries=["topic"],
            ))
        return out

    monkeypatch.setattr(N, "multi_source_search", lambda queries, settings=None: _fake_pool())
    monkeypatch.setattr(N, "enrich_citations", lambda papers, limit=25: 0)
    monkeypatch.setattr(N, "enrich_topn_fulltext", lambda papers, settings=None: 0)


# --------------------------------------------------------------------------
# node_guard：错误隔离
# --------------------------------------------------------------------------
def test_node_guard_catches_exception_records_error():
    """节点抛异常时，guard 应记录结构化错误并以安全空更新继续。"""

    def bad(state):
        raise RuntimeError("kaboom: simulated failure")

    guarded = node_guard("extractor", bad)
    out = guarded({"papers": []})

    assert "run_errors" in out
    err = out["run_errors"][0]
    assert err["node"] == "extractor"
    assert "kaboom" in err["error"]
    assert err["kind"] == "RuntimeError"
    assert "time" in err and err["time"]
    assert any("extractor" in line for line in out.get("logs", []))


def test_node_guard_passes_through_interrupt():
    """HITL 的 interrupt() 必须原样透传，不能被 guard 吞掉。"""
    from langgraph.errors import GraphInterrupt

    def h(state):
        raise GraphInterrupt({"report_path": "/tmp/x.md"})

    guarded = node_guard("human_review", h)
    with pytest.raises(GraphInterrupt):
        guarded({})


def test_node_guard_success_passthrough():
    """正常节点应原样返回更新字典。"""
    guarded = node_guard("q", lambda s: {"queries": ["a"]})
    assert guarded({}) == {"queries": ["a"]}


# --------------------------------------------------------------------------
# LLM 瞬时错误重试判定
# --------------------------------------------------------------------------
def test_is_retryable_classification():
    """429 / 5xx / 连接重置 / 超时 应判定为可重试；4xx 客户端错误 / 普通异常不应。"""
    # 瞬时故障：应重试
    assert _is_retryable_error(Exception("Connection reset by peer"))
    assert _is_retryable_error(Exception("Request timeout"))
    assert _is_retryable_error(Exception("503 Service Unavailable"))
    assert _is_retryable_error(Exception("502 Bad Gateway"))
    assert _is_retryable_error(Exception("rate limit exceeded 429"))
    assert _is_retryable_error(Exception("timeout waiting for upstream"))

    # 不可重试：客户端错误 / 业务逻辑异常
    assert not _is_retryable_error(Exception("400 Bad Request"))
    assert not _is_retryable_error(Exception("404 Not Found"))
    assert not _is_retryable_error(ValueError("invalid json"))


# --------------------------------------------------------------------------
# 端到端：单节点失败仍产出最佳努力成稿
# --------------------------------------------------------------------------
def test_node_failure_isolated_end_to_end(stub_env, offline_retrieval, monkeypatch):
    """extractor 抛异常被 guard 降级，图仍跑完并成稿，run_errors 记录该节点失败。"""
    import src.agent.graph as G
    from src.agent.graph import build_graph

    stub_env.target_paper_count = 10
    stub_env.max_retrieval_rounds = 1

    def failing_extractor(state):
        raise RuntimeError("boom: simulated extractor failure")

    monkeypatch.setattr(G, "extractor", failing_extractor)
    graph = build_graph()

    final = graph.invoke(
        initial_state("robustness topic"),
        {"configurable": {"thread_id": "test-node-fail"}, "recursion_limit": 60},
    )

    # 关键：图没有崩溃，产出了成稿
    assert final.get("report"), "单节点失败不应中断成稿"
    assert final.get("artifacts"), "应落盘产物"

    # 失败被隔离并记录
    errs = final.get("run_errors") or []
    assert any(e["node"] == "extractor" for e in errs), "应记录 extractor 节点失败"
    # 不应出现 synthesizer 失败（定稿本身成功）
    assert not any(e["node"] == "synthesizer" for e in errs)


# --------------------------------------------------------------------------
# 端到端：超时看门狗生成最佳努力成稿
# --------------------------------------------------------------------------
def test_timeout_watchdog_best_effort(stub_env, offline_retrieval, monkeypatch):
    """运行超过 RUN_TIMEOUT_SECONDS 时，应中断流式循环并产出最佳努力成稿。"""
    import src.agent.graph as G
    from src.agent.graph import run_review

    monkeypatch.setenv("RUN_TIMEOUT_SECONDS", "1")
    get_settings(refresh=True)
    stub_env.target_paper_count = 10
    stub_env.max_retrieval_rounds = 1

    real_qe = G.query_expander

    def slow_query_expander(state):
        import time as _t

        _t.sleep(2)  # 故意超过 1s deadline
        return real_qe(state)

    monkeypatch.setattr(G, "query_expander", slow_query_expander)

    final = run_review("timeout topic", stream=True)

    assert final.get("timed_out") is True, "应标记 timed_out"
    errs = final.get("run_errors") or []
    assert any(e["node"] == "watchdog" for e in errs), "watchdog 应记录一条超时错误"
    assert final.get("report"), "超时后应产出最佳努力成稿"
    for path in final.get("artifacts", {}).values():
        assert os.path.exists(path), f"产物缺失: {path}"


def test_no_timeout_when_disabled(stub_env, offline_retrieval, monkeypatch):
    """RUN_TIMEOUT_SECONDS=0 时关闭看门狗，正常跑完且 timed_out=False。"""
    import src.agent.graph as G
    from src.agent.graph import run_review

    monkeypatch.setenv("RUN_TIMEOUT_SECONDS", "0")
    get_settings(refresh=True)
    stub_env.target_paper_count = 10
    stub_env.max_retrieval_rounds = 1

    final = run_review("normal topic", stream=True)
    assert final.get("timed_out") is not True
    assert final.get("report")
