"""状态机与节点单测：全程离线（stub LLM + 假检索），不触网。"""

from __future__ import annotations

import os

import pytest

from src.agent import nodes as N
from src.agent.graph import build_graph, route_after_critic, route_after_retrieval
from src.agent.llm import parse_json
from src.agent.state import initial_state
from src.ingest.base import make_paper
from src.config import get_settings


# --------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def stub_env(monkeypatch, tmp_path):
    """强制 stub LLM + 关闭全文下载 + 输出到临时目录。"""
    monkeypatch.setenv("LLM_PROVIDER", "stub")
    monkeypatch.setenv("ENABLE_FULLTEXT", "false")
    monkeypatch.setenv("CONTACT_EMAIL", "test@example.com")
    monkeypatch.setenv("OUTPUT_DIR", str(tmp_path / "out"))
    monkeypatch.setenv("CACHE_DIR", str(tmp_path / "cache"))
    settings = get_settings(refresh=True)
    settings.output_dir = tmp_path / "out"
    settings.cache_dir = tmp_path / "cache"
    yield settings
    get_settings(refresh=True)


def _fake_pool(n: int = 24):
    """构造两簇语义可分的假文献。"""
    out = []
    for i in range(n // 2):
        out.append(make_paper(
            paper_id=f"arxiv:vision.{i}", title=f"Convolutional network for image recognition {i}",
            abstract="convolutional neural network image recognition vision benchmark imagenet",
            year=2018 + i % 6, citation_count=100 - i, source="arxiv",
            authors=[f"Author {i}"], pdf_url=f"https://arxiv.org/pdf/v{i}",
            matched_queries=["computer vision"],
        ))
    for i in range(n - n // 2):
        out.append(make_paper(
            paper_id=f"doi:10.1/nlp.{i}", doi=f"10.1/nlp.{i}",
            title=f"Transformer language model pretraining study {i}",
            abstract="transformer language model pretraining corpus tokens translation",
            year=2019 + i % 6, citation_count=200 - i, source="openalex",
            authors=[f"Writer {i}"], venue="Proceedings of ACL",
            matched_queries=["language model"],
        ))
    return out


@pytest.fixture
def offline_retrieval(monkeypatch):
    """拦截所有网络调用。"""
    monkeypatch.setattr(N, "multi_source_search", lambda queries, settings=None: _fake_pool())
    monkeypatch.setattr(N, "enrich_citations", lambda papers, limit=25: 0)
    monkeypatch.setattr(N, "enrich_topn_fulltext", lambda papers, settings=None: 0)


# --------------------------------------------------------------------------
# JSON 容错解析
# --------------------------------------------------------------------------
@pytest.mark.parametrize("raw,expected", [
    ('{"a": 1}', {"a": 1}),
    ('```json\n{"a": 1}\n```', {"a": 1}),
    ('好的，结果如下：\n{"a": 1}\n希望有帮助', {"a": 1}),
    ('{"a": [1, 2,]}', {"a": [1, 2]}),           # 尾随逗号
    ('{"a": "带 } 的字符串"}', {"a": "带 } 的字符串"}),  # 字符串内的括号不算嵌套
])
def test_parse_json_variants(raw, expected):
    assert parse_json(raw) == expected


def test_parse_json_failure_returns_default():
    assert parse_json("完全不是 JSON", default={"fallback": True}) == {"fallback": True}


# --------------------------------------------------------------------------
# 引用防幻觉
# --------------------------------------------------------------------------
def test_strip_invalid_citations():
    text = "方法 A 有效[1][99]，方法 B 更好[2]。"
    out = N._strip_invalid_citations(text, {1, 2})
    assert "[99]" not in out
    assert "[1]" in out and "[2]" in out


def test_split_abstract_intro():
    raw = "## 摘要\n这是摘要正文。\n\n## 引言\n这是引言正文。"
    abstract, intro = N._split_abstract_intro(raw)
    assert abstract == "这是摘要正文。"
    assert intro == "这是引言正文。"


def test_split_abstract_intro_degraded():
    """模型未按格式输出时，全部内容当摘要，引言留空而不是崩溃。"""
    abstract, intro = N._split_abstract_intro("只有一段话。")
    assert abstract == "只有一段话。"
    assert intro == ""


# --------------------------------------------------------------------------
# 路由
# --------------------------------------------------------------------------
def test_route_inner_loop_when_pool_too_small(stub_env):
    stub_env.target_paper_count = 40
    stub_env.max_retrieval_rounds = 3
    state = {"papers": [1] * 10, "retrieval_round": 1}
    assert route_after_retrieval(state) == "retriever"


def test_route_inner_loop_stops_at_round_cap(stub_env):
    stub_env.target_paper_count = 40
    stub_env.max_retrieval_rounds = 3
    state = {"papers": [1] * 10, "retrieval_round": 3}
    assert route_after_retrieval(state) == "ranker"


def test_route_inner_loop_ends_when_nothing_found(stub_env):
    stub_env.max_retrieval_rounds = 1
    assert route_after_retrieval({"papers": [], "retrieval_round": 1}) == "__end__"


def test_route_outer_loop_on_need_more(stub_env):
    """max=2 允许 2 次打回，故第 1、2 次评审都可回到扩词。"""
    stub_env.max_critic_rounds = 2
    for round_no in (1, 2):
        state = {"critic": {"verdict": "need_more"}, "critic_round": round_no}
        assert route_after_critic(state) == "query_expander"


def test_route_outer_loop_stops_at_cap(stub_env):
    stub_env.max_critic_rounds = 2
    state = {"critic": {"verdict": "need_more"}, "critic_round": 3}
    assert route_after_critic(state) == "gap_analyzer"


def test_route_outer_loop_on_pass(stub_env):
    assert route_after_critic({"critic": {"verdict": "pass"}, "critic_round": 1}) == "gap_analyzer"


# --------------------------------------------------------------------------
# 单节点
# --------------------------------------------------------------------------
def test_query_expander_produces_queries(stub_env):
    out = N.query_expander(initial_state("graph neural networks"))
    assert len(out["queries"]) >= 3
    assert out["pending_queries"] == out["queries"]
    assert len(set(q.lower() for q in out["queries"])) == len(out["queries"])  # 无重复


def test_query_expander_refine_skips_existing(stub_env):
    state = initial_state("graph neural networks")
    state["queries"] = ["graph neural networks", "graph neural networks survey"]
    state["critic"] = {"verdict": "need_more", "missing_topics": ["dynamic graphs"],
                       "extra_queries": ["dynamic graph learning"]}
    out = N.query_expander(state)
    assert "dynamic graph learning" in out["pending_queries"]
    # 已检索过的不再重复
    assert "graph neural networks survey" not in out["pending_queries"]


def test_extractor_falls_back_to_abstract(stub_env):
    """LLM 抽取失败时应用摘要兜底，保证每篇文献都有证据。"""
    state = initial_state("t")
    state["papers"] = [make_paper(paper_id="p1", title="T", abstract="An informative abstract.")]
    state["evidence"] = []
    out = N.extractor(state)
    assert len(out["evidence"]) == 1
    assert out["evidence"][0]["paper_id"] == "p1"
    assert out["evidence"][0]["claim"]


def test_extractor_skips_already_extracted(stub_env):
    state = initial_state("t")
    state["papers"] = [make_paper(paper_id="p1", title="T", abstract="A")]
    state["evidence"] = [{"paper_id": "p1", "claim": "done"}]
    out = N.extractor(state)
    assert "evidence" not in out  # 无新增


def test_clusterer_assigns_unique_labels_and_citations(stub_env):
    state = initial_state("t")
    state["papers"] = _fake_pool(12)
    state["evidence"] = [{"paper_id": p["paper_id"], "claim": "c"} for p in state["papers"]]
    out = N.clusterer(state)

    labels = [c["label"] for c in out["clusters"]]
    assert len(labels) == len(set(labels))                      # 标题互斥
    assert all(labels)                                          # 无空标题
    nums = sorted(out["citation_map"].values())
    assert nums == list(range(1, len(nums) + 1))                # 引用编号连续且从 1 开始


# --------------------------------------------------------------------------
# 端到端
# --------------------------------------------------------------------------
def test_full_graph_end_to_end(stub_env, offline_retrieval, tmp_path):
    stub_env.target_paper_count = 10
    stub_env.max_retrieval_rounds = 1

    graph = build_graph()
    final = graph.invoke(
        initial_state("multimodal representation learning"),
        {"configurable": {"thread_id": "test-e2e"}, "recursion_limit": 60},
    )

    # 各阶段产物齐全
    assert len(final["papers"]) == 24
    assert len(final["evidence"]) == 24
    assert len(final["clusters"]) >= 2
    assert len(final["sections"]) >= 2
    assert final["gaps"] and final["trends"]
    assert final["critic"]["verdict"] == "pass"

    # 成稿结构完整
    report = final["report"]
    for heading in ("## 摘要", "## 1. 引言", "参考文献", "附录 A"):
        assert heading in report, f"成稿缺少 {heading}"

    # 引用可追溯：正文出现的每个 [n] 都必须在参考文献表里
    import re
    body, refs = report.split("参考文献", 1)
    for num in set(re.findall(r"\[(\d+)\]", body)):
        assert f"**[{num}]**" in refs, f"引用 [{num}] 在参考文献表中不存在"

    # BibTeX 与产物落盘
    assert final["bibtex"].count("@") >= len(final["citation_map"])
    for path in final["artifacts"].values():
        assert os.path.exists(path), f"产物缺失: {path}"


def test_graph_with_human_interrupt(stub_env, offline_retrieval):
    """开启人工审核后，图应在 human_review 之前挂起。"""
    stub_env.target_paper_count = 10
    stub_env.max_retrieval_rounds = 1

    graph = build_graph(with_human=True)
    config = {"configurable": {"thread_id": "test-human"}, "recursion_limit": 60}
    state = graph.invoke(initial_state("test topic"), config)

    assert state["report"], "挂起前应已完成成稿"
    snapshot = graph.get_state(config)
    assert snapshot.next == ("human_review",), f"应挂起在 human_review，实际 {snapshot.next}"

    resumed = graph.invoke(None, config)  # 续跑收尾
    assert any("Human" in log for log in resumed["logs"])


def test_critic_loop_triggers_second_retrieval(stub_env, offline_retrieval, monkeypatch):
    """Critic 判定 need_more 时，应回到 QueryExpander 并再跑一轮检索。"""
    stub_env.target_paper_count = 10
    stub_env.max_retrieval_rounds = 1
    stub_env.max_critic_rounds = 1

    calls = {"n": 0}
    real_critic = N.critic

    def flaky_critic(state):
        calls["n"] += 1
        out = real_critic(state)
        if calls["n"] == 1:  # 第一轮强制打回
            out["critic"] = {"verdict": "need_more", "coverage_score": 4,
                             "missing_topics": ["cross-modal alignment"],
                             "extra_queries": ["cross modal alignment"], "comments": "补文献"}
        return out

    # graph.py 在导入时就绑定了节点函数对象，须在 graph 模块命名空间打补丁
    import src.agent.graph as G

    monkeypatch.setattr(G, "critic", flaky_critic)
    graph = build_graph()
    final = graph.invoke(
        initial_state("multimodal learning"),
        {"configurable": {"thread_id": "test-loop"}, "recursion_limit": 60},
    )

    assert calls["n"] == 2, "外环应触发第二轮评审"
    assert final["critic_round"] == 2
    assert "cross modal alignment" in final["queries"]
    assert final["report"]
