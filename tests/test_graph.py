"""状态机与节点单测：全程离线（stub LLM + 假检索），不触网。"""

from __future__ import annotations

import os

import pytest

from src.agent import nodes as N
from src.agent.graph import (
    build_graph,
    route_after_critic,
    route_after_human,
    route_after_retrieval,
)
from src.agent.llm import parse_json
from src.agent.state import initial_state
from src.ingest.base import make_paper
from src.config import get_settings
from langgraph.graph import END

import src.cluster.theme_cluster as TC


# --------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def stub_env(monkeypatch, tmp_path):
    """强制 stub LLM + 关闭全文下载 + 输出到临时目录。"""
    monkeypatch.setenv("LLM_PROVIDER", "stub")
    monkeypatch.setenv("ENABLE_FULLTEXT", "false")
    monkeypatch.setenv("RELEVANCE_GATE", "0")   # 离线桩测试关闭相关性闸门（闸门逻辑见 test_p0.py）
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
    monkeypatch.setattr(TC, "_EMBEDDER_TRIED", True)
    monkeypatch.setattr(TC, "_EMBEDDER", None)


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
    """开启人工审核后，图应在 human_review 内部挂起，且可用 Command(resume=...) 续跑。"""
    stub_env.target_paper_count = 10
    stub_env.max_retrieval_rounds = 1

    graph = build_graph(with_human=True)
    config = {"configurable": {"thread_id": "test-human"}, "recursion_limit": 60}
    state = graph.invoke(initial_state("test topic"), config)

    assert state["report"], "挂起前应已完成成稿"
    assert "__interrupt__" in state, "应在 human_review 处挂起（含 __interrupt__）"
    snapshot = graph.get_state(config)
    assert snapshot.next == ("human_review",), f"应挂起在 human_review，实际 {snapshot.next}"

    from langgraph.types import Command
    resumed = graph.invoke(Command(resume="approve"), config)  # 续跑收尾
    assert any("Human" in log for log in resumed["logs"])


def test_run_review_human_resume(stub_env, offline_retrieval):
    """run_review 在 --human 时挂起，同一 thread_id + feedback 续跑可定稿（断点续跑）。"""
    from src.agent.graph import run_review

    stub_env.target_paper_count = 10
    stub_env.max_retrieval_rounds = 1
    tid = "resume-e2e"

    # 第一次：开启 human，应在 human_review 处挂起
    first = run_review("a test topic", thread_id=tid, with_human=True, stream=False)
    assert first.get("report"), "首次运行应已完成成稿并挂起"
    assert first.get("interrupted"), "首次运行应标记为挂起态"

    # 续跑：同 thread_id，用 feedback 恢复
    second = run_review(
        "a test topic", thread_id=tid, with_human=True,
        feedback="approve", stream=False,
    )
    assert not second.get("interrupted"), "续跑不应再挂起"
    assert any("Human" in log for log in second.get("logs", [])), "应记录人工审核节点"
    assert second.get("report"), "续跑应产出最终成稿"
    for path in second.get("artifacts", {}).values():
        assert os.path.exists(path), f"产物缺失: {path}"


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


# --------------------------------------------------------------------------
# 方向 A：引用编号跨轮次稳定
# --------------------------------------------------------------------------
def test_clusterer_preserves_existing_citation_numbers(stub_env):
    """Critic 打回重聚类后，旧论文的引用编号应保持不变，新论文仅追加。"""
    st = initial_state("t")
    st["papers"] = _fake_pool(12)
    st["evidence"] = [{"paper_id": p["paper_id"], "claim": "c"} for p in st["papers"]]
    m1 = dict(N.clusterer(st)["citation_map"])

    new_paper = make_paper(
        paper_id="arxiv:new.99", title="New method paper", abstract="novel approach benchmark",
        year=2023, citation_count=10, source="arxiv", authors=["N"], matched_queries=["x"],
    )
    st2 = initial_state("t")
    st2["papers"] = st["papers"] + [new_paper]
    st2["evidence"] = st["evidence"] + [{"paper_id": "arxiv:new.99", "claim": "c"}]
    st2["citation_map"] = m1  # 携带上一轮编号
    m2 = N.clusterer(st2)["citation_map"]

    for pid, num in m1.items():
        assert m2.get(pid) == num, f"{pid} 的引用编号应跨轮次保持不变"
    assert m2.get("arxiv:new.99") == max(m1.values()) + 1, "新论文应追加在已用编号之后"


# --------------------------------------------------------------------------
# 方向 A：Human-in-the-loop 闭环改写路由
# --------------------------------------------------------------------------
def test_route_after_human_approve_ends(stub_env):
    assert route_after_human({"human_feedback": "", "human_round": 0}) == END
    assert route_after_human({"human_feedback": "approve", "human_round": 0}) == END


def test_route_after_human_feedback_triggers_rewrite(stub_env):
    stub_env.max_human_rounds = 2
    assert route_after_human({"human_feedback": "补充对比实验", "human_round": 0}) == "parse_human_feedback"


def test_route_after_human_respects_cap(stub_env):
    stub_env.max_human_rounds = 2
    assert route_after_human({"human_feedback": "再改一次", "human_round": 2}) == END


def test_parse_human_feedback_produces_targets(stub_env):
    st = initial_state("t")
    st["sections"] = {"视觉识别路线": "x", "语言模型路线": "y"}
    st["human_feedback"] = "请在视觉识别路线补充与对比方法的实验分析"
    out = N.parse_human_feedback(st)
    targets = out["rewrite_targets"]
    assert len(targets) >= 1
    assert targets[0]["action"] in ("rewrite", "add")
    assert targets[0]["section"] or targets[0]["instruction"]


def test_rewrite_sections_updates_only_target(stub_env):
    st = initial_state("vision")
    p1 = make_paper(paper_id="arxiv:1", title="Vision A", abstract="vision recognition",
                    citation_count=50, year=2020)
    p2 = make_paper(paper_id="doi:10.1/b", title="Vision B", abstract="vision transformer",
                    citation_count=80, year=2021)
    st["papers"] = [p1, p2]
    st["citation_map"] = {"arxiv:1": 1, "doi:10.1/b": 2}
    st["evidence"] = [
        {"paper_id": "arxiv:1", "claim": "CNN 提升识别", "method": "m", "dataset": "d", "metric": "k", "section": ""},
        {"paper_id": "doi:10.1/b", "claim": "ViT 领先", "method": "m", "dataset": "d", "metric": "k", "section": ""},
    ]
    st["clusters"] = [{"cluster_id": 0, "label": "视觉识别路线", "keywords": ["vision"],
                       "paper_ids": ["arxiv:1", "doi:10.1/b"], "size": 2}]
    st["sections"] = {"视觉识别路线": "原内容 [1][2]。", "其它小节": "保持不变 [1]。"}
    st["rewrite_targets"] = [{"action": "rewrite", "section": "视觉识别路线",
                              "instruction": "补充对比实验"}]
    out = N.rewrite_sections(st)
    assert out["human_round"] == 1
    assert out["human_feedback"] == ""  # 改写后清除，避免误判
    assert "视觉识别路线" in out["sections"]
    # 非目标小节内容不变
    assert out["sections"]["其它小节"] == "保持不变 [1]。"


def test_human_targeted_rewrite_loop(stub_env, offline_retrieval):
    """--human 挂起后，带意见续跑应触发 parse→rewrite→...→再次挂起（闭环重生成）。"""
    from src.agent.graph import run_review

    stub_env.target_paper_count = 10
    stub_env.max_retrieval_rounds = 1
    stub_env.max_human_rounds = 2
    tid = "rewrite-e2e"

    first = run_review("a test topic", thread_id=tid, with_human=True, stream=False)
    assert first.get("interrupted"), "首次运行应挂起"
    first_sections = dict(first.get("sections", {}))

    second = run_review(
        "a test topic", thread_id=tid, with_human=True,
        feedback="请在第 1 个主题小节补充与对比方法的实验分析", stream=False,
    )
    assert second.get("interrupted"), "改写后应再次挂起等待下一轮审核"
    assert second.get("human_round", 0) == 1, "应完成 1 轮 targeted 改写"
    assert any("RewriteSections" in log for log in second.get("logs", [])), "应记录改写节点"
    assert len(second.get("sections", {})) == len(first_sections), "小节数应保持一致"
    for path in second.get("artifacts", {}).values():
        assert os.path.exists(path), f"产物缺失: {path}"


def test_faithfulness_appears_in_report(stub_env, offline_retrieval):
    """faithfulness 节点结果应写入成稿附录 A.7。"""
    stub_env.target_paper_count = 10
    stub_env.max_retrieval_rounds = 1
    graph = build_graph()
    final = graph.invoke(
        initial_state("topic x"),
        {"configurable": {"thread_id": "faith-e2e"}, "recursion_limit": 60},
    )
    report = final["report"]
    assert "A.7" in report
    assert "一致性得分" in report
    assert final["faithfulness"]["checked"] >= 1
