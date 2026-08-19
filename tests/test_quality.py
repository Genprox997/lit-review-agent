"""方向 F' 质量评估仪表盘：compute_quality_report 单元测试（纯函数，离线）。

不依赖 LLM / 网络 / embedding：直接构造最小输入验证维度计算、薄弱项生成与缺失数据安全降级。
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.agent import quality as QL  # noqa: E402


def _papers():
    return [
        {"paper_id": "P1", "title": "A", "year": 2023, "citation_count": 100, "hub_score": 0.5},
        {"paper_id": "P2", "title": "B", "year": 2022, "citation_count": 50, "hub_score": 0.2},
        {"paper_id": "P3", "title": "C", "year": 2021, "citation_count": 10, "hub_score": 0.0},
        {"paper_id": "P4", "title": "D", "year": 2018, "citation_count": 5, "hub_score": 0.0},
    ]


def _sections():
    return {
        "主题一": "本文讨论方法 [1]，并与 [2] 对比。",
        "主题二": "最新进展见 [3]。",
    }


def test_quality_report_all_six_dimensions_present():
    r = QL.compute_quality_report(
        papers=_papers(),
        clusters=[{"label": "x", "paper_ids": ["P1", "P2"]}, {"label": "y", "paper_ids": ["P3", "P4"]}],
        sections=_sections(),
        faithfulness={"score": 0.95, "checked": 4, "flagged": []},
        citation_analysis={"dropped_high_hub": [], "top_hub": []},
        citation_graph={"stats": {"edge_count": 3}},
        grounded_claims=[{"section": "主题一", "claims": [
            {"text": "c1", "paper_ids": ["P1"], "confidence": "high"},
            {"text": "c2", "paper_ids": ["P2"], "confidence": "medium"},
        ]}],
        gaps=[],
    )
    keys = {d["key"] for d in r["dimensions"]}
    assert keys == {
        "faithfulness", "citation_coverage", "network_hub",
        "topic_balance", "recency", "grounding",
    }
    assert 0 <= r["overall"] <= 100
    assert r["grade"] in ("A", "B", "C", "D")
    # 引用覆盖：2/2 小节含标注
    cov = next(d for d in r["dimensions"] if d["key"] == "citation_coverage")
    assert cov["score"] == 100
    # 引用网络：有边且无剔除 → 100
    net = next(d for d in r["dimensions"] if d["key"] == "network_hub")
    assert net["score"] == 100


def test_quality_report_missing_data_is_safe():
    r = QL.compute_quality_report()  # 全空
    # 无引用边时 network_hub 为唯一可用维度（中性 60），其余缺失即降级，不抛错
    assert 0 <= r["overall"] <= 100
    assert r["grade"] in ("A", "B", "C", "D")
    avail = [d for d in r["dimensions"] if d["available"]]
    assert [d["key"] for d in avail] == ["network_hub"]
    assert next(d for d in r["dimensions"] if d["key"] == "network_hub")["score"] == 60
    for d in r["dimensions"]:
        assert d["available"] in (True, False)
        # 不可用时 score 为 None，可用时 0-100
        assert d["score"] is None or (0 <= d["score"] <= 100)
    assert isinstance(r["weaknesses"], list)
    assert isinstance(r["highlights"], list)


def test_quality_report_faithfulness_skipped_flagged_in_weaknesses():
    r = QL.compute_quality_report(
        papers=_papers(),
        sections=_sections(),
        faithfulness={"score": 0.6, "checked": 5, "flagged": [{"text": "x"}]},
        citation_graph={"stats": {"edge_count": 0}},
    )
    f = next(d for d in r["dimensions"] if d["key"] == "faithfulness")
    assert f["score"] == 60
    assert any("一致性 60%" in w for w in r["weaknesses"])


def test_quality_report_dropped_high_hub_generates_weakness():
    r = QL.compute_quality_report(
        papers=_papers(),
        sections=_sections(),
        faithfulness={"score": 1.0, "checked": 0, "flagged": []},
        citation_analysis={"dropped_high_hub": [{"title": "奠基工作 X", "hub": 0.9}]},
        citation_graph={"stats": {"edge_count": 2}},
    )
    net = next(d for d in r["dimensions"] if d["key"] == "network_hub")
    assert net["score"] < 100
    assert any("奠基工作 X" in w for w in r["weaknesses"])


def test_quality_report_gaps_reduce_topic_balance_and_warn():
    r = QL.compute_quality_report(
        papers=_papers(),
        clusters=[{"label": "x", "paper_ids": ["P1"]}, {"label": "y", "paper_ids": ["P2"]}],
        sections=_sections(),
        citation_graph={"stats": {"edge_count": 1}},
        gaps=["共引子群（2 篇）未被覆盖：Z1、Z2"],
    )
    topic = next(d for d in r["dimensions"] if d["key"] == "topic_balance")
    # 有研究空白 → 分数被乘 0.85
    assert topic["score"] == 85
    assert any("研究空白" in w for w in r["weaknesses"])


def test_quality_report_high_faithfulness_highlighted():
    r = QL.compute_quality_report(
        papers=_papers(),
        sections=_sections(),
        faithfulness={"score": 0.98, "checked": 10, "flagged": []},
        citation_graph={"stats": {"edge_count": 1}},
    )
    assert any("一致性高" in h for h in r["highlights"])
