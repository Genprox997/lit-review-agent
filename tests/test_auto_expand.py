"""方向 H'：检索式自动扩词（伪相关反馈 PRF）测试。"""

from __future__ import annotations

from src.agent.nodes import auto_expand, query_expander
from src.agent.tools import auto_expand_queries
from src.config import Settings, get_settings


def _mk(pid, title, abstract, relevance=0.6, score=1.0, cites=10):
    return {
        "paper_id": pid,
        "title": title,
        "abstract": abstract,
        "relevance": relevance,
        "score": score,
        "citation_count": cites,
        "year": 2020,
        "matched_queries": [],
    }


DOCS = [
    ("Adaptive optics with deformable mirror control for vortex beam shaping",
     "We use a deformable mirror to shape the vortex beam; holography enables phase control."),
    ("Deformable mirror calibration via holography",
     "Holography based calibration of the deformable mirror improves wavefront correction."),
    ("Vortex beam generation using deformable mirror",
     "A deformable mirror generates a high order vortex beam with holography encoded phase."),
    ("Wavefront sensing for deformable mirror",
     "Closed loop deformable mirror control with holography feedback for vortex beams."),
    ("Holography assisted deformable mirror",
     "Holography assists deformable mirror in vortex beam applications."),
]


def test_auto_expand_queries_mines_terms():
    papers = [_mk(f"p{i}", t, a) for i, (t, a) in enumerate(DOCS)]
    q = auto_expand_queries(
        papers, topic="vortex beam shaping", existing_queries=[], settings=get_settings(),
    )
    assert q, "should mine at least one query"
    joined = " ".join(q)
    assert "deformable mirror" in joined, q
    assert "holography" in joined, q


def test_auto_expand_queries_dedup_existing():
    papers = [_mk(f"p{i}", t, a) for i, (t, a) in enumerate(DOCS)]
    q = auto_expand_queries(
        papers, topic="vortex beam shaping",
        existing_queries=["deformable mirror", "holography"], settings=get_settings(),
    )
    # 已存在的词不应再被挖出（其余区分度高的词仍可保留）
    assert "deformable mirror" not in q, q
    assert "holography" not in q, q


def test_auto_expand_queries_dedup_topic():
    papers = [_mk(f"p{i}", t, a) for i, (t, a) in enumerate(DOCS)]
    q = auto_expand_queries(
        papers, topic="deformable mirror holography", existing_queries=[], settings=get_settings(),
    )
    # 主题已覆盖的词不应再被挖出（其余区分度高的词仍可保留）
    assert "deformable mirror" not in q, q
    assert "holography" not in q, q


def test_auto_expand_queries_empty():
    assert auto_expand_queries([], topic="x", settings=get_settings()) == []


def test_auto_expand_queries_disabled():
    s = Settings(enable_auto_expand=False)
    papers = [_mk(f"p{i}", t, a) for i, (t, a) in enumerate(DOCS)]
    assert auto_expand_queries(papers, topic="vortex beam shaping", settings=s) == []


def test_auto_expand_queries_respects_max():
    s = Settings(max_auto_queries=2)
    papers = [_mk(f"p{i}", t, a) for i, (t, a) in enumerate(DOCS)]
    q = auto_expand_queries(papers, topic="vortex beam shaping", existing_queries=[], settings=s)
    assert len(q) <= 2, q


def test_auto_expand_node_fires_once():
    papers = [_mk(f"p{i}", t, a) for i, (t, a) in enumerate(DOCS)]
    state = {
        "topic": "vortex beam shaping",
        "papers": papers,
        "queries": ["vortex beam shaping"],
        "auto_expanded": False,
        "auto_expanded_queries": [],
    }
    out = auto_expand(state)
    assert out.get("pending_queries"), out
    assert out.get("auto_expanded_queries"), out
    assert out.get("auto_expanded") is True
    # 二次调用应被 auto_expanded 标志拦截
    out2 = auto_expand({
        **state,
        "auto_expanded": True,
        "auto_expanded_queries": out["auto_expanded_queries"],
    })
    assert "pending_queries" not in out2, out2


def test_auto_expand_node_skips_when_disabled():
    papers = [_mk(f"p{i}", t, a) for i, (t, a) in enumerate(DOCS)]
    state = {
        "topic": "vortex beam shaping",
        "papers": papers,
        "queries": ["vortex beam shaping"],
        "auto_expanded": False,
        "auto_expanded_queries": [],
    }
    import src.agent.nodes as N
    orig = get_settings().enable_auto_expand
    try:
        get_settings().enable_auto_expand = False
        out = auto_expand(state)
    finally:
        get_settings().enable_auto_expand = orig
    assert "pending_queries" not in out, out


def test_query_expander_merges_prf_on_refine(monkeypatch):
    import src.agent.nodes as N

    monkeypatch.setattr(N, "chat_json", lambda *a, **k: {"queries": []})
    papers = [_mk(f"p{i}", t, a) for i, (t, a) in enumerate(DOCS)]
    state = {
        "topic": "vortex beam shaping",
        "queries": ["vortex beam shaping"],
        "critic": {"verdict": "need_more"},
        "papers": papers,
        "auto_expanded_queries": [],
    }
    out = query_expander(state)
    assert out.get("auto_expanded_queries"), out
    assert any(
        "deformable mirror" in q or "holography" in q for q in out["pending_queries"]
    ), out
