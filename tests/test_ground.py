"""离线测试：Claim 级证据锚定节点（P3-2）。"""

from __future__ import annotations

import pytest

from src.agent import nodes as N
from src.agent.state import initial_state
from src.config import get_settings
from src.ingest.base import make_paper


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


def _state_with_sections():
    p1 = make_paper(paper_id="arxiv:1", title="Vision method A",
                    abstract="vision recognition", citation_count=50, year=2020)
    p2 = make_paper(paper_id="doi:10.1/b", title="Vision method B",
                    abstract="vision transformer", citation_count=80, year=2021)
    st = initial_state("vision")
    st["papers"] = [p1, p2]
    st["citation_map"] = {"arxiv:1": 1, "doi:10.1/b": 2}
    st["evidence"] = [
        {"paper_id": "arxiv:1", "claim": "CNN 提升了图像识别", "method": "m", "dataset": "d", "metric": "k", "section": ""},
        {"paper_id": "doi:10.1/b", "claim": "ViT 在基准上领先", "method": "m", "dataset": "d", "metric": "k", "section": ""},
    ]
    st["sections"] = {"视觉识别路线": "CNN 在图像识别上表现好 [1]，ViT 在基准领先 [2]。"}
    return st


def test_ground_claims_populates(stub_env):
    st = _state_with_sections()
    out = N.ground_claims(st)
    grounded = out["grounded_claims"]
    assert len(grounded) == 1
    g = grounded[0]
    assert g["section"] == "视觉识别路线"
    claims = g["claims"]
    assert len(claims) >= 1
    # 每条 claim 必须绑定有效 paper_id 且只引用该小节出现的论文
    allowed = {"arxiv:1", "doi:10.1/b"}
    for c in claims:
        assert c["confidence"] in ("high", "medium", "low")
        assert set(c["paper_ids"]) <= allowed


def test_ground_claims_empty_section_skipped(stub_env):
    st = _state_with_sections()
    st["sections"] = {"空小节": "没有任何引用编号的文字。"}  # 无 [n]
    out = N.ground_claims(st)
    assert out["grounded_claims"] == []
