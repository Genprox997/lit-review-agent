"""方向 J'：PDF 深度解析（启发式）单元测试——纯文本逻辑 + 入口优雅降级。

不需要真实 PDF 文件：核心逻辑走 ``deep_parse_text``（确定性），
PDF 读取路径用 monkeypatch 注入假 ``PdfReader`` 验证串流。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.ingest import pdf_parser as PP


SAMPLE = """
Abstract
We propose a novel deformable mirror control method for vortex beam shaping.
Experiments show a 30% improvement in Strehl ratio.

1 Introduction
Adaptive optics is important. Prior work used fixed mirrors.

2 Related Work
Many methods exist. We build on Schmidt et al.

3 Method
Our approach uses a Kalman filter to drive the ALPAO DM.
The control law minimizes wavefront error.

3.1 Implementation
We deployed on Matlab R2016a.

4 Experiments
We tested on 120 samples.

Table 1 Comparison of methods

Method   RMSE   Time(s)
Ours     0.12   1.3
Baseline 0.45   2.1

Figure 2 shows the recovered wavefront.

5 Conclusion
The method is effective and real-time capable.

References
[1] Author et al. Paper.
"""


def test_deep_parse_text_detects_sections():
    st = PP.deep_parse_text(SAMPLE)
    headings = [s["heading"] for s in st["sections"]]
    assert "Introduction" in headings
    assert "Method" in headings
    assert "Conclusion" in headings
    # 编号标题识别层级
    method = next(s for s in st["sections"] if s["heading"] == "Method")
    assert method["level"] == 1
    impl = next(s for s in st["sections"] if s["heading"] == "Implementation")
    assert impl["level"] == 2


def test_deep_parse_text_extracts_abstract():
    st = PP.deep_parse_text(SAMPLE)
    assert "deformable mirror" in st["abstract"]
    assert "Strehl" in st["abstract"]


def test_deep_parse_text_detects_table_and_figure():
    st = PP.deep_parse_text(SAMPLE)
    assert len(st["tables"]) >= 1
    tbl = st["tables"][0]
    assert "Comparison" in tbl["caption"]
    # 表体行被聚合（含数字列）
    assert any("Ours" in r for r in tbl["rows"])
    # 真实 PDF 中图注多独立成行；本样本无独立图注行，故 figures 可能为空，
    # 仅校验「有图注时被正确捕获」的逻辑单独覆盖（此处不强求）。


def test_deep_parse_text_degrades_when_no_headings():
    st = PP.deep_parse_text("Just a plain paragraph with no structure at all.\n" * 5)
    assert len(st["sections"]) == 1
    assert st["sections"][0]["heading"] == "Full Text"
    assert st["parse_method"] == "heuristic"


def test_deep_parse_text_empty():
    st = PP.deep_parse_text("")
    assert st["sections"] == []
    assert st["parse_method"] == "empty"


def test_deep_parse_text_standalone_figure_caption():
    text = (
        "3 Results\nWe observed clear vortex beams.\n\n"
        "Figure 3: Recovered wavefront after correction.\n\n"
        "The correction improves Strehl ratio.\n"
    )
    st = PP.deep_parse_text(text)
    assert len(st["figures"]) >= 1
    assert "Recovered wavefront" in st["figures"][0]["caption"]


def test_deep_parse_text_chinese_headings():
    text = "摘要\n本文研究涡旋光调控。\n\n1 引言\n背景介绍。\n\n2 方法\n采用变形镜。\n\n3 结论\n有效。\n"
    st = PP.deep_parse_text(text)
    headings = [s["heading"] for s in st["sections"]]
    assert "摘要" in headings
    assert "方法" in headings
    assert "结论" in headings
    assert "涡旋光" in st["abstract"]


def test_extractor_body_from_struct_uses_priority_sections():
    st = PP.deep_parse_text(SAMPLE)
    body = PP.extractor_body_from_struct(st, budget=5000)
    assert body is not None
    assert "## 摘要" in body
    assert "## Method" in body
    assert "## Conclusion" in body
    # 跳过 References 等非内容小节
    assert "References" not in body.split("## ")[-1] or "References" not in body


def test_extractor_body_from_struct_budget_truncation():
    st = PP.deep_parse_text(SAMPLE)
    body = PP.extractor_body_from_struct(st, budget=120)
    assert body is not None
    assert len(body) <= 200  # 预算 + 少量截断余量


def test_extractor_body_from_struct_none_when_empty():
    assert PP.extractor_body_from_struct({}, budget=5000) is None
    assert PP.extractor_body_from_struct(None, budget=5000) is None


def test_deep_parse_pdf_missing_file_returns_none(tmp_path):
    assert PP.deep_parse_pdf(tmp_path / "nope.pdf") is None


def test_deep_parse_pdf_non_pdf_returns_none(tmp_path):
    f = tmp_path / "x.txt"
    f.write_text("not a pdf", encoding="utf-8")
    assert PP.deep_parse_pdf(f) is None


def test_deep_parse_pdf_streams_through_pypdf(monkeypatch):
    """用假 PdfReader 验证 deep_parse_pdf 正确串流到 deep_parse_text。"""
    import tempfile

    class _FakePage:
        def __init__(self, text):
            self._t = text

        def extract_text(self):
            return self._t

    class _FakeReader:
        def __init__(self, path):
            self.pages = [_FakePage("1 Introduction\nWe study vortex beams.\n"),
                          _FakePage("2 Conclusion\nIt works well.\n")]

    import pypdf

    monkeypatch.setattr(pypdf, "PdfReader", _FakeReader)

    # 写一个最小占位文件以满足「存在且 >=1024 字节」的前置检查
    d = Path(tempfile.mkdtemp())
    f = d / "fake.pdf"
    f.write_bytes(b"%PDF-1.4 " + b"x" * 1200)

    st = PP.deep_parse_pdf(f)
    assert st is not None
    headings = [s["heading"] for s in st["sections"]]
    assert "Introduction" in headings
    assert "Conclusion" in headings
    assert st["n_pages"] == 2


def test_parse_pdf_and_condense_still_work():
    """旧接口保持兼容。"""
    text = PP.clean_pdf_text("Hello.\n\n\n\nWorld.\nreferences\n[1] x")
    assert "references" not in text
    assert PP.condense_fulltext("a" * 20000) != "a" * 20000
