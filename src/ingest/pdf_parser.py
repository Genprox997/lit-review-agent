"""PDF 解析：pypdf 抽取文本 + 面向 LLM 的结构化深度解析（方向 J'）。

提供两条路径：
- ``parse_pdf`` / ``clean_pdf_text`` / ``condense_fulltext``：兼容旧逻辑，抽取纯文本并裁剪；
- ``deep_parse_pdf`` / ``deep_parse_text``：在纯文本之上做**布局感知**的结构化解析——
  切分章节（编号标题 + 关键词标题）、抽取摘要与关键词、识别量表（Table/Figure 图注），
  输出可供 Extractor 直接消费的章节化正文。离线默认可用，无任何重依赖。

若用户安装了 ``unstructured`` 并通过环境变量 ``PDF_DEEP_PARSER=unstructured`` 开启，
则优先用其版面元素做解析（更准）；未安装或缺省时回退到纯启发式解析器，保证离线绿灯。
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# 参考文献之后的内容对 Extractor 无价值，直接截断
_REF_HEAD = re.compile(
    r"\n\s*(references|bibliography|参考文献)\s*\n", re.IGNORECASE
)
_MULTI_NL = re.compile(r"\n{3,}")
_WS = re.compile(r"[ \t]{2,}")

# --------------------------------------------------------------------------
# 旧接口（兼容）
# --------------------------------------------------------------------------
def parse_pdf(path: str | Path, max_pages: int = 30) -> Optional[str]:
    """抽取 PDF 文本，失败返回 None。"""
    try:
        from pypdf import PdfReader
    except ImportError:  # pragma: no cover
        logger.warning("未安装 pypdf，跳过全文解析")
        return None

    path = Path(path)
    if not path.exists() or path.stat().st_size < 1024:
        return None

    try:
        reader = PdfReader(str(path))
        pages = reader.pages[:max_pages]
        text = "\n".join((p.extract_text() or "") for p in pages)
    except Exception as exc:
        logger.warning("PDF 解析失败 %s: %s", path.name, exc)
        return None

    return clean_pdf_text(text)


def clean_pdf_text(text: str) -> str:
    """去掉参考文献段、压缩空白。"""
    if not text:
        return ""
    m = _REF_HEAD.search(text)
    if m and m.start() > len(text) * 0.4:  # 防止误把正文里的 "References" 当尾部
        text = text[: m.start()]
    text = _MULTI_NL.sub("\n\n", text)
    text = _WS.sub(" ", text)
    return text.strip()


def condense_fulltext(text: str, head_chars: int = 6000, tail_chars: int = 3000) -> str:
    """长全文压缩：保留开头（摘要+引言+方法）与结尾（实验结论）。"""
    if not text:
        return ""
    if len(text) <= head_chars + tail_chars:
        return text
    return (
        text[:head_chars]
        + "\n\n...[中间内容省略]...\n\n"
        + text[-tail_chars:]
    )


# --------------------------------------------------------------------------
# 深度解析（方向 J'）：布局感知结构化
# --------------------------------------------------------------------------

# 编号标题：1 / 1.2 / 1.2.3 后接标题（含中文）
_NUMBERED_RE = re.compile(
    r"^\s*(?P<num>\d{1,2}(?:\.\d+){0,3})[\.\)]?\s+"
    r"(?P<title>[\w\s:\-–&/,()\u4e00-\u9fff\uff00-\uffef]{2,70})\s*$"
)

# 已知「小节标题词」——仅当独立成行时才视作标题，避免正文首句误判
_KEYWORD_HEADINGS = {
    "abstract", "summary",
    "introduction", "intro", "background",
    "related work", "prior work",
    "methods", "methodology", "method",
    "materials and methods", "approach", "proposed method", "proposed approach",
    "model", "models", "architecture",
    "experiments", "experimental setup", "experimental", "experiment",
    "evaluation", "evaluations", "results", "results and discussion",
    "discussion", "analysis", "conclusions", "conclusion", "concluding remarks",
    "future work", "limitations",
    "acknowledgments", "acknowledgements", "appendix", "appendices",
    "references", "bibliography", "references cited",
    # 中文常见标题
    "摘要", "引言", "背景", "相关工作", "方法", "方法学", "模型", "实验",
    "实验设置", "评估", "结果", "结果与讨论", "讨论", "结论", "总结",
    "未来工作", "致谢", "附录", "参考文献",
}

# 摘要之后的正文里常见的「非内容」小节（萃取时跳过）
_SKIP_SECTIONS = {"references", "bibliography", "references cited", "acknowledgments",
                  "acknowledgements", "appendix", "appendices", "致谢", "附录", "参考文献"}

# 萃取时优先挑选的内容小节（命中即纳入萃取正文）
_PRIORITY_SECTIONS = (
    "method", "methods", "methodology", "approach", "proposed method", "proposed approach",
    "model", "models", "architecture", "experiment", "experiments", "experimental setup",
    "experimental", "evaluation", "evaluations", "result", "results", "results and discussion",
    "discussion", "analysis", "conclusion", "conclusions", "concluding remarks", "summary",
    "方法", "方法学", "模型", "实验", "实验设置", "评估", "结果", "结果与讨论",
    "讨论", "分析", "结论", "总结", "摘要",
)

_CAP_TABLE_RE = re.compile(r"^\s*(table|tab\.?)\s*\d+", re.IGNORECASE)
_CAP_FIG_RE = re.compile(r"^\s*(figure|fig\.?)\s*\d+", re.IGNORECASE)
_KEYWORDS_RE = re.compile(
    r"^\s*(keywords|index terms|key words|关键词)\s*[:：\-]\s*(.+)$", re.IGNORECASE
)
_LEAD_NUM_RE = re.compile(r"^\d{1,2}(?:\.\d+){0,3}[\.\)]?\s+")


def _normalize_heading(s: str) -> str:
    """去掉前导编号，转小写用于匹配。"""
    return _LEAD_NUM_RE.sub("", s).strip().lower().rstrip(". ").strip()


def _is_heading(line: str) -> Optional[str]:
    """若是标题行返回标题文本，否则返回 None。"""
    s = line.strip()
    if not s or len(s) > 90:
        return None
    m = _NUMBERED_RE.match(line)
    if m:
        return m.group("title").strip()
    low = _normalize_heading(s)
    if low in _KEYWORD_HEADINGS:
        return s
    return None


def _detect_tables_figures(raw_lines: List[str], norm_lines: List[str]) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """扫描行，识别 Table/Figure 图注，并尽量聚合表体。

    ``raw_lines`` 用于表体识别（保留列间多空格对齐信号），``norm_lines``
    用于图注/章节文本（已压缩空白）。
    """
    tables: List[Dict[str, Any]] = []
    figures: List[Dict[str, Any]] = []

    def _is_table_row(line: str) -> bool:
        """表体行：保留原始空格以识别列对齐（≥2 个 2+ 空格间隔且含 ≥2 数字列，
        或 ≥4 token 且 ≥2 数字）。避免把含多个数字的句子误判为表。"""
        if _CAP_TABLE_RE.match(line) or _CAP_FIG_RE.match(line):
            return False  # 图注/表题本身不是表体行
        if "|" in line:
            return line.count("|") >= 2
        gaps = len(_WS.findall(line))  # 2+ 空格的列间隔数
        if gaps >= 2:
            parts = _WS.split(line.strip())
            nums = sum(1 for p in parts if re.search(r"\d", p))
            if nums >= 2:
                return True
        toks = line.split()
        nums = sum(1 for t in toks if re.search(r"\d", t))
        if len(toks) >= 4 and nums >= 2:
            return True
        return False

    def _cap_text(raw: str, norm: str) -> str:
        return raw.strip() if _WS.search(raw) else norm.strip()

    i = 0
    n = len(norm_lines)
    while i < n:
        # 用 norm 判定标题/图注触发，用 raw 保留对齐
        line = norm_lines[i]
        mt = _CAP_TABLE_RE.match(line)
        mf = _CAP_FIG_RE.match(line)
        if mt or mf:
            caption = _cap_text(raw_lines[i], line)
            j = i + 1
            sub = 0
            while j < n and sub < 2:
                lj = norm_lines[j]
                if not lj.strip():
                    j += 1
                    continue
                if _is_heading(lj) or _CAP_TABLE_RE.match(lj) or _CAP_FIG_RE.match(lj):
                    break
                caption += " " + lj.strip()
                sub += 1
                j += 1
                if _is_table_row(raw_lines[j]):
                    break
            if mt:
                rows: List[str] = []
                k = j
                while k < n and _is_table_row(raw_lines[k]) and len(rows) < 40:
                    rows.append(raw_lines[k].strip())
                    k += 1
                tables.append({"caption": caption, "rows": rows})
            else:
                figures.append({"caption": caption})
            i = max(j, i + 1)
            continue
        if _is_table_row(raw_lines[i]):
            rows = [raw_lines[i].strip()]
            k = i + 1
            while k < n and _is_table_row(raw_lines[k]) and len(rows) < 40:
                rows.append(raw_lines[k].strip())
                k += 1
            tables.append({"caption": "", "rows": rows})
            i = k
            continue
        i += 1
    return tables, figures


def deep_parse_text(text: str, n_pages: int = 0) -> Dict[str, Any]:
    """对纯文本做布局感知结构化解析，返回 JSON 友好的字典。

    输出 schema::
        {
          "title": str, "abstract": str, "keywords": [str],
          "sections": [{"heading": str, "level": int, "text": str}],
          "tables": [{"caption": str, "rows": [str]}],
          "figures": [{"caption": str}],
          "n_pages": int, "parse_method": str,
        }
    """
    if not text or not text.strip():
        return {
            "title": "", "abstract": "", "keywords": [],
            "sections": [], "tables": [], "figures": [],
            "n_pages": n_pages, "parse_method": "empty",
        }

    # 参考资料段之后的内容无价值
    m = _REF_HEAD.search(text)
    if m and m.start() > len(text) * 0.4:
        text = text[: m.start()]
    text = _MULTI_NL.sub("\n", text)

    raw_lines = text.split("\n")
    # 压缩空白用于章节/图注文本；表体识别仍用 raw 保留列对齐
    norm_lines = [_WS.sub(" ", ln).strip() for ln in raw_lines]

    # 章节切分
    sections: List[Dict[str, Any]] = []
    cur_heading = "Full Text"
    cur_level = 0
    cur_buf: List[str] = []
    keywords: List[str] = []

    def _flush() -> None:
        body = "\n".join(cur_buf).strip()
        if body:
            sections.append({"heading": cur_heading, "level": cur_level, "text": body})

    for line in norm_lines:
        # 关键词行（多出现在摘要后 / 独立行）
        km = _KEYWORDS_RE.match(line)
        if km and not keywords:
            raw = km.group(2).strip()
            for part in re.split(r"[;,]", raw):
                part = part.strip().strip(".")
                if part and len(part) <= 60:
                    keywords.append(part)
        head = _is_heading(line)
        if head is not None:
            _flush()
            cur_heading = head
            num = _NUMBERED_RE.match(line)
            cur_level = (num.group("num").count(".") + 1) if num else 1
            cur_buf = []
        else:
            if line.strip():
                cur_buf.append(line.strip())
    _flush()

    if not sections:
        sections = [{"heading": "Full Text", "level": 0, "text": text.strip()}]

    # 摘要抽取：优先命中 abstract 标题小节
    abstract = ""
    for sec in sections:
        if _normalize_heading(sec["heading"]) in ("abstract", "摘要", "summary"):
            abstract = sec["text"]
            break

    tables, figures = _detect_tables_figures(raw_lines, norm_lines)

    return {
        "title": "",
        "abstract": abstract,
        "keywords": keywords,
        "sections": sections,
        "tables": tables,
        "figures": figures,
        "n_pages": n_pages,
        "parse_method": "heuristic",
    }


def deep_parse_pdf(path: str | Path, max_pages: int = 30) -> Optional[Dict[str, Any]]:
    """深度解析 PDF：抽取文本后做结构化解析；失败返回 None。

    ``PDF_DEEP_PARSER=unstructured`` 且已安装 ``unstructured`` 时优先走其版面解析，
    否则走纯启发式（离线默认可用）。
    """
    import os

    path = Path(path)
    if not path.exists() or path.stat().st_size < 1024:
        return None

    parser = os.environ.get("PDF_DEEP_PARSER", "auto").lower()
    if parser in ("unstructured", "auto"):
        struct = _deep_parse_with_unstructured(path, max_pages)
        if struct is not None:
            return struct

    try:
        from pypdf import PdfReader
    except ImportError:  # pragma: no cover
        logger.warning("未安装 pypdf，跳过全文解析")
        return None

    try:
        reader = PdfReader(str(path))
        pages = reader.pages[:max_pages]
        n_pages = len(reader.pages)
        text = "\n".join((p.extract_text() or "") for p in pages)
    except Exception as exc:  # noqa: BLE001
        logger.warning("PDF 解析失败 %s: %s", path.name, exc)
        return None

    if not text or not text.strip():
        return None
    struct = deep_parse_text(text, n_pages=n_pages)
    return struct


def _deep_parse_with_unstructured(path: str | Path, max_pages: int) -> Optional[Dict[str, Any]]:
    """用 unstructured 的版面元素映射到统一 schema（可选加速）。"""
    try:
        from unstructured.partition.pdf import partition_pdf
    except Exception:  # noqa: BLE001 - 仅在显式开启且已安装时调用
        return None

    try:
        elements = partition_pdf(
            filename=str(path), strategy="fast",
            infer_table_structure=True, max_pages=max_pages or None,
        )
    except Exception as exc:  # noqa: BLE001
        logger.debug("unstructured 解析失败，回退启发式: %s", exc)
        return None

    sections: List[Dict[str, Any]] = []
    tables: List[Dict[str, Any]] = []
    figures: List[Dict[str, Any]] = []
    abstract = ""
    keywords: List[str] = []
    cur = {"heading": "Full Text", "level": 0, "text": ""}

    try:
        from unstructured.documents.elements import (
            NarrativeText, Table, Title, FigureCaption,
        )
    except Exception:  # noqa: BLE001
        Title = NarrativeText = Table = FigureCaption = None  # type: ignore[assignment]

    for el in elements:
        etype = type(el).__name__
        txt = (getattr(el, "text", "") or "").strip()
        if not txt:
            continue
        if Title is not None and isinstance(el, Title):
            if cur["text"]:
                sections.append(cur)
            cur = {"heading": txt, "level": 1, "text": ""}
        elif Table is not None and isinstance(el, Table):
            tables.append({"caption": "", "rows": [txt]})
        elif FigureCaption is not None and isinstance(el, FigureCaption):
            figures.append({"caption": txt})
        else:
            cur["text"] = (cur["text"] + "\n" + txt).strip()
            km = _KEYWORDS_RE.match(txt)
            if km and not keywords:
                for part in re.split(r"[;,]", km.group(2)):
                    part = part.strip().strip(".")
                    if part and len(part) <= 60:
                        keywords.append(part)
    if cur["text"]:
        sections.append(cur)

    for sec in sections:
        if _normalize_heading(sec["heading"]) in ("abstract", "摘要", "summary"):
            abstract = sec["text"]
            break

    return {
        "title": "",
        "abstract": abstract,
        "keywords": keywords,
        "sections": sections,
        "tables": tables,
        "figures": figures,
        "n_pages": len(elements),
        "parse_method": "unstructured",
    }


def extractor_body_from_struct(struct: Dict[str, Any], budget: int = 5000) -> Optional[str]:
    """从结构化解析里挑选「摘要 + 方法/结果相关小节」拼装萃取正文。

    返回带章节标签的正文；若无可用内容小节则返回 None（调用方退回 head/tail 行为）。
    """
    if not struct:
        return None
    parts: List[str] = []
    used_chars = 0

    abstract = (struct.get("abstract") or "").strip()
    if abstract:
        parts.append(f"## 摘要\n{abstract}")
        used_chars += len(abstract) + 12

    for sec in struct.get("sections", []):
        head_low = _normalize_heading(sec.get("heading", ""))
        if head_low in _SKIP_SECTIONS:
            continue
        if not any(k in head_low for k in _PRIORITY_SECTIONS):
            continue
        body = (sec.get("text") or "").strip()
        if not body:
            continue
        head = sec.get("heading", "").strip()
        block = f"## {head}\n{body}"
        if used_chars + len(block) > budget:
            # 截断到预算内
            remain = budget - used_chars
            if remain > 200:
                block = block[:remain].rstrip() + "\n...[截断]"
                parts.append(block)
            break
        parts.append(block)
        used_chars += len(block)

    # 量表信息（精简）补充到末尾，帮助 Extractor 抓方法/指标
    figs = struct.get("figures") or []
    tbls = struct.get("tables") or []
    if figs or tbls:
        tail = "\n## 图表线索\n"
        for f in figs[:8]:
            cap = (f.get("caption") or "").strip()
            if cap:
                tail += f"- 图：{cap[:160]}\n"
        for t in tbls[:8]:
            cap = (t.get("caption") or "").strip()
            if cap:
                tail += f"- 表：{cap[:160]}\n"
        if used_chars <= budget and used_chars + len(tail) <= budget + 400:
            parts.append(tail)

    return "\n\n".join(parts).strip() if parts else None
