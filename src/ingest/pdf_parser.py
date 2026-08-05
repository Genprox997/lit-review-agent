"""PDF 解析：pypdf 抽取文本，并做面向 LLM 的裁剪。"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# 参考文献之后的内容对 Extractor 无价值，直接截断
_REF_HEAD = re.compile(
    r"\n\s*(references|bibliography|参考文献)\s*\n", re.IGNORECASE
)
_MULTI_NL = re.compile(r"\n{3,}")
_WS = re.compile(r"[ \t]{2,}")


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
    """长全文压缩：保留开头（摘要+引言+方法）与结尾（实验结论）。

    综述抽取真正需要的是「做了什么 + 得到什么结论」，中间的公式推导可丢弃。
    """
    if not text:
        return ""
    if len(text) <= head_chars + tail_chars:
        return text
    return (
        text[:head_chars]
        + "\n\n...[中间内容省略]...\n\n"
        + text[-tail_chars:]
    )
