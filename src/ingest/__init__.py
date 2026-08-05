"""文献获取层：多源元数据检索 + OA 全文下载解析。"""

from src.ingest.base import Paper, clean_text, make_paper, normalize_doi, normalize_title_key

__all__ = [
    "Paper",
    "clean_text",
    "make_paper",
    "normalize_doi",
    "normalize_title_key",
]
