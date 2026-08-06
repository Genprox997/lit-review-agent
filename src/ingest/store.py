"""跨主题持久化文献池（设计文档 §7/§9：向量库持久化文献池）。

默认用 SQLite（标准库，零额外依赖）落盘所有检索到的文献，按规范化标识
（DOI / arXiv ID / 标题指纹）去重。下游检索后能：
1. ``hydrate`` —— 用历史缓存回填引用数 / 摘要 / DOI，避免重复打 OpenAlex、重复下载；
2. ``recall``  —— 给定检索式，从本地池召回语义相近的历史文献，实现跨主题复用。

语义召回默认走 sklearn 的 TF-IDF 余弦；若装了 ``[store]`` extra 的 chromadb，
可换更强的句向量召回（本文件预留接口，缺省不强制依赖）。
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from src.config import get_settings
from src.ingest.base import Paper, normalize_title_key

logger = logging.getLogger(__name__)

# 需要从缓存回填 / 落盘的字段（不含 fulltext 原文，避免库体膨胀）
_PERSIST_FIELDS = (
    "paper_id", "title", "authors", "year", "venue", "citation_count", "abstract",
    "url", "pdf_url", "doi", "source", "score", "matched_queries",
)


def _paper_key(p: Paper) -> str:
    """规范化主键：DOI > arXiv ID > 标题指纹。"""
    if p.get("doi"):
        return f"doi:{p['doi']}"
    pid = p.get("paper_id", "")
    if pid.startswith("arxiv:"):
        return pid
    tk = normalize_title_key(p.get("title", ""))
    if len(tk) >= 12:
        return f"title:{tk}"
    return pid or normalize_title_key(p.get("title", ""))


class PaperStore:
    """基于 SQLite 的轻量文献池，跨运行复用。"""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self._conn.execute(
            """CREATE TABLE IF NOT EXISTS papers (
                key TEXT PRIMARY KEY,
                data TEXT NOT NULL,
                updated_at REAL NOT NULL
            )"""
        )
        self._conn.commit()

    # ------------------------------------------------------------------ 写
    def upsert(self, papers: Sequence[Paper]) -> int:
        n = 0
        with self._lock:
            for p in papers:
                key = _paper_key(p)
                if not key:
                    continue
                blob = {k: p.get(k) for k in _PERSIST_FIELDS if k in p}
                self._conn.execute(
                    "INSERT INTO papers(key, data, updated_at) VALUES(?,?,?) "
                    "ON CONFLICT(key) DO UPDATE SET data=excluded.data, updated_at=excluded.updated_at",
                    (key, json.dumps(blob, ensure_ascii=False), time.time()),
                )
                n += 1
            self._conn.commit()
        if n:
            logger.info("文献池持久化：写入/更新 %d 条 → %s", n, self.path)
        return n

    # ------------------------------------------------------------------ 读
    def get(self, paper: Paper) -> Optional[Paper]:
        key = _paper_key(paper)
        if not key:
            return None
        row = self._conn.execute("SELECT data FROM papers WHERE key=?", (key,)).fetchone()
        return json.loads(row[0]) if row else None

    def hydrate(self, papers: Sequence[Paper]) -> List[Paper]:
        """用本地缓存回填缺失字段（引用数 / 摘要 / DOI 等），不覆盖已有更全信息。"""
        out: List[Paper] = []
        for p in papers:
            cached = self.get(p)
            if cached:
                merged = dict(p)
                for f in ("citation_count", "abstract", "doi", "pdf_url", "venue", "year"):
                    if not merged.get(f) and cached.get(f):
                        merged[f] = cached[f]
                if len(cached.get("authors") or []) > len(merged.get("authors") or []):
                    merged["authors"] = cached["authors"]
                out.append(merged)
            else:
                out.append(dict(p))
        return out

    def recall(self, query: str, top_k: int = 10) -> List[Paper]:
        """从本地池召回与 query 语义相近的文献（TF-IDF 余弦；sklearn 缺失则退回近期）。"""
        rows = self._conn.execute("SELECT data FROM papers").fetchall()
        papers = [json.loads(r[0]) for r in rows]
        if not papers:
            return []
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.metrics.pairwise import cosine_similarity
        except ImportError:  # pragma: no cover
            return papers[:top_k]
        docs = [f"{p.get('title', '')} {p.get('abstract', '')}" for p in papers]
        try:
            vec = TfidfVectorizer(stop_words="english", max_features=20000).fit(docs + [query])
            sims = cosine_similarity(vec.transform([query]), vec.transform(docs)).ravel()
        except Exception:
            return papers[:top_k]
        order = [i for i in sims.argsort()[::-1][:top_k] if sims[i] > 0]
        return [papers[i] for i in order]

    def reset(self) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM papers")
            self._conn.commit()


# --------------------------------------------------------------------------
# 单例访问（读写全局配置开关）
# --------------------------------------------------------------------------
_store: Optional[PaperStore] = None


def get_paper_store() -> Optional[PaperStore]:
    """返回全局文献池；`PAPER_STORE_ENABLED=false` 时返回 None。"""
    global _store
    settings = get_settings()
    if not settings.paper_store_enabled:
        return None
    if _store is None:
        _store = PaperStore(Path(settings.paper_store_path))
    return _store


def reset_paper_store() -> None:
    """测试用：清空单例，下次 get_paper_store 重新构造。"""
    global _store
    _store = None
