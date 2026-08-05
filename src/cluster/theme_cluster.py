"""主题聚类：embedding + KMeans 把文献池分成若干主题簇。

Embedding 策略（自动降级，保证零额外配置也能跑）：
1. `sentence-transformers` 若已安装 → 语义向量，效果最好；
2. 否则 → TF-IDF + TruncatedSVD（LSA），纯 scikit-learn，无需下载模型。

簇数 k 支持自动推断：在候选区间内用轮廓系数（silhouette）选优。
"""

from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)

_EMBEDDER = None
_EMBEDDER_TRIED = False

# 学术文本里的高频无信息词，加入停用词避免污染簇关键词
_EXTRA_STOPWORDS = [
    "paper", "propose", "proposed", "method", "methods", "approach", "approaches",
    "results", "result", "show", "shows", "shown", "based", "using", "used", "use",
    "novel", "new", "study", "studies", "work", "works", "model", "models",
    "state", "art", "performance", "task", "tasks", "data", "dataset", "datasets",
    "experiments", "experimental", "achieve", "achieves", "compared", "existing",
]


@dataclass
class Cluster:
    """一个主题簇。"""

    cluster_id: int
    paper_ids: List[str] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)
    label: str = ""            # 由 LLM 命名，初始为空

    @property
    def size(self) -> int:
        return len(self.paper_ids)

    def to_dict(self) -> dict:
        return {
            "cluster_id": self.cluster_id,
            "label": self.label,
            "keywords": self.keywords,
            "paper_ids": self.paper_ids,
            "size": self.size,
        }


# --------------------------------------------------------------------------
# Embedding
# --------------------------------------------------------------------------
def _load_sentence_transformer():
    global _EMBEDDER, _EMBEDDER_TRIED
    if _EMBEDDER_TRIED:
        return _EMBEDDER
    _EMBEDDER_TRIED = True
    try:
        from sentence_transformers import SentenceTransformer

        _EMBEDDER = SentenceTransformer("all-MiniLM-L6-v2")
        logger.info("使用 sentence-transformers 语义向量")
    except Exception as exc:
        logger.info("未启用 sentence-transformers（%s），降级为 TF-IDF + SVD", type(exc).__name__)
        _EMBEDDER = None
    return _EMBEDDER


def _tfidf_matrix(texts: Sequence[str]):
    from sklearn.feature_extraction.text import TfidfVectorizer

    stopwords = list(_sklearn_english_stopwords()) + _EXTRA_STOPWORDS
    vec = TfidfVectorizer(
        max_features=8000,
        ngram_range=(1, 2),
        stop_words=stopwords,
        min_df=1,
        max_df=0.85,
        sublinear_tf=True,
    )
    return vec, vec.fit_transform(texts)


def _sklearn_english_stopwords():
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

    return ENGLISH_STOP_WORDS


def embed_texts(texts: Sequence[str]) -> np.ndarray:
    """把文本编码为稠密向量矩阵（已 L2 归一化）。"""
    if not texts:
        return np.zeros((0, 1), dtype=np.float32)

    model = _load_sentence_transformer()
    if model is not None:
        vecs = model.encode(list(texts), show_progress_bar=False, normalize_embeddings=True)
        return np.asarray(vecs, dtype=np.float32)

    from sklearn.decomposition import TruncatedSVD
    from sklearn.preprocessing import normalize

    _, matrix = _tfidf_matrix(texts)
    n_comp = int(max(2, min(128, matrix.shape[1] - 1, len(texts) - 1)))
    if n_comp < 2:
        return normalize(matrix.toarray().astype(np.float32))
    svd = TruncatedSVD(n_components=n_comp, random_state=42)
    return normalize(svd.fit_transform(matrix)).astype(np.float32)


# --------------------------------------------------------------------------
# 聚类
# --------------------------------------------------------------------------
def choose_k(n_samples: int, requested: int = 0) -> int:
    """决定簇数：显式指定优先，否则按规模启发式给一个候选上界。"""
    if n_samples <= 2:
        return 1
    if requested and requested > 0:
        return int(max(1, min(requested, n_samples - 1)))
    # 经验值：sqrt(n/2)，限制在 [2, 8]，避免综述章节过碎
    k = int(round(math.sqrt(n_samples / 2)))
    return max(2, min(8, k, n_samples - 1))


def _best_k_by_silhouette(vectors: np.ndarray, k_max: int) -> int:
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    n = vectors.shape[0]
    best_k, best_score = 2, -1.0
    for k in range(2, min(k_max, n - 1) + 1):
        try:
            labels = KMeans(n_clusters=k, n_init=10, random_state=42).fit_predict(vectors)
            if len(set(labels)) < 2:
                continue
            score = silhouette_score(vectors, labels)
        except Exception:
            continue
        logger.debug("k=%d silhouette=%.4f", k, score)
        if score > best_score:
            best_k, best_score = k, score
    logger.info("自动选定簇数 k=%d (silhouette=%.3f)", best_k, best_score)
    return best_k


def _cluster_keywords(texts: Sequence[str], labels: np.ndarray, top_n: int = 8) -> dict:
    """用簇内 TF-IDF 均值提取每簇的代表性关键词。"""
    try:
        vec, matrix = _tfidf_matrix(texts)
        vocab = np.array(vec.get_feature_names_out())
    except Exception as exc:
        logger.debug("关键词提取失败: %s", exc)
        return {}

    out = {}
    dense = matrix.toarray()
    for cid in sorted(set(labels.tolist())):
        mask = labels == cid
        if not mask.any():
            out[int(cid)] = []
            continue
        centroid = dense[mask].mean(axis=0)
        top_idx = np.argsort(centroid)[::-1][:top_n]
        out[int(cid)] = [str(vocab[i]) for i in top_idx if centroid[i] > 0]
    return out


def _paper_text(paper: dict) -> str:
    """聚类用文本：标题权重加倍 + 摘要。"""
    title = paper.get("title") or ""
    abstract = paper.get("abstract") or ""
    return re.sub(r"\s+", " ", f"{title}. {title}. {abstract}").strip()


def cluster_papers(
    papers: List[dict],
    n_clusters: int = 0,
    auto_tune: bool = True,
) -> List[Cluster]:
    """把文献池按主题分簇。

    Args:
        papers: Paper 字典列表（需含 paper_id / title / abstract）
        n_clusters: 0 表示自动推断
        auto_tune: 自动模式下是否用轮廓系数搜索最优 k

    Returns:
        按簇规模降序排列的 Cluster 列表
    """
    papers = [p for p in papers if p.get("paper_id")]
    if not papers:
        return []
    if len(papers) <= 3:
        return [Cluster(0, [p["paper_id"] for p in papers], keywords=[])]

    texts = [_paper_text(p) for p in papers]
    vectors = embed_texts(texts)

    k_cap = choose_k(len(papers), n_clusters)
    if n_clusters and n_clusters > 0:
        k = k_cap
    elif auto_tune:
        k = _best_k_by_silhouette(vectors, k_cap)
    else:
        k = k_cap

    if k <= 1:
        return [Cluster(0, [p["paper_id"] for p in papers], keywords=[])]

    from sklearn.cluster import KMeans

    labels = KMeans(n_clusters=k, n_init=10, random_state=42).fit_predict(vectors)
    keywords = _cluster_keywords(texts, labels)

    clusters: List[Cluster] = []
    for cid in sorted(set(labels.tolist())):
        ids = [papers[i]["paper_id"] for i in range(len(papers)) if labels[i] == cid]
        clusters.append(Cluster(int(cid), ids, keywords=keywords.get(int(cid), [])))

    clusters.sort(key=lambda c: c.size, reverse=True)
    # 重新编号，保证 0 号是最大簇
    for new_id, c in enumerate(clusters):
        c.cluster_id = new_id
    logger.info("聚类完成：%d 篇 → %d 个主题簇 %s", len(papers), len(clusters),
                [c.size for c in clusters])
    return clusters
