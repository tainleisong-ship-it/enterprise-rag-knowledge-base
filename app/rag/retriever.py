import re
from typing import List, Tuple

from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

from app.core.config import settings
from app.rag.store import documents, vectorstore


def _tokenize(text: str) -> List[str]:
    """中文按字符级切分，同时保留英文单词，适配 BM25。"""
    tokens = re.findall(r"[a-zA-Z0-9]+|[\u4e00-\u9fff]", text.lower())
    return tokens


class HybridRetriever:
    """向量检索 + BM25 关键词检索的混合检索器。"""

    def __init__(self) -> None:
        self._docs: List[Document] = documents
        self._vectorstore: FAISS = vectorstore
        self._bm25: BM25Okapi = BM25Okapi([_tokenize(d.page_content) for d in self._docs])

    def retrieve(self, query: str, top_k: int | None = None) -> List[Tuple[Document, float]]:
        k = top_k or settings.top_k_retrieval

        vec_results = self._vectorstore.similarity_search_with_score(query, k=k)
        # FAISS 返回的是 L2 距离，越小越相似；转换为 0~1 的相似度
        vec_map: dict[str, float] = {}
        for doc, dist in vec_results:
            sim = 1.0 / (1.0 + float(dist))
            vec_map[doc.page_content] = max(vec_map.get(doc.page_content, 0.0), sim)

        bm25_scores = self._bm25.get_scores(_tokenize(query))
        max_bm25 = max(bm25_scores) if bm25_scores.size and max(bm25_scores) > 0 else 1.0
        bm25_map: dict[str, float] = {}
        for doc, score in zip(self._docs, bm25_scores):
            bm25_map[doc.page_content] = float(score) / max_bm25

        all_contents = set(vec_map.keys()) | set(bm25_map.keys())
        merged: List[Tuple[Document, float]] = []
        for content in all_contents:
            doc = next(d for d in self._docs if d.page_content == content)
            v = vec_map.get(content, 0.0)
            b = bm25_map.get(content, 0.0)
            combined = 0.6 * v + 0.4 * b
            merged.append((doc, combined))

        merged.sort(key=lambda x: x[1], reverse=True)
        return merged[:k]
