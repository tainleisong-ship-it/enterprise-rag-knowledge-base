from typing import List, Tuple

from langchain_core.documents import Document


class RuleBasedReranker:
    """基于规则的模拟 Reranker：综合精确匹配、关键词重叠、原始排名与长度因子重新打分。"""

    def __init__(self) -> None:
        self._weights = {
            "exact_match": 0.35,
            "keyword_overlap": 0.30,
            "position": 0.20,
            "length": 0.15,
        }

    def _score(self, query: str, doc: str, rank: int, total: int) -> float:
        q_lower = query.lower()
        d_lower = doc.lower()
        q_terms = set(q_lower.split())
        d_terms = set(d_lower.split())

        exact = 1.0 if q_lower and q_lower in d_lower else 0.0
        overlap = len(q_terms & d_terms) / max(len(q_terms), 1)
        position = 1.0 - (rank / max(total, 1))
        length = min(len(doc) / 300.0, 1.0)

        return (
            self._weights["exact_match"] * exact
            + self._weights["keyword_overlap"] * overlap
            + self._weights["position"] * position
            + self._weights["length"] * length
        )

    def rerank(
        self,
        query: str,
        candidates: List[Tuple[Document, float]],
        top_k: int,
    ) -> List[Tuple[Document, float]]:
        total = len(candidates)
        scored: List[Tuple[Document, float]] = []
        for idx, (doc, orig_score) in enumerate(candidates):
            rule_score = self._score(query, doc.page_content, idx, total)
            final = 0.5 * orig_score + 0.5 * rule_score
            scored.append((doc, final))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]
