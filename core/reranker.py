from __future__ import annotations

import os

from core.schemas import SearchMatch


class ScoreReranker:
    """Fallback reranker that orders candidates by the score already attached to them."""

    def rerank(self, query: str, matches: list[SearchMatch], top_k: int) -> list[SearchMatch]:
        return sorted(matches, key=lambda match: match.score, reverse=True)[:top_k]


class CrossEncoderReranker:
    """Cross-encoder reranker for stronger query/chunk relevance scoring."""

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> None:
        from sentence_transformers import CrossEncoder

        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, matches: list[SearchMatch], top_k: int) -> list[SearchMatch]:
        if not matches:
            return []

        pairs = [(query, match.chunk.text) for match in matches]
        scores = self.model.predict(pairs)
        reranked = [
            SearchMatch(chunk=match.chunk, similarity_score=float(score))
            for match, score in zip(matches, scores)
        ]
        return sorted(reranked, key=lambda match: match.similarity_score, reverse=True)[:top_k]


def build_default_reranker():
    if os.getenv("ENABLE_CROSS_ENCODER_RERANKER", "").lower() in {"1", "true", "yes"}:
        return CrossEncoderReranker()
    return ScoreReranker()
