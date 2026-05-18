from __future__ import annotations

import re

from core.config import MIN_SIMILARITY, TOP_K
from core.ports import VectorSearchStore
from core.schemas import RetrievalMode, RetrievalOutput, SearchMatch, Source


class RetrievalAgent:
    """Agent 4A: classify query intent and retrieve context with the right strategy."""

    def __init__(
        self,
        vector_store: VectorSearchStore,
        top_k: int = TOP_K,
        min_similarity: float = MIN_SIMILARITY,
    ) -> None:
        self.vector_store = vector_store
        self.top_k = top_k
        self.min_similarity = min_similarity

    def retrieve(self, query: str) -> RetrievalOutput:
        mode = self.select_mode(query)
        matches = self._retrieve_by_mode(query, mode)
        return RetrievalOutput(mode=mode, matches=matches, sources=self.build_sources(matches))

    def select_mode(self, query: str) -> RetrievalMode:
        normalized = query.lower()
        if self._contains_any(
            normalized,
            {
                "all",
                "every",
                "complete",
                "entire",
                "total",
                "count",
                "how many",
                "number of",
                "list all",
                "full list",
            },
        ):
            return RetrievalMode.EXHAUSTIVE
        if self._contains_any(
            normalized,
            {
                "summary",
                "summarize",
                "overview",
                "compare",
                "comparison",
                "differences",
                "explain",
                "main points",
                "key points",
                "pros and cons",
            },
        ):
            return RetrievalMode.BROAD
        return RetrievalMode.FOCUSED

    def _retrieve_by_mode(self, query: str, mode: RetrievalMode) -> list[SearchMatch]:
        if mode is RetrievalMode.FOCUSED:
            return self.vector_store.search(query, self.top_k, self.min_similarity)
        if mode is RetrievalMode.BROAD:
            return self.vector_store.search(
                query,
                top_k=max(self.top_k * 2, 8),
                min_similarity=max(self.min_similarity * 0.6, 0.10),
            )
        return self.vector_store.search(query, top_k=10_000, min_similarity=0.0)

    def build_sources(self, matches: list[SearchMatch]) -> list[Source]:
        seen: set[tuple[str, int, str]] = set()
        sources: list[Source] = []
        for match in matches:
            key = (match.chunk.filename, match.chunk.page_number, match.chunk.chunk_id)
            if key in seen:
                continue
            seen.add(key)
            sources.append(
                Source(
                    document=match.chunk.filename,
                    page=match.chunk.page_number,
                    chunk_id=match.chunk.chunk_id,
                    similarity_score=match.similarity_score,
                )
            )
        return sources

    @staticmethod
    def _contains_any(text: str, phrases: set[str]) -> bool:
        for phrase in phrases:
            if " " in phrase and phrase in text:
                return True
            if " " not in phrase and re.search(rf"\b{re.escape(phrase)}\b", text):
                return True
        return False
