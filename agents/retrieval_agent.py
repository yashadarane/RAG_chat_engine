from __future__ import annotations

from core.config import MIN_SIMILARITY, TOP_K
from core.ports import VectorSearchStore
from core.schemas import SearchMatch, Source


class RetrievalAgent:
    """Agent 4A: retrieve relevant chunks and source metadata."""

    def __init__(
        self,
        vector_store: VectorSearchStore,
        top_k: int = TOP_K,
        min_similarity: float = MIN_SIMILARITY,
    ) -> None:
        self.vector_store = vector_store
        self.top_k = top_k
        self.min_similarity = min_similarity

    def retrieve(self, query: str) -> list[SearchMatch]:
        return self.vector_store.search(query, self.top_k, self.min_similarity)

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
