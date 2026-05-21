from __future__ import annotations

from typing import Protocol

from core.schemas import SearchMatch


class LanguageModelClient(Protocol):
    """Boundary for any text-generation provider used by the RAG agent."""

    def generate(self, prompt: str) -> str:
        """Generate a grounded answer from a fully constructed prompt."""


class VectorSearchStore(Protocol):
    """Boundary for vector stores that can retrieve source chunks."""

    def search(self, query: str, top_k: int, min_similarity: float) -> list[SearchMatch]:
        """Return ranked chunks above the relevance threshold."""

    def keyword_search(self, query: str, top_k: int) -> list[SearchMatch]:
        """Return keyword/BM25 candidates."""

    def get_all(self) -> list[SearchMatch]:
        """Return all indexed chunks for exhaustive tasks."""

    def get_by_filenames(self, filenames: list[str]) -> list[SearchMatch]:
        """Return all chunks belonging to the selected documents."""


class QueryRewriter(Protocol):
    """Boundary for query rewriting strategies."""

    def rewrite(self, query: str) -> list[str]:
        """Return query variants that preserve the user's intent."""


class Reranker(Protocol):
    """Boundary for candidate reranking strategies."""

    def rerank(self, query: str, matches: list[SearchMatch], top_k: int) -> list[SearchMatch]:
        """Return the highest relevance matches for the original query."""
