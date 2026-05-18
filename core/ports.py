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
