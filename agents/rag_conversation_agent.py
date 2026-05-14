from __future__ import annotations

from collections.abc import Callable

from core.config import MIN_SIMILARITY, OLLAMA_MODEL, TOP_K
from core.llm_client import OllamaLLMClient
from core.prompts import build_grounded_prompt
from core.schemas import ChatTurn, RagOutput, SearchMatch, Source
from core.vector_store import ChromaVectorStore


class RagConversationAgent:
    """Agent 4: retrieve ChromaDB context and generate grounded answers with Ollama."""

    def __init__(
        self,
        vector_store: ChromaVectorStore,
        llm_client: OllamaLLMClient | None = None,
        top_k: int = TOP_K,
        min_similarity: float = MIN_SIMILARITY,
    ) -> None:
        self.vector_store = vector_store
        self.llm_client = llm_client or OllamaLLMClient(model=OLLAMA_MODEL)
        self.top_k = top_k
        self.min_similarity = min_similarity

    def answer(self, query: str, history: list[ChatTurn]) -> RagOutput:
        matches = self.vector_store.search(query, self.top_k, self.min_similarity)
        prompt = build_grounded_prompt(query, history[-6:], matches)
        sources = build_sources(matches)

        if not matches:
            return RagOutput(
                answer="I could not find that in the uploaded documents.",
                sources=[],
                prompt=prompt,
                matches=[],
            )

        try:
            answer = self.llm_client.generate(prompt)
        except Exception as exc:
            answer = (
                "Ollama could not generate a response. Start the Ollama service and pull the "
                f"`{OLLAMA_MODEL}` model. Error: {exc}"
            )

        return RagOutput(answer=answer, sources=sources, prompt=prompt, matches=matches)


def build_sources(matches: list[SearchMatch]) -> list[Source]:
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
