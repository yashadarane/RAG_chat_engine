from __future__ import annotations

import re
from collections import OrderedDict

from core.config import MIN_SIMILARITY, TOP_K
from core.ports import QueryRewriter, Reranker, VectorSearchStore
from core.reranker import ScoreReranker
from core.schemas import RetrievalMode, RetrievalOutput, SearchMatch, Source


class RetrievalAgent:
    """Agent 4A: classify query intent and retrieve context with the right strategy."""

    def __init__(
        self,
        vector_store: VectorSearchStore,
        query_rewriter: QueryRewriter | None = None,
        reranker: Reranker | None = None,
        top_k: int = TOP_K,
        min_similarity: float = MIN_SIMILARITY,
    ) -> None:
        self.vector_store = vector_store
        self.query_rewriter = query_rewriter
        self.reranker = reranker or ScoreReranker()
        self.top_k = top_k
        self.min_similarity = min_similarity

    def retrieve(self, query: str) -> RetrievalOutput:
        mode = self.select_mode(query)
        if mode is RetrievalMode.FOCUSED:
            matches = self._retrieve_focused(query)
        elif mode is RetrievalMode.BROAD:
            matches = self._retrieve_broad(query)
        else:
            matches = self._retrieve_exhaustive()
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

    def _retrieve_focused(self, query: str) -> list[SearchMatch]:
        query_variants = self._get_query_variants(query)
        candidates: list[SearchMatch] = []

        for variant in query_variants:
            candidates.extend(self.retrieve_dense(variant, top_k=max(self.top_k * 2, 8)))
            candidates.extend(self.retrieve_sparse(variant, top_k=max(self.top_k * 2, 8)))

        deduped = self.merge_and_deduplicate(candidates)
        return self.rerank(query, deduped, self.top_k)

    def _retrieve_broad(self, query: str) -> list[SearchMatch]:
        all_matches = self.vector_store.get_all()
        if all_matches:
            return all_matches

        query_variants = self._get_query_variants(query)
        candidates: list[SearchMatch] = []
        for variant in query_variants:
            candidates.extend(
                self.retrieve_dense(
                    variant,
                    top_k=max(self.top_k * 4, 20),
                    min_similarity=max(self.min_similarity * 0.5, 0.05),
                )
            )
            candidates.extend(self.retrieve_sparse(variant, top_k=max(self.top_k * 4, 20)))
        return self.merge_and_deduplicate(candidates)

    def _retrieve_exhaustive(self) -> list[SearchMatch]:
        return self.vector_store.get_all()

    def _get_query_variants(self, query: str) -> list[str]:
        if not self.query_rewriter:
            return [query]

        try:
            variants = self.query_rewriter.rewrite(query)
        except Exception:
            return [query]

        cleaned: list[str] = []
        seen: set[str] = set()
        for variant in [query, *variants]:
            normalized = variant.strip()
            key = normalized.lower()
            if normalized and key not in seen:
                seen.add(key)
                cleaned.append(normalized)
        return cleaned[:5]

    def retrieve_dense(
        self,
        query: str,
        top_k: int,
        min_similarity: float | None = None,
    ) -> list[SearchMatch]:
        threshold = self.min_similarity if min_similarity is None else min_similarity
        return self.vector_store.search(query, top_k=top_k, min_similarity=max(threshold * 0.7, 0.10))

    def retrieve_sparse(self, query: str, top_k: int) -> list[SearchMatch]:
        return self.vector_store.keyword_search(query, top_k=top_k)

    def merge_and_deduplicate(self, matches: list[SearchMatch]) -> list[SearchMatch]:
        deduped: OrderedDict[tuple[str, int, str], SearchMatch] = OrderedDict()
        for match in matches:
            key = (match.chunk.filename, match.chunk.page_number, match.chunk.chunk_id)
            existing = deduped.get(key)
            if existing is None or match.similarity_score > existing.similarity_score:
                deduped[key] = match
        return list(deduped.values())

    def rerank(self, query: str, matches: list[SearchMatch], top_k: int) -> list[SearchMatch]:
        return self.reranker.rerank(query, matches, top_k)

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
