from __future__ import annotations

import re
from collections import OrderedDict

from core.config import MIN_SIMILARITY, TOP_K
from core.ports import QueryRewriter, Reranker, VectorSearchStore
from core.reranker import ScoreReranker
from core.schemas import RetrievalMode, RetrievalOutput, RetrievalScope, SearchMatch, Source


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
        scope, selected_documents = self.select_scope(query)
        if mode is RetrievalMode.FOCUSED:
            matches = self._retrieve_focused(query)
        elif mode is RetrievalMode.BROAD:
            matches = self._retrieve_broad(query, scope, selected_documents)
        else:
            matches = self._retrieve_exhaustive(scope, selected_documents)
        return RetrievalOutput(
            mode=mode,
            scope=scope,
            selected_documents=selected_documents,
            matches=matches,
            sources=self.build_sources(matches),
        )

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

    def select_scope(self, query: str) -> tuple[RetrievalScope, list[str]]:
        if self._requires_all_documents(query):
            return RetrievalScope.ALL_DOCUMENTS, self._all_document_names()

        selected_documents = self._select_relevant_documents(query)
        return RetrievalScope.SELECTED_DOCUMENTS, selected_documents

    def _retrieve_focused(self, query: str) -> list[SearchMatch]:
        query_variants = self._get_query_variants(query)
        candidates: list[SearchMatch] = []

        for variant in query_variants:
            candidates.extend(self.retrieve_dense(variant, top_k=max(self.top_k * 2, 8)))
            candidates.extend(self.retrieve_sparse(variant, top_k=max(self.top_k * 2, 8)))

        deduped = self.merge_and_deduplicate(candidates)
        return self.rerank(query, deduped, self.top_k)

    def _retrieve_broad(
        self,
        query: str,
        scope: RetrievalScope,
        selected_documents: list[str],
    ) -> list[SearchMatch]:
        if scope is RetrievalScope.ALL_DOCUMENTS:
            return self.vector_store.get_all()
        if selected_documents:
            return self.vector_store.get_by_filenames(selected_documents)
        return []

    def _retrieve_exhaustive(
        self,
        scope: RetrievalScope,
        selected_documents: list[str],
    ) -> list[SearchMatch]:
        if scope is RetrievalScope.ALL_DOCUMENTS:
            return self.vector_store.get_all()
        if selected_documents:
            return self.vector_store.get_by_filenames(selected_documents)
        return []

    def _select_relevant_documents(self, query: str) -> list[str]:
        candidates: list[SearchMatch] = []
        for variant in self._get_query_variants(query):
            candidates.extend(
                self.vector_store.search(
                    query=variant,
                    top_k=max(self.top_k * 3, 12),
                    min_similarity=max(self.min_similarity * 0.5, 0.10),
                )
            )
            candidates.extend(self.vector_store.keyword_search(query=variant, top_k=max(self.top_k * 3, 12)))

        candidates = self.merge_and_deduplicate(candidates)
        if not candidates:
            return []

        doc_scores: dict[str, float] = {}
        for match in candidates:
            filename = match.chunk.filename
            doc_scores[filename] = max(doc_scores.get(filename, 0.0), match.score)

        ranked = sorted(doc_scores.items(), key=lambda item: item[1], reverse=True)
        if not ranked:
            return []

        top_score = ranked[0][1]
        if top_score <= 0:
            return []

        selected = [filename for filename, score in ranked if score >= top_score * 0.90]
        return selected

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

    def _requires_all_documents(self, query: str) -> bool:
        return self._contains_any(
            query.lower(),
            {
                "all documents",
                "all uploaded documents",
                "all files",
                "all uploaded files",
                "all three documents",
                "three documents",
                "each document",
                "every document",
                "compare documents",
                "compare all",
                "summarize all",
                "across all documents",
            },
        )

    def _all_document_names(self) -> list[str]:
        seen: set[str] = set()
        names: list[str] = []
        for match in self.vector_store.get_all():
            filename = match.chunk.filename
            if filename not in seen:
                seen.add(filename)
                names.append(filename)
        return names

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
                    retrieval_reason=match.retrieval_reason,
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
