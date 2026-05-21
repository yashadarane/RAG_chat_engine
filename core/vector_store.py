from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Callable

from core.config import CHROMA_DIR, EMBEDDING_MODEL
from core.schemas import SearchMatch, TextChunk


class SentenceTransformerEmbeddingFunction:
    def __init__(self, model_name: str = EMBEDDING_MODEL) -> None:
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name)

    def __call__(self, texts: list[str]) -> list[list[float]]:
        return self.model.encode(texts, normalize_embeddings=True).tolist()


class ChromaVectorStore:
    def __init__(
        self,
        collection_name: str,
        embedding_function: Callable[[list[str]], list[list[float]]] | None = None,
    ) -> None:
        self.collection_name = collection_name
        self.embedding_function = embedding_function or SentenceTransformerEmbeddingFunction()
        import chromadb

        self.client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        self.collection = self._fresh_collection(collection_name)
        self.chunks: list[TextChunk] = []
        self.keyword_index: BM25Index | None = None

    def _fresh_collection(self, collection_name: str):
        try:
            self.client.delete_collection(collection_name)
        except Exception:
            pass
        return self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def add_chunks(self, chunks: list[TextChunk]) -> None:
        self.chunks = chunks
        self.keyword_index = BM25Index(chunks)
        if not chunks:
            return

        documents = [chunk.text for chunk in chunks]
        embeddings = self.embedding_function(documents)
        metadatas = [
            {
                "doc_id": chunk.doc_id,
                "filename": chunk.filename,
                "page_number": chunk.page_number,
            }
            for chunk in chunks
        ]
        ids = [chunk.chunk_id for chunk in chunks]
        self.collection.add(documents=documents, embeddings=embeddings, metadatas=metadatas, ids=ids)

    def search(self, query: str, top_k: int, min_similarity: float) -> list[SearchMatch]:
        if not self.chunks:
            return []

        query_embedding = self.embedding_function([query])[0]
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, len(self.chunks)),
            include=["documents", "metadatas", "distances"],
        )

        matches: list[SearchMatch] = []
        ids = results.get("ids", [[]])[0]
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        for chunk_id, text, metadata, distance in zip(ids, documents, metadatas, distances):
            similarity = max(0.0, 1.0 - float(distance))
            if similarity < min_similarity:
                continue
            chunk = TextChunk(
                chunk_id=chunk_id,
                doc_id=str(metadata["doc_id"]),
                filename=str(metadata["filename"]),
                page_number=int(metadata["page_number"]),
                text=str(text),
            )
            matches.append(SearchMatch(chunk=chunk, similarity_score=similarity, retrieval_reason="semantic_match"))
        return matches

    def keyword_search(self, query: str, top_k: int) -> list[SearchMatch]:
        if self.keyword_index is None:
            return []
        return self.keyword_index.search(query, top_k)

    def get_all(self) -> list[SearchMatch]:
        return [
            SearchMatch(chunk=chunk, similarity_score=None, retrieval_reason="complete_context")
            for chunk in self.chunks
        ]

    def get_by_filenames(self, filenames: list[str]) -> list[SearchMatch]:
        selected = set(filenames)
        return [
            SearchMatch(chunk=chunk, similarity_score=None, retrieval_reason="complete_context")
            for chunk in self.chunks
            if chunk.filename in selected
        ]


class BM25Index:
    def __init__(self, chunks: list[TextChunk], k1: float = 1.5, b: float = 0.75) -> None:
        self.chunks = chunks
        self.k1 = k1
        self.b = b
        self.tokenized_chunks = [self._tokenize(chunk.text) for chunk in chunks]
        self.doc_lengths = [len(tokens) for tokens in self.tokenized_chunks]
        self.avg_doc_length = sum(self.doc_lengths) / len(self.doc_lengths) if self.doc_lengths else 0.0
        self.document_frequencies = self._build_document_frequencies()

    def search(self, query: str, top_k: int) -> list[SearchMatch]:
        query_terms = self._tokenize(query)
        if not query_terms:
            return []

        raw_scores = [self._score(query_terms, index) for index in range(len(self.chunks))]
        max_score = max(raw_scores, default=0.0)
        if max_score <= 0:
            return []

        ranked = sorted(
            zip(self.chunks, raw_scores),
            key=lambda item: item[1],
            reverse=True,
        )
        matches: list[SearchMatch] = []
        for chunk, score in ranked[:top_k]:
            if score <= 0:
                continue
            matches.append(
                SearchMatch(
                    chunk=chunk,
                    similarity_score=float(score / max_score),
                    retrieval_reason="keyword_match",
                )
            )
        return matches

    def _score(self, query_terms: list[str], doc_index: int) -> float:
        tokens = self.tokenized_chunks[doc_index]
        if not tokens:
            return 0.0

        frequencies = Counter(tokens)
        score = 0.0
        for term in query_terms:
            if term not in frequencies:
                continue
            idf = self._idf(term)
            tf = frequencies[term]
            doc_length = self.doc_lengths[doc_index]
            denominator = tf + self.k1 * (1 - self.b + self.b * doc_length / (self.avg_doc_length or 1))
            score += idf * ((tf * (self.k1 + 1)) / denominator)
        return score

    def _idf(self, term: str) -> float:
        total_docs = len(self.chunks)
        docs_with_term = self.document_frequencies.get(term, 0)
        return math.log(1 + (total_docs - docs_with_term + 0.5) / (docs_with_term + 0.5))

    def _build_document_frequencies(self) -> dict[str, int]:
        frequencies: dict[str, int] = {}
        for tokens in self.tokenized_chunks:
            for token in set(tokens):
                frequencies[token] = frequencies.get(token, 0) + 1
        return frequencies

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return re.findall(r"\b\w+\b", text.lower())
