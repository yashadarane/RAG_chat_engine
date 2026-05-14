from __future__ import annotations

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
            matches.append(SearchMatch(chunk=chunk, similarity_score=similarity))
        return matches
