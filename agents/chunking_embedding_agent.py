from __future__ import annotations

from collections.abc import Callable

from core.config import CHUNK_OVERLAP_WORDS, CHUNK_WORDS
from core.schemas import ExtractedDocument, IndexingOutput, TextChunk
from core.vector_store import ChromaVectorStore
from utils.text_utils import chunk_text_by_boundaries


class ChunkingEmbeddingAgent:
    """Agent 3: chunk text, embed with all-MiniLM-L6-v2, and persist in ChromaDB."""

    def __init__(
        self,
        chunk_words: int = CHUNK_WORDS,
        overlap_words: int = CHUNK_OVERLAP_WORDS,
        embedding_function: Callable[[list[str]], list[list[float]]] | None = None,
    ) -> None:
        self.chunk_words = chunk_words
        self.overlap_words = overlap_words
        self.embedding_function = embedding_function

    def index(self, session_id: str, documents: list[ExtractedDocument]) -> tuple[ChromaVectorStore, IndexingOutput]:
        collection_name = f"session_{session_id}"
        vector_store = ChromaVectorStore(collection_name, embedding_function=self.embedding_function)
        chunks = self._build_chunks(documents)
        vector_store.add_chunks(chunks)
        return vector_store, IndexingOutput(
            session_id=session_id,
            collection_name=collection_name,
            chunks_indexed=len(chunks),
            status="indexed",
        )

    def _build_chunks(self, documents: list[ExtractedDocument]) -> list[TextChunk]:
        chunks: list[TextChunk] = []
        for document in documents:
            for page in document.pages:
                chunk_texts = chunk_text_by_boundaries(page.text, self.chunk_words, self.overlap_words)
                for index, text in enumerate(chunk_texts, start=1):
                    chunks.append(
                        TextChunk(
                            chunk_id=f"{document.doc_id}_p{page.page_number}_c{index}",
                            doc_id=document.doc_id,
                            filename=document.filename,
                            page_number=page.page_number,
                            text=text,
                        )
                    )
        return chunks
