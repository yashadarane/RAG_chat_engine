from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path


@dataclass(frozen=True)
class ValidatedDocument:
    session_id: str
    doc_id: str
    filename: str
    file_path: Path
    file_type: str
    page_count: int | None
    status: str
    issues: list[str] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return self.status == "validated"


@dataclass(frozen=True)
class IngestionOutput:
    session_id: str
    documents: list[ValidatedDocument]


@dataclass(frozen=True)
class ExtractedPage:
    page_number: int
    text: str


@dataclass(frozen=True)
class ExtractedDocument:
    doc_id: str
    filename: str
    file_type: str
    pages: list[ExtractedPage]
    warnings: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class ExtractionOutput:
    session_id: str
    documents: list[ExtractedDocument]


@dataclass(frozen=True)
class TextChunk:
    chunk_id: str
    doc_id: str
    filename: str
    page_number: int
    text: str

    @property
    def source_label(self) -> str:
        return f"{self.filename}, page {self.page_number}"


@dataclass(frozen=True)
class IndexingOutput:
    session_id: str
    collection_name: str
    chunks_indexed: int
    status: str


@dataclass(frozen=True)
class SearchMatch:
    chunk: TextChunk
    similarity_score: float

    @property
    def score(self) -> float:
        return self.similarity_score


@dataclass(frozen=True)
class Source:
    document: str
    page: int
    chunk_id: str
    similarity_score: float

    @property
    def label(self) -> str:
        return f"{self.document}, page {self.page} ({self.similarity_score:.2f})"


class RetrievalMode(str, Enum):
    FOCUSED = "focused"
    BROAD = "broad"
    EXHAUSTIVE = "exhaustive"


@dataclass(frozen=True)
class RetrievalOutput:
    mode: RetrievalMode
    matches: list[SearchMatch]
    sources: list[Source]


@dataclass(frozen=True)
class ChatTurn:
    role: str
    content: str


@dataclass(frozen=True)
class RagOutput:
    answer: str
    sources: list[Source]
    prompt: str
    matches: list[SearchMatch]
    retrieval_mode: RetrievalMode = RetrievalMode.FOCUSED
