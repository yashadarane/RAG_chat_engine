from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class ValidatedDocument:
    path: Path
    name: str
    file_type: str
    page_count: int | None
    size_bytes: int
    is_valid: bool
    issues: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class ValidationReport:
    documents: list[ValidatedDocument]


@dataclass(frozen=True)
class ExtractedPage:
    page_number: int
    text: str


@dataclass(frozen=True)
class ExtractedDocument:
    name: str
    file_type: str
    pages: list[ExtractedPage]
    warnings: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class ExtractionReport:
    documents: list[ExtractedDocument]


@dataclass(frozen=True)
class TextChunk:
    id: str
    document_name: str
    page_number: int
    text: str

    @property
    def source_label(self) -> str:
        return f"{self.document_name}, page {self.page_number}"


@dataclass(frozen=True)
class SearchMatch:
    chunk: TextChunk
    score: float


@dataclass(frozen=True)
class Source:
    document_name: str
    page_number: int

    @property
    def label(self) -> str:
        return f"{self.document_name}, page {self.page_number}"


@dataclass(frozen=True)
class ChatTurn:
    role: str
    content: str


@dataclass(frozen=True)
class AnswerResponse:
    answer: str
    prompt: str
    matches: list[SearchMatch]
    sources: list[Source]
