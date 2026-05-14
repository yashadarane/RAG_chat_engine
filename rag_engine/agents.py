from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pdfplumber
from PIL import Image, UnidentifiedImageError
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .models import (
    AnswerResponse,
    ChatTurn,
    ExtractedDocument,
    ExtractedPage,
    ExtractionReport,
    SearchMatch,
    Source,
    TextChunk,
    ValidatedDocument,
    ValidationReport,
)


SUPPORTED_IMAGE_TYPES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
SUPPORTED_TYPES = {".pdf", *SUPPORTED_IMAGE_TYPES}
MAX_DOCUMENTS = 3
MAX_PAGES = 5
MAX_FILE_SIZE_BYTES = 20 * 1024 * 1024


class DocumentIngestionAgent:
    """Validates uploaded files and emits a structured handoff payload."""

    def validate(self, paths: list[Path]) -> ValidationReport:
        documents: list[ValidatedDocument] = []

        if len(paths) > MAX_DOCUMENTS:
            overflow = paths[MAX_DOCUMENTS:]
            paths = paths[:MAX_DOCUMENTS]
            for path in overflow:
                documents.append(
                    ValidatedDocument(
                        path=path,
                        name=path.name,
                        file_type=path.suffix.lower(),
                        page_count=None,
                        size_bytes=path.stat().st_size if path.exists() else 0,
                        is_valid=False,
                        issues=[f"Maximum {MAX_DOCUMENTS} documents are allowed per session."],
                    )
                )

        for path in paths:
            documents.append(self._validate_one(path))

        return ValidationReport(documents=documents)

    def _validate_one(self, path: Path) -> ValidatedDocument:
        issues: list[str] = []
        suffix = path.suffix.lower()
        size = path.stat().st_size if path.exists() else 0
        page_count: int | None = None

        if suffix not in SUPPORTED_TYPES:
            issues.append("Unsupported format. Upload PDF, PNG, JPG, TIFF, or BMP.")
        if size > MAX_FILE_SIZE_BYTES:
            issues.append("File is larger than the 20 MB prototype limit.")

        if suffix == ".pdf":
            try:
                reader = PdfReader(str(path))
                page_count = len(reader.pages)
                if page_count > MAX_PAGES:
                    issues.append(f"PDF has {page_count} pages; maximum allowed is {MAX_PAGES}.")
            except Exception as exc:
                issues.append(f"PDF is corrupt or unreadable: {exc}")
        elif suffix in SUPPORTED_IMAGE_TYPES:
            try:
                with Image.open(path) as image:
                    image.verify()
                page_count = 1
            except (UnidentifiedImageError, OSError) as exc:
                issues.append(f"Image is corrupt or unreadable: {exc}")

        return ValidatedDocument(
            path=path,
            name=path.name,
            file_type=suffix.removeprefix("."),
            page_count=page_count,
            size_bytes=size,
            is_valid=not issues,
            issues=issues,
        )


class TextExtractionAgent:
    """Extracts page-level text from validated PDFs and images."""

    def extract(self, documents: list[ValidatedDocument]) -> ExtractionReport:
        extracted = [self._extract_one(doc) for doc in documents]
        return ExtractionReport(documents=extracted)

    def _extract_one(self, document: ValidatedDocument) -> ExtractedDocument:
        if document.file_type == "pdf":
            return self._extract_pdf(document)
        return self._extract_image(document)

    def _extract_pdf(self, document: ValidatedDocument) -> ExtractedDocument:
        pages: list[ExtractedPage] = []
        warnings: list[str] = []

        try:
            with pdfplumber.open(document.path) as pdf:
                for index, page in enumerate(pdf.pages[:MAX_PAGES], start=1):
                    text = page.extract_text(x_tolerance=1, y_tolerance=3) or ""
                    pages.append(ExtractedPage(page_number=index, text=clean_text(text)))
                    if not text.strip():
                        warnings.append(
                            f"Page {index} has no embedded text. OCR is required for scanned content."
                        )
        except Exception as exc:
            warnings.append(f"Could not extract PDF text: {exc}")

        return ExtractedDocument(
            name=document.name,
            file_type=document.file_type,
            pages=pages,
            warnings=warnings,
        )

    def _extract_image(self, document: ValidatedDocument) -> ExtractedDocument:
        warnings: list[str] = []
        text = ""
        try:
            import pytesseract

            text = pytesseract.image_to_string(Image.open(document.path))
            if not text.strip():
                warnings.append("OCR produced no text for this image.")
        except Exception as exc:
            warnings.append(
                "Image OCR is unavailable. Install the Tesseract binary and ensure it is on PATH. "
                f"Underlying error: {exc}"
            )

        return ExtractedDocument(
            name=document.name,
            file_type=document.file_type,
            pages=[ExtractedPage(page_number=1, text=clean_text(text))],
            warnings=warnings,
        )


class ChunkingEmbeddingAgent:
    """Chunks extracted text and builds a local vector index."""

    def __init__(self, chunk_words: int = 420, overlap_words: int = 60) -> None:
        self.chunk_words = chunk_words
        self.overlap_words = overlap_words

    def index(self, documents: list[ExtractedDocument]) -> "VectorStore":
        chunks: list[TextChunk] = []
        for document in documents:
            for page in document.pages:
                for idx, chunk_text in enumerate(
                    chunk_text_by_boundaries(page.text, self.chunk_words, self.overlap_words), start=1
                ):
                    chunks.append(
                        TextChunk(
                            id=f"{document.name}:{page.page_number}:{idx}",
                            document_name=document.name,
                            page_number=page.page_number,
                            text=chunk_text,
                        )
                    )
        return VectorStore(chunks)


@dataclass
class VectorStore:
    chunks: list[TextChunk]

    def __post_init__(self) -> None:
        self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=12000)
        corpus = [chunk.text for chunk in self.chunks] or ["empty"]
        self.matrix = self.vectorizer.fit_transform(corpus)

    def search(self, query: str, top_k: int = 5, min_score: float = 0.08) -> list[SearchMatch]:
        if not self.chunks:
            return []
        query_vector = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vector, self.matrix).flatten()
        ranked = np.argsort(scores)[::-1][:top_k]
        return [
            SearchMatch(chunk=self.chunks[index], score=float(scores[index]))
            for index in ranked
            if scores[index] >= min_score
        ]


class RetrievalConversationAgent:
    """Retrieves relevant chunks and composes grounded, explainable answers."""

    def __init__(self, vector_store: VectorStore, top_k: int = 5, min_score: float = 0.08) -> None:
        self.vector_store = vector_store
        self.top_k = top_k
        self.min_score = min_score

    def answer(self, query: str, history: list[ChatTurn]) -> AnswerResponse:
        matches = self.vector_store.search(query, self.top_k, self.min_score)
        prompt = build_prompt(query, history[-6:], matches)

        if not matches:
            return AnswerResponse(
                answer="I could not find that in the uploaded documents.",
                prompt=prompt,
                matches=[],
                sources=[],
            )

        answer = compose_extractive_answer(query, matches)
        sources = dedupe_sources(
            Source(match.chunk.document_name, match.chunk.page_number) for match in matches[:3]
        )
        return AnswerResponse(answer=answer, prompt=prompt, matches=matches, sources=sources)


def clean_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def chunk_text_by_boundaries(text: str, chunk_words: int, overlap_words: int) -> list[str]:
    words = text.split()
    if not words:
        return []
    if len(words) <= chunk_words:
        return [" ".join(words)]

    chunks: list[str] = []
    start = 0
    while start < len(words):
        end = min(start + chunk_words, len(words))
        chunk = " ".join(words[start:end])
        chunks.append(chunk.strip())
        if end == len(words):
            break
        start = max(end - overlap_words, start + 1)
    return chunks


def build_prompt(query: str, history: list[ChatTurn], matches: list[SearchMatch]) -> str:
    context = "\n\n".join(
        f"[{idx}] {match.chunk.source_label} (score {match.score:.3f})\n{match.chunk.text}"
        for idx, match in enumerate(matches, start=1)
    )
    turns = "\n".join(f"{turn.role}: {turn.content}" for turn in history)
    return (
        "You are a document-grounded assistant. Answer only from the retrieved context. "
        "If the answer is not present, say it is not found in the uploaded documents.\n\n"
        f"Recent conversation:\n{turns or 'No prior turns.'}\n\n"
        f"Retrieved context:\n{context or 'No relevant context retrieved.'}\n\n"
        f"User question: {query}"
    )


def compose_extractive_answer(query: str, matches: list[SearchMatch]) -> str:
    query_terms = set(re.findall(r"[a-zA-Z0-9]+", query.lower()))
    sentences: list[tuple[float, str, str]] = []

    for match in matches:
        for sentence in split_sentences(match.chunk.text):
            sentence_terms = set(re.findall(r"[a-zA-Z0-9]+", sentence.lower()))
            overlap = len(query_terms & sentence_terms)
            score = match.score + (overlap * 0.04)
            if overlap or match == matches[0]:
                sentences.append((score, sentence, match.chunk.source_label))

    if not sentences:
        return "I found related context, but it does not contain a clear answer to the question."

    selected = sorted(sentences, key=lambda item: item[0], reverse=True)[:4]
    selected = sorted(selected, key=lambda item: matches_source_order(item[2], matches))
    answer_lines = [f"{sentence} [{source}]" for _, sentence, source in selected]
    return " ".join(answer_lines)


def split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [part.strip() for part in parts if len(part.strip()) > 20]


def matches_source_order(label: str, matches: list[SearchMatch]) -> int:
    for index, match in enumerate(matches):
        if match.chunk.source_label == label:
            return index
    return len(matches)


def dedupe_sources(sources) -> list[Source]:
    seen: set[tuple[str, int]] = set()
    unique: list[Source] = []
    for source in sources:
        key = (source.document_name, source.page_number)
        if key not in seen:
            seen.add(key)
            unique.append(source)
    return unique
