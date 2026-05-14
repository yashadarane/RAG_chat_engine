from __future__ import annotations

import fitz

from core.config import MAX_PAGES, SUPPORTED_IMAGE_TYPES
from core.schemas import ExtractedDocument, ExtractedPage, ExtractionOutput, ValidatedDocument
from utils.ocr_utils import ocr_image, ocr_pdf_page
from utils.text_utils import clean_text


class TextExtractionAgent:
    """Agent 2: extract page-level text with PyMuPDF and EasyOCR."""

    def extract(self, session_id: str, documents: list[ValidatedDocument]) -> ExtractionOutput:
        valid_documents = [document for document in documents if document.is_valid]
        return ExtractionOutput(
            session_id=session_id,
            documents=[self._extract_one(document) for document in valid_documents],
        )

    def _extract_one(self, document: ValidatedDocument) -> ExtractedDocument:
        if document.file_type == "pdf":
            return self._extract_pdf(document)
        if f".{document.file_type}" in SUPPORTED_IMAGE_TYPES:
            return self._extract_image(document)
        return ExtractedDocument(
            doc_id=document.doc_id,
            filename=document.filename,
            file_type=document.file_type,
            pages=[],
            warnings=["Unsupported document type reached extraction."],
        )

    def _extract_pdf(self, document: ValidatedDocument) -> ExtractedDocument:
        pages: list[ExtractedPage] = []
        warnings: list[str] = []

        try:
            with fitz.open(document.file_path) as pdf:
                for page_number in range(min(pdf.page_count, MAX_PAGES)):
                    page = pdf.load_page(page_number)
                    text = clean_text(page.get_text("text"))
                    if not text:
                        text = ocr_pdf_page(page)
                        if not text:
                            warnings.append(f"Page {page_number + 1} produced no text after OCR.")
                    pages.append(ExtractedPage(page_number=page_number + 1, text=text))
        except Exception as exc:
            warnings.append(f"Could not extract PDF text: {exc}")

        return ExtractedDocument(
            doc_id=document.doc_id,
            filename=document.filename,
            file_type=document.file_type,
            pages=pages,
            warnings=warnings,
        )

    def _extract_image(self, document: ValidatedDocument) -> ExtractedDocument:
        warnings: list[str] = []
        text = ""
        try:
            text = ocr_image(document.file_path)
            if not text:
                warnings.append("EasyOCR produced no text for this image.")
        except Exception as exc:
            warnings.append(f"EasyOCR failed for this image: {exc}")

        return ExtractedDocument(
            doc_id=document.doc_id,
            filename=document.filename,
            file_type=document.file_type,
            pages=[ExtractedPage(page_number=1, text=text)],
            warnings=warnings,
        )
