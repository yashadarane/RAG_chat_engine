from __future__ import annotations

from pathlib import Path

from core.config import MAX_DOCUMENTS, ensure_data_dirs
from core.schemas import IngestionOutput, ValidatedDocument
from utils.file_utils import make_session_id, save_uploaded_file, validate_file


class DocumentIngestionAgent:
    """Agent 1: validate uploaded files and produce a session-level document contract."""

    def validate_paths(self, paths: list[Path], session_id: str | None = None) -> IngestionOutput:
        ensure_data_dirs()
        session_id = session_id or make_session_id()
        documents: list[ValidatedDocument] = []

        for index, path in enumerate(paths, start=1):
            if index > MAX_DOCUMENTS:
                documents.append(
                    ValidatedDocument(
                        session_id=session_id,
                        doc_id=f"doc_{index}",
                        filename=path.name,
                        file_path=path,
                        file_type=path.suffix.lower().removeprefix("."),
                        page_count=None,
                        status="rejected",
                        issues=[f"Maximum {MAX_DOCUMENTS} documents are allowed per session."],
                    )
                )
                continue

            page_count, issues = validate_file(path)
            documents.append(
                ValidatedDocument(
                    session_id=session_id,
                    doc_id=f"doc_{index}",
                    filename=path.name,
                    file_path=path,
                    file_type=path.suffix.lower().removeprefix("."),
                    page_count=page_count,
                    status="validated" if not issues else "rejected",
                    issues=issues,
                )
            )

        return IngestionOutput(session_id=session_id, documents=documents)

    def validate_uploads(self, uploaded_files) -> IngestionOutput:
        session_id = make_session_id()
        paths = [save_uploaded_file(uploaded, session_id) for uploaded in uploaded_files]
        return self.validate_paths(paths, session_id=session_id)
