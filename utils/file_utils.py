from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import fitz
from PIL import Image, UnidentifiedImageError

from core.config import MAX_FILE_SIZE_BYTES, MAX_PAGES, SUPPORTED_IMAGE_TYPES, SUPPORTED_TYPES, UPLOAD_DIR


def make_session_id() -> str:
    return uuid4().hex[:12]


def save_uploaded_file(uploaded_file, session_id: str) -> Path:
    session_dir = UPLOAD_DIR / session_id
    session_dir.mkdir(parents=True, exist_ok=True)
    target = session_dir / uploaded_file.name
    target.write_bytes(uploaded_file.getbuffer())
    return target


def validate_file(path: Path) -> tuple[int | None, list[str]]:
    issues: list[str] = []
    suffix = path.suffix.lower()
    page_count: int | None = None

    if suffix not in SUPPORTED_TYPES:
        issues.append("Unsupported format. Upload PDF, PNG, JPG, TIFF, or BMP.")
        return page_count, issues

    if path.stat().st_size > MAX_FILE_SIZE_BYTES:
        issues.append("File is larger than the 20 MB prototype limit.")

    if suffix == ".pdf":
        try:
            with fitz.open(path) as pdf:
                page_count = pdf.page_count
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

    return page_count, issues
