from __future__ import annotations

from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
CHROMA_DIR = DATA_DIR / "chroma_db"

MAX_DOCUMENTS = 3
MAX_PAGES = 5
MAX_FILE_SIZE_BYTES = 20 * 1024 * 1024
SUPPORTED_IMAGE_TYPES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
SUPPORTED_TYPES = {".pdf", *SUPPORTED_IMAGE_TYPES}

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
OLLAMA_MODEL = "mistral"
TOP_K = 5
MIN_SIMILARITY = 0.25
CHUNK_WORDS = 420
CHUNK_OVERLAP_WORDS = 60


def ensure_data_dirs() -> None:
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
