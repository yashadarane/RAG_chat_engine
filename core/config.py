from __future__ import annotations

import os
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

EMBEDDING_MODEL = str(BASE_DIR / "models" / "all-MiniLM-L6-v2")
GROQ_API_BASE_URL = os.getenv("GROQ_API_BASE_URL", "https://api.groq.com/openai/v1")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
GROQ_TIMEOUT_SECONDS = int(os.getenv("GROQ_TIMEOUT_SECONDS", "120"))
TOP_K = 5
MIN_SIMILARITY = 0.25
CHUNK_WORDS = 420
CHUNK_OVERLAP_WORDS = 60


def ensure_data_dirs() -> None:
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
