from __future__ import annotations

import os
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent


def load_environment() -> None:
    env_path = BASE_DIR / ".env"
    if not env_path.exists():
        return

    try:
        from dotenv import load_dotenv
    except ImportError:
        for line in env_path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or "=" not in stripped:
                continue

            key, value = stripped.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip("\"'"))
        return

    load_dotenv(env_path, override=False)


load_environment()

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
GROQ_MAX_RETRIES = int(os.getenv("GROQ_MAX_RETRIES", "3"))
GROQ_RETRY_BASE_SECONDS = float(os.getenv("GROQ_RETRY_BASE_SECONDS", "2"))
TOP_K = 5
MIN_SIMILARITY = 0.25
CHUNK_WORDS = 420
CHUNK_OVERLAP_WORDS = 60


def ensure_data_dirs() -> None:
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
