from __future__ import annotations

from pathlib import Path

import easyocr
import fitz
import numpy as np
from PIL import Image

from utils.text_utils import clean_text


_reader: easyocr.Reader | None = None


def get_easyocr_reader() -> easyocr.Reader:
    global _reader
    if _reader is None:
        _reader = easyocr.Reader(["en"], gpu=False)
    return _reader


def ocr_image(path: Path) -> str:
    reader = get_easyocr_reader()
    results = reader.readtext(str(path), detail=0, paragraph=True)
    return clean_text("\n".join(results))


def ocr_pdf_page(page: fitz.Page) -> str:
    pixmap = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
    mode = "RGB" if pixmap.n < 4 else "RGBA"
    image = Image.frombytes(mode, [pixmap.width, pixmap.height], pixmap.samples)
    reader = get_easyocr_reader()
    results = reader.readtext(np.array(image), detail=0, paragraph=True)
    return clean_text("\n".join(results))
