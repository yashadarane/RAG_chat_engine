from __future__ import annotations

import re


def clean_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def chunk_text_by_boundaries(text: str, chunk_words: int, overlap_words: int) -> list[str]:
    words = clean_text(text).split()
    if not words:
        return []
    if len(words) <= chunk_words:
        return [" ".join(words)]

    chunks: list[str] = []
    start = 0
    while start < len(words):
        end = min(start + chunk_words, len(words))
        chunks.append(" ".join(words[start:end]).strip())
        if end == len(words):
            break
        start = max(end - overlap_words, start + 1)
    return chunks


def split_sentences(text: str) -> list[str]:
    sentences = re.split(r"(?<=[.!?])\s+", clean_text(text))
    return [sentence.strip() for sentence in sentences if len(sentence.strip()) > 20]
