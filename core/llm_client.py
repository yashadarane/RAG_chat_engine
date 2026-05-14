from __future__ import annotations

import ollama

from core.config import OLLAMA_MODEL


class OllamaLLMClient:
    def __init__(self, model: str = OLLAMA_MODEL) -> None:
        self.model = model

    def generate(self, prompt: str) -> str:
        response = ollama.generate(model=self.model, prompt=prompt)
        return str(response.get("response", "")).strip()
