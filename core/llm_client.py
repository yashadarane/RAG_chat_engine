from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Protocol
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from core.config import GROQ_API_BASE_URL, GROQ_MODEL, GROQ_TIMEOUT_SECONDS


class LLMConfigurationError(RuntimeError):
    """Raised when required LLM configuration is absent."""


class LLMProviderError(RuntimeError):
    """Raised when the external LLM provider cannot produce a response."""


@dataclass(frozen=True)
class GroqClientConfig:
    api_key: str
    model: str = GROQ_MODEL
    base_url: str = GROQ_API_BASE_URL
    timeout_seconds: int = GROQ_TIMEOUT_SECONDS


class JsonHttpTransport(Protocol):
    """Small HTTP boundary so the provider client is easy to test or replace."""

    def post_json(self, url: str, payload: dict[str, Any], headers: dict[str, str], timeout: int) -> dict[str, Any]:
        """POST JSON and return a decoded JSON object."""


class UrllibJsonHttpTransport:
    """Standard-library HTTP transport used to avoid an extra runtime dependency."""

    def post_json(self, url: str, payload: dict[str, Any], headers: dict[str, str], timeout: int) -> dict[str, Any]:
        request = Request(
            url=url,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )

        try:
            with urlopen(request, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            detail = extract_error_detail(body)
            if exc.code == 403 and "1010" in body:
                detail = (
                    f"{detail}. Groq's edge rejected the request before normal API handling. "
                    "Confirm your network/VPN is allowed by Groq, then retry. If it persists, "
                    "install Groq's official SDK and swap this transport."
                )
            raise LLMProviderError(f"Groq API request failed with HTTP {exc.code}: {detail}") from exc
        except URLError as exc:
            raise LLMProviderError(f"Could not reach Groq API: {exc.reason}") from exc
        except TimeoutError as exc:
            raise LLMProviderError("Groq API request timed out.") from exc


class GroqLLMClient:
    """Groq chat-completions adapter behind the LanguageModelClient port."""

    def __init__(
        self,
        config: GroqClientConfig | None = None,
        transport: JsonHttpTransport | None = None,
    ) -> None:
        self.config = config or GroqClientConfig(api_key=self._read_api_key())
        self.transport = transport or UrllibJsonHttpTransport()

    def generate(self, prompt: str) -> str:
        payload = {
            "model": self.config.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.1,
            "stream": False,
        }
        response = self._post_json("/chat/completions", payload)
        try:
            return str(response["choices"][0]["message"]["content"]).strip()
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMProviderError("Groq returned an unexpected response shape.") from exc

    @staticmethod
    def _read_api_key() -> str:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise LLMConfigurationError("Missing Groq API key. Set GROQ_API_KEY in your environment.")
        return api_key

    def _post_json(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        return self.transport.post_json(
            url=f"{self.config.base_url.rstrip('/')}{path}",
            payload=payload,
            headers={
                "Authorization": f"Bearer {self.config.api_key}",
                "Accept": "application/json",
                "Content-Type": "application/json",
                "User-Agent": "rag-chat-engine/1.0",
            },
            timeout=self.config.timeout_seconds,
        )


class StaticLLMClient:
    """Small test double used by unit tests and local diagnostics."""

    def __init__(self, response: str) -> None:
        self.response = response

    def generate(self, prompt: str) -> str:
        return self.response


def extract_error_detail(body: str) -> str:
    try:
        parsed = json.loads(body)
    except json.JSONDecodeError:
        return body.strip() or "No error body returned."

    error = parsed.get("error")
    if isinstance(error, dict):
        message = error.get("message") or error.get("code") or error
        return str(message)
    return str(error or parsed)
