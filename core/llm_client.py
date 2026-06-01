from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from typing import Any, Protocol
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from core.config import (
    GROQ_API_BASE_URL,
    GROQ_MAX_RETRIES,
    GROQ_MODEL,
    GROQ_RETRY_BASE_SECONDS,
    GROQ_TIMEOUT_SECONDS,
)


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
    max_retries: int = GROQ_MAX_RETRIES
    retry_base_seconds: float = GROQ_RETRY_BASE_SECONDS


class JsonHttpTransport(Protocol):
    """Small HTTP boundary so the provider client is easy to test or replace."""

    def post_json(
        self,
        url: str,
        payload: dict[str, Any],
        headers: dict[str, str],
        timeout: int,
        max_retries: int,
        retry_base_seconds: float,
    ) -> dict[str, Any]:
        """POST JSON and return a decoded JSON object."""


class UrllibJsonHttpTransport:
    """Standard-library HTTP transport used to avoid an extra runtime dependency."""

    def post_json(
        self,
        url: str,
        payload: dict[str, Any],
        headers: dict[str, str],
        timeout: int,
        max_retries: int,
        retry_base_seconds: float,
    ) -> dict[str, Any]:
        request = Request(
            url=url,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )

        for attempt in range(max_retries + 1):
            try:
                with urlopen(request, timeout=timeout) as response:
                    return json.loads(response.read().decode("utf-8"))
            except HTTPError as exc:
                body = exc.read().decode("utf-8", errors="replace")
                if exc.code == 429 and attempt < max_retries:
                    time.sleep(retry_delay_seconds(body, attempt, retry_base_seconds))
                    continue
                detail = extract_error_detail(body)
                if exc.code == 403 and "1010" in body:
                    detail = (
                        f"{detail}. Groq's edge rejected the request before normal API handling. "
                        "Confirm your network/VPN is allowed by Groq, then retry. If it persists, "
                        "install Groq's official SDK and swap this transport."
                    )
                raise LLMProviderError(f"Groq API request failed with HTTP {exc.code}: {detail}") from exc
            except URLError as exc:
                if attempt < max_retries:
                    time.sleep(retry_base_seconds * (attempt + 1))
                    continue
                raise LLMProviderError(f"Could not reach Groq API: {exc.reason}") from exc
            except TimeoutError as exc:
                if attempt < max_retries:
                    time.sleep(retry_base_seconds * (attempt + 1))
                    continue
                raise LLMProviderError("Groq API request timed out.") from exc

        raise LLMProviderError("Groq API request failed after retries.")


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
            max_retries=self.config.max_retries,
            retry_base_seconds=self.config.retry_base_seconds,
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


def retry_delay_seconds(body: str, attempt: int, retry_base_seconds: float) -> float:
    match = re.search(r"try again in ([0-9.]+)s", body, flags=re.IGNORECASE)
    if match:
        return float(match.group(1)) + 0.5
    return retry_base_seconds * (2**attempt)
