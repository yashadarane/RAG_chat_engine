from __future__ import annotations

import json

from core.llm_client import GroqLLMClient
from core.ports import LanguageModelClient


QUERY_REWRITER_PROMPT = """You are a query rewriting agent for a document-grounded RAG system.
Rewrite the user's question into multiple search queries that may retrieve relevant document chunks
even when the document uses different wording.

Rules:
- Preserve the user's intent.
- Include the original question as the first query.
- Generate synonyms, paraphrases, and related terminology.
- Do not answer the question.
- Do not add assumptions.
- Return only a JSON list of 3 to 5 strings.

User question:
{query}
"""


class LLMQueryRewriter:
    """LLM-backed query rewriting adapter."""

    def __init__(self, llm_client: LanguageModelClient | None = None) -> None:
        self.llm_client = llm_client

    def rewrite(self, query: str) -> list[str]:
        client = self.llm_client or GroqLLMClient()
        raw_response = client.generate(QUERY_REWRITER_PROMPT.format(query=query))
        parsed = json.loads(raw_response)
        if not isinstance(parsed, list):
            return [query]
        return [str(item).strip() for item in parsed if str(item).strip()]
