from __future__ import annotations

from core.llm_client import GroqLLMClient
from core.ports import LanguageModelClient
from core.prompts import build_grounded_prompt
from core.schemas import ChatTurn, RagOutput, SearchMatch, Source


class ConversationAgent:
    """Agent 4B: build grounded prompts and generate final responses."""

    def __init__(self, llm_client: LanguageModelClient | None = None) -> None:
        self.llm_client = llm_client or GroqLLMClient()

    def answer(
        self,
        query: str,
        history: list[ChatTurn],
        matches: list[SearchMatch],
        sources: list[Source],
    ) -> RagOutput:
        prompt = build_grounded_prompt(query, history[-6:], matches)

        if not matches:
            return RagOutput(
                answer="I could not find that in the uploaded documents.",
                sources=[],
                prompt=prompt,
                matches=[],
            )

        try:
            answer = self.llm_client.generate(prompt)
        except Exception as exc:
            answer = (
                "Groq could not generate a response. Check that GROQ_API_KEY is set "
                f"and that the API is reachable. Error: {exc}"
            )

        return RagOutput(answer=answer, sources=sources, prompt=prompt, matches=matches)
