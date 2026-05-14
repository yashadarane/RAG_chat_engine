from pathlib import Path

from agents import ChunkingEmbeddingAgent, DocumentIngestionAgent, RagConversationAgent
from core.schemas import ChatTurn, ExtractedDocument, ExtractedPage


class FakeEmbeddings:
    terms = ["claims", "waiting", "period", "dental", "coverage", "car", "premium"]

    def __call__(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for text in texts:
            lower = text.lower()
            vector = [1.0 if term in lower else 0.0 for term in self.terms]
            vectors.append(vector)
        return vectors


class FakeLLM:
    def generate(self, prompt: str) -> str:
        if "thirty days" in prompt:
            return "The claims waiting period is thirty days. [policy.pdf, page 2]"
        return "I could not find that in the uploaded documents."


def test_ingestion_rejects_unsupported_file(tmp_path: Path) -> None:
    bad_file = tmp_path / "notes.txt"
    bad_file.write_text("hello", encoding="utf-8")

    report = DocumentIngestionAgent().validate_paths([bad_file], session_id="testreject")

    assert not report.documents[0].is_valid
    assert "Unsupported format" in report.documents[0].issues[0]


def test_retrieval_returns_grounded_source() -> None:
    document = ExtractedDocument(
        doc_id="doc_1",
        filename="policy.pdf",
        file_type="pdf",
        pages=[
            ExtractedPage(
                page_number=2,
                text="The claims waiting period is thirty days. Dental exclusions are listed separately.",
            )
        ],
    )
    store, _ = ChunkingEmbeddingAgent(
        chunk_words=80,
        overlap_words=10,
        embedding_function=FakeEmbeddings(),
    ).index("testretrieval", [document])

    response = RagConversationAgent(store, llm_client=FakeLLM()).answer(
        "What is the claims waiting period?",
        [ChatTurn(role="user", content="What is the claims waiting period?")],
    )

    assert "thirty days" in response.answer
    assert response.sources[0].document == "policy.pdf"
    assert response.sources[0].page == 2


def test_no_match_avoids_hallucination() -> None:
    document = ExtractedDocument(
        doc_id="doc_1",
        filename="policy.pdf",
        file_type="pdf",
        pages=[ExtractedPage(page_number=1, text="Only dental coverage information appears here.")],
    )
    store, _ = ChunkingEmbeddingAgent(embedding_function=FakeEmbeddings()).index("testnomatch", [document])

    response = RagConversationAgent(store, llm_client=FakeLLM(), min_similarity=0.5).answer(
        "What is the car premium?",
        [],
    )

    assert response.answer == "I could not find that in the uploaded documents."
