from pathlib import Path

from rag_engine.agents import ChunkingEmbeddingAgent, DocumentIngestionAgent, RetrievalConversationAgent
from rag_engine.models import ChatTurn, ExtractedDocument, ExtractedPage


def test_ingestion_rejects_unsupported_file(tmp_path: Path) -> None:
    bad_file = tmp_path / "notes.txt"
    bad_file.write_text("hello", encoding="utf-8")

    report = DocumentIngestionAgent().validate([bad_file])

    assert not report.documents[0].is_valid
    assert "Unsupported format" in report.documents[0].issues[0]


def test_retrieval_returns_grounded_source() -> None:
    document = ExtractedDocument(
        name="policy.pdf",
        file_type="pdf",
        pages=[
            ExtractedPage(
                page_number=2,
                text="The claims waiting period is thirty days. Dental exclusions are listed separately.",
            )
        ],
    )
    store = ChunkingEmbeddingAgent(chunk_words=80, overlap_words=10).index([document])

    response = RetrievalConversationAgent(store).answer(
        "What is the claims waiting period?",
        [ChatTurn(role="user", content="What is the claims waiting period?")],
    )

    assert "thirty days" in response.answer
    assert response.sources[0].document_name == "policy.pdf"
    assert response.sources[0].page_number == 2


def test_no_match_avoids_hallucination() -> None:
    document = ExtractedDocument(
        name="policy.pdf",
        file_type="pdf",
        pages=[ExtractedPage(page_number=1, text="Only dental coverage information appears here.")],
    )
    store = ChunkingEmbeddingAgent().index([document])

    response = RetrievalConversationAgent(store, min_score=0.5).answer("What is the car premium?", [])

    assert response.answer == "I could not find that in the uploaded documents."
