from __future__ import annotations

import json

import streamlit as st

from agents import (
    ChunkingEmbeddingAgent,
    DocumentIngestionAgent,
    RagConversationAgent,
    TextExtractionAgent,
)
from core.config import OLLAMA_MODEL, ensure_data_dirs
from core.schemas import ChatTurn


st.set_page_config(page_title="RAG Chat Engine", page_icon="A", layout="wide")


def initialize_state() -> None:
    ensure_data_dirs()
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None
    if "pipeline_report" not in st.session_state:
        st.session_state.pipeline_report = None


def process_uploads(uploaded_files) -> None:
    ingestion_agent = DocumentIngestionAgent()
    extraction_agent = TextExtractionAgent()
    chunking_agent = ChunkingEmbeddingAgent()

    ingestion = ingestion_agent.validate_uploads(uploaded_files)
    extraction = extraction_agent.extract(ingestion.session_id, ingestion.documents)
    vector_store, indexing = chunking_agent.index(ingestion.session_id, extraction.documents)

    st.session_state.vector_store = vector_store
    st.session_state.pipeline_report = {
        "ingestion": ingestion,
        "extraction": extraction,
        "indexing": indexing,
    }
    st.session_state.chat_history = []


def render_pipeline_report() -> None:
    report = st.session_state.pipeline_report
    if not report:
        st.info("Upload documents to build the retrieval index.")
        return

    ingestion = report["ingestion"]
    extraction = report["extraction"]
    indexing = report["indexing"]
    st.success(
        f"Indexed {indexing.chunks_indexed} chunks into `{indexing.collection_name}` "
        f"from {len(extraction.documents)} document(s)."
    )

    with st.expander("Agent handoff report", expanded=False):
        st.write("**Agent 1 - validation**")
        st.code(
            json.dumps(
                {
                "session_id": ingestion.session_id,
                "documents": [
                    {
                        "doc_id": doc.doc_id,
                        "filename": doc.filename,
                        "file_path": str(doc.file_path),
                        "file_type": doc.file_type,
                        "page_count": doc.page_count,
                        "status": doc.status,
                    }
                    for doc in ingestion.documents
                ],
            },
                indent=2,
            ),
            language="json",
        )
        for doc in ingestion.documents:
            icon = "OK" if doc.is_valid else "Skipped"
            st.write(f"{icon}: {doc.filename} ({doc.file_type}, {doc.page_count or 'unknown'} page(s))")
            for issue in doc.issues:
                st.warning(f"{doc.filename}: {issue}")

        st.write("**Agent 2 - extraction**")
        for doc in extraction.documents:
            extracted_chars = sum(len(page.text) for page in doc.pages)
            st.write(f"{doc.filename}: {len(doc.pages)} page(s), {extracted_chars:,} extracted characters")
            for warning in doc.warnings:
                st.warning(f"{doc.filename}: {warning}")

        st.write("**Agent 3 - chunk + embed**")
        st.code(
            json.dumps(
                {
                "chunks_indexed": indexing.chunks_indexed,
                "collection_name": indexing.collection_name,
                "status": indexing.status,
            },
                indent=2,
            ),
            language="json",
        )


def render_chat() -> None:
    vector_store = st.session_state.vector_store
    if vector_store is None:
        st.chat_message("assistant").write("Upload and index documents first, then ask me about them.")
        return

    agent = RagConversationAgent(vector_store=vector_store)

    for turn in st.session_state.chat_history:
        st.chat_message(turn.role).write(turn.content)

    query = st.chat_input("Ask a question grounded in the uploaded documents")
    if not query:
        return

    st.session_state.chat_history.append(ChatTurn(role="user", content=query))
    st.chat_message("user").write(query)

    response = agent.answer(query, st.session_state.chat_history[-6:])
    st.session_state.chat_history.append(ChatTurn(role="assistant", content=response.answer))

    with st.chat_message("assistant"):
        st.write(response.answer)
        if response.sources:
            st.caption("Sources: " + "; ".join(source.label for source in response.sources))
        with st.expander("Retrieval transparency", expanded=False):
            st.write("**Agent 4 output**")
            st.code(
                json.dumps(
                    {
                    "answer": response.answer,
                    "sources": [
                        {
                            "document": source.document,
                            "page": source.page,
                            "chunk_id": source.chunk_id,
                            "similarity_score": round(source.similarity_score, 3),
                        }
                        for source in response.sources
                    ],
                },
                    indent=2,
                ),
                language="json",
            )
            st.write("**Structured prompt**")
            st.code(response.prompt, language="text")
            st.write("**Retrieved chunks**")
            for match in response.matches:
                st.markdown(f"- `{match.similarity_score:.3f}` {match.chunk.source_label}")


def main() -> None:
    initialize_state()

    st.title("Multi-Agent RAG Chat Engine")
    st.caption(
        "Streamlit + PyMuPDF + EasyOCR + sentence-transformers/all-MiniLM-L6-v2 + "
        f"ChromaDB + Ollama `{OLLAMA_MODEL}`"
    )

    left, right = st.columns([0.34, 0.66], gap="large")
    with left:
        st.subheader("Documents")
        uploaded_files = st.file_uploader(
            "Upload up to 3 PDFs or images",
            type=["pdf", "png", "jpg", "jpeg", "tif", "tiff", "bmp"],
            accept_multiple_files=True,
        )
        if uploaded_files and st.button("Process documents", type="primary", use_container_width=True):
            with st.spinner("Agents are validating, extracting, chunking, and indexing..."):
                process_uploads(uploaded_files)
        render_pipeline_report()

    with right:
        st.subheader("Conversation")
        render_chat()


if __name__ == "__main__":
    main()
