from __future__ import annotations

import tempfile
from pathlib import Path

import streamlit as st

from rag_engine.agents import (
    ChunkingEmbeddingAgent,
    DocumentIngestionAgent,
    RetrievalConversationAgent,
    TextExtractionAgent,
)
from rag_engine.models import ChatTurn


st.set_page_config(page_title="RAG Chat Engine", page_icon="A", layout="wide")


def initialize_state() -> None:
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

    with tempfile.TemporaryDirectory() as tmpdir:
        saved_paths: list[Path] = []
        for uploaded in uploaded_files:
            path = Path(tmpdir) / uploaded.name
            path.write_bytes(uploaded.getbuffer())
            saved_paths.append(path)

        validation = ingestion_agent.validate(saved_paths)
        valid_docs = [doc for doc in validation.documents if doc.is_valid]
        extracted = extraction_agent.extract(valid_docs)
        vector_store = chunking_agent.index(extracted.documents)

    st.session_state.vector_store = vector_store
    st.session_state.pipeline_report = {
        "validation": validation,
        "extraction": extracted,
        "chunks": len(vector_store.chunks),
    }
    st.session_state.chat_history = []


def render_pipeline_report() -> None:
    report = st.session_state.pipeline_report
    if not report:
        st.info("Upload documents to build the retrieval index.")
        return

    validation = report["validation"]
    extraction = report["extraction"]
    st.success(f"Indexed {report['chunks']} chunks from {len(extraction.documents)} document(s).")

    with st.expander("Agent handoff report", expanded=False):
        st.write("**Agent 1 - validation**")
        for doc in validation.documents:
            icon = "OK" if doc.is_valid else "Skipped"
            st.write(f"{icon}: {doc.name} ({doc.file_type}, {doc.page_count or 'unknown'} page(s))")
            for issue in doc.issues:
                st.warning(f"{doc.name}: {issue}")

        st.write("**Agent 2 - extraction**")
        for doc in extraction.documents:
            extracted_chars = sum(len(page.text) for page in doc.pages)
            st.write(f"{doc.name}: {len(doc.pages)} page(s), {extracted_chars:,} extracted characters")
            for warning in doc.warnings:
                st.warning(f"{doc.name}: {warning}")


def render_chat() -> None:
    vector_store = st.session_state.vector_store
    if vector_store is None:
        st.chat_message("assistant").write("Upload and index documents first, then ask me about them.")
        return

    agent = RetrievalConversationAgent(vector_store=vector_store)

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
            st.write("**Structured prompt**")
            st.code(response.prompt, language="text")
            st.write("**Retrieved chunks**")
            for match in response.matches:
                st.markdown(f"- `{match.score:.3f}` {match.chunk.source_label}")


def main() -> None:
    initialize_state()

    st.title("Multi-Agent RAG Chat Engine")

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
