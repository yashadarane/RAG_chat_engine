from __future__ import annotations

import json

import streamlit as st

from agents import (
    DocumentIngestionAgent,
)
from core.config import GROQ_MODEL, ensure_data_dirs
from core.graph_orchestration import RagGraphOrchestrator
from core.schemas import ChatTurn, RagOutput
from utils.transcript_export import conversation_to_pdf, conversation_to_text


st.set_page_config(page_title="RAG Chat Engine", page_icon="A", layout="wide")


def initialize_state() -> None:
    ensure_data_dirs()
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None
    if "pipeline_report" not in st.session_state:
        st.session_state.pipeline_report = None
    if "latest_query_state" not in st.session_state:
        st.session_state.latest_query_state = None
    if "latest_question" not in st.session_state:
        st.session_state.latest_question = None
    if "latest_response" not in st.session_state:
        st.session_state.latest_response = None
    if "live_ragas_result" not in st.session_state:
        st.session_state.live_ragas_result = None


def process_uploads(uploaded_files) -> None:
    ingestion_agent = DocumentIngestionAgent()
    upload_ingestion = ingestion_agent.validate_uploads(uploaded_files)
    pipeline = RagGraphOrchestrator().process_documents([doc.file_path for doc in upload_ingestion.documents])
    ingestion = pipeline.get("ingestion", upload_ingestion)
    extraction = pipeline.get("extraction")
    vector_store = pipeline.get("vector_store")
    indexing = pipeline.get("indexing")

    st.session_state.vector_store = vector_store
    st.session_state.pipeline_report = {
        "ingestion": ingestion,
        "extraction": extraction,
        "indexing": indexing,
        "graph_state": pipeline,
    }
    st.session_state.chat_history = []
    st.session_state.latest_question = None
    st.session_state.latest_response = None
    st.session_state.latest_query_state = None
    st.session_state.live_ragas_result = None


def render_pipeline_report() -> None:
    report = st.session_state.pipeline_report
    if not report:
        st.info("Upload documents to build the retrieval index.")
        return

    if report.get("graph_state", {}).get("errors"):
        st.error("Document processing stopped before the retrieval index was ready.")
        with st.expander("Agent handoff report", expanded=True):
            render_agent_state(report.get("graph_state", {}))
        return

    ingestion = report["ingestion"]
    extraction = report["extraction"]
    indexing = report["indexing"]
    if extraction is None or indexing is None:
        st.warning("Document processing did not produce a complete pipeline report.")
        with st.expander("Agent handoff report", expanded=True):
            render_agent_state(report.get("graph_state", {}))
        return

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
        render_agent_state(report.get("graph_state", {}))


def render_chat() -> None:
    vector_store = st.session_state.vector_store
    if vector_store is None:
        st.chat_message("assistant").write("Upload and index documents first, then ask me about them.")
        return

    message_container = st.container()
    export_container = st.container()

    query = st.chat_input("Ask a question grounded in the uploaded documents", key="document_chat_input")

    latest_response: RagOutput | None = None
    if query:
        st.session_state.chat_history.append(ChatTurn(role="user", content=query))
        graph_state = RagGraphOrchestrator().answer_question(query, st.session_state.chat_history[-6:], vector_store)
        st.session_state.latest_query_state = graph_state
        latest_response = graph_state.get("response")
        if latest_response is None:
            errors = graph_state.get("errors", ["The query pipeline did not return a response."])
            latest_response = RagOutput(
                answer=f"Agent flow failed: {errors[-1]}",
                sources=[],
                prompt="",
                matches=[],
            )
        st.session_state.chat_history.append(ChatTurn(role="assistant", content=latest_response.answer))
        st.session_state.latest_question = query
        st.session_state.latest_response = latest_response
        st.session_state.live_ragas_result = None

    response_for_tools = latest_response or st.session_state.latest_response

    with message_container:
        for turn in st.session_state.chat_history:
            st.chat_message(turn.role).write(turn.content)

        if response_for_tools:
            if response_for_tools.sources:
                st.caption("Sources: " + "; ".join(source.label for source in response_for_tools.sources))
            render_live_ragas(response_for_tools)
            with st.expander("Retrieval transparency", expanded=False):
                render_agent_state(st.session_state.latest_query_state or {})
                render_retrieval_transparency(response_for_tools)

    with export_container:
        render_conversation_exports()


def render_agent_state(graph_state: dict) -> None:
    if not graph_state:
        return

    st.write("**LangGraph state**")
    st.code(
        json.dumps(
            {
                "current_step": graph_state.get("current_step"),
                "completed_steps": graph_state.get("completed_steps", []),
                "errors": graph_state.get("errors", []),
                "agent_steps": graph_state.get("agent_steps", []),
            },
            indent=2,
        ),
        language="json",
    )


def render_live_ragas(response: RagOutput) -> None:
    question = st.session_state.latest_question
    evaluate_clicked = st.button(
        "Evaluate Answer Quality",
        key="evaluate_latest_answer_quality",
        use_container_width=True,
    )

    if evaluate_clicked:
        from evals.live_ragas import evaluate_live_answer

        contexts = [match.chunk.text for match in response.matches]
        with st.spinner("Evaluating answer quality against retrieved context..."):
            st.session_state.live_ragas_result = evaluate_live_answer(
                question=question or "",
                answer=response.answer,
                retrieved_contexts=contexts,
            )

    result = st.session_state.live_ragas_result
    if not result:
        return

    if result.get("status") == "success":
        metrics = result.get("metrics", {})
        faithfulness = metrics.get("faithfulness")
        response_relevancy = metrics.get("response_relevancy")
        faith_col, relevancy_col = st.columns(2)

        with faith_col:
            st.metric(
                "Faithfulness",
                format_metric_score(faithfulness),
                help="Measures whether the answer is supported by retrieved document context.",
            )
        with relevancy_col:
            st.metric(
                "Response Relevancy",
                format_metric_score(response_relevancy),
                help="Measures whether the answer addresses the user question.",
            )

        st.info(
            "These scores are evaluator-based estimates using retrieved context only. "
            "They do not guarantee that retrieval found every relevant document."
        )
        return

    st.warning(f"RAGAS evaluation unavailable: {result.get('reason', 'Unknown error')}")


def format_metric_score(value: float | None) -> str:
    if value is None:
        return "N/A"
    return f"{value:.3f}"


def render_retrieval_transparency(response: RagOutput) -> None:
    st.write(f"**Agent 4A retrieval mode:** `{response.retrieval_mode.value}`")
    st.write(f"**Agent 4A document scope:** `{response.retrieval_scope.value}`")
    if response.selected_documents:
        st.write("**Selected documents:** " + ", ".join(response.selected_documents))
    st.write("**Agent 4B output**")
    st.code(
        json.dumps(
            {
                "answer": response.answer,
                "sources": [
                    {
                        "document": source.document,
                        "page": source.page,
                        "chunk_id": source.chunk_id,
                        "similarity_score": (
                            round(source.similarity_score, 3)
                            if source.similarity_score is not None
                            else None
                        ),
                        "retrieval_reason": source.retrieval_reason,
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
    st.write("**Agent 4A retrieved chunks**")
    for match in response.matches:
        score = "complete_context" if match.similarity_score is None else f"{match.similarity_score:.3f}"
        st.markdown(f"- `{score}` {match.chunk.source_label} ({match.retrieval_reason})")


def render_conversation_exports() -> None:
    history = st.session_state.chat_history
    if not history:
        return

    txt_data = conversation_to_text(history).encode("utf-8")
    pdf_data = conversation_to_pdf(history)
    txt_col, pdf_col = st.columns(2)
    with txt_col:
        st.download_button(
            "Download TXT",
            data=txt_data,
            file_name="rag_chat_conversation.txt",
            mime="text/plain",
            use_container_width=True,
        )
    with pdf_col:
        st.download_button(
            "Download PDF",
            data=pdf_data,
            file_name="rag_chat_conversation.pdf",
            mime="application/pdf",
            use_container_width=True,
        )


def main() -> None:
    initialize_state()

    st.title("Multi-Agent RAG Chat Engine")
    st.caption(
        "Streamlit + PyMuPDF + EasyOCR + sentence-transformers/all-MiniLM-L6-v2 + "
        f"ChromaDB + Groq `{GROQ_MODEL}`"
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
