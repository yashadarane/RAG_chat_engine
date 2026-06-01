from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, TypedDict

from langgraph.graph import END, StateGraph

from agents import ChunkingEmbeddingAgent, ConversationAgent, DocumentIngestionAgent, RetrievalAgent, TextExtractionAgent
from core.query_rewriter import LLMQueryRewriter
from core.reranker import build_default_reranker
from core.schemas import ChatTurn, RagOutput, RetrievalOutput


class AgentStepRecord(TypedDict):
    step: str
    agent: str
    status: Literal["completed", "failed"]
    message: str


class DocumentPipelineState(TypedDict, total=False):
    doc_paths: list[Path]
    ingestion: Any
    extraction: Any
    indexing: Any
    vector_store: Any
    current_step: str
    completed_steps: list[str]
    agent_steps: list[AgentStepRecord]
    errors: list[str]


class QueryPipelineState(TypedDict, total=False):
    query: str
    history: list[ChatTurn]
    vector_store: Any
    retrieval: RetrievalOutput
    response: RagOutput
    current_step: str
    completed_steps: list[str]
    agent_steps: list[AgentStepRecord]
    errors: list[str]


class RagGraphOrchestrator:
    """LangGraph coordinator for document and question-answering pipelines."""

    def __init__(
        self,
        query_rewriter=None,
        reranker=None,
        conversation_agent: ConversationAgent | None = None,
    ) -> None:
        self.query_rewriter = query_rewriter
        self.reranker = reranker
        self.conversation_agent = conversation_agent
        self.document_graph = self._build_document_graph()
        self.query_graph = self._build_query_graph()

    def process_documents(self, doc_paths: list[Path]) -> DocumentPipelineState:
        return self.document_graph.invoke(
            {
                "doc_paths": doc_paths,
                "current_step": "upload_received",
                "completed_steps": [],
                "agent_steps": [],
                "errors": [],
            }
        )

    def answer_question(self, query: str, history: list[ChatTurn], vector_store) -> QueryPipelineState:
        return self.query_graph.invoke(
            {
                "query": query,
                "history": history,
                "vector_store": vector_store,
                "current_step": "question_received",
                "completed_steps": [],
                "agent_steps": [],
                "errors": [],
            }
        )

    def _build_document_graph(self):
        graph = StateGraph(DocumentPipelineState)
        graph.add_node("ingest", self._ingest)
        graph.add_node("extract", self._extract)
        graph.add_node("index", self._index)
        graph.set_entry_point("ingest")
        graph.add_conditional_edges("ingest", self._next_document_step, {"extract": "extract", "end": END})
        graph.add_conditional_edges("extract", self._next_document_step, {"index": "index", "end": END})
        graph.add_edge("index", END)
        return graph.compile()

    def _build_query_graph(self):
        graph = StateGraph(QueryPipelineState)
        graph.add_node("retrieve", self._retrieve)
        graph.add_node("converse", self._converse)
        graph.set_entry_point("retrieve")
        graph.add_conditional_edges("retrieve", self._next_query_step, {"converse": "converse", "end": END})
        graph.add_edge("converse", END)
        return graph.compile()

    @staticmethod
    def _ingest(state: DocumentPipelineState) -> DocumentPipelineState:
        try:
            ingestion = DocumentIngestionAgent().validate_paths(state["doc_paths"])
            valid_count = sum(1 for document in ingestion.documents if document.is_valid)
            return record_step(
                state,
                step="ingest",
                agent="Agent 1 - Document Ingestion",
                message=f"Validated {valid_count} of {len(ingestion.documents)} uploaded document(s).",
                updates={"ingestion": ingestion},
            )
        except Exception as exc:
            return record_step_error(
                state,
                step="ingest",
                agent="Agent 1 - Document Ingestion",
                error=exc,
            )

    @staticmethod
    def _extract(state: DocumentPipelineState) -> DocumentPipelineState:
        try:
            ingestion = state["ingestion"]
            extraction = TextExtractionAgent().extract(ingestion.session_id, ingestion.documents)
            page_count = sum(len(document.pages) for document in extraction.documents)
            return record_step(
                state,
                step="extract",
                agent="Agent 2 - Text Extraction",
                message=f"Extracted text from {page_count} page(s).",
                updates={"extraction": extraction},
            )
        except Exception as exc:
            return record_step_error(
                state,
                step="extract",
                agent="Agent 2 - Text Extraction",
                error=exc,
            )

    @staticmethod
    def _index(state: DocumentPipelineState) -> DocumentPipelineState:
        try:
            ingestion = state["ingestion"]
            extraction = state["extraction"]
            vector_store, indexing = ChunkingEmbeddingAgent().index(ingestion.session_id, extraction.documents)
            return record_step(
                state,
                step="index",
                agent="Agent 3 - Chunking and Embedding",
                message=f"Indexed {indexing.chunks_indexed} chunk(s) into {indexing.collection_name}.",
                updates={"vector_store": vector_store, "indexing": indexing},
            )
        except Exception as exc:
            return record_step_error(
                state,
                step="index",
                agent="Agent 3 - Chunking and Embedding",
                error=exc,
            )

    def _retrieve(self, state: QueryPipelineState) -> QueryPipelineState:
        try:
            retrieval = RetrievalAgent(
                vector_store=state["vector_store"],
                query_rewriter=self.query_rewriter or LLMQueryRewriter(),
                reranker=self.reranker or build_default_reranker(),
            ).retrieve(state["query"])
            return record_step(
                state,
                step="retrieve",
                agent="Agent 4A - Retrieval",
                message=(
                    f"Retrieved {len(retrieval.matches)} chunk(s) with "
                    f"{retrieval.mode.value} mode and {retrieval.scope.value} scope."
                ),
                updates={"retrieval": retrieval},
            )
        except Exception as exc:
            return record_step_error(
                state,
                step="retrieve",
                agent="Agent 4A - Retrieval",
                error=exc,
            )

    def _converse(self, state: QueryPipelineState) -> QueryPipelineState:
        try:
            agent = self.conversation_agent or ConversationAgent()
            response = agent.answer(
                query=state["query"],
                history=state.get("history", []),
                retrieval=state["retrieval"],
            )
            return record_step(
                state,
                step="converse",
                agent="Agent 4B - Conversation",
                message=f"Generated an answer with {len(response.sources)} source citation(s).",
                updates={"response": response},
            )
        except Exception as exc:
            return record_step_error(
                state,
                step="converse",
                agent="Agent 4B - Conversation",
                error=exc,
            )

    @staticmethod
    def _next_document_step(state: DocumentPipelineState) -> str:
        if state.get("errors"):
            return "end"
        if "ingestion" in state and "extraction" not in state:
            return "extract"
        if "extraction" in state and "indexing" not in state:
            return "index"
        return "end"

    @staticmethod
    def _next_query_step(state: QueryPipelineState) -> str:
        if state.get("errors") or "retrieval" not in state:
            return "end"
        return "converse"


def record_step(
    state: DocumentPipelineState | QueryPipelineState,
    step: str,
    agent: str,
    message: str,
    updates: dict[str, Any],
) -> dict[str, Any]:
    return {
        **updates,
        "current_step": step,
        "completed_steps": [*state.get("completed_steps", []), step],
        "agent_steps": [
            *state.get("agent_steps", []),
            {
                "step": step,
                "agent": agent,
                "status": "completed",
                "message": message,
            },
        ],
        "errors": state.get("errors", []),
    }


def record_step_error(
    state: DocumentPipelineState | QueryPipelineState,
    step: str,
    agent: str,
    error: Exception,
) -> dict[str, Any]:
    message = f"{type(error).__name__}: {error}"
    return {
        "current_step": step,
        "completed_steps": state.get("completed_steps", []),
        "agent_steps": [
            *state.get("agent_steps", []),
            {
                "step": step,
                "agent": agent,
                "status": "failed",
                "message": message,
            },
        ],
        "errors": [*state.get("errors", []), message],
    }
