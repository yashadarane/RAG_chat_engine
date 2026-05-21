from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets import Dataset
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from ragas import evaluate

warnings.simplefilter("ignore", DeprecationWarning)
from ragas.metrics import Faithfulness, LLMContextPrecisionWithReference, LLMContextRecall, ResponseRelevancy

from agents import ChunkingEmbeddingAgent, ConversationAgent, DocumentIngestionAgent, RetrievalAgent, TextExtractionAgent
from core.config import EMBEDDING_MODEL, GROQ_API_BASE_URL, GROQ_MODEL
from core.query_rewriter import LLMQueryRewriter
from core.reranker import build_default_reranker
from core.schemas import ChatTurn, RetrievalOutput


def load_eval_set(path: Path) -> list[dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))


def build_pipeline(doc_paths: list[Path]):
    ingestion = DocumentIngestionAgent().validate_paths(doc_paths)
    extraction = TextExtractionAgent().extract(ingestion.session_id, ingestion.documents)
    vector_store, indexing = ChunkingEmbeddingAgent().index(ingestion.session_id, extraction.documents)
    return ingestion, extraction, indexing, vector_store


def run_rag(eval_set: list[dict[str, Any]], doc_paths: list[Path]) -> dict[str, Any]:
    ingestion, extraction, indexing, vector_store = build_pipeline(doc_paths)
    retrieval_agent = RetrievalAgent(
        vector_store=vector_store,
        query_rewriter=LLMQueryRewriter(),
        reranker=build_default_reranker(),
    )
    conversation_agent = ConversationAgent()

    records: list[dict[str, Any]] = []
    history: list[ChatTurn] = []

    for item in eval_set:
        question = item["question"]
        retrieval = retrieval_agent.retrieve(question)
        response = conversation_agent.answer(question, history[-6:], retrieval)
        history.extend(
            [
                ChatTurn(role="user", content=question),
                ChatTurn(role="assistant", content=response.answer),
            ]
        )

        records.append(
            {
                "id": item["id"],
                "question": question,
                "reference": item["reference"],
                "answer": response.answer,
                "expected_documents": item.get("expected_documents", []),
                "expected_answer_keywords": item.get("expected_answer_keywords", []),
                "retrieval": serialize_retrieval(retrieval),
                "contexts": [match.chunk.text for match in response.matches],
                "retrieved_documents": sorted({match.chunk.filename for match in response.matches}),
                "sources": [asdict(source) for source in response.sources],
            }
        )

    return {
        "run_at": datetime.now(timezone.utc).isoformat(),
        "ingestion": {
            "session_id": ingestion.session_id,
            "documents": [
                {
                    "doc_id": doc.doc_id,
                    "filename": doc.filename,
                    "status": doc.status,
                    "page_count": doc.page_count,
                    "issues": doc.issues,
                }
                for doc in ingestion.documents
            ],
        },
        "indexing": asdict(indexing),
        "records": records,
    }


def serialize_retrieval(retrieval: RetrievalOutput) -> dict[str, Any]:
    return {
        "mode": retrieval.mode.value,
        "scope": retrieval.scope.value,
        "selected_documents": retrieval.selected_documents,
        "matches": [
            {
                "chunk_id": match.chunk.chunk_id,
                "document": match.chunk.filename,
                "page": match.chunk.page_number,
                "similarity_score": match.similarity_score,
                "retrieval_reason": match.retrieval_reason,
                "text": match.chunk.text,
            }
            for match in retrieval.matches
        ],
    }


def build_ragas_dataset(records: list[dict[str, Any]]) -> Dataset:
    return Dataset.from_list(
        [
            {
                "user_input": record["question"],
                "response": record["answer"],
                "retrieved_contexts": record["contexts"],
                "reference": record["reference"],
            }
            for record in records
        ]
    )


def run_ragas(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not os.getenv("GROQ_API_KEY"):
        raise RuntimeError("GROQ_API_KEY is required for the RAGAS judge LLM.")

    evaluator_llm = ChatOpenAI(
        model=os.getenv("RAGAS_LLM_MODEL", GROQ_MODEL),
        api_key=os.getenv("GROQ_API_KEY"),
        base_url=os.getenv("GROQ_API_BASE_URL", GROQ_API_BASE_URL),
        temperature=0,
    )
    embeddings = build_ragas_embeddings()
    dataset = build_ragas_dataset(records)
    result = evaluate(
        dataset,
        metrics=[
            Faithfulness(),
            ResponseRelevancy(),
            LLMContextPrecisionWithReference(),
            LLMContextRecall(),
        ],
        llm=evaluator_llm,
        embeddings=embeddings,
        raise_exceptions=False,
    )
    return result.to_pandas().to_dict(orient="records")


def build_ragas_embeddings():
    if os.getenv("OPENAI_API_KEY") and os.getenv("RAGAS_USE_OPENAI_EMBEDDINGS", "").lower() in {"1", "true", "yes"}:
        return OpenAIEmbeddings(
            model=os.getenv("RAGAS_EMBEDDING_MODEL", "text-embedding-3-small"),
            api_key=os.getenv("OPENAI_API_KEY"),
        )

    return HuggingFaceEmbeddings(
        model_name=os.getenv("RAGAS_LOCAL_EMBEDDING_MODEL", EMBEDDING_MODEL),
        encode_kwargs={"normalize_embeddings": True},
    )


def custom_metrics(records: list[dict[str, Any]]) -> dict[str, float]:
    if not records:
        return {"retrieval_hit_rate": 0.0, "avg_retrieved_docs": 0.0}

    hits = 0
    retrieved_doc_counts: list[int] = []
    for record in records:
        expected = set(record.get("expected_documents", []))
        retrieved = set(record.get("retrieved_documents", []))
        if expected and expected.issubset(retrieved):
            hits += 1
        retrieved_doc_counts.append(len(retrieved))

    return {
        "retrieval_hit_rate": hits / len(records),
        "avg_retrieved_docs": sum(retrieved_doc_counts) / len(retrieved_doc_counts),
    }


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run manual RAG + RAGAS evaluation.")
    parser.add_argument("--eval-set", default="evals/manual_eval_set.json")
    parser.add_argument("--docs", nargs="+", required=True, help="PDF/image documents to evaluate against.")
    parser.add_argument("--output-dir", default="evals/outputs")
    parser.add_argument("--skip-ragas", action="store_true", help="Only save RAG run logs and custom metrics.")
    args = parser.parse_args()

    eval_set = load_eval_set(Path(args.eval_set))
    doc_paths = [Path(doc).resolve() for doc in args.docs]
    output_dir = Path(args.output_dir)

    run_payload = run_rag(eval_set, doc_paths)
    custom = custom_metrics(run_payload["records"])
    run_payload["custom_metrics"] = custom

    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logs_path = output_dir / f"rag_run_{run_stamp}.json"
    write_json(logs_path, run_payload)

    report: dict[str, Any] = {
        "logs_path": str(logs_path),
        "custom_metrics": custom,
        "target_table": {
            "faithfulness": "> 0.85",
            "response_relevancy": "> 0.80",
            "context_precision": "> 0.75",
            "context_recall": "> 0.75",
            "retrieval_hit_rate": "> 0.90",
            "avg_retrieved_docs": "lower is better",
        },
    }

    if not args.skip_ragas:
        ragas_rows = run_ragas(run_payload["records"])
        ragas_path = output_dir / f"ragas_scores_{run_stamp}.json"
        write_json(ragas_path, ragas_rows)
        report["ragas_path"] = str(ragas_path)
        report["ragas_scores"] = ragas_rows

    report_path = output_dir / f"eval_report_{run_stamp}.json"
    write_json(report_path, report)
    print(f"Saved RAG logs: {logs_path}")
    print(f"Saved evaluation report: {report_path}")


if __name__ == "__main__":
    main()
