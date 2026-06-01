from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import sys
import time
import warnings
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from openai import APIConnectionError, APITimeoutError, RateLimitError

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets import Dataset
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from ragas import evaluate

warnings.simplefilter("ignore", DeprecationWarning)

from ragas.metrics import (
    Faithfulness,
    LLMContextPrecisionWithReference,
    LLMContextRecall,
    ResponseRelevancy,
)

try:
    from ragas.exceptions import LLMDidNotFinishException
except Exception:
    LLMDidNotFinishException = RuntimeError

try:
    from ragas.run_config import RunConfig
except Exception:
    RunConfig = None

from core.config import EMBEDDING_MODEL, GROQ_API_BASE_URL
from core.graph_orchestration import RagGraphOrchestrator
from core.schemas import RetrievalOutput


DEFAULT_MAX_CONTEXTS_FOR_RAGAS = 1
DEFAULT_MAX_CONTEXT_CHARS_FOR_RAGAS = 300
DEFAULT_MAX_RESPONSE_CHARS_FOR_RAGAS = 500
DEFAULT_MAX_REFERENCE_CHARS_FOR_RAGAS = 500


def load_eval_set(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Evaluation set not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_rag_log(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"RAG log not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def build_pipeline(doc_paths: list[Path]):
    pipeline = RagGraphOrchestrator().process_documents(doc_paths)
    return (
        pipeline["ingestion"],
        pipeline["extraction"],
        pipeline["indexing"],
        pipeline["vector_store"],
    )


def run_rag(
    eval_set: list[dict[str, Any]],
    doc_paths: list[Path],
    delay_seconds: float,
) -> dict[str, Any]:
    ingestion, extraction, indexing, vector_store = build_pipeline(doc_paths)
    orchestrator = RagGraphOrchestrator()

    records: list[dict[str, Any]] = []

    for index, item in enumerate(eval_set):
        question = item["question"]
        graph_state = orchestrator.answer_question(question, [], vector_store)

        retrieval = graph_state["retrieval"]
        response = graph_state["response"]

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
                "retrieved_documents": sorted(
                    {match.chunk.filename for match in response.matches}
                ),
                "sources": [safe_to_dict(source) for source in response.sources],
                "failed_generation": is_failed_generation(response.answer),
            }
        )

        if delay_seconds > 0 and index < len(eval_set) - 1:
            time.sleep(delay_seconds)

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
        "indexing": safe_to_dict(indexing),
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


def trim_contexts_for_ragas(
    contexts: list[str],
    max_contexts: int,
    max_chars_per_context: int,
) -> list[str]:
    cleaned: list[str] = []

    for context in contexts:
        text = str(context).strip()
        if not text:
            continue

        cleaned.append(text[:max_chars_per_context])

        if len(cleaned) >= max_contexts:
            break

    return cleaned


def build_ragas_dataset(
    records: list[dict[str, Any]],
    max_contexts: int,
    max_context_chars: int,
    max_response_chars: int,
    max_reference_chars: int,
) -> Dataset:
    rows: list[dict[str, Any]] = []

    for record in records:
        contexts = trim_contexts_for_ragas(
            record.get("contexts", []),
            max_contexts=max_contexts,
            max_chars_per_context=max_context_chars,
        )

        question = str(record.get("question", "")).strip()
        answer = str(record.get("answer", "")).strip()
        reference = str(record.get("reference", "")).strip()

        if not question or not answer or not contexts:
            continue

        rows.append(
            {
                "user_input": question,
                "response": answer[:max_response_chars],
                "retrieved_contexts": contexts,
                "reference": reference[:max_reference_chars],
            }
        )

    return Dataset.from_list(rows)


def build_ragas_metrics(metric_group: str) -> list[Any]:
    if metric_group == "faithfulness":
        return [Faithfulness()]

    if metric_group == "relevancy":
        return [build_response_relevancy_metric()]

    if metric_group == "precision":
        return [LLMContextPrecisionWithReference()]

    if metric_group == "recall":
        return [LLMContextRecall()]

    if metric_group == "generation":
        return [Faithfulness(), build_response_relevancy_metric()]

    if metric_group == "retrieval":
        return [LLMContextPrecisionWithReference(), LLMContextRecall()]

    return [
        Faithfulness(),
        build_response_relevancy_metric(),
        LLMContextPrecisionWithReference(),
        LLMContextRecall(),
    ]


def build_response_relevancy_metric() -> Any:
    try:
        return ResponseRelevancy(strictness=1)
    except TypeError:
        metric = ResponseRelevancy()
        if hasattr(metric, "strictness"):
            metric.strictness = 1
        return metric


def build_ragas_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model=os.getenv("RAGAS_LLM_MODEL", "llama-3.1-8b-instant"),
        api_key=os.getenv("GROQ_API_KEY"),
        base_url=os.getenv("GROQ_API_BASE_URL", GROQ_API_BASE_URL),
        temperature=0,
        max_tokens=int(os.getenv("RAGAS_MAX_TOKENS", "1024")),
        n=1,
        timeout=float(os.getenv("RAGAS_REQUEST_TIMEOUT", "180")),
        max_retries=int(os.getenv("RAGAS_CLIENT_MAX_RETRIES", "0")),
    )


def build_ragas_run_config():
    if RunConfig is None:
        return None

    return RunConfig(
        timeout=int(os.getenv("RAGAS_TIMEOUT", "180")),
        max_retries=int(os.getenv("RAGAS_MAX_RETRIES", "0")),
        max_wait=int(os.getenv("RAGAS_MAX_WAIT", "30")),
    )


def build_ragas_embeddings():
    use_openai_embeddings = (
        os.getenv("OPENAI_API_KEY")
        and os.getenv("RAGAS_USE_OPENAI_EMBEDDINGS", "").lower()
        in {"1", "true", "yes"}
    )

    if use_openai_embeddings:
        return OpenAIEmbeddings(
            model=os.getenv("RAGAS_EMBEDDING_MODEL", "text-embedding-3-small"),
            api_key=os.getenv("OPENAI_API_KEY"),
        )

    return HuggingFaceEmbeddings(
        model_name=os.getenv("RAGAS_LOCAL_EMBEDDING_MODEL", EMBEDDING_MODEL),
        encode_kwargs={"normalize_embeddings": True},
    )


def extract_retry_seconds(error: Exception) -> float | None:
    message = str(error)

    match = re.search(r"try again in ([0-9.]+)s", message)
    if match:
        return float(match.group(1)) + 3.0

    match = re.search(r"try again in ([0-9.]+)m([0-9.]+)s", message)
    if match:
        return float(match.group(1)) * 60 + float(match.group(2)) + 5.0

    return None


def evaluate_with_retry(
    dataset: Dataset,
    metrics: list[Any],
    llm: Any,
    embeddings: Any,
    run_config: Any,
    max_attempts: int = 5,
) -> Any:
    attempt = 0

    while True:
        try:
            kwargs = {
                "dataset": dataset,
                "metrics": metrics,
                "llm": llm,
                "embeddings": embeddings,
                "raise_exceptions": True,
                "batch_size": 1,
            }

            if run_config is not None:
                kwargs["run_config"] = run_config

            return evaluate(**kwargs)

        except RateLimitError as error:
            attempt += 1
            if attempt >= max_attempts:
                raise

            wait_seconds = extract_retry_seconds(error) or min(180, 25 * attempt)
            wait_seconds += random.uniform(1.0, 3.0)

            print(
                f"Rate limit hit. Waiting {wait_seconds:.1f}s "
                f"before retry {attempt}/{max_attempts}..."
            )
            time.sleep(wait_seconds)

        except (APIConnectionError, APITimeoutError) as error:
            attempt += 1
            if attempt >= max_attempts:
                raise

            wait_seconds = min(120, 20 * attempt) + random.uniform(1.0, 3.0)

            print(
                f"Connection/timeout error. Waiting {wait_seconds:.1f}s "
                f"before retry {attempt}/{max_attempts}..."
            )
            print(f"Error: {error}")
            time.sleep(wait_seconds)

        except LLMDidNotFinishException as error:
            print("RAGAS judge output was cut off. Skipping this batch.")
            print(f"Error: {error}")
            return None


def run_ragas(
    records: list[dict[str, Any]],
    batch_size: int,
    delay_seconds: float,
    max_contexts: int,
    max_context_chars: int,
    max_response_chars: int,
    max_reference_chars: int,
    metric_group: str,
) -> list[dict[str, Any]]:
    if not os.getenv("GROQ_API_KEY"):
        raise RuntimeError("GROQ_API_KEY is required for the RAGAS judge LLM.")

    successful_records = [
        record
        for record in records
        if not record.get("failed_generation")
        and record.get("answer")
        and record.get("contexts")
    ]

    print(f"Total records: {len(records)}")
    print(f"RAGAS eligible records: {len(successful_records)}")
    print(f"RAGAS metric group: {metric_group}")

    if not successful_records:
        return []

    evaluator_llm = build_ragas_llm()
    embeddings = build_ragas_embeddings()
    metrics = build_ragas_metrics(metric_group)
    run_config = build_ragas_run_config()

    rows: list[dict[str, Any]] = []

    for start in range(0, len(successful_records), batch_size):
        batch = successful_records[start : start + batch_size]

        dataset = build_ragas_dataset(
            batch,
            max_contexts=max_contexts,
            max_context_chars=max_context_chars,
            max_response_chars=max_response_chars,
            max_reference_chars=max_reference_chars,
        )

        if len(dataset) == 0:
            continue

        print(
            f"Evaluating RAGAS records {start + 1} "
            f"to {start + len(batch)} of {len(successful_records)}..."
        )

        result = evaluate_with_retry(
            dataset=dataset,
            metrics=metrics,
            llm=evaluator_llm,
            embeddings=embeddings,
            run_config=run_config,
        )

        if result is None:
            skipped_row = {
                "user_input": batch[0].get("question"),
                "response": batch[0].get("answer"),
                "reference": batch[0].get("reference"),
                "ragas_error": "LLMDidNotFinishException",
                args_metric_key(metric_group): None,
            }
            rows.append(skipped_row)
        else:
            rows.extend(result.to_pandas().to_dict(orient="records"))

        if delay_seconds > 0 and start + batch_size < len(successful_records):
            print(f"Sleeping {delay_seconds}s before next RAGAS batch...")
            time.sleep(delay_seconds)

    return rows

def args_metric_key(metric_group: str) -> str:
    if metric_group == "faithfulness":
        return "faithfulness"
    if metric_group == "relevancy":
        return "answer_relevancy"
    if metric_group == "precision":
        return "llm_context_precision_with_reference"
    if metric_group == "recall":
        return "context_recall"
    return metric_group

def custom_metrics(records: list[dict[str, Any]]) -> dict[str, float]:
    if not records:
        return {
            "retrieval_hit_rate": 0.0,
            "source_precision": 0.0,
            "source_recall": 0.0,
            "avg_retrieved_docs": 0.0,
        }

    hits = 0
    precision_values: list[float] = []
    recall_values: list[float] = []
    retrieved_doc_counts: list[int] = []

    for record in records:
        expected = set(record.get("expected_documents", []))
        retrieved = set(record.get("retrieved_documents", []))

        if expected and expected.issubset(retrieved):
            hits += 1

        retrieved_doc_counts.append(len(retrieved))

        if expected:
            precision_values.append(
                len(expected & retrieved) / len(retrieved) if retrieved else 0.0
            )
            recall_values.append(len(expected & retrieved) / len(expected))
        else:
            precision_values.append(1.0 if not retrieved else 0.0)
            recall_values.append(1.0)

    return {
        "retrieval_hit_rate": hits / len(records),
        "source_precision": sum(precision_values) / len(precision_values),
        "source_recall": sum(recall_values) / len(recall_values),
        "avg_retrieved_docs": sum(retrieved_doc_counts) / len(retrieved_doc_counts),
    }


def get_first_existing(row: dict[str, Any], names: list[str]) -> Any:
    for name in names:
        if name in row:
            return row.get(name)
    return None


def ragas_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    metric_aliases = {
        "faithfulness": ["faithfulness"],
        "response_relevancy": ["response_relevancy", "answer_relevancy"],
        "context_precision": ["llm_context_precision_with_reference", "context_precision"],
        "context_recall": ["context_recall", "llm_context_recall"],
    }

    summary: dict[str, Any] = {}

    for metric_name, aliases in metric_aliases.items():
        values = [get_first_existing(row, aliases) for row in rows]

        valid_values = [
            float(value)
            for value in values
            if isinstance(value, (int, float)) and not math.isnan(float(value))
        ]

        summary[metric_name] = {
            "mean": sum(valid_values) / len(valid_values) if valid_values else None,
            "valid_count": len(valid_values),
            "nan_count": len(values) - len(valid_values),
        }

    return summary


def is_failed_generation(answer: str) -> bool:
    return str(answer).startswith("Groq could not generate a response.")


def safe_to_dict(value: Any) -> Any:
    try:
        return asdict(value)
    except TypeError:
        if hasattr(value, "model_dump"):
            return value.model_dump()
        if hasattr(value, "dict"):
            return value.dict()
        return value


def clean_for_json(value: Any) -> Any:
    if isinstance(value, float) and math.isnan(value):
        return None

    if isinstance(value, dict):
        return {key: clean_for_json(item) for key, item in value.items()}

    if isinstance(value, list):
        return [clean_for_json(item) for item in value]

    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(clean_for_json(payload), indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run manual RAG + RAGAS evaluation.")

    parser.add_argument("--eval-set", default="evals/manual_eval_set.json")
    parser.add_argument(
        "--docs",
        nargs="+",
        required=False,
        help="PDF/image documents to evaluate against. Required unless --input-rag-log is used.",
    )
    parser.add_argument("--output-dir", default="evals/outputs")
    parser.add_argument("--skip-ragas", action="store_true")
    parser.add_argument(
        "--input-rag-log",
        default=None,
        help="Use an existing rag_run_*.json file instead of running RAG again.",
    )

    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument("--rag-delay-seconds", type=float, default=8.0)

    parser.add_argument("--ragas-batch-size", type=int, default=1)
    parser.add_argument("--ragas-delay-seconds", type=float, default=90.0)
    parser.add_argument("--ragas-max-contexts", type=int, default=1)
    parser.add_argument("--ragas-max-context-chars", type=int, default=300)
    parser.add_argument("--ragas-max-response-chars", type=int, default=500)
    parser.add_argument("--ragas-max-reference-chars", type=int, default=500)

    parser.add_argument(
        "--ragas-metrics",
        default="faithfulness",
        choices=[
            "all",
            "faithfulness",
            "relevancy",
            "precision",
            "recall",
            "generation",
            "retrieval",
        ],
    )

    args = parser.parse_args()

    eval_set = load_eval_set(Path(args.eval_set))
    if args.max_records is not None:
        eval_set = eval_set[: args.max_records]

    output_dir = Path(args.output_dir)
    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if args.input_rag_log:
        run_payload = load_rag_log(Path(args.input_rag_log))

        if args.max_records is not None:
            run_payload["records"] = run_payload.get("records", [])[: args.max_records]

        logs_path = Path(args.input_rag_log)
    else:
        if not args.docs:
            raise ValueError("--docs is required when --input-rag-log is not provided.")

        doc_paths = [Path(doc).resolve() for doc in args.docs]
        run_payload = run_rag(
            eval_set=eval_set,
            doc_paths=doc_paths,
            delay_seconds=args.rag_delay_seconds,
        )

        logs_path = output_dir / f"rag_run_{run_stamp}.json"
        write_json(logs_path, run_payload)

    records = run_payload.get("records", [])
    custom = custom_metrics(records)
    run_payload["custom_metrics"] = custom

    failed_generation_count = sum(
        1 for record in records if record.get("failed_generation")
    )

    ragas_eligible_count = sum(
        1
        for record in records
        if not record.get("failed_generation")
        and record.get("answer")
        and record.get("contexts")
    )

    report: dict[str, Any] = {
        "logs_path": str(logs_path),
        "custom_metrics": custom,
        "failed_generation_count": failed_generation_count,
        "ragas_eligible_record_count": ragas_eligible_count,
        "ragas_settings": {
            "metric_group": args.ragas_metrics,
            "model": os.getenv("RAGAS_LLM_MODEL", "llama-3.1-8b-instant"),
            "max_tokens": os.getenv("RAGAS_MAX_TOKENS", "1024"),
            "max_contexts": args.ragas_max_contexts,
            "max_context_chars": args.ragas_max_context_chars,
            "max_response_chars": args.ragas_max_response_chars,
            "max_reference_chars": args.ragas_max_reference_chars,
            "batch_size": max(args.ragas_batch_size, 1),
            "delay_seconds": args.ragas_delay_seconds,
        },
    }

    if not args.skip_ragas:
        ragas_rows = run_ragas(
            records=records,
            batch_size=max(args.ragas_batch_size, 1),
            delay_seconds=args.ragas_delay_seconds,
            max_contexts=args.ragas_max_contexts,
            max_context_chars=args.ragas_max_context_chars,
            max_response_chars=args.ragas_max_response_chars,
            max_reference_chars=args.ragas_max_reference_chars,
            metric_group=args.ragas_metrics,
        )

        ragas_path = output_dir / f"ragas_{args.ragas_metrics}_{run_stamp}.json"
        write_json(ragas_path, ragas_rows)

        report["ragas_metric_group"] = args.ragas_metrics
        report["ragas_path"] = str(ragas_path)
        report["ragas_scores"] = ragas_rows
        report["ragas_summary"] = ragas_summary(ragas_rows)

    report_path = output_dir / f"eval_report_{args.ragas_metrics}_{run_stamp}.json"
    write_json(report_path, report)

    print(f"Saved RAG logs: {logs_path}")
    print(f"Saved evaluation report: {report_path}")

    if not args.skip_ragas:
        print("RAGAS summary:")
        print(json.dumps(report.get("ragas_summary", {}), indent=2))


if __name__ == "__main__":
    main()