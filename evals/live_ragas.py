from __future__ import annotations

import math
import os
import random
import re
import time
from typing import Any

from datasets import Dataset
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from openai import APIConnectionError, APITimeoutError, RateLimitError
from ragas import evaluate
from ragas.metrics import Faithfulness, ResponseRelevancy

try:
    from ragas.run_config import RunConfig
except Exception:
    RunConfig = None

from core.config import EMBEDDING_MODEL, GROQ_API_BASE_URL


DEFAULT_MAX_CONTEXTS = 3
DEFAULT_MAX_CONTEXT_CHARS = 1200


def build_live_ragas_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model=os.getenv("RAGAS_LLM_MODEL", "llama-3.1-8b-instant"),
        api_key=os.getenv("GROQ_API_KEY"),
        base_url=os.getenv("GROQ_API_BASE_URL", GROQ_API_BASE_URL),
        temperature=0,
        max_tokens=int(os.getenv("RAGAS_MAX_TOKENS", "1024")),
        n=1,
        timeout=float(os.getenv("RAGAS_REQUEST_TIMEOUT", "180")),
        max_retries=0,
    )


def build_live_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name=os.getenv("RAGAS_LOCAL_EMBEDDING_MODEL", EMBEDDING_MODEL),
        encode_kwargs={"normalize_embeddings": True},
    )


def build_live_run_config() -> Any:
    if RunConfig is None:
        return None

    return RunConfig(
        timeout=int(os.getenv("RAGAS_TIMEOUT", "180")),
        max_retries=int(os.getenv("RAGAS_MAX_RETRIES", "0")),
        max_wait=int(os.getenv("RAGAS_MAX_WAIT", "30")),
    )


def build_response_relevancy_metric() -> Any:
    try:
        return ResponseRelevancy(strictness=1)
    except TypeError:
        metric = ResponseRelevancy()
        if hasattr(metric, "strictness"):
            metric.strictness = 1
        return metric


def trim_contexts(
    contexts: list[str],
    max_contexts: int = DEFAULT_MAX_CONTEXTS,
    max_chars_per_context: int = DEFAULT_MAX_CONTEXT_CHARS,
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
    max_attempts: int = 3,
) -> Any:
    attempt = 0

    while True:
        try:
            kwargs = {
                "dataset": dataset,
                "metrics": metrics,
                "llm": llm,
                "embeddings": embeddings,
                "batch_size": 1,
                "raise_exceptions": True,
            }

            if run_config is not None:
                kwargs["run_config"] = run_config

            return evaluate(**kwargs)

        except RateLimitError as error:
            attempt += 1
            if attempt >= max_attempts:
                raise

            wait_seconds = extract_retry_seconds(error) or min(90, 20 * attempt)
            time.sleep(wait_seconds + random.uniform(1.0, 3.0))

        except (APIConnectionError, APITimeoutError):
            attempt += 1
            if attempt >= max_attempts:
                raise

            wait_seconds = min(60, 15 * attempt)
            time.sleep(wait_seconds + random.uniform(1.0, 3.0))


def coerce_score(value: Any) -> float | None:
    if not isinstance(value, (int, float)):
        return None

    score = float(value)
    if math.isnan(score):
        return None

    return score


def evaluate_live_answer(
    question: str,
    answer: str,
    retrieved_contexts: list[str],
) -> dict[str, Any]:
    if not os.getenv("GROQ_API_KEY"):
        return {
            "status": "skipped",
            "reason": "GROQ_API_KEY is required for the RAGAS judge LLM.",
            "metrics": {},
        }

    trimmed_contexts = trim_contexts(retrieved_contexts)

    if not question.strip() or not answer.strip() or not trimmed_contexts:
        return {
            "status": "skipped",
            "reason": "Question, answer, or retrieved contexts were empty.",
            "metrics": {},
        }

    dataset = Dataset.from_list(
        [
            {
                "user_input": question,
                "response": answer,
                "retrieved_contexts": trimmed_contexts,
            }
        ]
    )

    try:
        result = evaluate_with_retry(
            dataset=dataset,
            metrics=[Faithfulness(), build_response_relevancy_metric()],
            llm=build_live_ragas_llm(),
            embeddings=build_live_embeddings(),
            run_config=build_live_run_config(),
        )
        row = result.to_pandas().to_dict(orient="records")[0]

        return {
            "status": "success",
            "metrics": {
                "faithfulness": coerce_score(row.get("faithfulness")),
                "response_relevancy": coerce_score(
                    row.get("response_relevancy", row.get("answer_relevancy"))
                ),
            },
        }

    except Exception as error:
        return {
            "status": "failed",
            "reason": str(error),
            "metrics": {},
        }


def evaluate_single_answer(
    question: str,
    answer: str,
    contexts: list[str],
) -> dict[str, Any]:
    return evaluate_live_answer(
        question=question,
        answer=answer,
        retrieved_contexts=contexts,
    )
