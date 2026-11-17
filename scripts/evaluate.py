"""Evaluate retrieval and, optionally, answer quality on eval/sample_qa.json."""
import argparse
import json
import logging
import math
import re
import sys
import time
from pathlib import Path
from typing import Any

from langchain_core.documents import Document
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402
from src import generation, indexing, retrieval  # noqa: E402

logging.basicConfig(level=logging.WARNING)


def _expected_papers(item: dict) -> list[str]:
    expected = item.get("expected_papers", item.get("expected_paper"))
    if isinstance(expected, str):
        return [expected]
    if isinstance(expected, list) and all(isinstance(value, str) for value in expected):
        if expected:
            return expected
    raise ValueError("Each evaluation item needs expected_paper or expected_papers.")


def _matching_expected_papers(doc: Document, expected_papers: list[str]) -> set[str]:
    title = _normalize_title(str(doc.metadata.get("paper_title", "")))
    filename = _normalize_title(str(doc.metadata.get("filename", "")))
    return {
        _normalize_title(expected)
        for expected in expected_papers
        if _normalize_title(expected) in title or _normalize_title(expected) in filename
    }


def _normalize_title(value: str) -> str:
    return re.sub(r"[\W_]+", " ", value.casefold()).strip()


def _retrieval_metrics(
    items: list[tuple[list[Document], list[str]]],
    k: int,
) -> dict[str, float]:
    if not items:
        raise ValueError("The evaluation set contains no questions.")
    hits = 0
    recalls = []
    reciprocal_ranks = []
    ndcgs = []
    for docs, expected in items:
        ranked = docs[:k]
        relevant = []
        covered: set[str] = set()
        for doc in ranked:
            newly_covered = _matching_expected_papers(doc, expected) - covered
            relevant.append(newly_covered)
            covered.update(newly_covered)
        hits += bool(covered)
        expected_unique = {_normalize_title(paper) for paper in expected}
        recalls.append(len(covered) / len(expected_unique))
        first_rank = next((index + 1 for index, matches in enumerate(relevant) if matches), None)
        reciprocal_ranks.append(1 / first_rank if first_rank else 0.0)
        dcg = sum(bool(matches) / math.log2(index + 2) for index, matches in enumerate(relevant))
        ideal_count = min(len(expected_unique), k)
        ideal_dcg = sum(1 / math.log2(index + 2) for index in range(ideal_count))
        ndcgs.append(dcg / ideal_dcg if ideal_dcg else 0.0)
    count = len(items)
    return {
        f"hit_rate@{k}": hits / count,
        f"recall@{k}": sum(recalls) / count,
        "mrr": sum(reciprocal_ranks) / count,
        f"ndcg@{k}": sum(ndcgs) / count,
    }


def _latency_metrics(samples_seconds: list[float]) -> dict[str, float]:
    if not samples_seconds:
        return {"mean": 0.0, "p50": 0.0, "p95": 0.0}
    ordered_ms = sorted(sample * 1000 for sample in samples_seconds)

    def percentile(value: float) -> float:
        index = max(0, math.ceil(value * len(ordered_ms)) - 1)
        return round(ordered_ms[index], 2)

    return {
        "mean": round(sum(ordered_ms) / len(ordered_ms), 2),
        "p50": percentile(0.5),
        "p95": percentile(0.95),
    }


def _failure_cases(
    rows: list[tuple[dict[str, Any], list[str], list[Document]]],
    k: int,
    limit: int,
) -> list[dict[str, Any]]:
    if limit == 0:
        return []
    failures = []
    for item, expected, docs in rows:
        ranked = docs[:k]
        matched = set().union(
            *(_matching_expected_papers(doc, expected) for doc in ranked)
        ) if ranked else set()
        if matched:
            continue
        top_papers = list(
            dict.fromkeys(
                str(doc.metadata.get("paper_title", ""))
                for doc in ranked
                if doc.metadata.get("paper_title")
            )
        )
        failures.append(
            {
                "question": item["question"],
                "difficulty": item.get("difficulty", "unspecified"),
                "category": item.get("category", "unspecified"),
                "expected_papers": expected,
                "retrieved_papers": top_papers,
            }
        )
        if len(failures) >= limit:
            break
    return failures


def _dense_retrieve(question: str, k: int, collection_id: str | None):
    kwargs: dict[str, Any] = {"k": k}
    if collection_id and indexing.supports_metadata_filters():
        kwargs["filter"] = {"collection_id": collection_id}
    vectorstore = indexing.load_vectorstore()
    return vectorstore.as_retriever(search_kwargs=kwargs).invoke(question)


def evaluate(
    qa_path: Path,
    k: int = config.TOP_K_FINAL,
    collection_id: str | None = None,
    evaluate_answers: bool = False,
    limit: int | None = None,
    failure_limit: int = 3,
) -> dict:
    if k < 1:
        raise ValueError("k must be a positive integer.")
    if failure_limit < 0:
        raise ValueError("failure_limit cannot be negative.")
    qa_pairs = json.loads(qa_path.read_text(encoding="utf-8"))
    if not isinstance(qa_pairs, list) or not qa_pairs:
        raise ValueError(f"{qa_path} must contain a non-empty JSON list.")
    if limit is not None:
        if limit < 1:
            raise ValueError("limit must be a positive integer.")
        qa_pairs = qa_pairs[:limit]

    strategy_names = ("dense_only", "hybrid_only", "hybrid_rerank")
    strategy_items: dict[str, list[tuple[list[Document], list[str]]]] = {
        name: [] for name in strategy_names
    }
    strategy_rows: dict[str, list[tuple[dict[str, Any], list[str], list[Document]]]] = {
        name: [] for name in strategy_names
    }
    strategy_latencies: dict[str, list[float]] = {
        name: [] for name in strategy_names
    }
    quality_scores = []
    citation_correct = 0
    citation_total = 0
    generation_requests = 0
    judge_requests = 0
    answer_quality_latencies = []

    for item in qa_pairs:
        question = item.get("question")
        if not isinstance(question, str) or not question.strip():
            raise ValueError("Every evaluation item needs a non-empty question.")
        expected = _expected_papers(item)

        started = time.perf_counter()
        baseline_docs = _dense_retrieve(question, k, collection_id)
        strategy_latencies["dense_only"].append(time.perf_counter() - started)

        started = time.perf_counter()
        hybrid_docs = retrieval.retrieve_hybrid(
            question,
            candidate_k=max(config.TOP_K_CANDIDATES, k),
            final_k=k,
            collection_id=collection_id,
        )
        strategy_latencies["hybrid_only"].append(time.perf_counter() - started)

        started = time.perf_counter()
        full_docs = retrieval.retrieve(
            question,
            candidate_k=max(config.TOP_K_CANDIDATES, k),
            final_k=k,
            collection_id=collection_id,
        )
        strategy_latencies["hybrid_rerank"].append(time.perf_counter() - started)

        results_by_strategy = {
            "dense_only": baseline_docs,
            "hybrid_only": hybrid_docs,
            "hybrid_rerank": full_docs,
        }
        for name, docs in results_by_strategy.items():
            strategy_items[name].append((docs, expected))
            strategy_rows[name].append((item, expected, docs))

        if evaluate_answers:
            started = time.perf_counter()
            generation_requests += int(bool(full_docs))
            result = generation.generate_answer(question, full_docs)
            judge_requests += 1
            quality = generation.evaluate_answer_quality(
                question,
                result.answer,
                result.source_chunks,
            )
            citations = generation.citation_correctness(
                result.answer,
                result.source_chunks,
            )
            quality_scores.append(quality)
            citation_correct += int(citations["correct"])
            citation_total += int(citations["total"])
            answer_quality_latencies.append(time.perf_counter() - started)

    strategy_reports = {
        name: {
            "metrics": _retrieval_metrics(strategy_items[name], k),
            "latency_ms": _latency_metrics(strategy_latencies[name]),
        }
        for name in strategy_names
    }
    difficulties = sorted(
        {str(item.get("difficulty", "unspecified")) for item in qa_pairs}
    )
    metrics_by_difficulty = {
        name: {
            difficulty: _retrieval_metrics(
                [
                    row
                    for row, item in zip(strategy_items[name], qa_pairs)
                    if str(item.get("difficulty", "unspecified")) == difficulty
                ],
                k,
            )
            for difficulty in difficulties
            if any(
                str(item.get("difficulty", "unspecified")) == difficulty
                for item in qa_pairs
            )
        }
        for name in strategy_names
    }

    report = {
        "n_questions": len(qa_pairs),
        "k": k,
        "retrieval_strategies": strategy_reports,
        "metrics_by_difficulty": metrics_by_difficulty,
        "failure_cases": {
            name: _failure_cases(strategy_rows[name], k, failure_limit)
            for name in strategy_names
        },
        "api_usage": {
            "provider": "Gemini",
            "routing_requests": 0,
            "generation_requests": generation_requests,
            "judge_requests": judge_requests,
            "total_requests": generation_requests + judge_requests,
            "answer_quality_latency_ms": _latency_metrics(answer_quality_latencies),
            "tokens": "not collected",
        },
    }
    if evaluate_answers:
        report["answer_quality"] = {
            "faithfulness_mean_1_to_5": round(
                sum(score.faithfulness for score in quality_scores) / len(quality_scores),
                3,
            ),
            "answer_relevance_mean_1_to_5": round(
                sum(score.answer_relevance for score in quality_scores) / len(quality_scores),
                3,
            ),
            "citation_correctness": {
                "correct": citation_correct,
                "total": citation_total,
                "precision": citation_correct / citation_total if citation_total else 0.0,
            },
        }

    print(json.dumps(report, indent=2))
    config.EVAL_DIR.mkdir(parents=True, exist_ok=True)
    (config.EVAL_DIR / "results.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the academic RAG pipeline.")
    parser.add_argument("--k", type=int, default=config.TOP_K_FINAL)
    parser.add_argument("--collection-id")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--failure-limit", type=int, default=3)
    parser.add_argument(
        "--answer-quality",
        action="store_true",
        help=(
            "Generate answers and use Gemini to score faithfulness and relevance; "
            "makes generation and judge requests."
        ),
    )
    args = parser.parse_args()
    evaluate(
        config.EVAL_DIR / "sample_qa.json",
        k=args.k,
        collection_id=args.collection_id,
        evaluate_answers=args.answer_quality,
        limit=args.limit,
        failure_limit=args.failure_limit,
    )
