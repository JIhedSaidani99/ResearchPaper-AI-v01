"""
The single entry point the Streamlit app (and the evaluation script) calls.
Ties together: query routing -> intent-specific retrieval strategy ->
cross-encoder re-ranking -> Gemini generation with citations.
"""
from __future__ import annotations

import logging
from collections import OrderedDict
from copy import deepcopy
from threading import RLock
from typing import Sequence

import config
from src import generation, indexing, retrieval, router
from src.generation import AssistantTask, RagResult

logger = logging.getLogger(__name__)

_ANSWER_CACHE_MAX_ENTRIES = 64
_answer_cache: OrderedDict[tuple, RagResult] = OrderedDict()
_answer_cache_index_token: tuple[str | None, int | None] | None = None
_answer_cache_lock = RLock()


def _index_token() -> tuple[str | None, int | None]:
    return indexing.active_index_version(), indexing.active_index_revision()


def _answer_cache_key(
    question: str,
    collection_id: str | None,
    paper_titles: Sequence[str] | None,
    task: AssistantTask,
) -> tuple:
    return (
        _index_token(),
        config.GEMINI_MODEL,
        config.GEMINI_TEMPERATURE,
        collection_id,
        tuple(paper_titles or ()),
        task,
        question.strip(),
    )


def clear_answer_cache() -> None:
    global _answer_cache_index_token
    with _answer_cache_lock:
        _answer_cache.clear()
        _answer_cache_index_token = None


def _get_cached_answer(key: tuple) -> RagResult | None:
    global _answer_cache_index_token
    index_token = key[0]
    with _answer_cache_lock:
        if index_token != _answer_cache_index_token:
            _answer_cache.clear()
            _answer_cache_index_token = index_token
        result = _answer_cache.get(key)
        if result is None:
            return None
        _answer_cache.move_to_end(key)
        return deepcopy(result)


def _cache_answer(key: tuple, result: RagResult) -> None:
    with _answer_cache_lock:
        _answer_cache[key] = deepcopy(result)
        _answer_cache.move_to_end(key)
        while len(_answer_cache) > _ANSWER_CACHE_MAX_ENTRIES:
            _answer_cache.popitem(last=False)


def format_answering_error(error: Exception) -> str:
    message = str(error)
    normalized = message.casefold()
    status_code = getattr(error, "status_code", None)
    if (
        status_code == 429
        or "resource_exhausted" in normalized
        or "quota exceeded" in normalized
    ):
        return (
            "The Gemini quota for the configured model or project is exhausted. "
            "This request was not retried. Wait for the quota to reset, or select "
            "a model/project with available quota."
        )
    if isinstance(error, TimeoutError) or any(
        marker in normalized
        for marker in ("timed out", "timeout", "deadline exceeded", "deadlineexceeded")
    ):
        return (
            "The Gemini request timed out and was not retried automatically. "
            "Please try again later."
        )
    return f"Request failed: {message}"


def answer_question(
    question: str,
    collection_id: str | None = None,
    paper_titles: Sequence[str] | None = None,
    task: AssistantTask = "Ask",
) -> RagResult:
    cache_key = _answer_cache_key(question, collection_id, paper_titles, task)
    cached = _get_cached_answer(cache_key)
    if cached is not None:
        logger.info("Using cached answer for collection %s.", collection_id)
        return cached

    if task != "Ask":
        if not collection_id:
            raise ValueError("Select a collection before starting a research task.")
        if task == "Compare methodologies" and len(set(paper_titles or [])) < 2:
            raise ValueError("Select at least two papers to compare methodologies.")
        docs = retrieval.retrieve_for_task(
            question,
            collection_id=collection_id,
            paper_titles=paper_titles,
            per_paper_k=(
                config.SUMMARY_TOP_K
                if task == "Summary"
                else 6
                if task == "Compare methodologies"
                else 8
            ),
        )
        result = generation.generate_answer(question, docs, task=task)
        _cache_answer(cache_key, result)
        return result

    known_papers = indexing.list_indexed_papers(collection_id) if collection_id else []
    decision = router.route(question, paper_titles=known_papers)
    logger.info("Routed '%s' -> %s (paper_hint=%s)", question, decision.query_type, decision.paper_hint)

    if decision.query_type == "SUMMARY" and decision.paper_hint:
        docs = retrieval.retrieve_for_summary(
            decision.paper_hint,
            question,
            collection_id=collection_id,
        )
    elif decision.query_type == "COMPARISON" and decision.sub_questions:
        docs = retrieval.retrieve_multi(
            decision.sub_questions,
            collection_id=collection_id,
            paper_titles=paper_titles,
        )
    else:
        # FACT, GENERAL, or COMPARISON/SUMMARY without enough info to
        # specialize -> standard high-precision hybrid retrieval.
        docs = retrieval.retrieve(
            question,
            paper_hint=decision.paper_hint,
            collection_id=collection_id,
            paper_titles=paper_titles,
        )

    result = generation.generate_answer(question, docs)
    result.query_type = decision.query_type
    _cache_answer(cache_key, result)
    return result
