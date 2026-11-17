"""
Implements:

* Hybrid search — an EnsembleRetriever that merges the dense (Chroma)
  retriever and a sparse BM25 retriever with reciprocal-rank fusion
  ("RAG-Fusion" idea, but applied across two retrieval *methods* instead 
  of multiple query rewrites).
* A free local cross-encoder re-ranker that re-scores the top-N hybrid
  candidates against the raw query for higher precision before the LLM
  ever sees them ("Re-ranking", uses a free local model instead of 
  the paid Cohere Rerank API).
* Optional per-paper metadata filtering (a lightweight
  "Query Construction") so a question that names a paper only
  retrieves from that paper.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, List, Optional, Sequence

from langchain_core.documents import Document
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from sentence_transformers import CrossEncoder

import config
from src import indexing

logger = logging.getLogger(__name__)

_reranker = None


def get_reranker() -> CrossEncoder:
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoder(config.RERANKER_MODEL)
    return _reranker


def _matches_filters(
    doc: Document,
    collection_id: str | None,
    paper_titles: Sequence[str] | None,
) -> bool:
    metadata = doc.metadata
    source = metadata.get("source")
    if source and not Path(source).is_file():
        return False
    if collection_id and metadata.get("collection_id", config.DEFAULT_COLLECTION_ID) != collection_id:
        return False
    if paper_titles and metadata.get("paper_title") not in paper_titles:
        return False
    return True


def _chroma_filter(
    collection_id: str | None,
    paper_titles: Sequence[str] | None,
) -> dict | None:
    conditions = []
    if collection_id:
        conditions.append({"collection_id": collection_id})
    if paper_titles:
        if len(paper_titles) == 1:
            conditions.append({"paper_title": paper_titles[0]})
        else:
            conditions.append({"paper_title": {"$in": list(paper_titles)}})
    if not conditions:
        return None
    return conditions[0] if len(conditions) == 1 else {"$and": conditions}


def build_hybrid_retriever(
    k: int = config.TOP_K_CANDIDATES,
    collection_id: str | None = None,
    paper_titles: Sequence[str] | None = None,
):
    """Combine Chroma (dense) + BM25 (sparse) retrievers into one hybrid
    retriever using LangChain's EnsembleRetriever (reciprocal rank fusion)."""
    vectorstore = indexing.load_vectorstore()
    search_kwargs: dict[str, Any] = {"k": k}
    metadata_filter = _chroma_filter(collection_id, paper_titles)
    if metadata_filter and indexing.supports_metadata_filters():
        search_kwargs["filter"] = metadata_filter
    dense_retriever = vectorstore.as_retriever(search_kwargs=search_kwargs)

    bm25_data = indexing.load_bm25_index()
    sparse_docs = [
        doc for doc in bm25_data["docs"]
        if _matches_filters(doc, collection_id, paper_titles)
    ]

    if not sparse_docs:
        return dense_retriever
    bm25_retriever = BM25Retriever.from_documents(
        sparse_docs,
        preprocess_func=indexing.tokenize,
    )
    bm25_retriever.k = k
    return EnsembleRetriever(
        retrievers=[dense_retriever, bm25_retriever],
        weights=[config.HYBRID_VECTOR_WEIGHT, 1 - config.HYBRID_VECTOR_WEIGHT],
    )


def filter_by_paper(docs: List[Document], paper_title_hint: Optional[str]) -> List[Document]:
    """Narrow results to a named paper; do not silently broaden failed filters."""
    if not paper_title_hint:
        return docs
    hint = paper_title_hint.lower()
    return [
        d
        for d in docs
        if hint in d.metadata.get("paper_title", "").lower()
        or hint in d.metadata.get("filename", "").lower()
    ]


def rerank(query: str, docs: List[Document], top_k: int = config.TOP_K_FINAL) -> List[Document]:
    """Cross-encoder re-ranking: scores each (query, chunk) pair jointly,
    which is far more precise than embedding cosine similarity alone."""
    if not docs:
        return []
    reranker = get_reranker()
    pairs = [(query, d.page_content) for d in docs]
    scores = reranker.predict(pairs)
    scored = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
    return [d for d, _ in scored[:top_k]]


def _retrieve_hybrid_candidates(
    query: str,
    candidate_k: int,
    collection_id: str | None,
    paper_titles: Sequence[str] | None,
) -> List[Document]:
    if indexing.is_empty_index():
        return []
    hybrid = build_hybrid_retriever(
        k=candidate_k,
        collection_id=collection_id,
        paper_titles=paper_titles,
    )
    candidates = hybrid.invoke(query)
    return [
        doc for doc in candidates
        if _matches_filters(doc, collection_id, paper_titles)
    ]


def retrieve_hybrid(
    query: str,
    candidate_k: int = config.TOP_K_CANDIDATES,
    final_k: int = config.TOP_K_FINAL,
    collection_id: str | None = None,
    paper_titles: Sequence[str] | None = None,
) -> List[Document]:
    """Run dense + BM25 retrieval without the cross-encoder reranker."""
    candidates = _retrieve_hybrid_candidates(
        query,
        candidate_k,
        collection_id,
        paper_titles,
    )
    return candidates[:final_k]


def retrieve(
    query: str,
    paper_hint: Optional[str] = None,
    candidate_k: int = config.TOP_K_CANDIDATES,
    final_k: int = config.TOP_K_FINAL,
    collection_id: str | None = None,
    paper_titles: Sequence[str] | None = None,
) -> List[Document]:
    """Run filtered hybrid retrieval and cross-encoder ranking."""
    if indexing.is_empty_index():
        return []
    selected_titles = list(paper_titles or [])
    if paper_hint:
        docstore = indexing.load_docstore()
        hinted_titles: set[str] = set()
        for doc in docstore:
            title = doc.metadata.get("paper_title")
            filename = doc.metadata.get("filename")
            if (
                isinstance(title, str)
                and paper_hint.lower() in title.lower()
            ) or (
                isinstance(filename, str)
                and paper_hint.lower() in filename.lower()
            ):
                if isinstance(title, str):
                    hinted_titles.add(title)
        if not hinted_titles:
            return []
        selected_titles = (
            sorted(set(selected_titles).intersection(hinted_titles))
            if selected_titles
            else sorted(hinted_titles)
        )
        if not selected_titles:
            return []

    candidates = _retrieve_hybrid_candidates(
        query,
        candidate_k,
        collection_id,
        selected_titles or None,
    )
    return rerank(query, candidates, top_k=final_k)


def retrieve_for_summary(
    paper_hint: str,
    query: str,
    collection_id: str | None = None,
) -> List[Document]:
    """Broader retrieval used for whole-paper / concept summarization:
    pulls a wider pool from the target paper so the summary isn't based on
    a handful of possibly-unrepresentative chunks."""
    docstore = indexing.load_docstore()
    paper_docs = [
        doc for doc in filter_by_paper(docstore, paper_hint)
        if _matches_filters(doc, collection_id, None)
    ]
    if len(paper_docs) <= config.SUMMARY_TOP_K:
        return paper_docs
    return rerank(query, paper_docs, top_k=config.SUMMARY_TOP_K)


def retrieve_for_task(
    query: str,
    collection_id: str,
    paper_titles: Sequence[str] | None = None,
    per_paper_k: int = 8,
) -> List[Document]:
    """Retrieve balanced context for paper-level summaries and comparisons."""
    docstore = indexing.load_docstore()
    collection_docs = [
        doc for doc in docstore
        if _matches_filters(doc, collection_id, paper_titles)
    ]
    if paper_titles:
        selected_docs = []
        for title in paper_titles:
            paper_docs = [
                doc for doc in collection_docs
                if doc.metadata.get("paper_title") == title
            ]
            selected_docs.extend(rerank(query, paper_docs, top_k=per_paper_k))
        return selected_docs
    if collection_docs:
        return rerank(query, collection_docs, top_k=min(config.SUMMARY_TOP_K, len(collection_docs)))
    return retrieve(query, collection_id=collection_id)


def retrieve_multi(
    sub_queries: List[str],
    final_k: int = config.TOP_K_FINAL,
    collection_id: str | None = None,
    paper_titles: Sequence[str] | None = None,
) -> List[Document]:
    """Used for cross-paper / comparison questions (Part 7, Decomposition):
    retrieve separately for each sub-question, then de-duplicate and
    re-rank the union against the original combined intent."""
    if indexing.is_empty_index():
        return []
    hybrid = build_hybrid_retriever(
        k=config.TOP_K_CANDIDATES,
        collection_id=collection_id,
        paper_titles=paper_titles,
    )
    seen = {}
    for sq in sub_queries:
        for d in hybrid.invoke(sq):
            if not _matches_filters(d, collection_id, paper_titles):
                continue
            key = (d.metadata.get("source"), d.metadata.get("page"), d.page_content[:80])
            seen[key] = d
    merged_query = " ".join(sub_queries)
    return rerank(merged_query, list(seen.values()), top_k=final_k)
