"""
Builds and persists two indexes over the same chunk set so retrieval can be
hybrid:

* A dense vector index (Chroma + all-MiniLM-L6-v2 embeddings) for semantic
  similarity search.
* A sparse BM25 index (rank_bm25) for exact keyword / terminology matches,
  which academic text (method names, dataset names, acronyms) really
  benefits from.
"""
from __future__ import annotations

import logging
import json
import os
import pickle
import re
import shutil
import uuid
from pathlib import Path
from typing import Any, List

from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from rank_bm25 import BM25Okapi

import config

logger = logging.getLogger(__name__)

_embeddings = None


def get_embeddings() -> HuggingFaceEmbeddings:
    """instantiate the local embedding model"""
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(
            model_name=config.EMBEDDING_MODEL,
            encode_kwargs={"normalize_embeddings": True},
        )
    return _embeddings


def build_vectorstore(
    chunks: List[Document],
    persist_directory: Path | None = None,
) -> Chroma:
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=get_embeddings(),
        collection_name=config.COLLECTION_NAME,
        persist_directory=str(persist_directory or config.CHROMA_DIR),
    )
    return vectorstore


def _active_manifest() -> dict[str, Any] | None:
    if not config.INDEX_MANIFEST_PATH.exists():
        return None
    with config.INDEX_MANIFEST_PATH.open(encoding="utf-8") as source:
        manifest = json.load(source)
    version_id = manifest.get("version_id")
    if not isinstance(version_id, str) or not version_id.isalnum():
        raise ValueError("The active index manifest contains an invalid version ID.")
    return manifest


def active_index_revision() -> int | None:
    manifest = _active_manifest()
    if manifest is None:
        return None
    revision = manifest.get("library_revision")
    return revision if isinstance(revision, int) else None


def active_index_version() -> str | None:
    manifest = _active_manifest()
    if manifest is None:
        return None
    version_id = manifest.get("version_id")
    return version_id if isinstance(version_id, str) else None


def is_empty_index() -> bool:
    manifest = _active_manifest()
    return bool(manifest and manifest.get("empty"))


def supports_metadata_filters() -> bool:
    return _active_manifest() is not None


def list_indexed_papers(collection_id: str) -> list[str]:
    try:
        docs = load_docstore()
    except FileNotFoundError:
        return []
    return sorted(
        {
            doc.metadata["paper_title"]
            for doc in docs
            if doc.metadata.get("collection_id", config.DEFAULT_COLLECTION_ID) == collection_id
            and doc.metadata.get("paper_title")
        },
        key=str.casefold,
    )


def _active_version_dir() -> Path | None:
    manifest = _active_manifest()
    if manifest is None:
        return None
    return config.INDEX_VERSIONS_DIR / manifest["version_id"]


def load_vectorstore() -> Chroma:
    if is_empty_index():
        raise RuntimeError("The library contains no searchable PDF chunks.")
    version_dir = _active_version_dir()
    return Chroma(
        collection_name=config.COLLECTION_NAME,
        embedding_function=get_embeddings(),
        persist_directory=str(
            version_dir / "chroma" if version_dir is not None else config.CHROMA_DIR
        ),
    )


def tokenize(text: str) -> List[str]:
    tokens = []
    for token in re.findall(r"[\w]+(?:[-./][\w]+)*", text.casefold()):
        tokens.append(token)
        tokens.extend(part for part in re.split(r"[-./]", token) if part)
    return tokens


def build_bm25_index(
    chunks: List[Document],
    index_path: Path | None = None,
) -> None:
    corpus_tokens = [tokenize(d.page_content) for d in chunks]
    bm25 = BM25Okapi(corpus_tokens)
    target = index_path or config.BM25_INDEX_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("wb") as f:
        pickle.dump({"bm25": bm25, "docs": chunks}, f)
    logger.info("BM25 index built with %d documents -> %s", len(chunks), target)


def load_bm25_index() -> dict:
    if is_empty_index():
        return {"bm25": None, "docs": []}
    version_dir = _active_version_dir()
    path = version_dir / "bm25_index.pkl" if version_dir is not None else config.BM25_INDEX_PATH
    with path.open("rb") as f:
        return pickle.load(f)


def save_docstore(chunks: List[Document], docstore_path: Path | None = None) -> None:
    target = docstore_path or config.DOCSTORE_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("wb") as f:
        pickle.dump(chunks, f)


def load_docstore() -> List[Document]:
    if is_empty_index():
        return []
    version_dir = _active_version_dir()
    path = version_dir / "docstore.pkl" if version_dir is not None else config.DOCSTORE_PATH
    with path.open("rb") as f:
        return pickle.load(f)


def _write_manifest(
    version_id: str,
    library_revision: int | None,
    empty: bool = False,
) -> None:
    config.INDEX_MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary = config.INDEX_MANIFEST_PATH.with_name(
        f".{config.INDEX_MANIFEST_PATH.name}.{uuid.uuid4().hex}.tmp"
    )
    manifest = {
        "version_id": version_id,
        "library_revision": library_revision,
        "empty": empty,
    }
    try:
        temporary.write_text(json.dumps(manifest), encoding="utf-8")
        os.replace(temporary, config.INDEX_MANIFEST_PATH)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def write_empty_index(library_revision: int) -> None:
    """Make an empty library active without falling back to a legacy index."""
    _write_manifest(uuid.uuid4().hex, library_revision, empty=True)


def _cleanup_old_versions(active_version: str) -> None:
    if not config.INDEX_VERSIONS_DIR.exists():
        return
    versions = sorted(
        (
            path
            for path in config.INDEX_VERSIONS_DIR.iterdir()
            if path.is_dir() and path.name != active_version
        ),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for old_version in versions[1:]:
        try:
            shutil.rmtree(old_version)
        except OSError as exc:
            logger.warning("Could not remove old index version %s: %s", old_version, exc)


def build_all_indexes(
    chunks: List[Document],
    library_revision: int | None = None,
) -> None:
    """Convenience wrapper used by scripts/build_index.py."""
    if not chunks:
        raise ValueError(
            "No chunks to index. Add PDFs to data/pdfs/ and check data/skipped_pdfs.json."
        )
    version_id = uuid.uuid4().hex
    version_dir = config.INDEX_VERSIONS_DIR / version_id
    version_dir.mkdir(parents=True, exist_ok=False)
    try:
        logger.info("Building Chroma vector index (%d chunks)...", len(chunks))
        build_vectorstore(chunks, version_dir / "chroma")
        logger.info("Building BM25 keyword index...")
        build_bm25_index(chunks, version_dir / "bm25_index.pkl")
        save_docstore(chunks, version_dir / "docstore.pkl")
        _write_manifest(version_id, library_revision)
    except Exception:
        try:
            shutil.rmtree(version_dir)
        except OSError as cleanup_error:
            logger.warning("Could not clean failed index build %s: %s", version_dir, cleanup_error)
        raise
    _cleanup_old_versions(version_id)
    logger.info("All indexes built successfully (version %s).", version_id)
