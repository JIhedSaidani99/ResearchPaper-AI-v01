"""Local SQLite catalog for collections, PDFs, and indexing status."""
from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Generator

import config


class DuplicateDocumentError(ValueError):
    """Raised when the same PDF is already in a collection."""


@contextmanager
def _connect() -> Generator[sqlite3.Connection, None, None]:
    config.LIBRARY_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(config.LIBRARY_DB_PATH)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    try:
        with connection:
            yield connection
    finally:
        connection.close()


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _hash_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _bump_revision(connection: sqlite3.Connection) -> None:
    connection.execute(
        "UPDATE library_state SET value = CAST(value AS INTEGER) + 1 WHERE key = 'revision'"
    )


def _safe_source(relative_path: str) -> Path:
    root = config.PDF_DIR.resolve()
    source = (config.PDF_DIR / relative_path).resolve()
    if not source.is_relative_to(root):
        raise ValueError("The PDF path is outside the configured library directory.")
    return source


def ensure_library() -> None:
    """Create the catalog and register PDFs already present on disk."""
    config.PDF_DIR.mkdir(parents=True, exist_ok=True)
    with _connect() as connection:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS collections (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL COLLATE NOCASE UNIQUE,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS documents (
                id TEXT PRIMARY KEY,
                collection_id TEXT NOT NULL REFERENCES collections(id) ON DELETE CASCADE,
                filename TEXT NOT NULL,
                relative_path TEXT NOT NULL UNIQUE,
                sha256 TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'pending',
                status_message TEXT,
                indexed_at TEXT,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(collection_id, sha256)
            );
            CREATE TABLE IF NOT EXISTS library_state (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS index_state (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                status TEXT NOT NULL,
                error TEXT,
                indexed_revision INTEGER NOT NULL DEFAULT -1,
                started_at TEXT,
                completed_at TEXT
            );
            INSERT OR IGNORE INTO collections (id, name)
                VALUES ('general', 'General');
            INSERT OR IGNORE INTO library_state (key, value)
                VALUES ('revision', '0');
            INSERT OR IGNORE INTO index_state (id, status)
                VALUES (1, 'empty');
            """
        )

        known_paths = {
            row["relative_path"]
            for row in connection.execute("SELECT relative_path FROM documents")
        }
        added = 0
        for source in sorted(config.PDF_DIR.rglob("*.pdf")):
            relative_path = source.relative_to(config.PDF_DIR).as_posix()
            if relative_path in known_paths:
                continue
            collection_id = (
                relative_path.split("/", maxsplit=1)[0]
                if "/" in relative_path
                else config.DEFAULT_COLLECTION_ID
            )
            if collection_id != config.DEFAULT_COLLECTION_ID:
                connection.execute(
                    "INSERT OR IGNORE INTO collections (id, name) VALUES (?, ?)",
                    (collection_id, collection_id),
                )
            document_id = uuid.uuid5(uuid.NAMESPACE_URL, str(source.resolve())).hex
            connection.execute(
                """
                INSERT OR IGNORE INTO documents
                    (id, collection_id, filename, relative_path, sha256)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    document_id,
                    collection_id,
                    source.name,
                    relative_path,
                    _hash_file(source),
                ),
            )
            added += connection.execute("SELECT changes()").fetchone()[0]

        if added:
            _bump_revision(connection)
            connection.execute(
                "UPDATE index_state SET status = 'pending', error = NULL WHERE id = 1"
            )

        document_count = connection.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
        if document_count and connection.execute(
            "SELECT status FROM index_state WHERE id = 1"
        ).fetchone()["status"] == "empty":
            has_legacy_index = config.BM25_INDEX_PATH.exists() and config.CHROMA_DIR.exists()
            connection.execute(
                "UPDATE index_state SET status = ? WHERE id = 1",
                ("legacy" if has_legacy_index else "pending",),
            )


def create_collection(name: str) -> str:
    ensure_library()
    name = name.strip()
    if not name or len(name) > 80:
        raise ValueError("Collection names must be between 1 and 80 characters.")
    collection_id = uuid.uuid4().hex
    with _connect() as connection:
        try:
            connection.execute(
                "INSERT INTO collections (id, name) VALUES (?, ?)",
                (collection_id, name),
            )
        except sqlite3.IntegrityError as exc:
            raise ValueError(f"A collection named '{name}' already exists.") from exc
    (config.PDF_DIR / collection_id).mkdir(parents=True, exist_ok=True)
    return collection_id


def list_collections() -> list[dict[str, Any]]:
    ensure_library()
    with _connect() as connection:
        rows = connection.execute(
            """
            SELECT c.id, c.name, COUNT(d.id) AS document_count
            FROM collections AS c
            LEFT JOIN documents AS d ON d.collection_id = c.id
            GROUP BY c.id, c.name
            ORDER BY c.name COLLATE NOCASE
            """
        ).fetchall()
    return [dict(row) for row in rows]


def list_documents(collection_id: str | None = None) -> list[dict[str, Any]]:
    ensure_library()
    query = """
        SELECT d.*, c.name AS collection_name
        FROM documents AS d
        JOIN collections AS c ON c.id = d.collection_id
    """
    params: tuple[str, ...] = ()
    if collection_id is not None:
        query += " WHERE d.collection_id = ?"
        params = (collection_id,)
    query += " ORDER BY d.filename COLLATE NOCASE"
    with _connect() as connection:
        rows = connection.execute(query, params).fetchall()
    documents = []
    for row in rows:
        document = dict(row)
        document["path"] = _safe_source(document["relative_path"])
        documents.append(document)
    return documents


def add_pdf(collection_id: str, filename: str, content: bytes) -> dict[str, Any]:
    ensure_library()
    if not content:
        raise ValueError("The uploaded PDF is empty.")
    basename = filename.replace("\\", "/").rsplit("/", maxsplit=1)[-1].strip()
    if not basename or Path(basename).suffix.lower() != ".pdf":
        raise ValueError("Only PDF files can be added.")

    checksum = _hash_bytes(content)
    with _connect() as connection:
        collection = connection.execute(
            "SELECT id FROM collections WHERE id = ?", (collection_id,)
        ).fetchone()
        if collection is None:
            raise ValueError("The selected collection no longer exists.")
        duplicate = connection.execute(
            "SELECT filename FROM documents WHERE collection_id = ? AND sha256 = ?",
            (collection_id, checksum),
        ).fetchone()
        if duplicate:
            raise DuplicateDocumentError(
                f"'{duplicate['filename']}' is already in this collection."
            )

    directory = (config.PDF_DIR / collection_id).resolve()
    if not directory.is_relative_to(config.PDF_DIR.resolve()):
        raise ValueError("The selected collection path is invalid.")
    directory.mkdir(parents=True, exist_ok=True)

    stem, suffix = Path(basename).stem, Path(basename).suffix
    safe_stem = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", stem).strip(" .")
    if not safe_stem:
        raise ValueError("The uploaded PDF must have a valid filename.")
    target = directory / f"{safe_stem}{suffix.lower()}"
    counter = 2
    while target.exists():
        target = directory / f"{safe_stem} ({counter}){suffix.lower()}"
        counter += 1

    temporary = target.with_name(f".{uuid.uuid4().hex}.uploading")
    try:
        temporary.write_bytes(content)
        os.replace(temporary, target)
        relative_path = target.relative_to(config.PDF_DIR).as_posix()
        document_id = uuid.uuid4().hex
        with _connect() as connection:
            connection.execute(
                """
                INSERT INTO documents
                    (id, collection_id, filename, relative_path, sha256, status)
                VALUES (?, ?, ?, ?, ?, 'pending')
                """,
                (document_id, collection_id, target.name, relative_path, checksum),
            )
            _bump_revision(connection)
            connection.execute(
                "UPDATE index_state SET status = 'pending', error = NULL WHERE id = 1"
            )
    except Exception:
        temporary.unlink(missing_ok=True)
        if target.exists():
            target.unlink()
        raise

    return {
        "id": document_id,
        "collection_id": collection_id,
        "filename": target.name,
        "relative_path": relative_path,
        "path": target,
        "status": "pending",
    }


def remove_document(document_id: str) -> str:
    ensure_library()
    with _connect() as connection:
        row = connection.execute(
            "SELECT relative_path, filename FROM documents WHERE id = ?",
            (document_id,),
        ).fetchone()
        if row is None:
            raise ValueError("The selected document no longer exists.")
        source = _safe_source(row["relative_path"])
        if source.exists():
            source.unlink()
        connection.execute("DELETE FROM documents WHERE id = ?", (document_id,))
        _bump_revision(connection)
        connection.execute(
            "UPDATE index_state SET status = 'pending', error = NULL WHERE id = 1"
        )
    return row["filename"]


def library_revision() -> int:
    ensure_library()
    with _connect() as connection:
        row = connection.execute(
            "SELECT value FROM library_state WHERE key = 'revision'"
        ).fetchone()
    return int(row["value"])


def get_index_state() -> dict[str, Any]:
    ensure_library()
    with _connect() as connection:
        state = dict(connection.execute("SELECT * FROM index_state WHERE id = 1").fetchone())
        revision = int(
            connection.execute(
                "SELECT value FROM library_state WHERE key = 'revision'"
            ).fetchone()["value"]
        )
    state["revision"] = revision
    state["is_stale"] = (
        state["status"] in {"indexed", "legacy"} and state["indexed_revision"] != revision
    )
    if state["is_stale"]:
        state["status"] = "stale"
    return state


def mark_index_started() -> None:
    ensure_library()
    with _connect() as connection:
        connection.execute(
            """
            UPDATE index_state
            SET status = 'indexing', error = NULL, started_at = CURRENT_TIMESTAMP
            WHERE id = 1
            """
        )
        connection.execute(
            "UPDATE documents SET status = 'indexing', status_message = NULL"
        )


def mark_index_complete(indexed_paths: set[str], skipped: list[dict[str, str]]) -> None:
    ensure_library()
    normalized_indexed = {str(Path(path).resolve()).casefold() for path in indexed_paths}
    skipped_by_path = {
        str(Path(item["source"]).resolve()).casefold(): item["reason"]
        for item in skipped
        if item.get("source")
    }
    with _connect() as connection:
        revision = int(
            connection.execute(
                "SELECT value FROM library_state WHERE key = 'revision'"
            ).fetchone()["value"]
        )
        rows = connection.execute(
            "SELECT id, relative_path FROM documents"
        ).fetchall()
        for row in rows:
            source = str(_safe_source(row["relative_path"]).resolve()).casefold()
            if source in normalized_indexed:
                status, message = "indexed", None
            elif source in skipped_by_path:
                status, message = "failed", skipped_by_path[source]
            else:
                status, message = "failed", "No searchable text chunks were produced."
            connection.execute(
                """
                UPDATE documents
                SET status = ?, status_message = ?,
                    indexed_at = CASE WHEN ? = 'indexed' THEN CURRENT_TIMESTAMP ELSE NULL END
                WHERE id = ?
                """,
                (status, message, status, row["id"]),
            )
        connection.execute(
            """
            UPDATE index_state
            SET status = 'indexed', error = NULL, indexed_revision = ?,
                completed_at = CURRENT_TIMESTAMP
            WHERE id = 1
            """,
            (revision,),
        )


def mark_index_failed(error: str) -> None:
    ensure_library()
    with _connect() as connection:
        connection.execute(
            "UPDATE index_state SET status = 'failed', error = ? WHERE id = 1",
            (error,),
        )
        connection.execute(
            """
            UPDATE documents
            SET status = 'failed', status_message = ?
            WHERE status = 'indexing'
            """,
            (error,),
        )


def rebuild_library_index() -> int:
    """Rebuild all indexes from the registered PDFs and update their status."""
    ensure_library()
    mark_index_started()
    try:
        from src import indexing, ingestion

        chunks = ingestion.load_and_chunk_pdfs(config.PDF_DIR)
        if not chunks:
            indexing.write_empty_index(library_revision())
        else:
            indexing.build_all_indexes(chunks, library_revision())
        skipped = (
            json.loads(config.SKIPPED_LOG_PATH.read_text(encoding="utf-8"))
            if config.SKIPPED_LOG_PATH.exists()
            else []
        )
        indexed_paths = {str(chunk.metadata["source"]) for chunk in chunks}
        mark_index_complete(indexed_paths, skipped)
        return len(chunks)
    except Exception as exc:
        mark_index_failed(str(exc))
        raise
