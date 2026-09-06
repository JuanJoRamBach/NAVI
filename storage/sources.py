"""
storage/sources.py

Persistence for the Chat canvas's Sources tab — a "batch dispatch" run
(dispatcher/source_fetch.py) searches the user's Trusted Sites for each
term, judges relevance, fetches the good ones, and saves a document per
result. Two tables:

    source_batches(id, status, error, created_at, finished_at)
    source_documents(id, batch_id, term, title, url, domain, filen_path,
                      status, created_at)

`source_documents.status` is "pending_review" (default — a human hasn't
looked at it yet), "accepted", or "rejected". Only accepted documents are
meant to ever be used as real chat context — see server.py's routes for
where that gate actually lives.

Sync sqlite3, not aiosqlite — dispatcher/source_fetch.py runs inside a
plain background thread (same "threading.Thread, not asyncio.create_task"
reasoning as server.py's own research dispatch), calling into the same
synchronous provider/tool-loop chain as every other dispatcher module —
no asyncio anywhere in that call chain to hang off of.
"""

import sqlite3
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from urllib.parse import urlparse

DB_PATH = Path(__file__).parent.parent / "sources.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS source_batches (
    id TEXT PRIMARY KEY,
    status TEXT NOT NULL,
    error TEXT,
    created_at REAL NOT NULL,
    finished_at REAL
);
CREATE TABLE IF NOT EXISTS source_documents (
    id TEXT PRIMARY KEY,
    batch_id TEXT NOT NULL,
    term TEXT NOT NULL,
    title TEXT NOT NULL,
    url TEXT NOT NULL,
    domain TEXT NOT NULL,
    filen_path TEXT,
    status TEXT NOT NULL DEFAULT 'pending_review',
    created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_documents_batch ON source_documents(batch_id);
"""

_initialized = False


@contextmanager
def _connect():
    global _initialized
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        if not _initialized:
            conn.executescript(_SCHEMA)
            conn.commit()
            _initialized = True
        yield conn
    finally:
        conn.close()


def domain_of(url: str) -> str:
    """netloc without a leading 'www.' — the normalized form both the
    Trusted Sites registry and matching against it should compare against."""
    netloc = urlparse(url).netloc.lower()
    return netloc[4:] if netloc.startswith("www.") else netloc


def create_batch() -> str:
    batch_id = str(uuid.uuid4())
    with _connect() as conn:
        conn.execute(
            "INSERT INTO source_batches (id, status, created_at) VALUES (?, 'running', ?)",
            (batch_id, time.time()),
        )
        conn.commit()
    return batch_id


def finish_batch(batch_id: str, error: str | None = None) -> None:
    with _connect() as conn:
        conn.execute(
            "UPDATE source_batches SET status = ?, error = ?, finished_at = ? WHERE id = ?",
            ("error" if error else "done", error, time.time(), batch_id),
        )
        conn.commit()


def get_batch(batch_id: str) -> dict | None:
    with _connect() as conn:
        row = conn.execute("SELECT * FROM source_batches WHERE id = ?", (batch_id,)).fetchone()
        return dict(row) if row else None


def latest_batch() -> dict | None:
    """Most recently created batch, regardless of status — the PWA polls
    this (not a specific batch_id, which it never receives — see
    dispatcher/source_fetch.py's start_source_fetch_batch) to know
    whether Batch Dispatch is currently running. Single-user software:
    there's only ever meaningfully one batch "the one you're waiting on"
    at a time, same reasoning as dispatcher/research_status.py."""
    with _connect() as conn:
        row = conn.execute("SELECT * FROM source_batches ORDER BY created_at DESC LIMIT 1").fetchone()
        return dict(row) if row else None


def create_document(batch_id: str, term: str, title: str, url: str, filen_path: str | None) -> str:
    doc_id = str(uuid.uuid4())
    with _connect() as conn:
        conn.execute(
            "INSERT INTO source_documents (id, batch_id, term, title, url, domain, filen_path, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (doc_id, batch_id, term, title, url, domain_of(url), filen_path, time.time()),
        )
        conn.commit()
    return doc_id


def list_documents(batch_id: str | None = None, status: str | None = None) -> list[dict]:
    """Every saved document, newest first — optionally scoped to one
    batch and/or one status. No conversation-scoping yet: Sources is
    app-wide, not per-chat, until a real need for per-conversation
    scoping shows up.

    `status="accepted"` is the query a future context-building consumer
    needs: pull every accepted row, then fetch each one's real content
    from Filen via its `filen_path` — this table never stores the
    content itself, just the pointer to it (see tools/sources.py)."""
    if status is not None and status not in ("pending_review", "accepted", "rejected"):
        raise ValueError(f"Invalid status: {status}")
    clauses, params = [], []
    if batch_id:
        clauses.append("batch_id = ?")
        params.append(batch_id)
    if status:
        clauses.append("status = ?")
        params.append(status)
    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    with _connect() as conn:
        rows = conn.execute(f"SELECT * FROM source_documents {where} ORDER BY created_at DESC", params).fetchall()
        return [dict(r) for r in rows]


def get_document(doc_id: str) -> dict | None:
    """Single-row lookup — needed so the PWA can actually show a
    document's saved content before the user accepts/rejects it (2026-
    09-06, JuanJo: 'I can't see the documents it created, so I can't
    review them'). Mirrors get_batch's shape; list_documents alone never
    covered this since it always returns the whole set."""
    with _connect() as conn:
        row = conn.execute("SELECT * FROM source_documents WHERE id = ?", (doc_id,)).fetchone()
        return dict(row) if row else None


def set_document_status(doc_id: str, status: str) -> bool:
    if status not in ("pending_review", "accepted", "rejected"):
        raise ValueError(f"Invalid status: {status}")
    with _connect() as conn:
        cur = conn.execute("UPDATE source_documents SET status = ? WHERE id = ?", (status, doc_id))
        conn.commit()
        return cur.rowcount > 0
