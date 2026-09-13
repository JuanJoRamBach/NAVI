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
plain background thread ("threading.Thread, not asyncio.create_task",
same reasoning as server.py's other background dispatches), calling
into the same synchronous provider/tool-loop chain as every other
dispatcher module — no asyncio anywhere in that call chain to hang
off of.
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
    -- Vestigial since 2026-09-13: URLs are pasted directly, so there is
    -- no search term any more (dispatcher/source_ingest.py explains why
    -- a document must not depend on the term it was found under).
    --
    -- STILL NOT NULL, and it has to stay that way. Every database created
    -- before that date has NOT NULL here, and SQLite cannot drop a NOT
    -- NULL constraint with ALTER TABLE — only a full table rebuild can,
    -- which is not worth the risk to user data for a dead column. So this
    -- declaration matches what live databases actually have, and nothing
    -- may write NULL into it. Writers pass "" instead; see create_document.
    term TEXT NOT NULL DEFAULT '',
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
            # ADD COLUMN migration, not baked into _SCHEMA — CREATE TABLE
            # IF NOT EXISTS never alters an already-existing table on a
            # live database, only a brand-new one. `reason` explains an
            # AUTO-rejected document (2026-09-06, JuanJo: "why some were
            # rejected... must be informed to the user") — NULL for every
            # normal pending_review/human-reviewed row.
            #
            # The 2026-09-13 rebuild added the rest: a Source is no longer
            # a title and a file path, it is a structured document plus
            # the cleaned page it was distilled from. Keeping `markdown`
            # is load-bearing, not a nicety — it makes the structured
            # document a VIEW of the source rather than a replacement for
            # it, so a poor distillation costs a re-run and never the
            # material itself.
            for stmt in (
                "ALTER TABLE source_documents ADD COLUMN reason TEXT",
                # The distilled document, JSON (tools/source_document.py).
                "ALTER TABLE source_documents ADD COLUMN document TEXT",
                # The cleaned Markdown it was distilled FROM. Ground truth.
                "ALTER TABLE source_documents ADD COLUMN markdown TEXT",
                # Which extractor produced that markdown, so a document
                # built on the degraded regex fallback is identifiable
                # rather than silently indistinguishable.
                "ALTER TABLE source_documents ADD COLUMN extractor TEXT",
                # The page was longer than MAX_CHARS. A document distilled
                # from a truncated page is honest work on partial input and
                # has to say so.
                "ALTER TABLE source_documents ADD COLUMN truncated INTEGER DEFAULT 0",
                # Why a URL produced no document at all. A dead link still
                # gets a row — pasting five URLs and getting three cards
                # with no explanation reads as a bug.
                "ALTER TABLE source_documents ADD COLUMN fetch_error TEXT",
                # Grounding summary, JSON: how many claims verified, how
                # many disputed, whether it needs attention.
                "ALTER TABLE source_documents ADD COLUMN grounding TEXT",
            ):
                try:
                    conn.execute(stmt)
                except sqlite3.OperationalError:
                    pass  # already applied in a prior run
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
    at a time."""
    with _connect() as conn:
        row = conn.execute("SELECT * FROM source_batches ORDER BY created_at DESC LIMIT 1").fetchone()
        return dict(row) if row else None


def create_document(
    batch_id: str, term: str | None, title: str, url: str, filen_path: str | None,
    status: str = "pending_review", reason: str | None = None,
    document: str | None = None, markdown: str | None = None,
    extractor: str | None = None, truncated: bool = False,
    fetch_error: str | None = None, grounding: str | None = None,
) -> str:
    """`status`/`reason` default to the normal human-review flow — passed
    explicitly only for a document the DISPATCHER already auto-rejected
    (tools/registry.py's save_source content-quality guard) before a
    human ever saw it, so `reason` can explain why right in the UI
    instead of the row just silently never existing."""
    doc_id = str(uuid.uuid4())
    with _connect() as conn:
        conn.execute(
            "INSERT INTO source_documents (id, batch_id, term, title, url, domain, filen_path, status, reason, "
            "document, markdown, extractor, truncated, fetch_error, grounding, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            # term coerced to "" rather than passed through as None: the
            # column is a dead field but is NOT NULL on every pre-2026-09-13
            # database, and SQLite cannot drop that constraint. Writing NULL
            # fails the insert outright — which it did, live, the first
            # time a URL-driven batch ran.
            (doc_id, batch_id, term or "", title, url, domain_of(url), filen_path, status, reason,
             document, markdown, extractor, 1 if truncated else 0, fetch_error, grounding, time.time()),
        )
        conn.commit()
    return doc_id


def delete_document(doc_id: str) -> bool:
    """Permanent removal — 2026-09-06, JuanJo: 'I need to be able to
    eliminate rejected documents.' Doesn't touch the backing Filen file
    (if any); the DB row disappearing from every list/review view is
    the actual ask, not real storage reclamation."""
    with _connect() as conn:
        cur = conn.execute("DELETE FROM source_documents WHERE id = ?", (doc_id,))
        conn.commit()
        return cur.rowcount > 0


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

def find_by_url(url: str) -> dict | None:
    """The most recent document already produced for this exact URL.

    This is what stops a re-paste costing a fetch, a distillation call and
    a verification pass for a page already read — the dominant cost in
    this whole pipeline is the model call, and it is entirely avoidable
    when the answer is already stored.

    Exact URL match only, deliberately. Deciding that two URLs are "the
    same page" is genuinely hard (tracking parameters, trailing slashes,
    http vs https, AMP variants, redirects), and a wrong match silently
    serves a document for a DIFFERENT page — far worse than paying to
    fetch the same article twice. The fetcher already resolves redirects
    and stores the final URL, which catches the common case honestly.
    """
    with _connect() as conn:
        row = conn.execute(
            "SELECT * FROM source_documents WHERE url = ? AND fetch_error IS NULL "
            "ORDER BY created_at DESC LIMIT 1",
            (url,),
        ).fetchone()
    return dict(row) if row else None


def list_inspected(limit: int = 500) -> list[dict]:
    """Every URL already read, newest first — what the UI shows as
    Inspected Sources. One row per URL, not per attempt: re-reading a page
    should not make the list longer."""
    with _connect() as conn:
        rows = conn.execute(
            "SELECT url, domain, title, status, MAX(created_at) AS created_at, COUNT(*) AS times "
            "FROM source_documents WHERE fetch_error IS NULL "
            "GROUP BY url ORDER BY created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
    return [dict(r) for r in rows]
