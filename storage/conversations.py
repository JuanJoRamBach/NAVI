"""
storage/conversations.py

Server-side conversation memory — SQLite on the Lightsail disk (persistent,
unlike Render's old scratch-only free tier, so this needs no Filen backup
the way config/store.py does). Built generically for every chat mode, but
only Dev Slate actually reads/writes through it yet (2026-09-01) — Normal/
Research/Brainstorm chat keeps its existing single-message, no-history
behavior on /chat/send until that's deliberately retouched, per JuanJo's
own sequencing call. Wiring a mode in later just means calling these
functions from that mode's handler; nothing here is Dev-Slate-specific.

Async (aiosqlite) rather than the stdlib sqlite3 module, since this is
called from FastAPI's async request/websocket handlers — a blocking
sqlite3 call in an async def would stall the whole event loop for every
other concurrent connection, not just the one that issued it.

Schema:
    conversations(id, mode, project_id, parent_id, task_state, created_at, updated_at)
    messages(id, conversation_id, role, content, provider, model, created_at)

`task_state` is Layer 3 of the 4-layer context design (see the
navi-model-ranking-design-adjacent conversation, 2026-09-01) — one
JSON-serializable blob per conversation, rewritten wholesale via the
update_task_state tool, re-injected fresh each turn rather than living in
message history. `parent_id` exists now so a "Root Slate" vs. sub-Slate
hierarchy has somewhere to live, even though only Root Slates are created
by default (sub-Slates are an experienced-user opt-in, not yet wired to
any UI action).
"""

import json
import time
import uuid
from pathlib import Path

import aiosqlite

DB_PATH = Path(__file__).parent.parent / "conversations.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS conversations (
    id TEXT PRIMARY KEY,
    mode TEXT NOT NULL,
    project_id TEXT,
    parent_id TEXT,
    task_state TEXT,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS messages (
    id TEXT PRIMARY KEY,
    conversation_id TEXT NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    provider TEXT,
    model TEXT,
    created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_messages_conversation
    ON messages(conversation_id, created_at);
"""

_initialized = False


async def _ensure_schema(db: aiosqlite.Connection) -> None:
    global _initialized
    if _initialized:
        return
    await db.executescript(_SCHEMA)
    await db.commit()
    _initialized = True


async def create_conversation(mode: str, project_id: str | None = None, parent_id: str | None = None) -> str:
    conversation_id = str(uuid.uuid4())
    now = time.time()
    async with aiosqlite.connect(DB_PATH) as db:
        await _ensure_schema(db)
        await db.execute(
            "INSERT INTO conversations (id, mode, project_id, parent_id, task_state, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (conversation_id, mode, project_id, parent_id, None, now, now),
        )
        await db.commit()
    return conversation_id


async def get_conversation(conversation_id: str) -> dict | None:
    async with aiosqlite.connect(DB_PATH) as db:
        await _ensure_schema(db)
        db.row_factory = aiosqlite.Row
        async with db.execute(
            "SELECT id, mode, project_id, parent_id, task_state, created_at, updated_at "
            "FROM conversations WHERE id = ?",
            (conversation_id,),
        ) as cursor:
            row = await cursor.fetchone()
    if not row:
        return None
    return dict(row)


async def append_message(
    conversation_id: str, role: str, content: str,
    provider: str | None = None, model: str | None = None,
) -> None:
    now = time.time()
    async with aiosqlite.connect(DB_PATH) as db:
        await _ensure_schema(db)
        await db.execute(
            "INSERT INTO messages (id, conversation_id, role, content, provider, model, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (str(uuid.uuid4()), conversation_id, role, content, provider, model, now),
        )
        await db.execute(
            "UPDATE conversations SET updated_at = ? WHERE id = ?", (now, conversation_id),
        )
        await db.commit()


async def get_messages(conversation_id: str, limit: int | None = None) -> list[dict]:
    """Full-fidelity history, oldest first — no pruning happens to what's
    stored here. Layer-4-style windowing (see devslate_tools.py /
    dispatcher/devslate_chat.py) is applied only to what gets *sent to the
    model* on each call, never to what's persisted."""
    query = "SELECT role, content, provider, model, created_at FROM messages WHERE conversation_id = ? ORDER BY created_at ASC"
    params: tuple = (conversation_id,)
    if limit is not None:
        # Most-recent `limit` rows, still returned oldest-first.
        query = (
            "SELECT role, content, provider, model, created_at FROM ("
            "SELECT role, content, provider, model, created_at FROM messages "
            "WHERE conversation_id = ? ORDER BY created_at DESC LIMIT ?"
            ") ORDER BY created_at ASC"
        )
        params = (conversation_id, limit)
    async with aiosqlite.connect(DB_PATH) as db:
        await _ensure_schema(db)
        db.row_factory = aiosqlite.Row
        async with db.execute(query, params) as cursor:
            rows = await cursor.fetchall()
    return [dict(r) for r in rows]


async def set_task_state(conversation_id: str, state: dict) -> None:
    async with aiosqlite.connect(DB_PATH) as db:
        await _ensure_schema(db)
        await db.execute(
            "UPDATE conversations SET task_state = ?, updated_at = ? WHERE id = ?",
            (json.dumps(state), time.time(), conversation_id),
        )
        await db.commit()


async def get_task_state(conversation_id: str) -> dict | None:
    conversation = await get_conversation(conversation_id)
    if not conversation or not conversation.get("task_state"):
        return None
    try:
        return json.loads(conversation["task_state"])
    except json.JSONDecodeError:
        return None
