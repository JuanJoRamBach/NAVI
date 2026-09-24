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
CREATE TABLE IF NOT EXISTS chat_turns (
    client_message_id TEXT PRIMARY KEY,
    conversation_id TEXT,
    status TEXT NOT NULL,
    payload TEXT,
    created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_chat_turns_created
    ON chat_turns(created_at);
"""

# How long a completed turn's result stays replayable. Long enough to
# cover a user reconnecting after a dropped connection and retrying;
# short enough that this table stays small. A key older than this is
# simply treated as new, which at worst re-runs a turn nobody was still
# waiting for.
TURN_REPLAY_TTL_S = 30 * 60

# How long a duplicate request will WAIT for an in-flight turn to finish
# before giving up on replaying its result. Sized above the 15s chat
# budget so the ordinary case — a stream dropped at second 8, the client
# retries, the original turn is still going — resolves by handing back
# the real answer rather than by timing out into an error.
TURN_WAIT_TIMEOUT_S = 45.0

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


async def ensure_conversation(conversation_id: str, mode: str) -> None:
    """Like create_conversation, but for a caller that needs a KNOWN,
    stable id up front (e.g. Agent Vault chat using the saved agent's own
    id as its conversation id — see dispatcher/chat.py's
    run_agent_vault_chat) rather than one minted here. INSERT OR IGNORE
    so calling this on every turn (not just the first) is always safe."""
    now = time.time()
    async with aiosqlite.connect(DB_PATH) as db:
        await _ensure_schema(db)
        await db.execute(
            "INSERT OR IGNORE INTO conversations (id, mode, project_id, parent_id, task_state, created_at, updated_at) "
            "VALUES (?, ?, NULL, NULL, NULL, ?, ?)",
            (conversation_id, mode, now, now),
        )
        await db.commit()


async def set_conversation_project(conversation_id: str, project_id: str | None) -> None:
    """Puts a conversation in a project (storage/knowledge.py), or takes it
    out with None. The column existed from the start and nothing set it
    until projects were built (2026-09-24). Takes effect from the next
    turn: that turn reads the project's brief and searches its library.
    The caller has already checked the person may use the project."""
    async with aiosqlite.connect(DB_PATH) as db:
        await _ensure_schema(db)
        await db.execute(
            "UPDATE conversations SET project_id = ?, updated_at = ? WHERE id = ?",
            (project_id, time.time(), conversation_id),
        )
        await db.commit()


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


# ---- Turn idempotency ----
#
# WHY. A chat turn saves the user's message at the START and the reply at
# the END, and the work in between is paid for whether or not anyone is
# still listening. So a client that loses its connection mid-turn and
# retries would append the same message a second time, run a second full
# turn, and pay twice — while the first turn quietly finishes and saves
# its own reply anyway. The user ends up with their question twice and two
# answers to it.
#
# The fix is the standard one: the CLIENT names the turn. It generates an
# id per message; the server records that id before doing any work, and a
# repeat of the same id replays the first turn's result instead of
# starting a new one. That makes retrying safe, which matters more once
# replies stream — a long-lived connection has far more opportunity to
# break mid-flight than a single request does.
#
# The id must come from the client, not the server: the whole point is to
# survive the client never hearing the server's answer.


async def begin_turn(client_message_id: str, conversation_id: str | None) -> dict | None:
    """Claims this turn id. Returns None if the caller should proceed.

    A non-None return means this exact turn was already seen, and the dict
    is what to hand back instead of running it again:
      {"status": "done", "payload": {...}}  — finished; replay this
      {"status": "running"}                 — still going; wait for it
      {"status": "failed"}                  — it errored; safe to re-run
    """
    now = time.time()
    async with aiosqlite.connect(DB_PATH) as db:
        await _ensure_schema(db)
        # INSERT OR IGNORE is the claim, and it is atomic — two requests
        # racing with the same id cannot both win it. Doing this as a
        # SELECT-then-INSERT would leave exactly the gap this exists to
        # close.
        cursor = await db.execute(
            "INSERT OR IGNORE INTO chat_turns (client_message_id, conversation_id, status, payload, created_at) "
            "VALUES (?, ?, 'running', NULL, ?)",
            (client_message_id, conversation_id, now),
        )
        await db.commit()
        if cursor.rowcount == 1:
            # Won the claim. Opportunistic prune while we hold the
            # connection — cheap, and keeps this table from growing
            # forever without needing a job of its own.
            await db.execute("DELETE FROM chat_turns WHERE created_at < ?", (now - TURN_REPLAY_TTL_S,))
            await db.commit()
            return None

        db.row_factory = aiosqlite.Row
        async with db.execute(
            "SELECT status, payload FROM chat_turns WHERE client_message_id = ?",
            (client_message_id,),
        ) as c:
            row = await c.fetchone()
    if not row:
        return None  # pruned between the insert and the read; treat as new
    if row["status"] == "done" and row["payload"]:
        try:
            return {"status": "done", "payload": json.loads(row["payload"])}
        except json.JSONDecodeError:
            return None
    if row["status"] == "failed":
        # A turn that errored is worth retrying — the point is to stop
        # duplicate WORK, not to make one bad attempt permanent.
        return None
    return {"status": "running"}


async def finish_turn(client_message_id: str, payload: dict) -> None:
    async with aiosqlite.connect(DB_PATH) as db:
        await _ensure_schema(db)
        await db.execute(
            "UPDATE chat_turns SET status = 'done', payload = ? WHERE client_message_id = ?",
            (json.dumps(payload), client_message_id),
        )
        await db.commit()


async def fail_turn(client_message_id: str) -> None:
    """Releases the claim so a retry can genuinely re-run."""
    async with aiosqlite.connect(DB_PATH) as db:
        await _ensure_schema(db)
        await db.execute(
            "UPDATE chat_turns SET status = 'failed' WHERE client_message_id = ?",
            (client_message_id,),
        )
        await db.commit()


async def await_turn(client_message_id: str, timeout: float = TURN_WAIT_TIMEOUT_S) -> dict | None:
    """Waits for an in-flight turn and returns its payload, or None.

    Polling rather than an in-process event, deliberately: the original
    turn may be running in a DIFFERENT worker process, where an asyncio
    primitive would never be signalled. The database is the only thing
    both sides reliably share. Half a second is far below human notice
    for something the user is already waiting on.
    """
    import asyncio

    deadline = time.time() + timeout
    while time.time() < deadline:
        await asyncio.sleep(0.5)
        async with aiosqlite.connect(DB_PATH) as db:
            await _ensure_schema(db)
            db.row_factory = aiosqlite.Row
            async with db.execute(
                "SELECT status, payload FROM chat_turns WHERE client_message_id = ?",
                (client_message_id,),
            ) as c:
                row = await c.fetchone()
        if not row:
            return None
        if row["status"] == "done" and row["payload"]:
            try:
                return json.loads(row["payload"])
            except json.JSONDecodeError:
                return None
        if row["status"] == "failed":
            return None
    return None
