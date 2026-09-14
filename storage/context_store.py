"""
storage/context_store.py

`context.md`'s real storage — the per-conversation distilled memory that
rides along in every turn's prompt alongside the recency window, plus the
shared friction log. Full design and its open questions:
`how_to_handle_context.md`'s "Friction-driven context" section.

Three tables, three genuinely different jobs — do not collapse them:

1. `context_entries` — the RAW FLAGGED LOG. Every key insight the model
   flags in real time, tagged with the message it came from. Append-only,
   never compacted, never pruned. This is the ground-truth manifest that
   compaction's integrity check runs AGAINST (did every previously
   flagged item survive the new compacted version?) — which is exactly
   why compacting it would defeat its own purpose.
2. `context_snapshots` — the compacted versions. The LIVE one is the row
   with `superseded_at IS NULL`; prior ones are kept, not deleted
   (`how_to_handle_context.md`'s standing "previous summaries preserved,
   never overwritten" rule — functionally replaced going forward, still
   auditable if a pass looks wrong later).
3. `friction_events` — the shared friction signal. Written in this pass,
   deliberately NOT yet consumed: self-learning, compaction-quality
   tracking and dynamic-routing's model-trust signal are all designed to
   read this same table rather than each inventing their own notion of
   "something went wrong here."

`source` on an entry is load-bearing, not decorative: it's how compaction
avoids promoting a `tool_result`-sourced claim into a user-stated fact,
which is the consolidation-time half of the prompt-injection concern
`tools/content_safety.py` handles at ingestion time.
"""

import json
import sqlite3
import time
import uuid
from pathlib import Path

import aiosqlite

DB_PATH = Path(__file__).parent.parent / "conversations.db"

# Sources an entry can carry. Ordered loosely by how much a consolidation
# pass should trust the claim inside it.
SOURCE_USER = "user_message"
SOURCE_ASSISTANT = "assistant_reply"
SOURCE_TOOL = "tool_result"
# Inherited intent, not something said in this conversation. A branch
# chat opens with its spec already dissected into entries carrying this
# source, so consolidation can tell "this was handed down as the scope of
# the work" apart from "this came up while doing the work".
SOURCE_BRANCH_BRIEF = "branch_brief"
# The outcome of a completed branch, written back into its parent as a
# single entry. Same direction discipline as Agent_Work_Context.md: the
# parent receives the result, never the branch's whole history.
SOURCE_BRANCH_RESULT = "branch_result"

# Defined separately from _SCHEMA below, and then appended to it, because
# the SYNC writer (record_friction_sync) has to be able to create this one
# table on its own: a provider call can be the very first thing in a
# process to touch this database, before any async path has run the full
# schema script. One definition, two users — interpolated rather than
# copied so the two can't drift apart.
_FRICTION_SCHEMA = """
CREATE TABLE IF NOT EXISTS friction_events (
    id TEXT PRIMARY KEY,
    conversation_id TEXT,
    message_id TEXT,
    kind TEXT NOT NULL,
    severity INTEGER NOT NULL,
    detail TEXT,
    created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_friction_conversation
    ON friction_events(conversation_id, created_at);
CREATE INDEX IF NOT EXISTS idx_friction_kind
    ON friction_events(kind, created_at);
"""

# `wasted_tokens` — what this friction actually COST, in the only unit
# that is comparable across kinds.
#
# `severity` is a hand-assigned 1/2/3, decided by whoever wrote the call
# site. That is a judgement, and this codebase's standing rule is to
# measure rather than judge wherever a measurement exists. Tokens that
# were paid for and returned nothing usable is that measurement: a
# timeout on a 5,000-token prompt is genuinely five thousand tokens
# burned, and it is meaningfully worse than one on a 200-token prompt in
# a way no severity integer can express.
#
# NULL means "not measurable here", which is deliberately distinct from
# 0 ("measured, and nothing was wasted"). A failed call has no usage
# object to read, so those rows carry an ESTIMATE from the outgoing
# payload — flagged as such in `detail`, same honesty rule estimate_tokens
# itself follows. Severity is kept rather than replaced: it still
# captures how much a kind MATTERS, which is not the same question as
# what it cost.
#
# `provider` / `model` — WHICH model this happened on.
#
# Added 2026-09-14 so live evidence can reach jobs/model_ranking.py. Model
# attribution was the missing half: usage_calls already knows which model
# failed a CALL, but the quality signals that matter most for ranking a
# model — it repeated itself, it answered with nothing, it declared itself
# insufficient at the top tier — live here, and a kind with no model on it
# can describe a problem without ever identifying the thing to change.
# Nullable: a friction event that genuinely isn't about a model (a
# compaction pass that missed target, a user rejecting a plan) leaves both
# NULL rather than pretending to an attribution it doesn't have.
_FRICTION_MIGRATIONS = (
    "ALTER TABLE friction_events ADD COLUMN wasted_tokens INTEGER",
    "ALTER TABLE friction_events ADD COLUMN provider TEXT",
    "ALTER TABLE friction_events ADD COLUMN model TEXT",
)

# The kinds that say something about the MODEL, as opposed to the
# conversation or the user's judgement of it.
#
# A model that repeats itself, returns nothing, burns the whole tool loop
# or declares itself insufficient at the ceiling is telling you something
# about that model. A compaction pass that missed target, or a user
# sending a plan back, is not — the first is about how much material the
# conversation holds, the second is about whether a draft was right, and
# scoring a model down for either would be attributing a result to the
# wrong cause. fallback_used is excluded for a different reason: it
# describes the model that was SKIPPED, and the row is attributed to the
# one that answered.
MODEL_QUALITY_FRICTION = (
    "repetition_loop",
    "empty_response",
    "escalation_at_ceiling",
    "tool_loop_exhausted",
    "provider_timeout",
    "provider_error",
)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS context_entries (
    id TEXT PRIMARY KEY,
    conversation_id TEXT NOT NULL,
    message_id TEXT,
    source TEXT NOT NULL,
    content TEXT NOT NULL,
    created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_context_entries_conversation
    ON context_entries(conversation_id, created_at);

CREATE TABLE IF NOT EXISTS context_snapshots (
    id TEXT PRIMARY KEY,
    conversation_id TEXT NOT NULL,
    content TEXT NOT NULL,
    token_estimate INTEGER NOT NULL,
    created_at REAL NOT NULL,
    superseded_at REAL
);
CREATE INDEX IF NOT EXISTS idx_context_snapshots_live
    ON context_snapshots(conversation_id, superseded_at);

""" + _FRICTION_SCHEMA

_initialized = False


async def _ensure_schema(db: aiosqlite.Connection) -> None:
    global _initialized
    if _initialized:
        return
    await db.executescript(_SCHEMA)
    # ADD COLUMN migrations — CREATE TABLE IF NOT EXISTS never alters an
    # already-existing table on a live database, only a brand-new one.
    # Same pattern storage/sources.py already uses. `retired_at` /
    # `retired_reason` are the explicit forgetting path (2026-09-13): an
    # entry is never deleted, it stops being carried forward once the
    # compactor DECLARES it obsolete and says why.
    for stmt in (
        "ALTER TABLE context_entries ADD COLUMN retired_at REAL",
        "ALTER TABLE context_entries ADD COLUMN retired_reason TEXT",
        # `pinned` exempts an entry from retirement entirely (2026-09-13).
        # Built for a branch spec's acceptance criteria: everything else in
        # a spec can legitimately go stale as the work progresses, but the
        # criteria ARE the definition of finished — lose them and the
        # branch can no longer tell whether it is done.
        "ALTER TABLE context_entries ADD COLUMN pinned INTEGER NOT NULL DEFAULT 0",
    ) + _FRICTION_MIGRATIONS:
        try:
            await db.execute(stmt)
        except Exception:
            pass  # already applied in a prior run
    await db.commit()
    _initialized = True


# ---- Token estimation ----

_encoder = None
_encoder_tried = False

# Fallback divisor when no real tokenizer is available. ~4 chars/token is
# the usual English rule of thumb; 3.6 deliberately OVER-estimates a
# little, so the ceiling trips slightly early rather than slightly late —
# the failure we care about is an oversized context.md riding along on
# every turn, not one compaction firing sooner than strictly needed.
_CHARS_PER_TOKEN_FALLBACK = 3.6


def estimate_tokens(text: str) -> int:
    """Best-effort token count. Uses tiktoken when it's importable and its
    encoding is reachable, otherwise a character heuristic.

    Deliberately called an ESTIMATE, not a count: NAVI routes to gpt-oss,
    Qwen, Mistral and Nemotron, none of which tokenize identically to any
    single tiktoken encoding — so even the tiktoken path is an
    approximation of whichever model actually answers a given turn, not a
    precise figure for it. That's fine for a soft budget ceiling; it would
    not be fine for anything claiming to be an exact provider-side count.

    Never raises. tiktoken fetches its BPE data on first use, which can
    fail on a machine with no network — that must degrade to the
    heuristic, not break the chat turn that called it.
    """
    if not text:
        return 0
    global _encoder, _encoder_tried
    if not _encoder_tried:
        _encoder_tried = True
        try:
            import tiktoken
            _encoder = tiktoken.get_encoding("cl100k_base")
        except Exception:
            _encoder = None
    if _encoder is not None:
        try:
            return len(_encoder.encode(text))
        except Exception:
            pass
    return int(len(text) / _CHARS_PER_TOKEN_FALLBACK) + 1


# ---- The raw flagged log ----

async def append_entry(
    conversation_id: str, content: str, source: str = SOURCE_ASSISTANT,
    message_id: str | None = None, pinned: bool = False,
) -> None:
    """Records one flagged key insight. Never deduplicates — a repeated
    flag is itself real signal (it's how frequency scoring would later
    tell a reinforced fact from a one-off), and compaction is where
    duplicates actually get merged.

    `pinned` marks an entry no consolidation pass may retire. Use it only
    for something whose loss would break a mechanism rather than merely
    forget a fact — today that means a branch spec's acceptance criteria,
    which are what "done" is judged against."""
    if not (content or "").strip():
        return
    async with aiosqlite.connect(DB_PATH) as db:
        await _ensure_schema(db)
        await db.execute(
            "INSERT INTO context_entries (id, conversation_id, message_id, source, content, created_at, pinned) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (str(uuid.uuid4()), conversation_id, message_id, source, content.strip(), time.time(), 1 if pinned else 0),
        )
        await db.commit()


async def get_entries(
    conversation_id: str, since: float | None = None, include_retired: bool = False,
) -> list[dict]:
    """Oldest first. `since` filters to entries created after a timestamp —
    used to find the entries a live snapshot doesn't cover yet.

    Retired entries are excluded by default: they're still in the table
    (nothing here is ever deleted — that's the audit trail), they're just
    no longer carried forward into prompts or checked for survival.
    `include_retired=True` is for inspecting what was retired and why."""
    query = (
        "SELECT id, message_id, source, content, created_at, retired_at, retired_reason, pinned "
        "FROM context_entries WHERE conversation_id = ?"
    )
    params: tuple = (conversation_id,)
    if since is not None:
        query += " AND created_at > ?"
        params = (conversation_id, since)
    if not include_retired:
        query += " AND retired_at IS NULL"
    query += " ORDER BY created_at ASC"
    async with aiosqlite.connect(DB_PATH) as db:
        await _ensure_schema(db)
        db.row_factory = aiosqlite.Row
        async with db.execute(query, params) as cursor:
            rows = await cursor.fetchall()
    return [dict(r) for r in rows]


async def retire_entries(entry_ids: list[str], reason: str) -> int:
    """Marks entries as no longer worth carrying forward. Never deletes —
    the row stays, so "what did we used to remember, and why did we stop?"
    is always answerable.

    This is the ONLY sanctioned way for something to leave context.md.
    Compaction silently omitting an entry does NOT retire it (that entry
    gets rescued verbatim instead) — the compactor has to explicitly
    declare it obsolete and give a reason. Silent loss and deliberate
    forgetting look identical otherwise, and only one of them is safe.

    A pinned entry is refused outright — enforced here in the one place
    retirement can happen, rather than trusted to every caller and every
    future compaction prompt. Returns the number actually retired, so a
    refusal is visible to the caller as a smaller count rather than
    silently reported as success."""
    if not entry_ids:
        return 0
    now = time.time()
    placeholders = ",".join("?" for _ in entry_ids)
    async with aiosqlite.connect(DB_PATH) as db:
        await _ensure_schema(db)
        cursor = await db.execute(
            f"UPDATE context_entries SET retired_at = ?, retired_reason = ? "
            f"WHERE id IN ({placeholders}) AND retired_at IS NULL AND pinned = 0",
            (now, reason, *entry_ids),
        )
        await db.commit()
        return cursor.rowcount or 0


# ---- Compacted snapshots ----

async def get_live_snapshot(conversation_id: str) -> dict | None:
    async with aiosqlite.connect(DB_PATH) as db:
        await _ensure_schema(db)
        db.row_factory = aiosqlite.Row
        async with db.execute(
            "SELECT id, content, token_estimate, created_at FROM context_snapshots "
            "WHERE conversation_id = ? AND superseded_at IS NULL "
            "ORDER BY created_at DESC LIMIT 1",
            (conversation_id,),
        ) as cursor:
            row = await cursor.fetchone()
    return dict(row) if row else None


async def replace_live_snapshot(conversation_id: str, content: str) -> None:
    """Archives whatever was live (sets `superseded_at`, does NOT delete)
    and installs `content` as the new live snapshot."""
    now = time.time()
    async with aiosqlite.connect(DB_PATH) as db:
        await _ensure_schema(db)
        await db.execute(
            "UPDATE context_snapshots SET superseded_at = ? "
            "WHERE conversation_id = ? AND superseded_at IS NULL",
            (now, conversation_id),
        )
        await db.execute(
            "INSERT INTO context_snapshots (id, conversation_id, content, token_estimate, created_at, superseded_at) "
            "VALUES (?, ?, ?, ?, ?, NULL)",
            (str(uuid.uuid4()), conversation_id, content.strip(), estimate_tokens(content), now),
        )
        await db.commit()


# ---- The read path everything else uses ----

async def build_context_block(conversation_id: str) -> tuple[str, int]:
    """The actual text injected into a turn's prompt, plus its token
    estimate. Composed of the live compacted snapshot (if any) followed by
    every entry flagged SINCE that snapshot was taken — so a just-flagged
    insight is available on the very next turn without waiting for a
    compaction pass to fold it in.

    Returns ("", 0) for a conversation that has never flagged anything,
    so callers can skip injecting anything at all rather than adding an
    empty header to every prompt.
    """
    snapshot = await get_live_snapshot(conversation_id)
    entries = await get_entries(conversation_id, since=snapshot["created_at"] if snapshot else None)

    parts: list[str] = []
    if snapshot and snapshot["content"].strip():
        parts.append(snapshot["content"].strip())
    if entries:
        recent = "\n".join(f"- {e['content']}" for e in entries)
        # Header only when there's also a snapshot above it — otherwise
        # the whole block IS these entries and a section label is noise.
        parts.append(f"## Recent\n{recent}" if parts else recent)

    block = "\n\n".join(parts)
    return block, estimate_tokens(block)


# ---- The shared friction signal ----

async def record_friction(
    kind: str, severity: int = 1, conversation_id: str | None = None,
    message_id: str | None = None, detail: str | None = None,
    wasted_tokens: int | None = None,
    provider: str | None = None, model: str | None = None,
) -> None:
    """Logs one friction event. Written now, consumed later — see this
    module's own docstring. Never raises: a friction-logging failure must
    not break the turn that produced it, same rule
    `providers/base.py`'s usage tracking and `tools/registry.py`'s
    per-tool counter already follow."""
    try:
        async with aiosqlite.connect(DB_PATH) as db:
            await _ensure_schema(db)
            await db.execute(
                "INSERT INTO friction_events (id, conversation_id, message_id, kind, severity, detail, created_at, wasted_tokens, provider, model) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (str(uuid.uuid4()), conversation_id, message_id, kind, severity,
                 json.dumps(detail) if isinstance(detail, (dict, list)) else detail, time.time(),
                 wasted_tokens, provider, model),
            )
            await db.commit()
    except Exception as e:
        print(f"[context_store] friction logging failed (non-fatal): {e}")


def record_friction_sync(
    kind: str, severity: int = 1, conversation_id: str | None = None,
    message_id: str | None = None, detail: str | None = None,
    wasted_tokens: int | None = None,
    provider: str | None = None, model: str | None = None,
) -> None:
    """Same row, written from synchronous code.

    WHY THIS EXISTS. The friction signal's most valuable sources are in
    the synchronous layer — providers/base.py's Provider.chat() and
    dispatcher/executor.py's run_tool_loop, both plain functions running
    inside asyncio.to_thread. Neither can await the async writer above,
    and executor.py carried a comment saying exactly that: the tool loop
    hitting its ceiling "is a real friction signal, but it's recorded by
    the CALLER... this function is sync". That workaround is why only ONE
    of run_tool_loop's five callers ever logged it, and why four of
    NAVI's five subsystems could fail in silence.

    Concurrency: this writes to the same file aiosqlite writes to, from a
    worker thread. SQLite serializes writers itself; the busy timeout
    below is what turns "another writer holds the lock" from an error
    into a short wait. Five seconds is far beyond what a single-row
    insert into a tiny table needs, and on the rare occasion it is not
    enough, the except swallows it and one auxiliary row is lost — which
    is the correct trade for a signal that must never break the request
    that produced it.
    """
    try:
        conn = sqlite3.connect(DB_PATH, timeout=5.0)
        try:
            # The full schema script is owned by the async path; this one
            # table is created here too so a sync-first process (a
            # provider call before any chat turn) doesn't hit a missing
            # table. Both use _FRICTION_SCHEMA, so they cannot diverge.
            conn.executescript(_FRICTION_SCHEMA)
            for stmt in _FRICTION_MIGRATIONS:
                try:
                    conn.execute(stmt)
                except sqlite3.Error:
                    pass  # already applied
            conn.execute(
                "INSERT INTO friction_events (id, conversation_id, message_id, kind, severity, detail, created_at, wasted_tokens, provider, model) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (str(uuid.uuid4()), conversation_id, message_id, kind, severity,
                 json.dumps(detail) if isinstance(detail, (dict, list)) else detail, time.time(),
                 wasted_tokens, provider, model),
            )
            conn.commit()
        finally:
            conn.close()
    except Exception as e:
        print(f"[context_store] sync friction logging failed (non-fatal): {e}")


async def get_friction_since(since: float, conversation_id: str | None = None) -> list[dict]:
    query = (
        "SELECT id, conversation_id, message_id, kind, severity, detail, created_at, "
        "wasted_tokens, provider, model "
        "FROM friction_events WHERE created_at > ?"
    )
    params: tuple = (since,)
    if conversation_id:
        query += " AND conversation_id = ?"
        params = (since, conversation_id)
    query += " ORDER BY created_at ASC"
    async with aiosqlite.connect(DB_PATH) as db:
        await _ensure_schema(db)
        db.row_factory = aiosqlite.Row
        async with db.execute(query, params) as cursor:
            rows = await cursor.fetchall()
    return [dict(r) for r in rows]


async def get_friction_summary(days: int = 7) -> list[dict]:
    """Friction grouped by kind over the last `days`, worst first.

    Ordered by wasted tokens rather than by count, deliberately. Sorting
    by frequency puts the cheapest, most routine signal at the top and
    buries the one that actually cost something — the exact failure this
    column exists to avoid. `unmeasured` says how many rows in a kind
    carry no cost figure at all, so a small wasted_tokens total is never
    mistaken for a cheap problem when it is really an unmeasured one.
    """
    since = time.time() - days * 86_400
    async with aiosqlite.connect(DB_PATH) as db:
        await _ensure_schema(db)
        db.row_factory = aiosqlite.Row
        async with db.execute(
            """
            SELECT kind,
                   COUNT(*) AS events,
                   MAX(severity) AS severity,
                   COALESCE(SUM(wasted_tokens), 0) AS wasted_tokens,
                   SUM(CASE WHEN wasted_tokens IS NULL THEN 1 ELSE 0 END) AS unmeasured,
                   COUNT(DISTINCT conversation_id) AS conversations,
                   MAX(created_at) AS last_seen
            FROM friction_events
            WHERE created_at >= ?
            GROUP BY kind
            ORDER BY wasted_tokens DESC, events DESC
            """,
            (since,),
        ) as cursor:
            return [dict(r) for r in await cursor.fetchall()]


def get_model_friction_sync(days: int = 30) -> dict[tuple[str, str], dict]:
    """Model-quality friction per (provider, model) over the last `days`.

    Sync because its consumer is jobs/model_ranking.py, which is a plain
    script with no event loop — the same reason record_friction_sync
    exists, in the other direction.

    Only MODEL_QUALITY_FRICTION kinds are counted: this number is about to
    be used to move a model down a ranking, so a kind that describes the
    conversation rather than the model would put the blame in the wrong
    place. Rows with no attribution are skipped entirely rather than
    pooled into an "unknown" bucket that nothing could act on.
    """
    since = time.time() - days * 86_400
    placeholders = ",".join("?" for _ in MODEL_QUALITY_FRICTION)
    out: dict[tuple[str, str], dict] = {}
    try:
        conn = sqlite3.connect(DB_PATH, timeout=5.0)
        try:
            # Create-if-missing before reading, exactly as the sync writer
            # does. Without it, a box where no friction has EVER been
            # recorded raises "no such table" and logs it as a failure —
            # printing an alarm for the healthiest possible state, and
            # making a genuine read error indistinguishable from an empty
            # one. That confusion is the whole reason this log sat unread.
            conn.executescript(_FRICTION_SCHEMA)
            for stmt in _FRICTION_MIGRATIONS:
                try:
                    conn.execute(stmt)
                except sqlite3.Error:
                    pass  # already applied
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                f"""
                SELECT provider, model, kind, COUNT(*) AS events,
                       COALESCE(SUM(wasted_tokens), 0) AS wasted_tokens
                FROM friction_events
                WHERE created_at >= ? AND provider IS NOT NULL AND model IS NOT NULL
                  AND kind IN ({placeholders})
                GROUP BY provider, model, kind
                """,
                (since, *MODEL_QUALITY_FRICTION),
            ).fetchall()
        finally:
            conn.close()
    except Exception as e:
        print(f"[context_store] model friction read failed (non-fatal): {e}")
        return {}

    for r in rows:
        key = (r["provider"], r["model"])
        entry = out.setdefault(key, {"events": 0, "wasted_tokens": 0, "by_kind": {}})
        entry["events"] += r["events"]
        entry["wasted_tokens"] += r["wasted_tokens"]
        entry["by_kind"][r["kind"]] = r["events"]
    return out
