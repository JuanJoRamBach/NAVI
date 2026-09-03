"""
storage/agent_work.py

Persistence for "Agent Work" — native (not third-party-embedded) multi-step
agent task execution, with scheduling. Same aiosqlite + lazy-schema pattern
as storage/conversations.py, kept in its own DB file (agent_work.db) rather
than sharing conversations.db, since this is a genuinely separate concern
(workflow definitions and their runs, not chat history).

Schema:
    workflow_definitions(id, name, description, graph, trigger, creation_transcript, created_at, updated_at)
    agent_runs(id, workflow_id, status, trigger_source, started_at, finished_at, error)
    agent_run_steps(id, run_id, node_id, seq, status, input, output, error, started_at, finished_at)

`graph` is `{"nodes": [{"id", "label", "prompt", "role"?, "tools"?}], "edges": [{"from", "to"}]}`
— even a v1 workflow with one linear chain of nodes is stored as this
shape, not a bare list, specifically so a future node-graph visual builder
(JuanJo, 2026-09-01: "I would like a node-graph visual builder eventually,
so the backend would need to be laid out for that in mind") never needs a
schema migration — a straight line is just a degenerate graph. The executor
(dispatcher/agent_work.py) topologically sorts `nodes`/`edges` the same way
regardless of whether that graph is linear or branching.

`trigger` is `{"type": "manual"}` or `{"type": "scheduled", "interval_seconds",
"next_run_at"}` — epoch seconds, no cron-expression parsing (no dependency
for it anywhere in this codebase yet, and dispatcher/reminders.py already
sets the precedent of plain fire_at timestamps over cron syntax).

Every `agent_run_step` row's `node_id` ties it back to the specific graph
node that produced it — the hook a future visual canvas needs to color a
node by its live/last execution status without any further schema change.
"""

import json
import time
import uuid
from pathlib import Path

import aiosqlite

DB_PATH = Path(__file__).parent.parent / "agent_work.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS workflow_definitions (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    description TEXT,
    graph TEXT NOT NULL,
    trigger_json TEXT NOT NULL,
    creation_transcript TEXT,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS agent_runs (
    id TEXT PRIMARY KEY,
    workflow_id TEXT,
    status TEXT NOT NULL,
    trigger_source TEXT NOT NULL,
    started_at REAL NOT NULL,
    finished_at REAL,
    error TEXT
);
CREATE INDEX IF NOT EXISTS idx_runs_workflow ON agent_runs(workflow_id, started_at);
CREATE TABLE IF NOT EXISTS agent_run_steps (
    id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    node_id TEXT NOT NULL,
    seq INTEGER NOT NULL,
    status TEXT NOT NULL,
    input TEXT,
    output TEXT,
    error TEXT,
    started_at REAL NOT NULL,
    finished_at REAL
);
CREATE INDEX IF NOT EXISTS idx_steps_run ON agent_run_steps(run_id, seq);
"""

_initialized = False


async def _ensure_schema(db: aiosqlite.Connection) -> None:
    global _initialized
    if _initialized:
        return
    await db.executescript(_SCHEMA)
    # Idempotent add-column migration — covers a workflow_definitions table
    # that already existed on disk before creation_transcript was added
    # (CREATE TABLE IF NOT EXISTS above only shapes a brand-new table).
    async with db.execute("PRAGMA table_info(workflow_definitions)") as cursor:
        cols = {row[1] for row in await cursor.fetchall()}
    if "creation_transcript" not in cols:
        await db.execute("ALTER TABLE workflow_definitions ADD COLUMN creation_transcript TEXT")
    await db.commit()
    _initialized = True


# ---- workflow_definitions ----

async def create_workflow(
    name: str, description: str | None, graph: dict, trigger: dict, creation_transcript: str | None = None,
) -> str:
    """`creation_transcript` is the Agent Work Chat exchange that produced
    this workflow (user briefs + assistant replies, through the moment
    this tool was called) — set only when a workflow is built via chat;
    a manually-authored graph leaves it None. Agent Vault reads it as the
    starting "Instructions" text when a workflow is starred (see
    storage/agents.py)."""
    workflow_id = str(uuid.uuid4())
    now = time.time()
    async with aiosqlite.connect(DB_PATH) as db:
        await _ensure_schema(db)
        await db.execute(
            "INSERT INTO workflow_definitions (id, name, description, graph, trigger_json, creation_transcript, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (workflow_id, name, description, json.dumps(graph), json.dumps(trigger), creation_transcript, now, now),
        )
        await db.commit()
    return workflow_id


def _row_to_workflow(row: dict) -> dict:
    row = dict(row)
    row["graph"] = json.loads(row.pop("graph"))
    row["trigger"] = json.loads(row.pop("trigger_json"))
    return row


async def get_workflow(workflow_id: str) -> dict | None:
    async with aiosqlite.connect(DB_PATH) as db:
        await _ensure_schema(db)
        db.row_factory = aiosqlite.Row
        async with db.execute(
            "SELECT id, name, description, graph, trigger_json, creation_transcript, created_at, updated_at "
            "FROM workflow_definitions WHERE id = ?",
            (workflow_id,),
        ) as cursor:
            row = await cursor.fetchone()
    return _row_to_workflow(row) if row else None


async def list_workflows() -> list[dict]:
    async with aiosqlite.connect(DB_PATH) as db:
        await _ensure_schema(db)
        db.row_factory = aiosqlite.Row
        async with db.execute(
            "SELECT id, name, description, graph, trigger_json, creation_transcript, created_at, updated_at "
            "FROM workflow_definitions ORDER BY created_at DESC"
        ) as cursor:
            rows = await cursor.fetchall()
    return [_row_to_workflow(r) for r in rows]


async def delete_workflow(workflow_id: str) -> bool:
    """Deletes the workflow definition. Past agent_runs/agent_run_steps for
    it are left alone — a real audit trail of what already happened, not
    something an accidental double-click should be able to erase. Deleting
    the definition is also the entire "cancel its schedule" mechanism —
    there's no separate in-memory job to stop (dispatcher/scheduler.py's
    only registered job is the periodic check_due_workflows() poll itself);
    due_workflows() reads straight from this table, so a deleted workflow
    simply stops being returned by it, on the very next poll.
    Returns whether a row was actually deleted."""
    async with aiosqlite.connect(DB_PATH) as db:
        await _ensure_schema(db)
        cursor = await db.execute("DELETE FROM workflow_definitions WHERE id = ?", (workflow_id,))
        await db.commit()
        return cursor.rowcount > 0


async def update_workflow_trigger(workflow_id: str, trigger: dict) -> None:
    """Advances (or otherwise rewrites) a workflow's trigger — used after a
    scheduled run fires, to roll next_run_at forward by interval_seconds."""
    async with aiosqlite.connect(DB_PATH) as db:
        await _ensure_schema(db)
        await db.execute(
            "UPDATE workflow_definitions SET trigger_json = ?, updated_at = ? WHERE id = ?",
            (json.dumps(trigger), time.time(), workflow_id),
        )
        await db.commit()


async def due_workflows() -> list[dict]:
    """Scheduled workflows whose trigger.next_run_at has already passed —
    same shape as storage/reminders' due_reminders(), polled by an
    externally-pinged endpoint (no in-process scheduler exists anywhere in
    this codebase; see /reminders/check for the precedent)."""
    now = time.time()
    due = []
    for wf in await list_workflows():
        trigger = wf["trigger"]
        if trigger.get("type") != "scheduled":
            continue
        next_run_at = trigger.get("next_run_at")
        if next_run_at is not None and next_run_at <= now:
            due.append(wf)
    return due


# ---- agent_runs ----

async def create_run(workflow_id: str | None, trigger_source: str) -> str:
    run_id = str(uuid.uuid4())
    now = time.time()
    async with aiosqlite.connect(DB_PATH) as db:
        await _ensure_schema(db)
        await db.execute(
            "INSERT INTO agent_runs (id, workflow_id, status, trigger_source, started_at, finished_at, error) "
            "VALUES (?, ?, 'queued', ?, ?, NULL, NULL)",
            (run_id, workflow_id, trigger_source, now),
        )
        await db.commit()
    return run_id


async def update_run_status(run_id: str, status: str, error: str | None = None) -> None:
    terminal = status in ("completed", "failed")
    async with aiosqlite.connect(DB_PATH) as db:
        await _ensure_schema(db)
        if terminal:
            await db.execute(
                "UPDATE agent_runs SET status = ?, error = ?, finished_at = ? WHERE id = ?",
                (status, error, time.time(), run_id),
            )
        else:
            await db.execute(
                "UPDATE agent_runs SET status = ?, error = ? WHERE id = ?",
                (status, error, run_id),
            )
        await db.commit()


async def get_run(run_id: str) -> dict | None:
    async with aiosqlite.connect(DB_PATH) as db:
        await _ensure_schema(db)
        db.row_factory = aiosqlite.Row
        async with db.execute(
            "SELECT id, workflow_id, status, trigger_source, started_at, finished_at, error "
            "FROM agent_runs WHERE id = ?",
            (run_id,),
        ) as cursor:
            row = await cursor.fetchone()
    return dict(row) if row else None


async def list_runs(workflow_id: str | None = None, status: str | None = None, limit: int = 50) -> list[dict]:
    query = "SELECT id, workflow_id, status, trigger_source, started_at, finished_at, error FROM agent_runs WHERE 1=1"
    params: list = []
    if workflow_id is not None:
        query += " AND workflow_id = ?"
        params.append(workflow_id)
    if status is not None:
        query += " AND status = ?"
        params.append(status)
    query += " ORDER BY started_at DESC LIMIT ?"
    params.append(limit)
    async with aiosqlite.connect(DB_PATH) as db:
        await _ensure_schema(db)
        db.row_factory = aiosqlite.Row
        async with db.execute(query, tuple(params)) as cursor:
            rows = await cursor.fetchall()
    return [dict(r) for r in rows]


async def delete_run(run_id: str) -> bool:
    """Deletes one run and its steps (2026-09-04, JuanJo: "I don't
    actually wanna know which runs were done so long ago"). No FK
    cascade defined on agent_run_steps, so both deletes happen here
    explicitly, in the same connection. Returns whether a run row was
    actually deleted — deleting a run's steps when the run itself
    doesn't exist would silently no-op and report success either way,
    which is the wrong signal for the frontend's "not found" case."""
    async with aiosqlite.connect(DB_PATH) as db:
        await _ensure_schema(db)
        await db.execute("DELETE FROM agent_run_steps WHERE run_id = ?", (run_id,))
        cursor = await db.execute("DELETE FROM agent_runs WHERE id = ?", (run_id,))
        await db.commit()
        return cursor.rowcount > 0


async def delete_all_runs(workflow_id: str | None = None) -> int:
    """Bulk clear — every run (optionally scoped to one workflow) and
    all of their steps. Returns how many runs were deleted. Deliberately
    separate from delete_workflow, which already keeps a workflow's past
    runs on purpose as an audit trail when the WORKFLOW itself is
    deleted — this is a distinct, explicit "I don't need this history"
    action on runs alone, workflow untouched."""
    async with aiosqlite.connect(DB_PATH) as db:
        await _ensure_schema(db)
        if workflow_id is not None:
            await db.execute(
                "DELETE FROM agent_run_steps WHERE run_id IN (SELECT id FROM agent_runs WHERE workflow_id = ?)",
                (workflow_id,),
            )
            cursor = await db.execute("DELETE FROM agent_runs WHERE workflow_id = ?", (workflow_id,))
        else:
            await db.execute("DELETE FROM agent_run_steps")
            cursor = await db.execute("DELETE FROM agent_runs")
        await db.commit()
        return cursor.rowcount


# ---- agent_run_steps ----

async def create_step(run_id: str, node_id: str, seq: int) -> str:
    step_id = str(uuid.uuid4())
    async with aiosqlite.connect(DB_PATH) as db:
        await _ensure_schema(db)
        await db.execute(
            "INSERT INTO agent_run_steps (id, run_id, node_id, seq, status, input, output, error, started_at, finished_at) "
            "VALUES (?, ?, ?, ?, 'running', NULL, NULL, NULL, ?, NULL)",
            (step_id, run_id, node_id, seq, time.time()),
        )
        await db.commit()
    return step_id


async def set_step_input(step_id: str, input_data: dict) -> None:
    async with aiosqlite.connect(DB_PATH) as db:
        await _ensure_schema(db)
        await db.execute(
            "UPDATE agent_run_steps SET input = ? WHERE id = ?",
            (json.dumps(input_data), step_id),
        )
        await db.commit()


async def complete_step(step_id: str, status: str, output: str | None = None, error: str | None = None) -> None:
    async with aiosqlite.connect(DB_PATH) as db:
        await _ensure_schema(db)
        await db.execute(
            "UPDATE agent_run_steps SET status = ?, output = ?, error = ?, finished_at = ? WHERE id = ?",
            (status, output, error, time.time(), step_id),
        )
        await db.commit()


async def get_run_steps(run_id: str) -> list[dict]:
    async with aiosqlite.connect(DB_PATH) as db:
        await _ensure_schema(db)
        db.row_factory = aiosqlite.Row
        async with db.execute(
            "SELECT id, run_id, node_id, seq, status, input, output, error, started_at, finished_at "
            "FROM agent_run_steps WHERE run_id = ? ORDER BY seq ASC",
            (run_id,),
        ) as cursor:
            rows = await cursor.fetchall()
    return [dict(r) for r in rows]
