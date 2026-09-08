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

`trigger` is `{"type": "manual"}`, `{"type": "scheduled", "interval_seconds",
"next_run_at"}` — epoch seconds, no cron-expression parsing (no dependency
for it anywhere in this codebase yet, and dispatcher/reminders.py already
sets the precedent of plain fire_at timestamps over cron syntax) — or
`{"type": "webhook", "token"}` (2026-09-07): a random, unguessable token
that IS the credential (see server.py's /agent/webhooks/{token}), same
trust model as TELEGRAM_WEBHOOK_SECRET, no signature/HMAC scheme built
yet since nothing has needed one so far.

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
    updated_at REAL NOT NULL,
    deleted_at REAL
);
CREATE TABLE IF NOT EXISTS workflow_definition_versions (
    id TEXT PRIMARY KEY,
    workflow_id TEXT NOT NULL,
    version_number INTEGER NOT NULL,
    name TEXT NOT NULL,
    description TEXT,
    graph TEXT NOT NULL,
    trigger_json TEXT NOT NULL,
    edited_by TEXT,
    created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_versions_workflow ON workflow_definition_versions(workflow_id, version_number);
CREATE TABLE IF NOT EXISTS agent_runs (
    id TEXT PRIMARY KEY,
    workflow_id TEXT,
    status TEXT NOT NULL,
    trigger_source TEXT NOT NULL,
    started_at REAL NOT NULL,
    finished_at REAL,
    error TEXT,
    graph_snapshot TEXT
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
    # Same idempotent pattern for deleted_at (2026-09-08) — soft delete.
    # NULL = active. A real timestamp means "hidden from the normal
    # Workflows list (list_workflows filters it out by default) but not
    # actually gone" — see delete_workflow's own docstring for the full
    # reasoning (recoverable via restore_workflow, real permanent erasure
    # is the separate, more severe purge_workflow).
    if "deleted_at" not in cols:
        await db.execute("ALTER TABLE workflow_definitions ADD COLUMN deleted_at REAL")
    # Same idempotent pattern for agent_runs.graph_snapshot (2026-09-08) — a
    # run's own frozen copy of the graph it actually started with, so a
    # workflow edit made after a run starts can never retroactively change
    # what that run executes against (the version-skew failure mode Azure
    # Durable Functions' own docs warn about for exactly this shape of
    # problem: an in-flight orchestration replaying against source that
    # changed underneath it). Not read by _execute_run today — it already
    # gets `graph` as a plain in-memory value at start, which is already
    # safe for a run that's actively executing start-to-finish. This
    # column is what makes a FUTURE pause/resume (the human-in-the-loop
    # approval node) safe too, once a run can be suspended for real time
    # and needs its own durable snapshot to resume against instead of
    # re-fetching the (by then possibly-edited) live definition.
    async with db.execute("PRAGMA table_info(agent_runs)") as cursor:
        run_cols = {row[1] for row in await cursor.fetchall()}
    if "graph_snapshot" not in run_cols:
        await db.execute("ALTER TABLE agent_runs ADD COLUMN graph_snapshot TEXT")
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


async def update_workflow(
    workflow_id: str, name: str, description: str | None, graph: dict, trigger: dict, edited_by: str | None = None,
) -> bool:
    """Real update-in-place (2026-09-07) — until now the only way to
    change a saved workflow was create_workflow, which always makes a
    NEW row. That meant editing an existing workflow (e.g. to add a
    {{state...}} reference discovered via a real test run) silently
    forked a duplicate instead of changing the one you meant to — worse
    for a webhook-triggered workflow specifically, since the duplicate
    gets its own new token, orphaning whatever external service already
    has the original URL configured. Returns whether a row actually
    existed to update, same "did this really happen" convention
    delete_workflow below already uses.

    Version history (2026-09-08): the row's state is archived into
    workflow_definition_versions BEFORE being overwritten — every real
    "Save Edits" becomes a real, permanent, browsable version, for the
    same audit/legitimacy reasoning delete_workflow's soft-delete below
    follows. `edited_by` is real, plumbed all the way through, but stays
    None/unpopulated until real per-user accounts exist — same honest
    gap as everywhere else in this codebase that would otherwise need to
    fake an identity. revert_workflow_to_version below is just this same
    function called with an old version's fields — which means reverting
    ALSO archives whatever was live right before the revert as its own
    new version; reverting away from a version never destroys it, same
    principle `git revert` follows (a new commit, not erased history)."""
    async with aiosqlite.connect(DB_PATH) as db:
        await _ensure_schema(db)
        async with db.execute(
            "SELECT name, description, graph, trigger_json FROM workflow_definitions WHERE id = ?",
            (workflow_id,),
        ) as cursor:
            current = await cursor.fetchone()
        if current is None:
            return False
        async with db.execute(
            "SELECT COALESCE(MAX(version_number), 0) FROM workflow_definition_versions WHERE workflow_id = ?",
            (workflow_id,),
        ) as cursor:
            (max_version,) = await cursor.fetchone()
        await db.execute(
            "INSERT INTO workflow_definition_versions "
            "(id, workflow_id, version_number, name, description, graph, trigger_json, edited_by, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (str(uuid.uuid4()), workflow_id, max_version + 1, current[0], current[1], current[2], current[3], edited_by, time.time()),
        )
        cursor = await db.execute(
            "UPDATE workflow_definitions SET name = ?, description = ?, graph = ?, trigger_json = ?, updated_at = ? WHERE id = ?",
            (name, description, json.dumps(graph), json.dumps(trigger), time.time(), workflow_id),
        )
        await db.commit()
        return cursor.rowcount > 0


def _row_to_version(row: dict) -> dict:
    row = dict(row)
    row["graph"] = json.loads(row.pop("graph"))
    row["trigger"] = json.loads(row.pop("trigger_json"))
    return row


async def list_workflow_versions(workflow_id: str) -> list[dict]:
    async with aiosqlite.connect(DB_PATH) as db:
        await _ensure_schema(db)
        db.row_factory = aiosqlite.Row
        async with db.execute(
            "SELECT id, workflow_id, version_number, name, description, graph, trigger_json, edited_by, created_at "
            "FROM workflow_definition_versions WHERE workflow_id = ? ORDER BY version_number DESC",
            (workflow_id,),
        ) as cursor:
            rows = await cursor.fetchall()
    return [_row_to_version(dict(r)) for r in rows]


async def revert_workflow_to_version(workflow_id: str, version_id: str, edited_by: str | None = None) -> bool:
    """Restores an old version's name/description/graph as the new live
    definition — deliberately NOT its trigger. A revert must never
    silently change a workflow's webhook token or schedule out from
    under whatever's currently configured, same "never regenerate a
    token a real external service already has" rule Save Edits itself
    already follows for ordinary edits. Delegates to update_workflow, so
    this also gets the same automatic version-archiving (reverting away
    from a version doesn't destroy it) for free."""
    async with aiosqlite.connect(DB_PATH) as db:
        await _ensure_schema(db)
        db.row_factory = aiosqlite.Row
        async with db.execute(
            "SELECT name, description, graph FROM workflow_definition_versions WHERE id = ? AND workflow_id = ?",
            (version_id, workflow_id),
        ) as cursor:
            version_row = await cursor.fetchone()
        if version_row is None:
            return False
        async with db.execute("SELECT trigger_json FROM workflow_definitions WHERE id = ?", (workflow_id,)) as cursor:
            current_row = await cursor.fetchone()
        if current_row is None:
            return False
    return await update_workflow(
        workflow_id, version_row["name"], version_row["description"],
        json.loads(version_row["graph"]), json.loads(current_row["trigger_json"]), edited_by,
    )


def _row_to_workflow(row: dict) -> dict:
    row = dict(row)
    row["graph"] = json.loads(row.pop("graph"))
    row["trigger"] = json.loads(row.pop("trigger_json"))
    return row


async def get_workflow(workflow_id: str) -> dict | None:
    # Deliberately unfiltered by deleted_at — a direct by-ID lookup is
    # used internally (starting a run, resolving a webhook token via
    # list_workflows below, an admin viewing one specific deleted
    # workflow later) where "does this row exist at all" is the right
    # question, not "is it in the normal active list." Filtering belongs
    # specifically to list_workflows, the general browsing view.
    async with aiosqlite.connect(DB_PATH) as db:
        await _ensure_schema(db)
        db.row_factory = aiosqlite.Row
        async with db.execute(
            "SELECT id, name, description, graph, trigger_json, creation_transcript, created_at, updated_at, deleted_at "
            "FROM workflow_definitions WHERE id = ?",
            (workflow_id,),
        ) as cursor:
            row = await cursor.fetchone()
    return _row_to_workflow(row) if row else None


async def list_workflows(include_deleted: bool = False) -> list[dict]:
    query = (
        "SELECT id, name, description, graph, trigger_json, creation_transcript, created_at, updated_at, deleted_at "
        "FROM workflow_definitions"
    )
    if not include_deleted:
        query += " WHERE deleted_at IS NULL"
    query += " ORDER BY created_at DESC"
    async with aiosqlite.connect(DB_PATH) as db:
        await _ensure_schema(db)
        db.row_factory = aiosqlite.Row
        async with db.execute(query) as cursor:
            rows = await cursor.fetchall()
    return [_row_to_workflow(r) for r in rows]


async def delete_workflow(workflow_id: str) -> bool:
    """Soft delete (2026-09-08) — sets deleted_at instead of actually
    removing the row. The workflow disappears from the normal Workflows
    list (list_workflows filters it out by default) but the row and all
    its history (runs, versions) stay fully intact — recoverable via
    restore_workflow, or visible to an admin/owner-tier view via
    list_workflows(include_deleted=True) once real roles exist (today,
    that view sits behind the same single shared API key everyone has —
    an honest, known stopgap, not real access control yet). Real,
    permanent erasure is the separate, deliberately more severe
    purge_workflow below.

    Still the entire "cancel its schedule/webhook" mechanism, same as
    the old hard-delete was: due_workflows() and
    get_workflow_by_webhook_token() both read through list_workflows()'s
    default (filtered) behavior, so a soft-deleted workflow correctly
    stops firing on the very next poll — same real-world effect as
    before, just reversible now. Returns whether an active row was
    actually soft-deleted (a no-op, returning False, if it was already
    deleted or never existed)."""
    async with aiosqlite.connect(DB_PATH) as db:
        await _ensure_schema(db)
        cursor = await db.execute(
            "UPDATE workflow_definitions SET deleted_at = ? WHERE id = ? AND deleted_at IS NULL",
            (time.time(), workflow_id),
        )
        await db.commit()
        return cursor.rowcount > 0


async def restore_workflow(workflow_id: str) -> bool:
    """Undoes delete_workflow — clears deleted_at, making the workflow
    reappear in the normal list and, if it has a schedule/webhook
    trigger, eligible to fire again on the next poll. Returns whether a
    soft-deleted row was actually restored."""
    async with aiosqlite.connect(DB_PATH) as db:
        await _ensure_schema(db)
        cursor = await db.execute(
            "UPDATE workflow_definitions SET deleted_at = NULL WHERE id = ? AND deleted_at IS NOT NULL",
            (workflow_id,),
        )
        await db.commit()
        return cursor.rowcount > 0


async def purge_workflow(workflow_id: str) -> bool:
    """The real, permanent erasure — unlike delete_workflow (soft,
    reversible, audit-preserving), this actually removes the row AND its
    full history (runs, steps, versions), since the whole point of a
    purge is complete removal, not "hidden but still auditable." A
    deliberately separate, more severe action — see IDEAS.md's
    permission-catalog thread: `workflows.delete` (soft) vs
    `workflows.purge` (this), the latter meant to be grantable to
    Owner-tier only once real roles exist. Returns whether a row was
    actually removed."""
    async with aiosqlite.connect(DB_PATH) as db:
        await _ensure_schema(db)
        await db.execute(
            "DELETE FROM agent_run_steps WHERE run_id IN (SELECT id FROM agent_runs WHERE workflow_id = ?)",
            (workflow_id,),
        )
        await db.execute("DELETE FROM agent_runs WHERE workflow_id = ?", (workflow_id,))
        await db.execute("DELETE FROM workflow_definition_versions WHERE workflow_id = ?", (workflow_id,))
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


async def get_workflow_by_webhook_token(token: str) -> dict | None:
    """Webhook trigger lookup (2026-09-07) — `trigger` is a flat JSON blob
    (see module docstring), not a queryable column, so this scans
    workflow_definitions the same way due_workflows() below already scans
    for `scheduled` triggers. Fine at Agent Work's real scale; revisit
    only if workflow count ever grows enough for that to matter."""
    for wf in await list_workflows():
        trigger = wf["trigger"]
        if trigger.get("type") == "webhook" and trigger.get("token") == token:
            return wf
    return None


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

async def create_run(workflow_id: str | None, trigger_source: str, graph: dict | None = None) -> str:
    """`graph` (2026-09-08) is this run's own frozen snapshot, stored once
    at creation and never touched again — see _ensure_schema's own
    docstring on graph_snapshot for why. Optional (defaults to None/absent)
    only so any caller that genuinely has no graph handy at creation time
    doesn't break; every real caller in dispatcher/agent_work.py passes it."""
    run_id = str(uuid.uuid4())
    now = time.time()
    async with aiosqlite.connect(DB_PATH) as db:
        await _ensure_schema(db)
        await db.execute(
            "INSERT INTO agent_runs (id, workflow_id, status, trigger_source, started_at, finished_at, error, graph_snapshot) "
            "VALUES (?, ?, 'queued', ?, ?, NULL, NULL, ?)",
            (run_id, workflow_id, trigger_source, now, json.dumps(graph) if graph is not None else None),
        )
        await db.commit()
    return run_id


async def update_run_status(run_id: str, status: str, error: str | None = None) -> None:
    # "cancelled" (2026-09-08) is a real terminal status alongside
    # completed/failed — a run that stopped because a user asked it to,
    # not because it broke; finished_at gets set the same way so it stops
    # showing as still-running.
    terminal = status in ("completed", "failed", "cancelled")
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


async def get_latest_node_output(workflow_id: str, node_id: str) -> str | None:
    """The most recent real output a given node produced, across any past
    run of this workflow (2026-09-07) — what the frontend's reference
    picker uses to let someone browse an upstream node's actual JSON
    shape (a webhook payload, most commonly) instead of guessing field
    names blind, the same "test it once, then map real fields" flow
    Zapier's own product requires. Read-only convenience for the UI;
    _resolve_state_refs (dispatcher/agent_work.py) never reads history —
    it only ever reads the live run's own in-memory outputs dict."""
    async with aiosqlite.connect(DB_PATH) as db:
        await _ensure_schema(db)
        db.row_factory = aiosqlite.Row
        async with db.execute(
            "SELECT s.output FROM agent_run_steps s "
            "JOIN agent_runs r ON r.id = s.run_id "
            "WHERE r.workflow_id = ? AND s.node_id = ? AND s.status = 'completed' "
            "ORDER BY s.started_at DESC LIMIT 1",
            (workflow_id, node_id),
        ) as cursor:
            row = await cursor.fetchone()
    return row["output"] if row else None


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
