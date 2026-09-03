"""
storage/agents.py

Persistence for the Agent Vault — saved, reusable agent configs (name,
instructions, allowed skills, model tier, output type). Same aiosqlite +
lazy-schema pattern as storage/agent_work.py, own DB file since this is a
genuinely separate concern from workflow runs (a saved agent isn't a
workflow — see AgentWorkGraphEditor.tsx's "Open in canvas" for the
one-way fork that turns one into the other).

`tools` is a list of tool/skill names — plain tools/registry.py names
(e.g. "web_search") or namespaced MCP tool names (e.g.
"mcp__github__list_commits"), the same vocabulary an Agent Work node's
own `tools` field already uses. No new tool-naming scheme.

`output_type` is `null` (ask the user when a run finishes and nothing
was specified — see the Agent Inbox rail button design) or one of
"chat" | "pdf" | "markdown", set explicitly at creation time so this is
a deterministic dispatcher decision, never an LLM guessing whether to
ask.

`workflow_id` is set only for an agent created by starring a real
Agent Work workflow (see /agent/workflows/{id}/star in server.py) — a
reference, not a copy, so running the agent always executes the live
graph. Its "Tools/Nodes" list is derived from that graph at read time
(navi-pwa walks the fetched graph), never stored here. An agent built
through the Vault's own quick-create form instead leaves this null and
carries its own `tools`.
"""

import json
import time
import uuid
from pathlib import Path

import aiosqlite

DB_PATH = Path(__file__).parent.parent / "agents.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS saved_agents (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    instructions TEXT NOT NULL,
    tools_json TEXT NOT NULL,
    model TEXT,
    output_type TEXT,
    workflow_id TEXT,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);
"""

_initialized = False


async def _ensure_schema(db: aiosqlite.Connection) -> None:
    global _initialized
    if _initialized:
        return
    await db.executescript(_SCHEMA)
    async with db.execute("PRAGMA table_info(saved_agents)") as cursor:
        cols = {row[1] for row in await cursor.fetchall()}
    if "workflow_id" not in cols:
        await db.execute("ALTER TABLE saved_agents ADD COLUMN workflow_id TEXT")
    await db.commit()
    _initialized = True


def _row_to_agent(row: dict) -> dict:
    row = dict(row)
    row["tools"] = json.loads(row.pop("tools_json"))
    return row


async def create_agent(
    name: str, instructions: str, tools: list[str], model: str | None, output_type: str | None,
    workflow_id: str | None = None,
) -> str:
    agent_id = str(uuid.uuid4())
    now = time.time()
    async with aiosqlite.connect(DB_PATH) as db:
        await _ensure_schema(db)
        await db.execute(
            "INSERT INTO saved_agents (id, name, instructions, tools_json, model, output_type, workflow_id, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (agent_id, name, instructions, json.dumps(tools), model, output_type, workflow_id, now, now),
        )
        await db.commit()
    return agent_id


async def list_agents() -> list[dict]:
    async with aiosqlite.connect(DB_PATH) as db:
        await _ensure_schema(db)
        db.row_factory = aiosqlite.Row
        async with db.execute(
            "SELECT id, name, instructions, tools_json, model, output_type, workflow_id, created_at, updated_at "
            "FROM saved_agents ORDER BY created_at DESC"
        ) as cursor:
            rows = await cursor.fetchall()
    return [_row_to_agent(dict(r)) for r in rows]


async def get_agent(agent_id: str) -> dict | None:
    async with aiosqlite.connect(DB_PATH) as db:
        await _ensure_schema(db)
        db.row_factory = aiosqlite.Row
        async with db.execute(
            "SELECT id, name, instructions, tools_json, model, output_type, workflow_id, created_at, updated_at "
            "FROM saved_agents WHERE id = ?",
            (agent_id,),
        ) as cursor:
            row = await cursor.fetchone()
    return _row_to_agent(dict(row)) if row else None


async def get_agent_by_workflow_id(workflow_id: str) -> dict | None:
    """Lets the star toggle know whether a given workflow already has a
    Vault entry, without the frontend needing to scan the full list."""
    async with aiosqlite.connect(DB_PATH) as db:
        await _ensure_schema(db)
        db.row_factory = aiosqlite.Row
        async with db.execute(
            "SELECT id, name, instructions, tools_json, model, output_type, workflow_id, created_at, updated_at "
            "FROM saved_agents WHERE workflow_id = ?",
            (workflow_id,),
        ) as cursor:
            row = await cursor.fetchone()
    return _row_to_agent(dict(row)) if row else None


async def update_agent(agent_id: str, name: str, instructions: str, tools: list[str], model: str | None, output_type: str | None) -> None:
    async with aiosqlite.connect(DB_PATH) as db:
        await _ensure_schema(db)
        await db.execute(
            "UPDATE saved_agents SET name = ?, instructions = ?, tools_json = ?, model = ?, output_type = ?, updated_at = ? WHERE id = ?",
            (name, instructions, json.dumps(tools), model, output_type, time.time(), agent_id),
        )
        await db.commit()


async def delete_agent(agent_id: str) -> bool:
    async with aiosqlite.connect(DB_PATH) as db:
        await _ensure_schema(db)
        cursor = await db.execute("DELETE FROM saved_agents WHERE id = ?", (agent_id,))
        await db.commit()
        return cursor.rowcount > 0
