# Agent Work reliability/audit features — change log

Committed to the repo (not gitignored) on purpose, unlike `IDEAS.md`'s
scratch notes: this documents *shipped* reliability/audit behavior, kept
as a durable, versioned record — the actual audit trail the two features
below exist to build toward. Update this file, don't replace it, when
either feature is extended.

## 1. Run isolation + safe cancellation (2026-09-08, NAVI `f3d1748`, navi-ui `703007f`)

**Problem**: editing a workflow definition while a run of it was still
executing meant the running graph could shift under the running instance
mid-flight (a node the run hadn't reached yet could vanish, or get
reconnected differently), since every node execution read the *live*
`workflow_definitions.graph` for the whole duration of a run.

**What shipped**:
- `agent_runs.graph_snapshot` — the exact graph a run started with,
  frozen at `create_run()` and never re-read from the live definition
  again. A later edit to the workflow can never retroactively change a
  run already in flight.
- Safe, cooperative cancellation — `request_run_cancellation(run_id)`
  flags a run; `dispatcher/agent_work.py`'s execution loop checks the
  flag once per node boundary (Kahn's-algorithm topological walk), never
  mid-node. A currently-running node (e.g. a Delay node's own wait)
  always finishes before a cancellation takes effect. New terminal
  status: `"cancelled"`, distinct from `"failed"`.
- `POST /agent/runs/{run_id}/cancel` (server.py).
- Frontend (`AgentWorkRunHistory.tsx`): a cancel button on active runs,
  an inline warning banner when a run's workflow was edited while that
  run was still in flight (compares the run's `started_at` against the
  workflow's `updated_at`), and 3s polling while any run is active so
  status changes surface without a manual refresh.

**Live-verified** (no mocks): a 3-node, 2s-delay-each graph, cancelled
~1s into node 1 — node 1 finished, nodes 2/3 never started, run reached
`"cancelled"` in ~1.1s. Confirmed a plain, never-cancelled run still
completes normally (regression check on the new cancel-check-per-loop).

## 2. Workflow version history + soft-delete/restore/purge (2026-09-08, NAVI `49da20b`, navi-ui `7dbc3f9`)

**Problem**: "Save Edits" (added 2026-09-07, `0611cdb`) let a workflow be
updated in place — real fix for the fork-on-every-edit bug it replaced,
but it meant an edit silently and irreversibly overwrote the previous
graph, with no way to recover it. Likewise, deleting a workflow was a
real, permanent `DELETE` — one accidental click, no way back.

**Design principle** (JuanJo, this session): borrowed directly from
`git revert`'s "a new commit, not erased history" semantics, and from
Azure Durable Functions' replay-safety discipline researched earlier
this session — never let a piece of history be overwritten without
archiving it first.

**What shipped, `storage/agent_work.py` + `server.py`**:
- New `workflow_definition_versions` table (id, workflow_id,
  version_number, name, description, graph, trigger_json, edited_by,
  created_at). Every `update_workflow()` call archives the row's
  *pre-update* content into this table before applying the change — so
  a version holds what was true immediately before, not a copy of the
  update itself.
- `edited_by` — a real audit field, but there are no real user accounts
  anywhere in NAVI yet (one shared `NAVI_API_KEY` for the whole API).
  Populated only if the caller sends one (server.py's PUT/revert routes
  accept an optional `edited_by` in the payload); stays `NULL`
  otherwise. Honest gap, not faked — the schema is ready for real
  identity the moment accounts exist, nothing more.
- `list_workflow_versions(workflow_id)` /
  `GET /agent/workflows/{id}/versions` — most-recent-first.
- `revert_workflow_to_version(workflow_id, version_id, edited_by)` /
  `POST /agent/workflows/{id}/versions/{version_id}/revert` — restores
  an old version's name/description/graph as the new live state.
  Deliberately does **not** touch the workflow's *current* trigger (a
  webhook's token/URL, or a schedule) — reverting a graph edit can never
  silently orphan an already-configured external integration. Built on
  top of `update_workflow()`, so a revert itself archives what it
  overwrites — never a dead end, always reversible again.
- **Old versions are not runnable** — there is no route that accepts a
  `version_id` to execute; only the live workflow (`workflow_id`) can be
  run. This is structural, not a permission check that could be
  bypassed: a version row has no path into `start_workflow_run` at all.
- Soft-delete: `workflow_definitions.deleted_at` (nullable timestamp).
  `delete_workflow()` now sets it instead of removing the row;
  `list_workflows()` filters `WHERE deleted_at IS NULL` by default (new
  `include_deleted: bool` param, `GET /agent/workflows?include_deleted=true`).
  `get_workflow()` stays a direct, unfiltered by-id lookup (used
  internally — starting a run, resolving a webhook token) since "does
  this row exist" is the right question there, not "is it in the active
  list."
- `restore_workflow()` / `POST /agent/workflows/{id}/restore` — clears
  `deleted_at`, un-hides it, re-arms its schedule/webhook.
- `purge_workflow()` / `DELETE /agent/workflows/{id}/purge` — the real,
  permanent removal. Cascades to `agent_run_steps`, `agent_runs`,
  `workflow_definition_versions`, then the workflow row itself. Also
  best-effort removes any Agent Vault (`saved_agents`) entry pointing at
  it — a separate DB file (`agents.db`), so this can't be one atomic
  transaction with the purge; a leftover dangling reference on failure
  is harmless (the UI already treats a starred agent's instructions as
  standalone text) rather than a correctness bug.
- A soft-deleted workflow correctly stops firing on schedule/webhook —
  `due_workflows()` and `get_workflow_by_webhook_token()` both read
  through `list_workflows()`'s default (filtered) behavior, same real
  effect the old hard-delete had, just reversible now.
- **Deliberately deferred, not built**: who is *allowed* to see deleted
  workflows or purge them. Today `include_deleted=true` and `/purge` sit
  behind the same single shared `NAVI_API_KEY` everyone already has —
  not real access control. JuanJo's explicit call: "save deleted agents
  for now, then we make the permissions" — build the mechanism now,
  gate it by role once real multi-user accounts/roles exist (same
  sequencing already applied to the human-in-the-loop approval node's
  role-gating, see `IDEAS.md`).

**What shipped, frontend (`navi-pwa`)**:
- `SaveEditsConfirmDialog` (`AgentWorkGraphEditor.tsx`) — "Save Edits"
  on an already-loaded workflow now asks for confirmation before
  overwriting, mirroring the existing delete-confirmation dialog's
  reasoning ("a stray click shouldn't be able to do this"). Only shown
  on an edit-save; a first save has nothing to overwrite yet.
- `DeleteConfirmDialog`'s copy corrected — no longer says "permanently
  deleted" / "can't be undone" (both now false); says it disappears from
  the list but history/runs stay intact, since the delete itself is now
  a soft-delete.
- **Not built yet**: any UI surface for version history itself (browsing
  past versions, triggering a revert) or for restore/purge (correctly —
  those are the admin/owner-tier surface, deferred with the permissions
  work above). The backend routes exist and are live-verified; nothing
  on the canvas or in the Workflows list calls them yet.

**Live-verified** (no mocks): a real SQLite round-trip script
(`test_versions_and_soft_delete.py`) covering archive-on-update,
multi-version ordering, revert restoring content while preserving the
current trigger and itself archiving what it overwrote, soft-delete
hiding a workflow from the default list and from webhook resolution
while `get_workflow` still finds it directly, restore reversing all of
that, and purge actually cascading (workflow row, run, and both gone
after). Then re-verified the same behavior through real HTTP calls
against a local `uvicorn` server (create → update → list versions →
delete → list filtered/unfiltered → restore → purge → confirm 404),
using the real routes and the real `NAVI_API_KEY` gate, not a stubbed
client.

## Open, not yet decided

- Whether/when to build the version-history + revert UI, and the
  admin/owner "see + purge deleted workflows" surface — both wait on
  real accounts/roles, per JuanJo's explicit sequencing call above.
- Whether `edited_by` should default to *something* identifiable (e.g.
  a per-browser random id) before real accounts exist, or stay `NULL`
  until then — not raised or decided this session, flagged here so it
  isn't silently assumed either way later.
