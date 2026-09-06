# Handoff — NAVI, 2026-09-07 — next phase: Agent Work graphs & automation

Written to start a fresh chat because this one's context kept getting compacted.
Read this instead of re-deriving context. Two repos:
- Backend: `JuanJoRamBach/NAVI`, `C:\Users\juanj\Proyectos IA\NAVI`, branch `main`,
  deployed to AWS Lightsail at `api.getnavi.online`.
- Frontend: `navi-pwa` (React/TS), `C:\Users\juanj\Proyectos IA\navi-pwa`, deployed
  via GitHub Pages at `getnavi.online`, plus a Tauri desktop wrapper built earlier.

**Supersedes the 2026-09-01 HANDOFF.md** (still in git history if needed). This one's
narrowly scoped: JuanJo's explicit next-phase target is making **Agent Work's
workflow graphs and automation genuinely ready** — not a repeat of the whole
project's history. `IDEAS.md` and `how_to_handle_context.md` (both gitignored,
auto-loaded via `CLAUDE.md`) remain the authoritative running indexes for
everything outside this scope; this file is the on-ramp for this one thread.

## Where Agent Work actually stands right now (verified by reading the real code, not memory)

Backend: `dispatcher/agent_work.py` (899 lines) + `storage/agent_work.py`
(`workflow_definitions`/`agent_runs`/`agent_run_steps`). Frontend:
`navi-pwa/src/agentWorkNodeKinds.tsx` (153 lines, the node palette) +
`AgentWorkGraphEditor.tsx`/`AgentWorkGraphNode.tsx`/`agentWorkGraphConvert.ts`
(the `@xyflow/react` canvas). Both read in full this session.

- **Real, working today**: topological execution (Kahn's algorithm, cycle/dangling-
  edge detection), fan-out groups (sub-flows over a literal `items` list), a
  `choose_path` branching/pruning node, shared-state refs (`{{state.<node_id>}}`),
  a linear provider-fallback loop (`_call_for_node`) as the only existing retry
  mechanism, background-thread execution mirroring `/research`'s async pattern.
  The canvas is real and wired end-to-end (Agent Vault star flow creates graph
  nodes from chat, saves them as runnable agents) — not a placeholder.
- **Trigger types that exist: `manual` and `scheduled` only.** `scheduled` is
  externally polled (`GET /agent/workflows/due`, same shape as `/reminders/check`
  — no in-process scheduler anywhere in this codebase; something outside the repo
  must be pinging that route on a cron, unconfirmed whether it actually is —
  flagged as a real open question in `IDEAS.md` already).
- **Node palette, frontend (11 kinds)**: `writeText`, `generateAi`, `searchWeb`,
  `readPage`, `apiCall`, `sendMessage`, `sendMail`, `saveFile`, `choosePath`,
  `input`, `output`. Several are visual/definitional only — not fully wired to a
  backend handler (the file's own header comment says so).
- **Node handlers, backend**: `send_to_telegram`, `web_search`, `fetch_page`,
  `save_note`, `send_email`, `input`, `output`, `choose_path`, `text`, plus a
  generic multi-tool fallback. **Naming doesn't line up 1:1 with the frontend
  palette** (e.g. `apiCall`, `sendMail` vs `send_email`, `saveFile` vs
  `save_note`, `writeText` vs `text`) — this mismatch needs reconciling as part
  of "putting meat on the nodes," not assumed to already be wired.
- **Reliability gaps, concrete, from reading `_execute_run` and `_call_for_node`**:
  no per-node retry/backoff for deterministic action nodes (only LLM-backed nodes
  get the provider-fallback retry); no per-node timeout (a hanging `apiCall`/
  `fetch_page` can stall a run indefinitely); **any single node failure kills the
  entire run** (`_execute_run`'s `except WorkflowError: return` is unconditional
  — no continue-on-error, no partial resume); no circuit breaker / step cap at
  the workflow level (only `run_tool_loop`'s own `MAX_TOOL_ITERATIONS` guards a
  single node's internal tool-calling loop).

## This session's work: research + a full written plan (no code changed in Agent Work)

JuanJo's ask, verbatim shape: (1) a written accounts/audit-trail plan, plan-only,
no real provider accounts created yet; (2) "we need more reliability, and work on
the proper agentic work"; (3) his brother's question — can a workflow detect a
call/200 from an always-connected API and turn that into a flow; (4) build the
missing "pin"/webhook-receiver trigger node; (5) audit Agent Work's node system
broadly for reliability and "meat"; (6) research what LangChain/LangGraph do for
this and what `@xyflow/react` can support.

Everything below was **proposed in chat, not yet implemented** — standing rule
this session followed throughout: propose code changes and wait for go-ahead,
even small ones.

### 1. Accounts / audit trail — the plan (deferred, no provider accounts touched)

New tables: `accounts`, `users` (magic-link or OAuth login, no passwords —
avoids inventing a password-reset/hashing/breach-monitoring flow NAVI has
nothing for), `roles` (fixed v1 enum: owner/admin/member/viewer — gates exactly
the write/destructive tiers `dispatcher/mcp_client.py`'s trust model already
distinguishes), `sessions`, `audit_log` (append-only, one row per mutating/
side-effecting action — workflow create/run/delete, sent messages, provider-
connection changes, destructive MCP calls; plain chat replies don't get rows).

Auth: demote `NAVI_API_KEY` from "the only auth" to a service key for machine
callers only (webhooks, the Tauri app's bootstrap); humans get a real session
cookie from login.

Key point that unblocked this: **none of steps 1–4 below need any new Groq/
LLM7/Cloudflare accounts** — they run fine against NAVI's existing shared
provider pool. BYOK (each account bringing its own provider keys) is a real
step but explicitly last, built only once a second paying account needs it —
that's the part JuanJo didn't want to spend days provisioning for speculatively.

Build order when picked up: (1) `users`/`accounts`/`sessions` + magic-link login
+ PWA login screen → (2) `audit_log` table + one `record_audit(...)` helper
called from the existing write/destructive routes/tools → (3) role checks on
those same routes → (4) an audit viewer (likely just the existing Activity tab,
filtered) → (5) BYOK, deferred.

### 2. Webhook trigger — answers the brother's question, unblocks "pin" node

Yes, directly buildable, same pattern n8n/Zapier/Make all use (checked via
research, not assumed):

- New trigger type: `{"type": "webhook", "token": <random>}` on a workflow,
  generated when the trigger is added → stable URL
  `api.getnavi.online/agent/webhooks/{token}`.
- New route, **outside** `_require_api_key`'s gate (an external caller can't
  send NAVI's own header) — protected by the token itself being unguessable in
  the path, same trust model already used for the Telegram webhook secret
  (`TELEGRAM_WEBHOOK_SECRET`). Optional HMAC verification for services that sign
  payloads (Stripe, GitHub).
- New canvas node kind, **"Webhook Trigger"** (the "pin" node): graph-entry node,
  no input handle, only output — `@xyflow/react` supports this natively (a node
  with only a `source` Handle, no `target` Handle rendered), no library gap.
  Its output is the incoming payload. **Cheap to wire**: the webhook route just
  seeds `outputs["<trigger_node_id>"]` with the parsed payload before the run
  starts; every downstream node already reads prior context from `outputs` via
  the existing mechanism the `input` node uses today — almost no executor change.

### 3. Reliability — LangGraph's model, translated to concrete NAVI changes

Researched LangGraph's actual fault-tolerance primitives (RetryPolicy,
TimeoutPolicy, error_handler — real, current, from LangChain's own blog) and
checked each against `agent_work.py`'s real gaps listed above:

- **Retry with backoff for deterministic action nodes** (`_run_send_telegram_node`
  etc.) — wrap the existing node call in `_execute_run` with a small N-attempt/
  backoff loop. No new dependency.
- **Per-node timeout** — wrap the existing `asyncio.to_thread(_run_node, ...)`
  call in `asyncio.wait_for`.
- **Continue-on-error / error routing** — an optional `continue_on_error` flag on
  a node (skip to normal successors with a placeholder failure note instead of
  aborting the whole run), plus an optional "error" edge label reusing the exact
  label-matching mechanism `choose_path` edges already have.
- **Circuit breaker / step cap** — needed before any future loop-style node
  ships (see below); nothing analogous exists at the workflow level today.

Suggested build order for reliability specifically: retry + timeout first (zero
new node kinds, protects every existing workflow immediately) → continue-on-
error/error edges.

### 4. Node system audit — what's genuinely missing, prioritized

Beyond the naming reconciliation noted above:

- **Delay/Wait node** — nothing lets a workflow pause N seconds/until-a-time
  mid-run.
- **Dynamic loop/iterate node** — fan-out groups only iterate a literal,
  pre-set `items` list; nothing iterates over an array a *prior step* produced
  at run time. Since `_resolve_state_refs` and fan-out groups both already
  exist independently, letting a group's `items` be a `{{state.node_id}}`
  reference resolved to a JSON array at run time is a contained change, not a
  new subsystem.
- **Respond-to-webhook node** — once trigger #2 exists, a synchronous reply
  (Stripe-style) isn't expressible yet. v1 can just always ack 200 immediately
  and run async (matches `/research`'s own pattern) — build a real Respond node
  only once a specific integration needs a computed body back.
- **Merge/wait-for-all node** — parallel branches converging on one downstream
  node rely on topological-sort ordering today, not an explicit join. Low
  priority until branching graphs get common enough that ordering ambiguity
  actually bites someone.
- **Human-in-the-loop / approval node** — LangGraph has first-class pause/
  resume; Agent Work has no equivalent. Build once a real workflow needs it.

**React Flow/xyflow is not the bottleneck anywhere in this list** — checked
directly: distinct trigger-node shapes, live per-node run-status coloring,
edge validation, labeled/conditional edges (already used by `choose_path`) are
all natively supported. Every gap above is backend logic in
`dispatcher/agent_work.py` plus a new `agentWorkNodeKinds.tsx` entry — the
canvas plumbing to display new nodes already scales fine.

**Recommended overall order for the next phase**: retry + timeout → Webhook
Trigger node + route → continue-on-error/error edges → node-name reconciliation
(frontend↔backend) → Delay node → dynamic loop → Respond/Approval nodes only
once something concrete needs them. Accounts/audit-trail stays parked until
JuanJo raises it again — it's fully specced above, just not urgent for this
phase.

Sources checked this session: LangChain's own "Fault Tolerance in LangGraph"
blog post, a LangGraph error-handling/retry writeup, two n8n webhook-node
guides.

## Separately open, not part of this phase but don't lose track

- **Uncommitted, working, in-progress right now**: `server.py` and
  `storage/filen.py` have uncommitted local changes — `file_download_url` was
  moved from a private `_file_download_url` in `server.py` into
  `storage/filen.py` as a public function, specifically so `dispatcher/chat.py`'s
  new `create_document` tool can build a download link too (dispatcher modules
  can't import back from `server.py`). This refactor step is done; the rest of
  `create_document` (tool schema, dispatch branch, transcript-extraction logic,
  `NORMAL_CHAT.md` registration) was still in progress before this handoff was
  written — check `dispatcher/chat.py` for how far it actually got before
  continuing it, don't assume it's finished.
- Several pending NAVI backend deploys to Lightsail may still be queued from
  before this — confirm current deployed state before assuming local `main`
  matches production.
- `jobs/model_ranking.py`'s tier-fit fix (from the 2026-09-01 handoff) needed
  one more real rerun on AWS to confirm it actually changes model picks — status
  unconfirmed as of this handoff, check before assuming it's done.

## Process notes for whoever picks this up

- **Ask before code changes** — propose the edit, wait for go-ahead, even small/
  safe ones. Followed throughout this session (this handoff is itself the
  result of a planning-only pass, zero Agent Work code touched).
- **Verify against real code/data, not memory or docs** — every claim above
  about what exists/doesn't in Agent Work came from actually reading
  `dispatcher/agent_work.py` and `agentWorkNodeKinds.tsx` in full this session,
  not from recalling an earlier description of them.
- This chat was compacting too often to keep going — that's the whole reason
  this file exists. Start the next session by reading this file, then
  `IDEAS.md`'s "Agent Work backend" section for any updates since, before
  proposing which piece of the build order above to start with.
