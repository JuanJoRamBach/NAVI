"""
storage/usage.py

Persistence for the "Usage counters" panel (navi-pwa) — replaces the old
hardcoded USAGE_COUNTERS mock with real, server-side, per-provider numbers.

Sync sqlite3, not aiosqlite like storage/agent_work.py/conversations.py:
the write path here is providers/base.py's Provider.chat() and each
transport's _do_chat(), which are plain synchronous functions (real
`requests` calls, no asyncio anywhere in that layer) — forcing async I/O
into that call chain would mean an asyncio.run() per chat request for no
benefit. Async is the right tool where the rest of the module already is;
sync is the right tool here.

Schema, one row per (provider, model, day_utc) — "day_utc" is the ISO
date (YYYY-MM-DD) the request landed in, computed from UTC time so the
row boundary IS the reset boundary for providers with a real UTC-midnight
reset (Cloudflare, confirmed via developers.cloudflare.com; OpenRouter,
assumed since unconfirmed from a primary source — see the live /api/v1/key
fetch in providers/openrouter.py, which sidesteps needing this table's
day-key for OpenRouter entirely by asking OpenRouter directly instead):
    usage_daily(provider, model, day_utc, requests, tokens, neurons)

Groq is NOT tracked through this table's counting — its real per-model
remaining/limit/reset comes from response headers Groq returns on every
call (x-ratelimit-{limit,remaining,reset}-requests), which is strictly
more authoritative than anything summed locally could be. See
groq_rate_snapshots below and providers/groq.py's capture of it.

**Real, persisted token counts + a counterfactual baseline (2026-09-10,
NAVI reliability Stage 0)** — the "get the task done using fewer tokens"
claim needed an actual provable number behind it, not a description.
Two real gaps closed here:

1. `record_usage`'s `tokens` argument used to be populated by only 2 of
   NAVI's 7 providers (cloudflare.py, llm7.py calling it directly from
   their own _do_chat) — the other 5 (Groq, OpenRouter, Mistral, Ollama
   Cloud, GMI) never recorded a token count anywhere, despite every one
   of them returning a real, standard OpenAI-compatible `usage` object
   on every call. providers/base.py's Provider.chat() now extracts it
   uniformly, for every provider, in the one place every call already
   funnels through — closing that gap without touching 7 separate
   transport files' worth of call sites.
2. `usage_reference_costs` — what those exact same real prompt/
   completion token counts would have cost, had this call gone to a
   fixed, named, publicly-priced reference model instead. NOT a
   simulation of what that model would have generated (impossible to
   know) — the same methodology real published LLM-routing savings
   claims use: price the REAL token volume actually used at the
   expensive default's real rate, as the "what you'd have paid without
   routing" baseline. REFERENCE_MODELS below tracks the CURRENT
   flagship from each of the two most recognizable labs (GPT-6 Astra,
   Claude Fable 5.1 — both $10/$50 per million input/output tokens as
   of this writing) rather than just one, specifically so the claim
   reads as "vs. the current best from either lab," not "vs. one
   vendor's product." A real side table (not fixed columns) so a third
   reference model is a data addition, not a schema migration.
"""

import contextvars
import sqlite3
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

DB_PATH = Path(__file__).parent.parent / "usage.db"

# See module docstring — real, current (2026-09) pricing, not estimated.
# {reference_model_id: (input_usd_per_mtok, output_usd_per_mtok)}
REFERENCE_MODELS: dict[str, tuple[float, float]] = {
    "gpt-6-astra": (10.0, 50.0),
    "claude-fable-5.1": (10.0, 50.0),
}


def counterfactual_costs_usd(prompt_tokens: int, completion_tokens: int) -> dict[str, float]:
    """What this real token volume would have cost at EACH reference
    model's real published rate — see module docstring for why this
    (not a simulated response) is the honest, standard methodology."""
    return {
        ref: (prompt_tokens / 1_000_000) * in_price + (completion_tokens / 1_000_000) * out_price
        for ref, (in_price, out_price) in REFERENCE_MODELS.items()
    }


_SCHEMA = """
CREATE TABLE IF NOT EXISTS usage_daily (
    provider TEXT NOT NULL,
    model TEXT NOT NULL,
    day_utc TEXT NOT NULL,
    requests INTEGER NOT NULL DEFAULT 0,
    tokens INTEGER NOT NULL DEFAULT 0,
    neurons REAL NOT NULL DEFAULT 0,
    prompt_tokens INTEGER NOT NULL DEFAULT 0,
    completion_tokens INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (provider, model, day_utc)
);
CREATE TABLE IF NOT EXISTS usage_reference_costs (
    provider TEXT NOT NULL,
    model TEXT NOT NULL,
    day_utc TEXT NOT NULL,
    reference_model TEXT NOT NULL,
    usd REAL NOT NULL DEFAULT 0,
    PRIMARY KEY (provider, model, day_utc, reference_model)
);
CREATE TABLE IF NOT EXISTS groq_rate_snapshots (
    model TEXT PRIMARY KEY,
    limit_requests INTEGER,
    remaining_requests INTEGER,
    reset_requests_seconds REAL,
    updated_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS tool_calls_daily (
    tool_name TEXT NOT NULL,
    day_utc TEXT NOT NULL,
    calls INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (tool_name, day_utc)
);
CREATE TABLE IF NOT EXISTS usage_calls (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at REAL NOT NULL,
    day_utc TEXT NOT NULL,
    provider TEXT NOT NULL,
    model TEXT NOT NULL,
    role TEXT,
    mode TEXT,
    tier TEXT,
    attempt INTEGER NOT NULL DEFAULT 0,
    ok INTEGER NOT NULL DEFAULT 1,
    error_kind TEXT,
    prompt_tokens INTEGER NOT NULL DEFAULT 0,
    completion_tokens INTEGER NOT NULL DEFAULT 0,
    cached_tokens INTEGER NOT NULL DEFAULT 0,
    total_tokens INTEGER NOT NULL DEFAULT 0,
    latency_ms INTEGER NOT NULL DEFAULT 0,
    conversation_id TEXT
);
CREATE INDEX IF NOT EXISTS idx_usage_calls_day ON usage_calls(day_utc);
CREATE INDEX IF NOT EXISTS idx_usage_calls_role ON usage_calls(role, day_utc);
CREATE INDEX IF NOT EXISTS idx_usage_calls_model ON usage_calls(provider, model, day_utc);
"""

# How long a per-call row is kept. usage_daily is an aggregate and lives
# forever at a few rows a day; usage_calls is one row PER CALL, so it has
# to be bounded or it grows without limit for a table whose whole purpose
# is recent-window rates and percentiles. 90 days is long enough to see a
# seasonal pattern and to compare a routing change against the month
# before it, and short enough that the table stays small.
CALL_RETENTION_DAYS = 90

_initialized = False


@contextmanager
def _connect():
    global _initialized
    conn = sqlite3.connect(DB_PATH)
    try:
        if not _initialized:
            conn.executescript(_SCHEMA)
            # Idempotent migration for the two new columns, same
            # PRAGMA table_info pattern the aiosqlite stores use —
            # this file predates them, still worth matching the
            # convention now that it needs its first migration.
            cols = {row[1] for row in conn.execute("PRAGMA table_info(usage_daily)").fetchall()}
            for col in ("prompt_tokens", "completion_tokens"):
                if col not in cols:
                    conn.execute(f"ALTER TABLE usage_daily ADD COLUMN {col} INTEGER NOT NULL DEFAULT 0")
            # Prune usage_calls once per process rather than on every
            # write. A DELETE on every insert would make the hot path pay
            # for housekeeping it doesn't need — the table is read in
            # recent-window queries, so a row a few hours past retention
            # costs nothing, and this process restarts often enough in
            # practice (see IDEAS.md's scheduler incident) that once per
            # start is a real cadence, not a theoretical one.
            try:
                conn.execute(
                    "DELETE FROM usage_calls WHERE created_at < ?",
                    (time.time() - CALL_RETENTION_DAYS * 86_400,),
                )
            except sqlite3.Error:
                pass  # table may not exist yet on the very first run
            conn.commit()
            _initialized = True
        yield conn
    finally:
        conn.close()


def _today_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def record_usage(
    provider: str, model: str, requests: int = 0, tokens: int = 0, neurons: float = 0.0,
    prompt_tokens: int = 0, completion_tokens: int = 0,
) -> None:
    """Adds onto today's (UTC) row for (provider, model), creating it if
    this is the first call of the day — the UPSERT itself IS the daily
    reset: a new UTC day means a new row starting from zero, no separate
    reset job or cron needed for this table specifically.

    `prompt_tokens`/`completion_tokens` (2026-09-10) are optional and
    separate from the older flat `tokens` total — a caller that only has
    the total (or none at all, an old call site) still works exactly as
    before. When both are given, this also accumulates a real per-
    reference-model counterfactual cost (usage_reference_costs, one row
    per reference model — see counterfactual_costs_usd/module docstring)
    — computed once, here, from real token counts, not re-derived later
    from the flat total (which can't be split back into input/output)."""
    day = _today_utc()
    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO usage_daily (provider, model, day_utc, requests, tokens, neurons, prompt_tokens, completion_tokens)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(provider, model, day_utc) DO UPDATE SET
                requests = requests + excluded.requests,
                tokens = tokens + excluded.tokens,
                neurons = neurons + excluded.neurons,
                prompt_tokens = prompt_tokens + excluded.prompt_tokens,
                completion_tokens = completion_tokens + excluded.completion_tokens
            """,
            (provider, model, day, requests, tokens, neurons, prompt_tokens, completion_tokens),
        )
        if prompt_tokens or completion_tokens:
            for ref, usd in counterfactual_costs_usd(prompt_tokens, completion_tokens).items():
                conn.execute(
                    """
                    INSERT INTO usage_reference_costs (provider, model, day_utc, reference_model, usd)
                    VALUES (?, ?, ?, ?, ?)
                    ON CONFLICT(provider, model, day_utc, reference_model) DO UPDATE SET
                        usd = usd + excluded.usd
                    """,
                    (provider, model, day, ref, usd),
                )
        conn.commit()


# ---- Per-call rows, and the ambient context that tags them ----
#
# WHY A CONTEXTVAR AND NOT AN ARGUMENT. Who is calling (which role, which
# mode, which capability tier) is known high up — in run_stored_mode_chat,
# in compact_conversation, in a workflow node — and consumed at the very
# bottom, in Provider.chat(). Threading it through every frame in between
# would mean touching ~20 call sites across 12 files AND every future one,
# and the codebase already has a rule about exactly this shape of problem:
# Provider.chat() is concrete rather than abstract specifically so request
# counting "can't be reimplemented per-transport without someone
# eventually forgetting to" (providers/base.py's own docstring). Same
# reasoning applies to tagging. An untagged call still records a row — it
# just lands with NULL role/mode, which is visible in the data as a gap
# rather than silently absent.
#
# asyncio.to_thread propagates the caller's context into the worker
# thread (it copies the current contextvars.Context), which is the whole
# reason this works: every real provider call in NAVI is made through
# to_thread from an async dispatcher, and the tool loop's continuation
# calls inherit the same context without being tagged again.
_CALL_CONTEXT: contextvars.ContextVar[dict | None] = contextvars.ContextVar(
    "navi_call_context", default=None
)


@contextmanager
def call_context(**fields):
    """Tags every provider call made inside this block.

    Nests: inner fields win, outer fields survive, so a dispatcher can set
    role/mode/tier once around a fallback loop and each iteration can add
    its own `attempt=i` without repeating the rest. None values are
    dropped rather than overwriting an outer value with nothing.

    Recognized fields: role, mode, tier, attempt, conversation_id.
    Anything else is accepted and ignored by record_call — deliberately
    permissive, so adding a new dimension later means changing the schema
    and the reader, not auditing every caller.
    """
    parent = _CALL_CONTEXT.get() or {}
    merged = {**parent, **{k: v for k, v in fields.items() if v is not None}}
    token = _CALL_CONTEXT.set(merged)
    try:
        yield merged
    finally:
        _CALL_CONTEXT.reset(token)


def set_call_context(**fields) -> None:
    """Tags every subsequent provider call in THIS task, with no block.

    The `with` form above needs the calls it covers to sit inside it.
    NAVI's dispatchers are long functions whose provider call sits deep
    in a fallback loop, and wrapping those bodies would mean re-indenting
    a few hundred lines of working code to record a label — a diff whose
    risk is out of all proportion to what it buys.

    Safe because a contextvar set inside a coroutine is scoped to that
    coroutine's TASK: asyncio copies the context when a Task is created,
    so one request cannot leak its tags into another, and asyncio.to_thread
    copies it again into the worker thread that makes the actual call.
    Within a single request, carrying forward IS the intent — a tag set
    before a fallback loop should still apply on the third attempt.

    Use `call_context` instead when the tag genuinely must stop at the end
    of a block — a job that makes calls for several different roles in
    one process, say.
    """
    parent = _CALL_CONTEXT.get() or {}
    _CALL_CONTEXT.set({**parent, **{k: v for k, v in fields.items() if v is not None}})


def current_call_context() -> dict:
    """What the ambient tag is right now. Empty dict when untagged."""
    return dict(_CALL_CONTEXT.get() or {})


def record_call(
    provider: str, model: str, *, ok: bool = True,
    prompt_tokens: int = 0, completion_tokens: int = 0,
    cached_tokens: int = 0, total_tokens: int = 0,
    latency_ms: int = 0, error_kind: str | None = None,
    context: dict | None = None,
) -> None:
    """One row per real provider call, attributed to whoever asked for it.

    This is the denominator usage_daily can't provide. That table is
    aggregated by (provider, model, day) — enough to answer "how many
    tokens today", useless for "how often does the serious tier have to
    fall back", because it knows nothing about roles, modes or tiers and
    keeps no per-call granularity to compute a rate or a percentile from.

    FAILED CALLS ARE RECORDED TOO, and that is the point rather than a
    nicety: a table of successes only would make every rate wrong in the
    flattering direction. `ok=0` rows are what make "3 failures out of
    412 calls" expressible at all.

    Never raises — usage tracking must never break a real chat request,
    the same rule record_usage and tools/registry.py's per-tool counter
    already follow.
    """
    ctx = context if context is not None else current_call_context()
    try:
        with _connect() as conn:
            conn.execute(
                """
                INSERT INTO usage_calls (
                    created_at, day_utc, provider, model, role, mode, tier, attempt,
                    ok, error_kind, prompt_tokens, completion_tokens, cached_tokens,
                    total_tokens, latency_ms, conversation_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    time.time(), _today_utc(), provider, model,
                    ctx.get("role"), ctx.get("mode"), ctx.get("tier"),
                    int(ctx.get("attempt") or 0),
                    1 if ok else 0, error_kind,
                    prompt_tokens, completion_tokens, cached_tokens, total_tokens,
                    latency_ms, ctx.get("conversation_id"),
                ),
            )
            conn.commit()
    except Exception as e:
        print(f"[usage] per-call recording failed (non-fatal): {e}")


def get_call_stats(days: int = 7, group_by: str = "role") -> list[dict]:
    """Rates and latency percentiles per group over the last `days`.

    `group_by` is one of "role", "mode", "tier", or "model" (provider+
    model together). Returns, per group: calls, failures, fallback_calls
    (attempt > 0), failure_rate, fallback_rate, tokens, and p50/p95
    latency in ms.

    Rates, not counts, deliberately. A count answers "how much happened",
    which is a function of traffic; a rate answers "how often does this go
    wrong", which is the only form of the number that can be compared
    between two models with very different volumes — and comparing models
    is what this data exists to eventually do.
    """
    columns = {
        "role": "COALESCE(role, '(untagged)')",
        "mode": "COALESCE(mode, '(untagged)')",
        "tier": "COALESCE(tier, '(none)')",
        "model": "provider || '/' || model",
    }
    if group_by not in columns:
        raise ValueError(f"group_by must be one of {sorted(columns)}")
    expr = columns[group_by]
    since = time.time() - days * 86_400
    with _connect() as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            f"""
            SELECT {expr} AS grp,
                   COUNT(*) AS calls,
                   SUM(CASE WHEN ok = 0 THEN 1 ELSE 0 END) AS failures,
                   SUM(CASE WHEN attempt > 0 THEN 1 ELSE 0 END) AS fallback_calls,
                   COALESCE(SUM(prompt_tokens), 0) AS prompt_tokens,
                   COALESCE(SUM(completion_tokens), 0) AS completion_tokens,
                   COALESCE(SUM(cached_tokens), 0) AS cached_tokens
            FROM usage_calls WHERE created_at >= ?
            GROUP BY grp ORDER BY calls DESC
            """,
            (since,),
        ).fetchall()
        out = []
        for row in rows:
            d = dict(row)
            # Percentiles computed per group in Python rather than in SQL:
            # SQLite has no PERCENTILE_CONT, and the alternatives (a
            # window function over a self-join, or NTILE) are markedly
            # harder to read for a table this size. The latency list is
            # bounded by one group's calls in the window, not the whole
            # table.
            latencies = [
                r[0] for r in conn.execute(
                    f"SELECT latency_ms FROM usage_calls "
                    f"WHERE created_at >= ? AND {expr} = ? AND ok = 1 AND latency_ms > 0 "
                    f"ORDER BY latency_ms",
                    (since, d["grp"]),
                ).fetchall()
            ]
            d["p50_latency_ms"] = _percentile(latencies, 0.50)
            d["p95_latency_ms"] = _percentile(latencies, 0.95)
            d["failure_rate"] = round(d["failures"] / d["calls"], 4) if d["calls"] else 0.0
            d["fallback_rate"] = round(d["fallback_calls"] / d["calls"], 4) if d["calls"] else 0.0
            out.append(d)
        return out


def _percentile(sorted_values: list[int], q: float) -> int:
    """Nearest-rank percentile of an already-sorted list. 0 when empty."""
    if not sorted_values:
        return 0
    idx = max(0, min(len(sorted_values) - 1, int(round(q * (len(sorted_values) - 1)))))
    return int(sorted_values[idx])


def get_usage_today(provider: str | None = None) -> list[dict]:
    """Today's (UTC) rows, optionally filtered to one provider. Each row:
    {provider, model, requests, tokens, neurons, prompt_tokens, completion_tokens}."""
    day = _today_utc()
    cols = "provider, model, requests, tokens, neurons, prompt_tokens, completion_tokens"
    with _connect() as conn:
        conn.row_factory = sqlite3.Row
        if provider:
            cur = conn.execute(
                f"SELECT {cols} FROM usage_daily WHERE day_utc = ? AND provider = ?",
                (day, provider),
            )
        else:
            cur = conn.execute(
                f"SELECT {cols} FROM usage_daily WHERE day_utc = ?",
                (day,),
            )
        return [dict(row) for row in cur.fetchall()]


def get_savings_summary(days: int = 30) -> dict:
    """The real, provable number behind "fewer tokens" — real prompt/
    completion tokens actually used across the last `days` UTC days, and
    what that same real token volume would have cost against EACH
    tracked reference model's real published rate. Returns
    {days, total_requests, total_prompt_tokens, total_completion_tokens,
    counterfactual_usd: {reference_model: usd, ...}}. Doesn't attempt a
    real "actual USD spent" figure — most of NAVI's providers are free/
    near-free tier, so the honest, defensible claim is about token
    volume against real reference prices, not a real-vs-real dollar
    comparison that would need per-provider pricing this file doesn't
    track (a separate, larger piece of work, not this one).

    Bring-your-own-key providers are left out entirely (2026-09-23): a call
    on someone's own paid DeepSeek or Claude key cost real money, so
    counting it toward "what you'd have paid otherwise" would report
    savings on spend that actually happened."""
    from providers.byok import BYOK_TRANSPORTS, CUSTOM_PREFIX  # here, not at module top: providers.base imports this module

    paid = tuple(BYOK_TRANSPORTS)
    exclude = (
        f"AND provider NOT IN ({', '.join('?' * len(paid))}) "
        f"AND provider NOT LIKE '{CUSTOM_PREFIX}%'"
    )
    with _connect() as conn:
        conn.row_factory = sqlite3.Row
        totals = dict(conn.execute(
            f"""
            SELECT
                COALESCE(SUM(requests), 0) AS total_requests,
                COALESCE(SUM(prompt_tokens), 0) AS total_prompt_tokens,
                COALESCE(SUM(completion_tokens), 0) AS total_completion_tokens
            FROM usage_daily
            WHERE day_utc >= date('now', ?) {exclude}
            """,
            (f"-{days} days", *paid),
        ).fetchone())
        cf_rows = conn.execute(
            f"""
            SELECT reference_model, COALESCE(SUM(usd), 0) AS usd
            FROM usage_reference_costs
            WHERE day_utc >= date('now', ?) {exclude}
            GROUP BY reference_model
            """,
            (f"-{days} days", *paid),
        ).fetchall()
    totals["days"] = days
    totals["counterfactual_usd"] = {row["reference_model"]: row["usd"] for row in cf_rows}
    return totals


def record_groq_snapshot(model: str, limit_requests: int | None, remaining_requests: int | None, reset_requests_seconds: float | None) -> None:
    """Overwrites (not accumulates) — this is Groq's own live snapshot for
    this model as of the most recent call, not something NAVI sums itself."""
    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO groq_rate_snapshots (model, limit_requests, remaining_requests, reset_requests_seconds, updated_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(model) DO UPDATE SET
                limit_requests = excluded.limit_requests,
                remaining_requests = excluded.remaining_requests,
                reset_requests_seconds = excluded.reset_requests_seconds,
                updated_at = excluded.updated_at
            """,
            (model, limit_requests, remaining_requests, reset_requests_seconds, time.time()),
        )
        conn.commit()


def get_groq_snapshots() -> list[dict]:
    """Every Groq model with a known snapshot: {model, limit_requests,
    remaining_requests, reset_requests_seconds, updated_at}. A model NAVI
    hasn't called yet simply has no row — the frontend shows it as
    "not yet observed" rather than a fabricated number."""
    with _connect() as conn:
        conn.row_factory = sqlite3.Row
        cur = conn.execute(
            "SELECT model, limit_requests, remaining_requests, reset_requests_seconds, updated_at FROM groq_rate_snapshots"
        )
        return [dict(row) for row in cur.fetchall()]


def record_tool_call(tool_name: str) -> None:
    """Real per-tool dispatch frequency (2026-09-11) — deliberately
    separate from record_usage above, since tools/registry.py's dispatch()
    runs entirely dispatcher-side (a real web search, an MCP call) with
    ZERO LLM token cost of its own; this answers "whenever the dispatcher
    uses which tools," a genuinely different question from the token/cost
    tracking record_usage answers. Counts every dispatch attempt,
    regardless of whether it then succeeds or raises — same "count the
    attempt, not just the success" reasoning record_usage's own `requests`
    counter already uses, since the dispatcher genuinely did route to this
    tool either way. Never called for ask_user_choice (intercepted before
    dispatch() is ever reached — see tools/registry.py's own comment)."""
    day = _today_utc()
    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO tool_calls_daily (tool_name, day_utc, calls)
            VALUES (?, ?, 1)
            ON CONFLICT(tool_name, day_utc) DO UPDATE SET calls = calls + 1
            """,
            (tool_name, day),
        )
        conn.commit()


def get_most_used_tools(days: int = 30, limit: int = 10) -> list[dict]:
    """Real dispatch frequency per tool over the window, most-called
    first — the tool-side sibling of the "most used models + providers"
    idea (IDEAS.md, 2026-09-11): same kind of empirical usage signal,
    just for which tools the dispatcher actually reaches for rather than
    which models answer. Returns [{tool_name, calls}, ...]."""
    with _connect() as conn:
        conn.row_factory = sqlite3.Row
        cur = conn.execute(
            """
            SELECT tool_name, SUM(calls) AS calls
            FROM tool_calls_daily
            WHERE day_utc >= date('now', ?)
            GROUP BY tool_name
            ORDER BY calls DESC
            LIMIT ?
            """,
            (f"-{days} days", limit),
        )
        return [dict(row) for row in cur.fetchall()]
