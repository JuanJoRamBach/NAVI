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
"""

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
    track (a separate, larger piece of work, not this one)."""
    with _connect() as conn:
        conn.row_factory = sqlite3.Row
        totals = dict(conn.execute(
            """
            SELECT
                COALESCE(SUM(requests), 0) AS total_requests,
                COALESCE(SUM(prompt_tokens), 0) AS total_prompt_tokens,
                COALESCE(SUM(completion_tokens), 0) AS total_completion_tokens
            FROM usage_daily
            WHERE day_utc >= date('now', ?)
            """,
            (f"-{days} days",),
        ).fetchone())
        cf_rows = conn.execute(
            """
            SELECT reference_model, COALESCE(SUM(usd), 0) AS usd
            FROM usage_reference_costs
            WHERE day_utc >= date('now', ?)
            GROUP BY reference_model
            """,
            (f"-{days} days",),
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
