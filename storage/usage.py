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
"""

import sqlite3
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

DB_PATH = Path(__file__).parent.parent / "usage.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS usage_daily (
    provider TEXT NOT NULL,
    model TEXT NOT NULL,
    day_utc TEXT NOT NULL,
    requests INTEGER NOT NULL DEFAULT 0,
    tokens INTEGER NOT NULL DEFAULT 0,
    neurons REAL NOT NULL DEFAULT 0,
    PRIMARY KEY (provider, model, day_utc)
);
CREATE TABLE IF NOT EXISTS groq_rate_snapshots (
    model TEXT PRIMARY KEY,
    limit_requests INTEGER,
    remaining_requests INTEGER,
    reset_requests_seconds REAL,
    updated_at REAL NOT NULL
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
            conn.commit()
            _initialized = True
        yield conn
    finally:
        conn.close()


def _today_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def record_usage(provider: str, model: str, requests: int = 0, tokens: int = 0, neurons: float = 0.0) -> None:
    """Adds onto today's (UTC) row for (provider, model), creating it if
    this is the first call of the day — the UPSERT itself IS the daily
    reset: a new UTC day means a new row starting from zero, no separate
    reset job or cron needed for this table specifically."""
    day = _today_utc()
    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO usage_daily (provider, model, day_utc, requests, tokens, neurons)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(provider, model, day_utc) DO UPDATE SET
                requests = requests + excluded.requests,
                tokens = tokens + excluded.tokens,
                neurons = neurons + excluded.neurons
            """,
            (provider, model, day, requests, tokens, neurons),
        )
        conn.commit()


def get_usage_today(provider: str | None = None) -> list[dict]:
    """Today's (UTC) rows, optionally filtered to one provider. Each row:
    {provider, model, requests, tokens, neurons}."""
    day = _today_utc()
    with _connect() as conn:
        conn.row_factory = sqlite3.Row
        if provider:
            cur = conn.execute(
                "SELECT provider, model, requests, tokens, neurons FROM usage_daily WHERE day_utc = ? AND provider = ?",
                (day, provider),
            )
        else:
            cur = conn.execute(
                "SELECT provider, model, requests, tokens, neurons FROM usage_daily WHERE day_utc = ?",
                (day,),
            )
        return [dict(row) for row in cur.fetchall()]


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
