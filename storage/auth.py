"""
storage/auth.py

Real per-user accounts (2026-09-10) — the first real identity NAVI has ever
had. Everything before this (`edited_by` on workflow versions, the whole
API) ran behind one shared secret (`NAVI_API_KEY`, see server.py's
`_require_api_key`) with zero notion of "which person." That gate is
UNCHANGED and stays the outer boundary for the whole API; this module adds
a second, layered-on-top check — a real session token identifying a real
user — required only by the specific routes that need to know WHO, not
just "some legitimate caller."

Scope, deliberately narrow for this first pass (Owner/Admin/Member — the
same three-tier shape n8n and Notion both converge on for "one
organization, several people, tiered roles" — see IDEAS.md): ONE company
per running NAVI instance. A deployment that serves several separate
companies (AWS's own "Silo" tenant-isolation model: one dedicated backend
+ database per company, not one shared table with a company_id column) is
real, documented, future scope — not this file's job. Nothing here assumes
multi-company; there's no `company_id` anywhere, on purpose.

**Bootstrap problem, solved the ordinary way**: the very first account
created (via `register_first_owner`) becomes Owner automatically — there's
no one yet to grant that role. Every account after that is created by an
existing Owner/Admin (`create_user`), never self-registered — see
server.py's `/auth/register` route for the enforcement (count_users() == 0
gate).

**Password storage**: `hashlib.scrypt` (Python stdlib, no new dependency)
rather than bcrypt/argon2 — this is a genuine security-relevant choice,
not a shortcut: scrypt is one of OWASP's recommended password-hashing KDFs
alongside argon2/bcrypt. Cost parameters (N=2**14, r=8, p=1) are
deliberately on the lower/safer-to-run side of OWASP's guidance — enough
to require real computation per guess, small enough that
`hashlib.scrypt`'s default 32MB memory ceiling doesn't need overriding.
The stored format (`scrypt$n$r$p$salt_hex$hash_hex`) is versioned/self-
describing, same reasoning as HTTP's own `Content-Type` header, so a
future change to the cost parameters doesn't break verifying passwords
hashed under the old ones.

**Sessions, not JWTs**: a random opaque token
(`secrets.token_urlsafe(32)`, same generator agent_work.py's webhook
tokens already use) stored server-side in `sessions`, looked up per
request. Chosen over a JWT specifically so logout/revocation is a real
row delete, not "wait for expiry" — a genuine property JWTs give up for
statelessness NAVI's own single-database single-instance deployment
doesn't need anyway.
"""

import hashlib
import hmac
import secrets
import time
import uuid
from pathlib import Path

import aiosqlite

DB_PATH = Path(__file__).parent.parent / "auth.db"

# Ordered weakest to strongest — server.py's require_role() checks
# membership by name, but ROLE_RANK below exists for any future "at least
# this senior" comparison, so it isn't silently re-derived twice.
ROLES = ("member", "admin", "owner")
ROLE_RANK = {role: i for i, role in enumerate(ROLES)}

SESSION_LIFETIME_SECONDS = 30 * 24 * 3600  # 30 days — no refresh-token dance for a v1 this small

_SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id TEXT PRIMARY KEY,
    email TEXT NOT NULL UNIQUE,
    password_hash TEXT NOT NULL,
    name TEXT,
    role TEXT NOT NULL,
    is_active INTEGER NOT NULL DEFAULT 1,
    created_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS sessions (
    token TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    created_at REAL NOT NULL,
    expires_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_sessions_user ON sessions(user_id);
"""

_initialized = False


async def _ensure_schema(db: aiosqlite.Connection) -> None:
    global _initialized
    if _initialized:
        return
    await db.executescript(_SCHEMA)
    await db.commit()
    _initialized = True


def hash_password(password: str) -> str:
    n, r, p = 2 ** 14, 8, 1
    salt = secrets.token_bytes(16)
    digest = hashlib.scrypt(password.encode("utf-8"), salt=salt, n=n, r=r, p=p, dklen=64)
    return f"scrypt${n}${r}${p}${salt.hex()}${digest.hex()}"


def verify_password(password: str, stored_hash: str) -> bool:
    try:
        algo, n, r, p, salt_hex, digest_hex = stored_hash.split("$")
        assert algo == "scrypt"
        n, r, p = int(n), int(r), int(p)
        salt = bytes.fromhex(salt_hex)
        expected = bytes.fromhex(digest_hex)
    except (ValueError, AssertionError):
        return False
    actual = hashlib.scrypt(password.encode("utf-8"), salt=salt, n=n, r=r, p=p, dklen=len(expected))
    return hmac.compare_digest(actual, expected)


def _row_to_user(row: dict) -> dict:
    row = dict(row)
    row.pop("password_hash", None)
    row["is_active"] = bool(row["is_active"])
    return row


async def count_users() -> int:
    async with aiosqlite.connect(DB_PATH) as db:
        await _ensure_schema(db)
        async with db.execute("SELECT COUNT(*) FROM users") as cursor:
            (count,) = await cursor.fetchone()
    return count


async def create_user(email: str, password: str, name: str | None, role: str) -> dict | None:
    """Returns None if the email is already taken (case-insensitively —
    emails are normalized to lowercase on the way in, same as every real
    auth system does, so "Juan@x.com" and "juan@x.com" can't become two
    separate accounts)."""
    if role not in ROLES:
        raise ValueError(f"unknown role {role!r}")
    user_id = str(uuid.uuid4())
    now = time.time()
    async with aiosqlite.connect(DB_PATH) as db:
        await _ensure_schema(db)
        try:
            await db.execute(
                "INSERT INTO users (id, email, password_hash, name, role, is_active, created_at) "
                "VALUES (?, ?, ?, ?, ?, 1, ?)",
                (user_id, email.strip().lower(), hash_password(password), name, role, now),
            )
            await db.commit()
        except aiosqlite.IntegrityError:
            return None
    return await get_user_by_id(user_id)


async def get_user_by_id(user_id: str) -> dict | None:
    async with aiosqlite.connect(DB_PATH) as db:
        await _ensure_schema(db)
        db.row_factory = aiosqlite.Row
        async with db.execute(
            "SELECT id, email, password_hash, name, role, is_active, created_at FROM users WHERE id = ?",
            (user_id,),
        ) as cursor:
            row = await cursor.fetchone()
    return _row_to_user(dict(row)) if row else None


async def get_user_by_email(email: str) -> dict | None:
    """Includes `password_hash` (unlike get_user_by_id) — this is the one
    caller (login) that actually needs it, via _get_user_by_email_raw
    below; every other reader goes through the id lookup, which strips it."""
    return await _get_user_by_email_raw(email)


async def _get_user_by_email_raw(email: str) -> dict | None:
    async with aiosqlite.connect(DB_PATH) as db:
        await _ensure_schema(db)
        db.row_factory = aiosqlite.Row
        async with db.execute(
            "SELECT id, email, password_hash, name, role, is_active, created_at FROM users WHERE email = ?",
            (email.strip().lower(),),
        ) as cursor:
            row = await cursor.fetchone()
    return dict(row) if row else None


async def list_users() -> list[dict]:
    async with aiosqlite.connect(DB_PATH) as db:
        await _ensure_schema(db)
        db.row_factory = aiosqlite.Row
        async with db.execute(
            "SELECT id, email, password_hash, name, role, is_active, created_at FROM users ORDER BY created_at ASC"
        ) as cursor:
            rows = await cursor.fetchall()
    return [_row_to_user(dict(r)) for r in rows]


async def update_user_role(user_id: str, role: str) -> bool:
    if role not in ROLES:
        raise ValueError(f"unknown role {role!r}")
    async with aiosqlite.connect(DB_PATH) as db:
        await _ensure_schema(db)
        cursor = await db.execute("UPDATE users SET role = ? WHERE id = ?", (role, user_id))
        await db.commit()
        return cursor.rowcount > 0


async def set_user_active(user_id: str, is_active: bool) -> bool:
    """Deactivate/reactivate — same soft-delete philosophy as
    storage/agent_work.py's workflow delete/restore: a removed account's
    row (and everything it authored/edited) stays intact for audit
    purposes, it just can no longer log in (see get_session below, which
    checks is_active on every lookup)."""
    async with aiosqlite.connect(DB_PATH) as db:
        await _ensure_schema(db)
        cursor = await db.execute("UPDATE users SET is_active = ? WHERE id = ?", (1 if is_active else 0, user_id))
        await db.commit()
        return cursor.rowcount > 0


async def create_session(user_id: str) -> str:
    token = secrets.token_urlsafe(32)
    now = time.time()
    async with aiosqlite.connect(DB_PATH) as db:
        await _ensure_schema(db)
        await db.execute(
            "INSERT INTO sessions (token, user_id, created_at, expires_at) VALUES (?, ?, ?, ?)",
            (token, user_id, now, now + SESSION_LIFETIME_SECONDS),
        )
        await db.commit()
    return token


async def get_session_user(token: str) -> dict | None:
    """The real per-request check: a session must exist, not be expired,
    AND belong to a still-active user — a deactivated account's
    outstanding session tokens stop working immediately, not just its
    ability to log in again."""
    async with aiosqlite.connect(DB_PATH) as db:
        await _ensure_schema(db)
        db.row_factory = aiosqlite.Row
        async with db.execute(
            "SELECT user_id, expires_at FROM sessions WHERE token = ?", (token,)
        ) as cursor:
            session_row = await cursor.fetchone()
        if not session_row or session_row["expires_at"] < time.time():
            return None
        async with db.execute(
            "SELECT id, email, password_hash, name, role, is_active, created_at FROM users WHERE id = ?",
            (session_row["user_id"],),
        ) as cursor:
            user_row = await cursor.fetchone()
    if not user_row or not user_row["is_active"]:
        return None
    return _row_to_user(dict(user_row))


async def delete_session(token: str) -> bool:
    async with aiosqlite.connect(DB_PATH) as db:
        await _ensure_schema(db)
        cursor = await db.execute("DELETE FROM sessions WHERE token = ?", (token,))
        await db.commit()
        return cursor.rowcount > 0
