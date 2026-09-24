"""
storage/knowledge.py

Company and project knowledge (2026-09-24): what NAVI should know about the
company it works in and about each project (a client, an engagement),
kept apart from each conversation's own memory (storage/context_store.py's
context.md), which stays per conversation and is compacted on its own.
Nothing here is ever compacted: people curate it, so its size is theirs.

Two levels, one shape. The company and every project each have:
- a BRIEF: a short text NAVI reads on every turn in that scope. Hard size
  limit (BRIEF_LIMITS), since it rides along on every message.
- a LIBRARY: longer entries NAVI searches only when a question needs them
  (search_knowledge), so a big library costs a chat nothing until one
  entry is actually relevant.

Who may change what, decided 2026-09-24 after checking Claude and ChatGPT
projects (per-project "can chat" / "can edit"), SOC 2 CC8.1 and NIST AC-5
(separation of duties), and GitHub's review rules:

- Company brief and library: Owner and Admin only. Every member can read.
- A project: its "edit" members, plus every Owner and Admin. Its "chat"
  members can read and chat in it.
- EVERY brief change needs approval by someone other than its author, who
  can also edit that scope. The one exception is the Owner, whose change
  applies at once but is recorded as "applied without review" and can be
  reviewed afterwards by an Admin (the after-the-fact review SOC 2 accepts
  for emergency changes). Owner and Admin can edit every scope, so there
  is always someone able to approve.
- An approval covers exactly the text approved. A newer proposal replaces
  the pending one, and a proposal written against an older brief can't be
  approved once the brief has changed underneath it (GitHub's "dismiss
  stale approvals").
- Restoring an earlier APPROVED version is immediate for any editor: it
  brings back something a second person already approved, so a bad
  change that slipped through can be undone without waiting.
- Library entries: editors add them directly. Suggestions from "chat"
  members and from NAVI wait for an editor's approval.
- History is append-only. knowledge_events can't be updated or deleted:
  SQLite triggers refuse it, so the guarantee holds even for code that
  forgets, and nothing in this module offers a way to remove it.

Synchronous sqlite3 on purpose: the search tool runs inside the tool loop's
worker thread, and the routes call in through asyncio.to_thread.
"""

import re
import sqlite3
import threading
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

DB_PATH = Path(__file__).parent.parent / "knowledge.db"

COMPANY = "company"
PROJECT_PREFIX = "project:"

# Characters, not tokens: exact, and the same number on every model.
# Roughly 350 and 600 tokens. The company brief is smaller because it
# rides along on every chat in the company, project or not.
BRIEF_LIMITS = {"company": 1500, "project": 2500}
TITLE_LIMIT = 120
BODY_LIMIT = 2000
SEARCH_RESULTS = 5

# A version whose text a second person has vouched for. Only these can be
# restored instantly (see revert_brief).
_VOUCHED = ("approved", "reviewed")
# Statuses a version can hold while it is the brief in force.
_LIVE = ("approved", "applied_unreviewed", "reviewed", "flagged")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS projects (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    created_by TEXT NOT NULL,
    created_at REAL NOT NULL,
    archived_at REAL
);
CREATE TABLE IF NOT EXISTS project_members (
    project_id TEXT NOT NULL,
    user_id TEXT NOT NULL,
    access TEXT NOT NULL,
    added_by TEXT NOT NULL,
    added_at REAL NOT NULL,
    PRIMARY KEY (project_id, user_id)
);
CREATE TABLE IF NOT EXISTS brief_versions (
    id TEXT PRIMARY KEY,
    scope TEXT NOT NULL,
    text TEXT NOT NULL,
    status TEXT NOT NULL,
    author_id TEXT NOT NULL,
    author_email TEXT NOT NULL,
    created_at REAL NOT NULL,
    based_on TEXT,
    reverted_from TEXT,
    decided_by TEXT,
    decided_at REAL,
    note TEXT,
    live_at REAL,
    reviewed_by TEXT,
    reviewed_at REAL
);
CREATE INDEX IF NOT EXISTS idx_brief_scope ON brief_versions(scope, status);
CREATE TABLE IF NOT EXISTS knowledge_entries (
    id TEXT PRIMARY KEY,
    scope TEXT NOT NULL,
    title TEXT NOT NULL,
    body TEXT NOT NULL,
    status TEXT NOT NULL,
    source TEXT NOT NULL,
    author_email TEXT NOT NULL,
    conversation_id TEXT,
    created_at REAL NOT NULL,
    decided_by TEXT,
    decided_at REAL,
    note TEXT
);
CREATE INDEX IF NOT EXISTS idx_entries_scope ON knowledge_entries(scope, status);
CREATE TABLE IF NOT EXISTS knowledge_events (
    seq INTEGER PRIMARY KEY AUTOINCREMENT,
    at REAL NOT NULL,
    scope TEXT NOT NULL,
    action TEXT NOT NULL,
    actor TEXT NOT NULL,
    subject_id TEXT,
    detail TEXT
);
CREATE TRIGGER IF NOT EXISTS knowledge_events_no_update
    BEFORE UPDATE ON knowledge_events
    BEGIN SELECT RAISE(ABORT, 'knowledge history is append-only'); END;
CREATE TRIGGER IF NOT EXISTS knowledge_events_no_delete
    BEFORE DELETE ON knowledge_events
    BEGIN SELECT RAISE(ABORT, 'knowledge history is append-only'); END;
"""

_init_lock = threading.Lock()
_initialized = False
_fts = False


class KnowledgeError(Exception):
    """A request the rules refuse. The message is written for the person
    who made it and is shown to them as is."""

    def __init__(self, message: str, status: int = 400):
        super().__init__(message)
        self.status = status


@contextmanager
def _db():
    """One connection per operation, always closed. Commits on success and
    ALSO when the operation ends in a KnowledgeError: a refusal can follow
    a write that must stick (approve_brief marks a stale proposal before
    refusing it). Any other exception rolls back."""
    conn = _connect()
    try:
        yield conn
        conn.commit()
    except KnowledgeError:
        conn.commit()
        raise
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def _connect() -> sqlite3.Connection:
    global _initialized, _fts
    conn = sqlite3.connect(DB_PATH, timeout=5.0)
    conn.row_factory = sqlite3.Row
    if not _initialized:
        with _init_lock:
            if not _initialized:
                conn.executescript(_SCHEMA)
                try:
                    conn.execute(
                        "CREATE VIRTUAL TABLE IF NOT EXISTS entries_fts "
                        "USING fts5(title, body, entry_id UNINDEXED, scope UNINDEXED)"
                    )
                    _fts = True
                except sqlite3.OperationalError:
                    # An SQLite built without FTS5. Search falls back to
                    # plain LIKE matching: slower and cruder, still correct.
                    _fts = False
                conn.commit()
                _initialized = True
    return conn


def _now() -> float:
    return time.time()


def _log(conn: sqlite3.Connection, scope: str, action: str, actor: str, subject_id: str | None = None,
         detail: str | None = None) -> None:
    conn.execute(
        "INSERT INTO knowledge_events (at, scope, action, actor, subject_id, detail) VALUES (?, ?, ?, ?, ?, ?)",
        (_now(), scope, action, actor, subject_id, detail),
    )


# ---- Scopes and permissions ---------------------------------------------------

def project_scope(project_id: str) -> str:
    return f"{PROJECT_PREFIX}{project_id}"


def _project_id(scope: str) -> str | None:
    return scope[len(PROJECT_PREFIX):] if scope.startswith(PROJECT_PREFIX) else None


def _is_admin(user: dict) -> bool:
    return user.get("role") in ("owner", "admin")


def _is_owner(user: dict) -> bool:
    return user.get("role") == "owner"


def _project_row(conn: sqlite3.Connection, project_id: str) -> sqlite3.Row | None:
    return conn.execute(
        "SELECT * FROM projects WHERE id = ? AND archived_at IS NULL", (project_id,)
    ).fetchone()


def _project_access(conn: sqlite3.Connection, user: dict, project_id: str) -> str | None:
    if not _project_row(conn, project_id):
        return None
    if _is_admin(user):
        return "edit"
    row = conn.execute(
        "SELECT access FROM project_members WHERE project_id = ? AND user_id = ?", (project_id, user["id"])
    ).fetchone()
    return row["access"] if row else None


def project_access(user: dict, project_id: str) -> str | None:
    """"edit", "chat", or None when this person can't use the project."""
    with _db() as conn:
        return _project_access(conn, user, project_id)


def _can_view(conn: sqlite3.Connection, user: dict, scope: str) -> bool:
    if scope == COMPANY:
        return True
    pid = _project_id(scope)
    return bool(pid and _project_access(conn, user, pid))


def _can_edit(conn: sqlite3.Connection, user: dict, scope: str) -> bool:
    if scope == COMPANY:
        return _is_admin(user)
    pid = _project_id(scope)
    return bool(pid and _project_access(conn, user, pid) == "edit")


def _require_view(conn, user, scope):
    if not _can_view(conn, user, scope):
        raise KnowledgeError("You don't have access to this project.", 403)


def _require_edit(conn, user, scope):
    if not _can_edit(conn, user, scope):
        raise KnowledgeError(
            "Only an Owner or Admin can change company knowledge." if scope == COMPANY
            else "You can chat in this project but not change it. Ask one of its editors.",
            403,
        )


def _brief_limit(scope: str) -> int:
    return BRIEF_LIMITS["company" if scope == COMPANY else "project"]


# ---- Projects -------------------------------------------------------------------

def create_project(user: dict, name: str) -> dict:
    name = (name or "").strip()
    if not name:
        raise KnowledgeError("Give the project a name.")
    if len(name) > 80:
        raise KnowledgeError("Keep the project name under 80 characters.")
    pid = uuid.uuid4().hex[:12]
    with _db() as conn:
        conn.execute(
            "INSERT INTO projects (id, name, created_by, created_at) VALUES (?, ?, ?, ?)",
            (pid, name, user["email"], _now()),
        )
        # The creator can edit what they made. Owners and Admins can edit
        # every project anyway, so they aren't added as members.
        if not _is_admin(user):
            conn.execute(
                "INSERT INTO project_members (project_id, user_id, access, added_by, added_at) VALUES (?, ?, 'edit', ?, ?)",
                (pid, user["id"], user["email"], _now()),
            )
        _log(conn, project_scope(pid), "project_created", user["email"], pid, name)
    return {"id": pid, "name": name, "access": "edit"}


def list_projects(user: dict) -> list[dict]:
    """Projects this person can use, with their access. Owners and Admins
    see every project."""
    with _db() as conn:
        if _is_admin(user):
            rows = conn.execute(
                "SELECT id, name, created_at FROM projects WHERE archived_at IS NULL ORDER BY name COLLATE NOCASE"
            ).fetchall()
            return [{**dict(r), "access": "edit"} for r in rows]
        rows = conn.execute(
            "SELECT p.id, p.name, p.created_at, m.access FROM projects p "
            "JOIN project_members m ON m.project_id = p.id "
            "WHERE p.archived_at IS NULL AND m.user_id = ? ORDER BY p.name COLLATE NOCASE",
            (user["id"],),
        ).fetchall()
        return [dict(r) for r in rows]


def get_project(user: dict, project_id: str) -> dict:
    with _db() as conn:
        row = _project_row(conn, project_id)
        access = _project_access(conn, user, project_id)
        if not row or not access:
            raise KnowledgeError("You don't have access to this project.", 403)
        return {"id": row["id"], "name": row["name"], "access": access}


def project_name(project_id: str) -> str | None:
    with _db() as conn:
        row = _project_row(conn, project_id)
        return row["name"] if row else None


def archive_project(user: dict, project_id: str) -> None:
    if not _is_admin(user):
        raise KnowledgeError("Only an Owner or Admin can archive a project.", 403)
    with _db() as conn:
        if not _project_row(conn, project_id):
            raise KnowledgeError("That project doesn't exist.", 404)
        conn.execute("UPDATE projects SET archived_at = ? WHERE id = ?", (_now(), project_id))
        _log(conn, project_scope(project_id), "project_archived", user["email"], project_id)


def list_members(user: dict, project_id: str) -> list[dict]:
    """Rows of (user_id, access). The caller joins in names and emails from
    the accounts store, which lives in a different database."""
    with _db() as conn:
        _require_view(conn, user, project_scope(project_id))
        rows = conn.execute(
            "SELECT user_id, access, added_by, added_at FROM project_members WHERE project_id = ?", (project_id,)
        ).fetchall()
        return [dict(r) for r in rows]


def set_member(user: dict, project_id: str, target: dict, access: str | None) -> None:
    """Adds someone, changes their access, or (access=None) removes them.
    Logged: who can change a brief is itself part of the audit trail."""
    scope = project_scope(project_id)
    with _db() as conn:
        _require_edit(conn, user, scope)
        if access not in ("chat", "edit", None):
            raise KnowledgeError("Access is either chat or edit.")
        if _is_admin(target):
            raise KnowledgeError(f"{target['email']} is an {target['role'].title()} and can already edit every project.")
        if access is None:
            conn.execute("DELETE FROM project_members WHERE project_id = ? AND user_id = ?", (project_id, target["id"]))
            _log(conn, scope, "member_removed", user["email"], target["id"], target["email"])
            return
        conn.execute(
            "INSERT INTO project_members (project_id, user_id, access, added_by, added_at) VALUES (?, ?, ?, ?, ?) "
            "ON CONFLICT(project_id, user_id) DO UPDATE SET access = excluded.access",
            (project_id, target["id"], access, user["email"], _now()),
        )
        _log(conn, scope, "member_set", user["email"], target["id"], f"{target['email']}: {access}")


# ---- Briefs ------------------------------------------------------------------------

def _current(conn: sqlite3.Connection, scope: str) -> sqlite3.Row | None:
    return conn.execute(
        f"SELECT * FROM brief_versions WHERE scope = ? AND status IN ({','.join('?' * len(_LIVE))}) "
        "ORDER BY live_at DESC LIMIT 1",
        (scope, *_LIVE),
    ).fetchone()


def _pending(conn: sqlite3.Connection, scope: str) -> sqlite3.Row | None:
    return conn.execute(
        "SELECT * FROM brief_versions WHERE scope = ? AND status = 'pending' ORDER BY created_at DESC LIMIT 1",
        (scope,),
    ).fetchone()


def _version(conn: sqlite3.Connection, version_id: str) -> sqlite3.Row:
    row = conn.execute("SELECT * FROM brief_versions WHERE id = ?", (version_id,)).fetchone()
    if not row:
        raise KnowledgeError("That version doesn't exist.", 404)
    return row


def current_brief_text(scope: str) -> str:
    with _db() as conn:
        row = _current(conn, scope)
        return row["text"] if row else ""


def brief_state(user: dict, scope: str) -> dict:
    """Everything the brief screen needs, with what THIS person may do."""
    with _db() as conn:
        _require_view(conn, user, scope)
        can_edit = _can_edit(conn, user, scope)
        current = _current(conn, scope)
        pending = _pending(conn, scope)
        versions = conn.execute(
            "SELECT * FROM brief_versions WHERE scope = ? ORDER BY created_at DESC LIMIT 50", (scope,)
        ).fetchall()
        return {
            "scope": scope,
            "limit": _brief_limit(scope),
            "can_edit": can_edit,
            "current": dict(current) if current else None,
            "pending": dict(pending) if pending else None,
            # Not the author, and allowed to edit this scope. The UI uses it
            # to show Approve only to someone who may actually approve.
            "can_approve_pending": bool(pending and can_edit and pending["author_id"] != user["id"]),
            # An Owner's change applies without review; an Admin reviews it
            # afterwards. Listed so it can't sit unnoticed.
            "unreviewed": [dict(v) for v in versions if v["status"] == "applied_unreviewed"],
            "can_review": _is_admin(user),
            "versions": [dict(v) for v in versions],
            # Earlier versions a second person approved. Restoring one is
            # immediate (see revert_brief).
            "restorable": [
                v["id"] for v in versions
                if v["status"] in _VOUCHED and (not current or v["id"] != current["id"])
            ] if can_edit else [],
        }


def propose_brief(user: dict, scope: str, text: str) -> dict:
    """An Owner's change applies immediately, flagged as unreviewed.
    Everyone else's waits for someone else's approval."""
    text = (text or "").strip()
    with _db() as conn:
        _require_edit(conn, user, scope)
        limit = _brief_limit(scope)
        if len(text) > limit:
            raise KnowledgeError(f"The brief is {len(text)} characters; the limit is {limit}. Move detail into the library.")
        current = _current(conn, scope)
        if current and current["text"] == text:
            raise KnowledgeError("That's the same as the brief in force.")
        now = _now()
        # Any proposal still waiting is replaced: an approval covers exactly
        # the text approved, never a later edit of it.
        stale = _pending(conn, scope)
        if stale:
            conn.execute("UPDATE brief_versions SET status = 'superseded', decided_at = ? WHERE id = ?", (now, stale["id"]))
            _log(conn, scope, "brief_superseded", user["email"], stale["id"], "replaced by a newer proposal")
        vid = uuid.uuid4().hex
        owner = _is_owner(user)
        conn.execute(
            "INSERT INTO brief_versions (id, scope, text, status, author_id, author_email, created_at, based_on, live_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (vid, scope, text, "applied_unreviewed" if owner else "pending", user["id"], user["email"],
             now, current["id"] if current else None, now if owner else None),
        )
        _log(conn, scope, "brief_applied_unreviewed" if owner else "brief_proposed", user["email"], vid,
             f"{len(text)} characters")
        return dict(_version(conn, vid))


def approve_brief(user: dict, version_id: str) -> dict:
    with _db() as conn:
        v = _version(conn, version_id)
        scope = v["scope"]
        _require_edit(conn, user, scope)
        if v["status"] != "pending":
            raise KnowledgeError("This change isn't waiting for approval any more.", 409)
        if v["author_id"] == user["id"]:
            raise KnowledgeError("Someone other than the author has to approve a change.", 403)
        current = _current(conn, scope)
        if (current["id"] if current else None) != v["based_on"]:
            # The brief changed after this was written. Approving it now would
            # silently undo whatever went in since, so it has to be redone.
            now = _now()
            conn.execute("UPDATE brief_versions SET status = 'stale', decided_at = ? WHERE id = ?", (now, version_id))
            _log(conn, scope, "brief_stale", user["email"], version_id, "brief changed since this was proposed")
            raise KnowledgeError("The brief changed after this was proposed, so it can't be approved as is. "
                                 "The author needs to redo it against the current brief.", 409)
        now = _now()
        conn.execute(
            "UPDATE brief_versions SET status = 'approved', decided_by = ?, decided_at = ?, live_at = ? WHERE id = ?",
            (user["email"], now, now, version_id),
        )
        _log(conn, scope, "brief_approved", user["email"], version_id, f"authored by {v['author_email']}")
        return dict(_version(conn, version_id))


def reject_brief(user: dict, version_id: str, note: str | None = None) -> dict:
    """An editor rejects a pending change; its author withdraws it."""
    with _db() as conn:
        v = _version(conn, version_id)
        scope = v["scope"]
        if v["status"] != "pending":
            raise KnowledgeError("This change isn't waiting for approval any more.", 409)
        is_author = v["author_id"] == user["id"]
        if not is_author:
            _require_edit(conn, user, scope)
        status = "withdrawn" if is_author else "rejected"
        conn.execute(
            "UPDATE brief_versions SET status = ?, decided_by = ?, decided_at = ?, note = ? WHERE id = ?",
            (status, user["email"], _now(), (note or "").strip() or None, version_id),
        )
        _log(conn, scope, f"brief_{status}", user["email"], version_id, (note or "").strip() or None)
        return dict(_version(conn, version_id))


def review_brief(user: dict, version_id: str, flag: bool, note: str | None = None) -> dict:
    """An Admin's after-the-fact review of a change the Owner applied
    without one. Flagging doesn't undo it; it marks it for attention, and
    any editor can restore an earlier approved version."""
    with _db() as conn:
        v = _version(conn, version_id)
        if v["status"] != "applied_unreviewed":
            raise KnowledgeError("That change doesn't need a review.", 409)
        if not _is_admin(user) or v["author_id"] == user["id"]:
            raise KnowledgeError("An Admin other than the author reviews the Owner's changes.", 403)
        status = "flagged" if flag else "reviewed"
        conn.execute(
            "UPDATE brief_versions SET status = ?, reviewed_by = ?, reviewed_at = ?, note = ? WHERE id = ?",
            (status, user["email"], _now(), (note or "").strip() or None, version_id),
        )
        _log(conn, v["scope"], f"brief_{status}", user["email"], version_id, (note or "").strip() or None)
        return dict(_version(conn, version_id))


def revert_brief(user: dict, target_version_id: str) -> dict:
    """Brings back an earlier version a second person approved. Immediate,
    because that text has already been through approval."""
    with _db() as conn:
        target = _version(conn, target_version_id)
        scope = target["scope"]
        _require_edit(conn, user, scope)
        if target["status"] not in _VOUCHED:
            raise KnowledgeError("Only a version someone approved can be restored straight away. "
                                 "Propose it as a change instead.", 409)
        current = _current(conn, scope)
        if current and current["id"] == target_version_id:
            raise KnowledgeError("That version is already the brief in force.", 409)
        now = _now()
        stale = _pending(conn, scope)
        if stale:
            conn.execute("UPDATE brief_versions SET status = 'superseded', decided_at = ? WHERE id = ?", (now, stale["id"]))
            _log(conn, scope, "brief_superseded", user["email"], stale["id"], "brief restored to an earlier version")
        vid = uuid.uuid4().hex
        conn.execute(
            "INSERT INTO brief_versions (id, scope, text, status, author_id, author_email, created_at, based_on, "
            "reverted_from, decided_by, decided_at, live_at) VALUES (?, ?, ?, 'approved', ?, ?, ?, ?, ?, ?, ?, ?)",
            (vid, scope, target["text"], user["id"], user["email"], now, current["id"] if current else None,
             target_version_id, user["email"], now, now),
        )
        _log(conn, scope, "brief_restored", user["email"], vid, f"restored version {target_version_id[:8]}")
        return dict(_version(conn, vid))


# ---- Library -------------------------------------------------------------------------

def _clean_entry(title: str, body: str) -> tuple[str, str]:
    title, body = (title or "").strip(), (body or "").strip()
    if not title or not body:
        raise KnowledgeError("An entry needs a title and some text.")
    if len(title) > TITLE_LIMIT:
        raise KnowledgeError(f"Keep the title under {TITLE_LIMIT} characters.")
    if len(body) > BODY_LIMIT:
        raise KnowledgeError(f"Keep an entry under {BODY_LIMIT} characters; split longer material into several.")
    return title, body


def _index(conn: sqlite3.Connection, entry: sqlite3.Row) -> None:
    if _fts:
        conn.execute(
            "INSERT INTO entries_fts (title, body, entry_id, scope) VALUES (?, ?, ?, ?)",
            (entry["title"], entry["body"], entry["id"], entry["scope"]),
        )


def _unindex(conn: sqlite3.Connection, entry_id: str) -> None:
    if _fts:
        conn.execute("DELETE FROM entries_fts WHERE entry_id = ?", (entry_id,))


def _entry(conn: sqlite3.Connection, entry_id: str) -> sqlite3.Row:
    row = conn.execute("SELECT * FROM knowledge_entries WHERE id = ?", (entry_id,)).fetchone()
    if not row:
        raise KnowledgeError("That entry doesn't exist.", 404)
    return row


def add_entry(user: dict, scope: str, title: str, body: str) -> dict:
    """An editor's entry goes live at once. Anyone else's is a suggestion
    an editor approves."""
    title, body = _clean_entry(title, body)
    with _db() as conn:
        _require_view(conn, user, scope)
        editor = _can_edit(conn, user, scope)
        eid = uuid.uuid4().hex
        conn.execute(
            "INSERT INTO knowledge_entries (id, scope, title, body, status, source, author_email, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (eid, scope, title, body, "active" if editor else "pending", "editor" if editor else "member",
             user["email"], _now()),
        )
        entry = _entry(conn, eid)
        if editor:
            _index(conn, entry)
        _log(conn, scope, "entry_added" if editor else "entry_suggested", user["email"], eid, title)
        return dict(entry)


def suggest_entry_from_chat(scope: str, title: str, body: str, conversation_id: str | None) -> dict:
    """NAVI's own suggestion, from a chat. Always waits for an editor."""
    title, body = _clean_entry(title, body)
    with _db() as conn:
        eid = uuid.uuid4().hex
        conn.execute(
            "INSERT INTO knowledge_entries (id, scope, title, body, status, source, author_email, conversation_id, created_at) "
            "VALUES (?, ?, ?, ?, 'pending', 'navi', 'NAVI', ?, ?)",
            (eid, scope, title, body, conversation_id, _now()),
        )
        _log(conn, scope, "entry_suggested", "NAVI", eid, title)
        return dict(_entry(conn, eid))


def decide_entry(user: dict, entry_id: str, approve: bool, note: str | None = None) -> dict:
    with _db() as conn:
        e = _entry(conn, entry_id)
        _require_edit(conn, user, e["scope"])
        if e["status"] != "pending":
            raise KnowledgeError("That suggestion has already been decided.", 409)
        status = "active" if approve else "rejected"
        conn.execute(
            "UPDATE knowledge_entries SET status = ?, decided_by = ?, decided_at = ?, note = ? WHERE id = ?",
            (status, user["email"], _now(), (note or "").strip() or None, entry_id),
        )
        e = _entry(conn, entry_id)
        if approve:
            _index(conn, e)
        _log(conn, e["scope"], "entry_approved" if approve else "entry_rejected", user["email"], entry_id, e["title"])
        return dict(e)


def retire_entry(user: dict, entry_id: str, reason: str | None = None) -> dict:
    """Takes an entry out of use. Kept, never deleted, so what NAVI used to
    know stays answerable."""
    with _db() as conn:
        e = _entry(conn, entry_id)
        _require_edit(conn, user, e["scope"])
        if e["status"] != "active":
            raise KnowledgeError("Only an entry in use can be retired.", 409)
        conn.execute(
            "UPDATE knowledge_entries SET status = 'retired', decided_by = ?, decided_at = ?, note = ? WHERE id = ?",
            (user["email"], _now(), (reason or "").strip() or None, entry_id),
        )
        _unindex(conn, entry_id)
        _log(conn, e["scope"], "entry_retired", user["email"], entry_id, (reason or "").strip() or e["title"])
        return dict(_entry(conn, entry_id))


def list_entries(user: dict, scope: str) -> dict:
    with _db() as conn:
        _require_view(conn, user, scope)
        editor = _can_edit(conn, user, scope)
        active = conn.execute(
            "SELECT * FROM knowledge_entries WHERE scope = ? AND status = 'active' ORDER BY created_at DESC", (scope,)
        ).fetchall()
        # Editors see every suggestion waiting; anyone else sees their own.
        pending = conn.execute(
            "SELECT * FROM knowledge_entries WHERE scope = ? AND status = 'pending' "
            + ("" if editor else "AND author_email = ? ") + "ORDER BY created_at DESC",
            (scope,) if editor else (scope, user["email"]),
        ).fetchall()
        return {"can_edit": editor, "active": [dict(r) for r in active], "pending": [dict(r) for r in pending]}


def _fts_query(text: str) -> str | None:
    words = [w for w in re.findall(r"\w+", (text or "").lower()) if len(w) > 1][:12]
    return " OR ".join(f'"{w}"' for w in words) if words else None


def search(scopes: list[str], query: str, limit: int = SEARCH_RESULTS) -> list[dict]:
    """Active entries in these scopes, best match first."""
    if not scopes:
        return []
    marks = ",".join("?" * len(scopes))
    with _db() as conn:
        if _fts:
            q = _fts_query(query)
            if not q:
                return []
            rows = conn.execute(
                f"SELECT e.* FROM entries_fts f JOIN knowledge_entries e ON e.id = f.entry_id "
                f"WHERE entries_fts MATCH ? AND f.scope IN ({marks}) AND e.status = 'active' "
                f"ORDER BY bm25(entries_fts) LIMIT ?",
                (q, *scopes, limit),
            ).fetchall()
        else:
            words = re.findall(r"\w+", (query or "").lower())[:6]
            if not words:
                return []
            clause = " OR ".join("(lower(title) LIKE ? OR lower(body) LIKE ?)" for _ in words)
            params = [p for w in words for p in (f"%{w}%", f"%{w}%")]
            rows = conn.execute(
                f"SELECT * FROM knowledge_entries WHERE status = 'active' AND scope IN ({marks}) AND ({clause}) "
                f"ORDER BY created_at DESC LIMIT ?",
                (*scopes, *params, limit),
            ).fetchall()
        return [dict(r) for r in rows]


# ---- History and what's waiting --------------------------------------------------------

def history(user: dict, scope: str, limit: int = 200) -> list[dict]:
    with _db() as conn:
        _require_view(conn, user, scope)
        rows = conn.execute(
            "SELECT seq, at, scope, action, actor, subject_id, detail FROM knowledge_events "
            "WHERE scope = ? ORDER BY seq DESC LIMIT ?",
            (scope, limit),
        ).fetchall()
        return [dict(r) for r in rows]


def waiting_for(user: dict) -> dict:
    """What this person can act on right now, per scope: brief changes to
    approve, suggestions to decide, Owner changes to review."""
    out: dict[str, int] = {}
    scopes = [COMPANY] + [project_scope(p["id"]) for p in list_projects(user)]
    with _db() as conn:
        for scope in scopes:
            if not _can_edit(conn, user, scope):
                continue
            n = conn.execute(
                "SELECT COUNT(*) FROM brief_versions WHERE scope = ? AND status = 'pending' AND author_id != ?",
                (scope, user["id"]),
            ).fetchone()[0]
            n += conn.execute(
                "SELECT COUNT(*) FROM knowledge_entries WHERE scope = ? AND status = 'pending'", (scope,)
            ).fetchone()[0]
            if _is_admin(user):
                n += conn.execute(
                    "SELECT COUNT(*) FROM brief_versions WHERE scope = ? AND status = 'applied_unreviewed' AND author_id != ?",
                    (scope, user["id"]),
                ).fetchone()[0]
            if n:
                out[scope] = n
    return out


# ---- What a chat reads ------------------------------------------------------------------

def brief_block(project_id: str | None) -> str:
    """The company brief, plus the project's when the chat is in one, as it
    goes into the system message. Empty when there's nothing to say."""
    parts = []
    company = current_brief_text(COMPANY)
    if company:
        parts.append(f"## Company brief\n{company}")
    if project_id:
        name = project_name(project_id)
        text = current_brief_text(project_scope(project_id)) if name else ""
        if name:
            parts.append(f"## Project: {name}\n{text or '(No brief written for this project yet.)'}")
    if not parts:
        return ""
    return (
        "\n\n".join(parts)
        + "\n\nThis is company knowledge, set by the people who run this company. It is authoritative on "
        "facts about the company and this project. If this conversation's own memory contradicts it, "
        "say so plainly instead of silently picking one."
    )
