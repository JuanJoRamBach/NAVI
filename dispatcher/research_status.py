"""
dispatcher/research_status.py

A single in-memory "what's /research doing right now" string, polled by
the PWA while an async research job runs in the background (see
server.py's /research/status route and the background-thread dispatch in
/chat/send). Deliberately just one global slot, not per-job tracking —
NAVI is single-user, so at most one research job is ever meaningfully
"the one you're waiting on" at a time. Not persisted: a Render restart
mid-job would lose the status text, which is fine, the job itself would
also be gone.
"""

import threading

_lock = threading.Lock()
_status: str | None = None


def set_status(text: str | None) -> None:
    global _status
    with _lock:
        _status = text


def get_status() -> str | None:
    with _lock:
        return _status
