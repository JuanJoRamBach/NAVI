"""
dispatcher/reminders.py

Persisted state for /remind. Piggybacks on config/store.py's existing
Filen-backed persistence (a plain "reminders" key via ConfigStore's
generic get/set) rather than building a separate storage path — every
write already gets backed up to Filen and restored on restart for free.

Reminders are checked and delivered by the live Render server (see
server.py's /reminders/check, hit periodically by a GitHub Actions cron —
see .github/workflows/check_reminders.yml) rather than a standalone job
script, so this module never needs its own Filen round-trip: it just
reads/writes the config singleton that's already loaded in-process.
"""

import uuid
from datetime import datetime, timezone

from config.store import config


def add_reminder(fire_at: datetime, message: str) -> dict:
    reminder = {
        "id": str(uuid.uuid4()),
        "fire_at": fire_at.astimezone(timezone.utc).isoformat(),
        "message": message,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "delivered": False,
    }
    reminders = config.get("reminders", [])
    reminders.append(reminder)
    config.set("reminders", reminders)
    return reminder


def due_reminders() -> list[dict]:
    """Undelivered reminders whose fire_at has already passed."""
    now = datetime.now(timezone.utc)
    due = []
    for r in config.get("reminders", []):
        if r.get("delivered"):
            continue
        try:
            fire_at = datetime.fromisoformat(r["fire_at"])
        except (KeyError, ValueError):
            continue
        if fire_at <= now:
            due.append(r)
    return due


def mark_delivered(reminder_id: str) -> None:
    reminders = config.get("reminders", [])
    for r in reminders:
        if r.get("id") == reminder_id:
            r["delivered"] = True
    config.set("reminders", reminders)
