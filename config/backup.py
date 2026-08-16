"""
config/backup.py

Fixes the gap flagged in config/store.py's own docstring: Render's free
tier has no persistent disk, so agent_config.json gets wiped on every
restart/redeploy unless something restores it from Filen at startup and
keeps pushing it back up on every write.

Uses the same rclone-subprocess approach as storage/filen.py, but a fixed
flat path (not the dated command/topic-slug structure results use) since
this is a single file that should always overwrite in place — a "current
config", not a dated archive.
"""

import subprocess
from pathlib import Path

RCLONE_REMOTE = "filen"
BACKUP_PATH = f"{RCLONE_REMOTE}:navi-config/agent_config.json"


class BackupError(Exception):
    pass


def restore_from_backup(local_path: Path) -> bool:
    """
    Called at startup, before the local config is read. If a backup exists
    on Filen, pulls it down to local_path. Returns True if a restore
    happened, False if there was nothing to restore (first run ever, or
    Filen/rclone unavailable) — either way, ConfigStore falls back to
    DEFAULTS, so this is best-effort, not fatal.
    """
    try:
        result = subprocess.run(
            ["rclone", "copyto", BACKUP_PATH, str(local_path)],
            capture_output=True, text=True, timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return False  # rclone not installed/reachable — proceed with DEFAULTS

    return result.returncode == 0 and local_path.exists()


def backup_to_filen(local_path: Path) -> None:
    """
    Called after every config write. Best-effort: a failed backup doesn't
    raise into the caller's write path (config/store.py callers shouldn't
    have to handle a Filen outage just to set an API key) but IS raised as
    BackupError for callers that want to surface it, e.g. a chat reply
    disclosing "key saved locally, but backup to Filen failed."
    """
    try:
        result = subprocess.run(
            ["rclone", "copyto", str(local_path), BACKUP_PATH],
            capture_output=True, text=True, timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as e:
        raise BackupError(f"rclone backup failed: {e}")

    if result.returncode != 0:
        raise BackupError(f"rclone backup failed: {result.stderr.strip()}")
