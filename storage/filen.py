"""
storage/filen.py

Wraps rclone (already configured against the "filen" remote — see the manual
setup we did earlier) to save finished results. Render's disk is scratch-only
(no persistent disk on the free tier), so nothing should be treated as
"saved" until it's actually landed here.

Folder structure (decided earlier in the session):
    filen:<command>/<date>/<topic-slug>/<HH-MM-SS>_<filename>

e.g. filen:research/2026-08-16/larian-studios-hiring/14-32-07_report.md

The time prefix matters: topic_slug is deterministic (see
dispatcher/slugify.py — same first-5-meaningful-words always produces the
same slug), so two separate runs of the same command on the same day
with similar-enough phrasing land in the exact same folder. Without a
per-run-unique filename, the second run silently overwrites the first's
saved artifact — a real bug hit in practice (a re-run's research.md and
gathered.md clobbered the previous run's). Fixed centrally here rather
than in each command, since every command that saves a result shares
this same path-building logic and was equally exposed.
"""

import subprocess
import tempfile
from datetime import date, datetime
from pathlib import Path

RCLONE_REMOTE = "filen"

# Maps each command to its top-level Filen folder. create-image maps to
# "images" per the earlier folder-naming decision (command name doesn't
# match folder name 1:1 for that one case).
FOLDER_FOR_COMMAND = {
    "research": "research",
    "code": "code",
    "graph-data": "graph-data",
    "create-image": "images",
    "brainstorm": "brainstorm",
}


class StorageError(Exception):
    pass


def _rclone_path(command: str, topic_slug: str, filename: str) -> str:
    folder = FOLDER_FOR_COMMAND.get(command, command)
    today = date.today().isoformat()
    time_prefix = datetime.now().strftime("%H-%M-%S")
    return f"{RCLONE_REMOTE}:{folder}/{today}/{topic_slug}/{time_prefix}_{filename}"


def save_result(command: str, topic_slug: str, filename: str, content: str) -> str:
    """
    Writes `content` to a temp file, then rclone-copies it to the correct
    Filen path. Returns the remote path it was saved to.

    Raises StorageError on failure — callers should treat a save failure as
    real (don't tell the user something is "saved" when it isn't).
    """
    remote_path = _rclone_path(command, topic_slug, filename)

    with tempfile.NamedTemporaryFile(mode="w", suffix=f"_{filename}", delete=False) as f:
        f.write(content)
        tmp_path = f.name

    try:
        try:
            result = subprocess.run(
                ["rclone", "copyto", tmp_path, remote_path],
                capture_output=True, text=True, timeout=60,
            )
        except OSError as e:
            raise StorageError(f"rclone not available: {e}")
        if result.returncode != 0:
            raise StorageError(f"rclone failed: {result.stderr.strip()}")
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    return remote_path


def save_bytes(command: str, topic_slug: str, filename: str, content: bytes) -> str:
    """Same as save_result, but for binary content (e.g. a rendered chart
    PNG) rather than text."""
    remote_path = _rclone_path(command, topic_slug, filename)

    with tempfile.NamedTemporaryFile(mode="wb", suffix=f"_{filename}", delete=False) as f:
        f.write(content)
        tmp_path = f.name

    try:
        try:
            result = subprocess.run(
                ["rclone", "copyto", tmp_path, remote_path],
                capture_output=True, text=True, timeout=60,
            )
        except OSError as e:
            raise StorageError(f"rclone not available: {e}")
        if result.returncode != 0:
            raise StorageError(f"rclone failed: {result.stderr.strip()}")
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    return remote_path


def save_file(command: str, topic_slug: str, local_path: str, filename: str | None = None) -> str:
    """Same as save_result, but for a file that already exists on disk (e.g.
    an image, a generated code file already written locally)."""
    filename = filename or Path(local_path).name
    remote_path = _rclone_path(command, topic_slug, filename)

    try:
        result = subprocess.run(
            ["rclone", "copyto", local_path, remote_path],
            capture_output=True, text=True, timeout=120,
        )
    except OSError as e:
        raise StorageError(f"rclone not available: {e}")
    if result.returncode != 0:
        raise StorageError(f"rclone failed: {result.stderr.strip()}")

    return remote_path


def download_for_reply(remote_path: str) -> bytes:
    """Pulls a file back down from Filen so it can be attached to a Telegram/
    Discord reply directly — no public link needed, per the earlier decision."""
    with tempfile.NamedTemporaryFile(delete=False) as f:
        tmp_path = f.name

    try:
        try:
            result = subprocess.run(
                ["rclone", "copyto", remote_path, tmp_path],
                capture_output=True, text=True, timeout=60,
            )
        except OSError as e:
            raise StorageError(f"rclone not available: {e}")
        if result.returncode != 0:
            raise StorageError(f"rclone download failed: {result.stderr.strip()}")
        return Path(tmp_path).read_bytes()
    finally:
        Path(tmp_path).unlink(missing_ok=True)
