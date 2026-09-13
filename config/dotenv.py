"""
config/dotenv.py

Loads the server's .env for code that systemd did not start.

WHY THIS EXISTS. The live server gets its secrets from systemd's
EnvironmentFile, which is not a shell script and does not reach an
interactive shell — systemd never exports into your session, and your
session never exports into the service. So `python -m jobs.something` run
by hand has none of the keys the running server has, even on the same
box, in the same directory.

That produced a genuinely bad failure (2026-09-13): a full benchmark ran
to completion with every chat call answered by LLM7, the second fallback,
because Gemini's key lives only in .env and Cloudflare's account id is
read straight from the environment, while LLM7's key happened to be in
the config database and so worked anywhere. Nothing errored. The numbers
were simply about the wrong models.

The documented workaround is `set -a && source .env && set +a`, which
works right up until the file is somewhere else, unreadable by the
current user, or was saved with Windows line endings — at which point it
fails quietly in a different way. A job that needs the environment should
just load it.

DELIBERATELY NEVER OVERRIDES an existing variable. A value already in the
environment was set by someone on purpose — systemd, a deploy script, or
a deliberate one-off override on the command line — and a file on disk
must not silently win over it. Same precedence as the config store over
env in config/store.py, one level further out.
"""

from __future__ import annotations

import os
from pathlib import Path

# The project root: this file lives at <root>/config/dotenv.py.
DEFAULT_ENV_PATH = Path(__file__).resolve().parent.parent / ".env"


def load_dotenv(path: Path | None = None, override: bool = False) -> dict[str, str]:
    """Reads a KEY=value file into os.environ. Returns what it set.

    Parses the same shape systemd's EnvironmentFile accepts, plus the
    `export` prefix that shell-sourced files often carry — the two formats
    overlap in practice and a file written for one is usually fed to the
    other.

    Never raises. A missing or unreadable .env is normal: on the live
    server systemd has already supplied everything, and there may be no
    file to read at all.
    """
    path = path or DEFAULT_ENV_PATH
    try:
        raw = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return {}

    loaded: dict[str, str] = {}
    for line in raw.splitlines():
        # Strips a trailing \r too, which is the whole reason this is
        # worth writing rather than shelling out: a .env saved from
        # Windows leaves one on every value, and a key with a carriage
        # return in it is accepted everywhere and rejected by the API.
        line = line.strip().lstrip("﻿")
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export "):]
        key, _, value = line.partition("=")
        key = key.strip()
        if not key:
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]
        if not override and os.environ.get(key):
            continue
        os.environ[key] = value
        loaded[key] = value
    return loaded


def ensure_env(path: Path | None = None) -> Path | None:
    """Loads .env and returns the file it used, or None if there was
    nothing to load. Call this at the top of any job meant to be run by
    hand against the live configuration."""
    path = path or DEFAULT_ENV_PATH
    loaded = load_dotenv(path)
    return path if loaded else None
