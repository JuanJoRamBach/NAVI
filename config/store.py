"""
config/store.py

A small persistent JSON store for everything that used to live in env vars:
provider API keys, which model is pinned to which role, and misc settings.

Why this exists: on Render, changing an env var means a redeploy. That's the
exact friction JuanJo wanted to avoid — "here's my Groq key" typed into chat
should just work, not require a dashboard trip. So this store lives on disk
(fine, since Render's disk is scratch-only anyway — this file gets rewritten
whenever a value changes, and if the service restarts, the config should be
re-synced from Filen at startup, see restore_from_backup below).
"""

import json
import threading
from pathlib import Path
from typing import Any

from config.backup import BackupError, backup_to_filen, restore_from_backup

CONFIG_PATH = Path(__file__).parent / "agent_config.json"

_lock = threading.Lock()

DEFAULTS = {
    "providers": {
        # name -> { "api_key": str | None, "enabled": bool }
        "openrouter": {"api_key": None, "enabled": True},
        "groq": {"api_key": None, "enabled": True},
        "nvidia_nim": {"api_key": None, "enabled": False},
    },
    "roles": {
        # Two SEPARATE dispatcher roles, deliberately not sharing a quota bucket.
        #
        # dispatcher_chat: handles live, interactive messages (whatever you
        # type). Pinned to the officially-supported model only — no fallback
        # to the deprecated one, since this is the role you'll actually
        # notice and feel if something's wrong.
        #
        # dispatcher_autonomous: handles the two GitHub Actions jobs (daily
        # model digest, daily opportunity scan). Deliberately uses the
        # deprecated-but-generous model instead — low stakes if it breaks
        # (you just miss a day's digest), and keeping it off the chat
        # quota means a chatty day never eats into what the autonomous
        # jobs need, or vice versa.
        #
        # Current free-tier numbers (2026-08-16, verify in Groq console —
        # these move):
        #   openai/gpt-oss-20b:    30 RPM / 1,000 RPD / 200K TPD
        #   llama-3.1-8b-instant:  30 RPM / 14,400 RPD / 500K TPD (deprecated)
        "dispatcher_chat": {"provider": "groq", "model": "openai/gpt-oss-20b"},
        "dispatcher_autonomous": {"provider": "groq", "model": "llama-3.1-8b-instant"},
    },
    "task_routing": {
        # Placeholder routing per command until the live daily-ranked model
        # list (pulled from ClawLabs' free-model feed) is wired in. Each
        # command gets a primary (provider, model) and an ordered fallback
        # chain. When one fails, the executor rotates through the fallback
        # list and flags the step as degraded in the final reply.
        "research": {
            # Staged trial on Ollama Cloud (2026-08-17): minimax-m3:cloud this
            # week, then gemma4:31b-cloud — swap the model string below to
            # rotate. deepseek-v4-flash:cloud was the original third candidate
            # but returned 403 "requires a subscription" on this account when
            # tested directly, so it's dropped from the trial entirely, not
            # just deprioritized. Picking a permanent winner once real usage
            # (via Ollama's /api/usage) shows which one actually holds up.
            # No fallback configured on purpose: a failure here IS the signal
            # we're trying to observe, not something to mask behind an
            # equally-untested backup.
            "primary": {"provider": "ollama_cloud", "model": "minimax-m3:cloud"},
            "fallback": [],
        },
        "code": {
            "primary": {"provider": "openrouter", "model": "openai/gpt-oss-120b:free"},
            "fallback": [{"provider": "openrouter", "model": "meta-llama/llama-3.3-70b-instruct:free"}],
        },
        "graph-data": {
            # Verified against OpenRouter's live /api/v1/models on 2026-08-17 —
            # free pricing AND supports the "tools" param (required for the
            # forced render_chart call). The two originally hardcoded here
            # went stale within the same day they were written — exactly
            # the churn problem the daily-ranking job is meant to solve.
            "primary": {"provider": "openrouter", "model": "nvidia/nemotron-3.5-lightning:free"},
            "fallback": [{"provider": "openrouter", "model": "nvidia/nemotron-3-ultra-550b-a55b:free"}],
        },
        "create-image": {
            "primary": {"provider": "openrouter", "model": None},  # filled in only on days one's free
            "fallback": [],
        },
        "brainstorm": {
            "primary": {"provider": "openrouter", "model": "openai/gpt-oss-120b:free"},
            "fallback": [{"provider": "openrouter", "model": "meta-llama/llama-3.3-70b-instruct:free"}],
        },
    },
    "storage": {
        "filen_configured": False,
    },
}


class ConfigStore:
    def __init__(self, path: Path = CONFIG_PATH):
        self.path = path
        self.last_backup_error: str | None = None
        self._data = self._load()

    def _load(self) -> dict:
        if not self.path.exists():
            # Local file is gone (fresh clone, or a Render restart on the
            # free tier's scratch-only disk) — try Filen before giving up
            # to DEFAULTS, so a restart doesn't silently forget every key
            # and routing choice that was configured via chat.
            restore_from_backup(self.path)

        if self.path.exists():
            try:
                return json.loads(self.path.read_text())
            except json.JSONDecodeError:
                pass
        return json.loads(json.dumps(DEFAULTS))  # deep copy of defaults

    def _save(self):
        with _lock:
            self.path.write_text(json.dumps(self._data, indent=2))
        try:
            backup_to_filen(self.path)
            self.last_backup_error = None
        except BackupError as e:
            # Local write already succeeded — don't raise into callers that
            # just wanted to save a key. Stash the error so a chat reply
            # can disclose "saved, but Filen backup failed" if it wants to.
            self.last_backup_error = str(e)

    # ---- Provider keys ----

    def set_provider_key(self, provider: str, api_key: str):
        """Called when the user sends the agent a new API key in chat."""
        self._data.setdefault("providers", {}).setdefault(provider, {})
        self._data["providers"][provider]["api_key"] = api_key
        self._data["providers"][provider]["enabled"] = True
        self._save()

    def get_provider_key(self, provider: str) -> str | None:
        return self._data.get("providers", {}).get(provider, {}).get("api_key")

    def is_provider_enabled(self, provider: str) -> bool:
        return self._data.get("providers", {}).get(provider, {}).get("enabled", False)

    def enabled_providers(self) -> list[str]:
        return [
            name for name, cfg in self._data.get("providers", {}).items()
            if cfg.get("enabled") and cfg.get("api_key")
        ]

    # ---- Fixed roles (e.g. dispatcher) ----

    def get_role(self, role: str) -> dict | None:
        return self._data.get("roles", {}).get(role)

    def set_role(self, role: str, provider: str, model: str):
        self._data.setdefault("roles", {})[role] = {"provider": provider, "model": model}
        self._save()

    # ---- Task routing (per-command primary/fallback) ----

    def get_task_routing(self, command: str) -> dict | None:
        return self._data.get("task_routing", {}).get(command)

    def set_task_routing(self, command: str, primary: dict, fallback: list[dict]):
        self._data.setdefault("task_routing", {})[command] = {
            "primary": primary, "fallback": fallback,
        }
        self._save()

    # ---- Generic get/set for anything else ----

    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)

    def set(self, key: str, value: Any):
        self._data[key] = value
        self._save()

    def as_dict(self) -> dict:
        return json.loads(json.dumps(self._data))


# Module-level singleton — the rest of the app imports this directly.
config = ConfigStore()
