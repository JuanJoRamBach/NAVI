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
        "llm7": {"api_key": None, "enabled": True},
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
        # model digest, daily opportunity scan). Kept off the chat quota so
        # a chatty day never eats into what the autonomous jobs need, or
        # vice versa. Uses gpt-oss-120b (2026-08-17) rather than the
        # deprecated llama-3.1-8b-instant: each job runs once/day, so the
        # smaller 1,000 RPD ceiling is irrelevant capacity-wise, and the
        # opportunity-scan job specifically benefits from gpt-oss-120b's
        # stronger instruction-following (it has to reliably refuse to
        # speculate, not just summarize).
        #
        # Current free-tier numbers (2026-08-17, verify in Groq console —
        # these move):
        #   openai/gpt-oss-20b:    30 RPM / 1,000 RPD / 200K TPD
        #   openai/gpt-oss-120b:   30 RPM / 1,000 RPD / 200K TPD
        #   llama-3.1-8b-instant:  30 RPM / 14,400 RPD / 500K TPD (deprecated)
        # Tried DeepSeek-V4-Flash-0731 via LLM7 (2026-08-26) — benchmarks
        # clearly ahead of gpt-oss-20b/120b on paper (52 vs 24 Artificial
        # Analysis Intelligence Index, 1M context vs 130k), but real
        # traffic that same day hit rate-limit + 503 "model temporarily
        # busy" three times in a row. dispatcher_chat has no fallback on
        # purpose (see above) specifically so a bad model choice here is
        # felt immediately rather than silently degrading — reverted to
        # gpt-oss-120b (proven-reliable Groq, and a real upgrade over the
        # original 20b) rather than leave chat flaky. LLM7/DeepSeek stays
        # registered as a provider — worth revisiting for a role with
        # fallback room (task_routing), just not this one.
        "dispatcher_chat": {"provider": "groq", "model": "openai/gpt-oss-120b"},
        "dispatcher_autonomous": {"provider": "groq", "model": "openai/gpt-oss-120b"},
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
            # Verified directly against Cloudflare Workers AI on 2026-08-17 —
            # real coding-specialist model, real 200 response, 2.7 Neurons for
            # the test call (10,000/day free). No fallback yet since this is
            # the first real thing wired in for /code.
            "primary": {"provider": "cloudflare", "model": "@cf/qwen/qwen2.5-coder-32b-instruct"},
            "fallback": [],
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
        "summarize": {
            # Own quota bucket, same reasoning as dispatcher_autonomous vs
            # dispatcher_chat: a plain digest call is frequent and cheap
            # enough that it shouldn't compete with dispatcher_chat's
            # (no-fallback, felt-immediately) quota. gpt-oss-20b rather
            # than 120b — summarization doesn't need the bigger model.
            "primary": {"provider": "groq", "model": "openai/gpt-oss-20b"},
            "fallback": [],
        },
        "recap": {
            # Same reasoning as /summarize's routing — own quota, small
            # model, single-phase call.
            "primary": {"provider": "groq", "model": "openai/gpt-oss-20b"},
            "fallback": [],
        },
        "note": {
            "primary": {"provider": "groq", "model": "openai/gpt-oss-20b"},
            "fallback": [],
        },
        "remind": {
            # Needs a forced tool_choice call (same requirement as
            # graph-data's render_chart) — reusing the same openrouter
            # models already verified to support that.
            "primary": {"provider": "openrouter", "model": "nvidia/nemotron-3.5-lightning:free"},
            "fallback": [{"provider": "openrouter", "model": "nvidia/nemotron-3-ultra-550b-a55b:free"}],
        },
        "tailor": {
            "primary": {"provider": "groq", "model": "openai/gpt-oss-20b"},
            "fallback": [],
        },
        "design-read": {
            # The only vision-capable model in the current roster — LLM7's
            # turbo-tier gemini-3.1-flash-lite. Nothing else configured
            # (Groq/OpenRouter/Cloudflare/Ollama Cloud roster here) does
            # vision, so no fallback chain yet.
            "primary": {"provider": "llm7", "model": "gemini-3.1-flash-lite"},
            "fallback": [],
        },
        # No "brainstorm" entry — retired as a standalone command (2026-08-27):
        # Brainstorm mode's own conversational chat (dispatcher/modes/
        # BRAINSTORM.md) does its job better, since the command was a
        # one-shot fire with no continuity while the mode explicitly
        # supports iterating on ideas across turns.
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


def _migrate_dispatcher_chat_to_llm7():
    """
    One-time switch (2026-08-26) for instances that already had
    dispatcher_chat persisted (via Filen backup) before DEFAULTS changed
    above — editing DEFAULTS alone only affects a brand new store, not
    one already materialized on disk. Guarded so it only runs once; if
    dispatcher_chat gets manually reassigned later, this won't fight it.
    """
    if config.get("migrated_dispatcher_chat_to_llm7"):
        return
    config.set_role("dispatcher_chat", "llm7", "DeepSeek-V4-Flash-0731")
    config.set("migrated_dispatcher_chat_to_llm7", True)


_migrate_dispatcher_chat_to_llm7()


def _migrate_dispatcher_chat_off_llm7():
    """
    One-time revert (same day, 2026-08-26) — the LLM7 switch above hit
    rate-limit + repeated 503 "model temporarily busy" under real traffic
    within hours of shipping. dispatcher_chat has no fallback on purpose,
    so this needs to be felt and fixed immediately, not left flaky. Own
    guard flag, separate from the migration above, so both stay in the
    history and this doesn't just silently undo it for instances that
    never got the LLM7 migration in the first place.
    """
    if config.get("migrated_dispatcher_chat_off_llm7"):
        return
    config.set_role("dispatcher_chat", "groq", "openai/gpt-oss-120b")
    config.set("migrated_dispatcher_chat_off_llm7", True)


_migrate_dispatcher_chat_off_llm7()


def _migrate_add_summarize_routing():
    """
    One-time addition (2026-08-26) for instances whose config.json already
    existed before /summarize's task_routing entry was added to DEFAULTS —
    editing DEFAULTS alone only materializes for a brand-new store.
    """
    if config.get("migrated_add_summarize_routing"):
        return
    if not config.get_task_routing("summarize"):
        config.set_task_routing(
            "summarize", {"provider": "groq", "model": "openai/gpt-oss-20b"}, [],
        )
    config.set("migrated_add_summarize_routing", True)


_migrate_add_summarize_routing()


def _migrate_add_recap_note_routing():
    """One-time addition (2026-08-26) — same reasoning as
    _migrate_add_summarize_routing, for /recap and /note."""
    if config.get("migrated_add_recap_note_routing"):
        return
    if not config.get_task_routing("recap"):
        config.set_task_routing("recap", {"provider": "groq", "model": "openai/gpt-oss-20b"}, [])
    if not config.get_task_routing("note"):
        config.set_task_routing("note", {"provider": "groq", "model": "openai/gpt-oss-20b"}, [])
    config.set("migrated_add_recap_note_routing", True)


_migrate_add_recap_note_routing()


def _migrate_add_remind_routing():
    """One-time addition (2026-08-26) — same reasoning as
    _migrate_add_summarize_routing, for /remind."""
    if config.get("migrated_add_remind_routing"):
        return
    if not config.get_task_routing("remind"):
        config.set_task_routing(
            "remind",
            {"provider": "openrouter", "model": "nvidia/nemotron-3.5-lightning:free"},
            [{"provider": "openrouter", "model": "nvidia/nemotron-3-ultra-550b-a55b:free"}],
        )
    config.set("migrated_add_remind_routing", True)


_migrate_add_remind_routing()


def _migrate_add_tailor_routing():
    """One-time addition (2026-08-26) — same reasoning as
    _migrate_add_summarize_routing, for /tailor."""
    if config.get("migrated_add_tailor_routing"):
        return
    if not config.get_task_routing("tailor"):
        config.set_task_routing("tailor", {"provider": "groq", "model": "openai/gpt-oss-20b"}, [])
    config.set("migrated_add_tailor_routing", True)


_migrate_add_tailor_routing()


def _migrate_add_design_read_routing():
    """One-time addition (2026-08-26) — same reasoning as
    _migrate_add_summarize_routing, for /design-read."""
    if config.get("migrated_add_design_read_routing"):
        return
    if not config.get_task_routing("design-read"):
        config.set_task_routing("design-read", {"provider": "llm7", "model": "gemini-3.1-flash-lite"}, [])
    config.set("migrated_add_design_read_routing", True)


_migrate_add_design_read_routing()
