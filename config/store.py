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

import base64
import hashlib
import json
import os
import threading
from pathlib import Path
from typing import Any

from config.backup import BackupError, backup_to_filen, restore_from_backup

CONFIG_PATH = Path(__file__).parent / "agent_config.json"
_MCP_KEY_PATH = Path(__file__).parent / ".mcp_secret.key"

_lock = threading.Lock()
_fernet_cache = None


def _get_fernet():
    """Lazily builds the Fernet cipher used to encrypt MCP connection
    credentials (bearer tokens) at rest — these are third-party secrets
    (GitHub/Google/Slack/etc tokens) NAVI holds on the company's behalf,
    a different risk class than the plaintext-JSON pattern the rest of
    this store already uses for its own provider keys (that's a known,
    separately-flagged gap — see IDEAS.md — not fixed here).

    Prefers NAVI_MCP_SECRET_KEY (set once on Lightsail, never written to
    disk or backed up to Filen) over a locally-generated key file — a key
    that lives on the same disk as the data it encrypts only protects
    against a narrower set of leaks (e.g. the Filen backup, since
    backup_to_filen() ships this file's plaintext bytes as-is), which is
    still real protection, just not as strong as a key kept elsewhere."""
    global _fernet_cache
    if _fernet_cache is not None:
        return _fernet_cache
    from cryptography.fernet import Fernet

    key_env = os.environ.get("NAVI_MCP_SECRET_KEY")
    if key_env:
        try:
            _fernet_cache = Fernet(key_env.encode())
        except Exception:
            # Accept a human-typed passphrase, not just a properly-formed
            # Fernet key — derive a valid 32-byte urlsafe-base64 key from it.
            digest = hashlib.sha256(key_env.encode()).digest()
            _fernet_cache = Fernet(base64.urlsafe_b64encode(digest))
        return _fernet_cache

    if _MCP_KEY_PATH.exists():
        _fernet_cache = Fernet(_MCP_KEY_PATH.read_bytes())
        return _fernet_cache

    key = Fernet.generate_key()
    _MCP_KEY_PATH.write_bytes(key)
    print(
        f"[config.store] WARNING: NAVI_MCP_SECRET_KEY not set — generated a local key file "
        f"at {_MCP_KEY_PATH}. Set the env var in production so the key isn't stored on the "
        f"same disk (and same Filen backup) as the data it protects."
    )
    _fernet_cache = Fernet(key)
    return _fernet_cache


def _encrypt_secret(value: str | None) -> str | None:
    if value is None:
        return None
    return _get_fernet().encrypt(value.encode()).decode()


def _decrypt_secret(value: str | None) -> str | None:
    if value is None:
        return None
    from cryptography.fernet import InvalidToken

    try:
        return _get_fernet().decrypt(value.encode()).decode()
    except InvalidToken:
        # Either the wrong/rotated key, or a value saved before this
        # encryption existed — treat as legacy plaintext rather than
        # silently breaking every connection saved before this change.
        return value

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
        # busy" three times in a row. Reverted to gpt-oss-120b (proven-
        # reliable Groq, and a real upgrade over the original 20b) rather
        # than leave chat flaky on a *different* model.
        #
        # 2026-08-28: hit a real "Groq rate limited" hard failure live —
        # Groq's own status page showed no outage, so this was ordinary
        # free-tier rate-limiting (30 RPM / 1,000 RPD on gpt-oss-120b),
        # not a broken model choice. Added a same-model fallback to
        # LLM7's "gpt-oss" (same family, 131k context, tools+reasoning,
        # 100% recent availability, free-tier covered) — this is
        # deliberately NOT the earlier "different model, might silently
        # degrade" risk, just a second door to the same room when Groq's
        # free tier is briefly saturated.
        # Renamed from "dispatcher_chat" to "normal_chat" (2026-09-01) —
        # "dispatcher" in the name was confusing once NAVI had multiple
        # named chat modes (Normal/Research/Brainstorm/Plan); this role
        # specifically backs Normal Chat, which is what the name should
        # say. dispatcher_autonomous keeps its name — it isn't a chat
        # mode, it backs the two GitHub Actions jobs, "dispatcher" still
        # fits there.
        # Groq/OpenRouter deliberately excluded, primary or fallback
        # (2026-09-01) — same reasoning dev_slate_chat below already
        # established: Groq's free tier caps at 8K tokens/minute, and
        # now that run_stored_mode_chat (dispatcher/chat.py) replays a
        # real 20-message window every turn instead of one bare message,
        # that cap is genuinely reachable, not theoretical. LLM7
        # (already proven as this role's prior fallback) promoted to
        # primary; Mistral's free tier is far more generous on tokens/
        # minute (~500K TPM) and mistral-small-latest is already a known
        # free-tier general-purpose model in this repo (see
        # jobs/model_ranking.py's free-tier prefix list) — not
        # code-specific like Codestral, which is why dev_slate_chat's own
        # fallback isn't reused here as-is.
        # Fallback chain widened 2026-09-04 (JuanJo): Cloudflare added as a
        # second door before Mistral — llama-3.1-8b-instruct-fp8-fast has
        # real native tool-calling support and is cheap (4,119/34,868
        # neurons per M in/out — a typical exchange costs ~50 neurons, well
        # inside the shared 10,000/day budget alongside dev_slate_chat's
        # coding role). No prior reason ruled Cloudflare out here — it
        # just hadn't been considered when this role was first wired.
        "normal_chat": {
            "provider": "llm7", "model": "gpt-oss",
            "fallback": [
                {"provider": "cloudflare", "model": "@cf/meta/llama-3.1-8b-instruct-fp8-fast"},
                {"provider": "mistral", "model": "mistral-small-latest"},
            ],
        },
        "dispatcher_autonomous": {"provider": "groq", "model": "openai/gpt-oss-120b"},
        # dev_slate_chat: backs Dev Slate's own chat (dispatcher/devslate_chat.py),
        # a separate role from normal_chat since it needs a genuinely
        # coding-capable model, not whatever's cheapest for everyday
        # questions. Cloudflare's qwen2.5-coder, with Mistral's Codestral
        # as fallback — the only coding-model role left in this store
        # since /code (a separate, redundant one-shot command) was
        # retired 2026-09-04. Deliberately NO Groq anywhere in this role, primary
        # or fallback: Groq's free tier caps at 8K tokens/minute, and
        # Dev Slate's baseline turn (mode brief + task-state block + real
        # conversation history) realistically exceeds that before any
        # file content even enters the picture — the same reasoning that
        # already ruled Groq out for Plan Chat's whole-conversation calls.
        "dev_slate_chat": {
            "provider": "cloudflare", "model": "@cf/qwen/qwen2.5-coder-32b-instruct",
            "fallback": [{"provider": "mistral", "model": "codestral-latest"}],
        },
        # agent_work: backs each node of an Agent Work workflow run
        # (dispatcher/agent_work.py) AND Agent Work's own chat
        # (run_stored_mode_chat). Moved BACK onto Groq's gpt-oss-120b as
        # primary (2026-09-04, JuanJo) — the reasoning that excluded Groq
        # here on 2026-09-01 (a real 20-message history replay blowing
        # past Groq's 8K tokens/minute cap) no longer applies: agent_work
        # went deliberately stateless on 2026-09-03 (see dispatcher/
        # chat.py's own comment, "agent_work is deliberately stateless"),
        # so there's no big replayed window here anymore, just one bare
        # prompt per node/turn. gpt-oss-120b over 20b since agent_work's
        # job (planning workflow steps, deciding branches) benefits from
        # the bigger model, and both share the same 30 RPM/1,000 RPD/
        # 200K TPD free-tier ceiling — no quota cost to picking the
        # stronger one. Fallback: same Cloudflare model as normal_chat's
        # fallback (see that role's comment), then Mistral's Ministral as
        # a second door.
        "agent_work": {
            "provider": "groq", "model": "openai/gpt-oss-120b",
            "fallback": [
                {"provider": "cloudflare", "model": "@cf/meta/llama-3.1-8b-instruct-fp8-fast"},
                {"provider": "mistral", "model": "ministral-8b-latest"},
            ],
        },
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
        "graph-data": {
            # Verified against OpenRouter's live /api/v1/models on 2026-08-17 —
            # free pricing AND supports the "tools" param (required for the
            # forced render_chart call). The two originally hardcoded here
            # went stale within the same day they were written — exactly
            # the churn problem the daily-ranking job is meant to solve.
            "primary": {"provider": "openrouter", "model": "nvidia/nemotron-3.5-lightning:free"},
            "fallback": [{"provider": "openrouter", "model": "nvidia/nemotron-3-ultra-550b-a55b:free"}],
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
        # No "brainstorm" entry — retired as a standalone command (2026-08-27):
        # Brainstorm mode's own conversational chat (dispatcher/modes/
        # BRAINSTORM.md) does its job better, since the command was a
        # one-shot fire with no continuity while the mode explicitly
        # supports iterating on ideas across turns.
    },
    "storage": {
        "filen_configured": False,
    },
    # MCP connections — server_name -> connection config + per-tool
    # security baseline. Separate from "providers" above on purpose:
    # provider keys authenticate NAVI's own LLM calls, mcp_connections
    # authenticate NAVI's dispatcher to a real third-party system on the
    # company's behalf (see dispatcher/mcp_client.py's own docstring for
    # the full security model — rug-pull hash pinning, read/write
    # classification, tool poisoning defense).
    #
    # server_name -> {
    #   "transport": "stdio" | "http",
    #   "command": str | None, "args": list[str] | None,  # stdio only
    #   "url": str | None,                                 # http only
    #   "auth_header": str | None,   # e.g. "Bearer <token>" — encrypted at
    #                                 # rest (see _encrypt_secret above),
    #                                 # decrypted only by get_mcp_connection
    #   "connected": bool,
    #   "tools": {
    #     tool_name: {
    #       "hash": str,             # sha256 of name+description+inputSchema at approval time
    #       "description": str,      # sanitized, pinned at approval time — schema-building
    #       "input_schema": dict,    # reads this instead of re-fetching live every call
    #       "read_only": bool,       # NAVI's own effective classification, seeded from
    #                                 # the server's readOnlyHint but not blindly trusted after
    #       "destructive": bool,     # seeded from destructiveHint (MCP spec default: True
    #                                 # for anything not read-only) — the ONE tier that still
    #                                 # needs per-call confirmation; read-only and plain-write
    #                                 # tools run autonomously once approved here
    #       "approved_at": float,    # epoch seconds
    #     }
    #   }
    # }
    "mcp_connections": {},
    # Standard 5-field Unix cron syntax — dispatcher/scheduler.py parses
    # this (via croniter) to fire check_due_workflows() in-process, no
    # external ping/crontab entry needed (2026-09-01, JuanJo's call:
    # "a python function that saves the cron... and it's fire per the
    # syntax" instead of an OS-level cron job on the Lightsail box).
    # Real setting, not hardcoded, same reasoning as everything else in
    # this store — changeable without a redeploy.
    "agent_work_due_check_cron": "*/5 * * * *",
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

    def set_role(self, role: str, provider: str, model: str, fallback: list[dict] | None = None):
        self._data.setdefault("roles", {})[role] = {
            "provider": provider, "model": model, "fallback": fallback or [],
        }
        self._save()

    # ---- Task routing (per-command primary/fallback) ----

    def get_task_routing(self, command: str) -> dict | None:
        return self._data.get("task_routing", {}).get(command)

    def set_task_routing(self, command: str, primary: dict, fallback: list[dict]):
        self._data.setdefault("task_routing", {})[command] = {
            "primary": primary, "fallback": fallback,
        }
        self._save()

    def remove_task_routing(self, command: str):
        self._data.get("task_routing", {}).pop(command, None)
        self._save()

    # ---- MCP connections (see DEFAULTS["mcp_connections"] for shape) ----

    def set_mcp_connection(
        self, name: str, transport: str, *,
        command: str | None = None, args: list[str] | None = None, env: dict | None = None,
        url: str | None = None, auth_header: str | None = None,
    ):
        """Registers or updates a connection's transport config. Does NOT
        mark it connected or touch its tool baseline — that happens once
        dispatcher/mcp_client.py actually completes a handshake and lists
        tools, via set_mcp_tool_baseline below. `env` is extra environment
        variables for a stdio server beyond what mcp_client.py already
        forces (PYTHONUNBUFFERED) — most connections won't need this."""
        existing = self._data.setdefault("mcp_connections", {}).get(name, {})
        self._data["mcp_connections"][name] = {
            **existing,
            "transport": transport, "command": command, "args": args, "env": env,
            "url": url,
            "auth_header": _encrypt_secret(auth_header) if auth_header is not None else existing.get("auth_header"),
            "connected": existing.get("connected", False),
            "tools": existing.get("tools", {}),
        }
        self._save()

    def get_mcp_connection(self, name: str) -> dict | None:
        """Decrypts auth_header for actual use (dispatcher/mcp_client.py's
        live handshake) — list_mcp_connections below stays encrypted since
        nothing reads auth_header off it (the REST list route never echoes
        it to the client either way)."""
        conn = self._data.get("mcp_connections", {}).get(name)
        if conn is None:
            return None
        return {**conn, "auth_header": _decrypt_secret(conn.get("auth_header"))}

    def list_mcp_connections(self) -> dict[str, dict]:
        return self._data.get("mcp_connections", {})

    def remove_mcp_connection(self, name: str):
        self._data.get("mcp_connections", {}).pop(name, None)
        self._save()

    def set_mcp_connected(self, name: str, connected: bool):
        conn = self._data.get("mcp_connections", {}).get(name)
        if conn is None:
            return
        conn["connected"] = connected
        self._save()

    def set_mcp_tool_baseline(
        self, server: str, tool_name: str, tool_hash: str, description: str, input_schema: dict,
        read_only: bool, destructive: bool,
    ):
        """Pins a tool's approved definition — the rug-pull defense. Called
        once per tool, right after its description has been sanitized and
        hashed at connect/re-approval time (never from inside a live call).
        Stores description/input_schema alongside the hash so schema-
        building (tools/mcp_registry.py) reads the pinned snapshot instead
        of a live server round-trip on every request. `destructive` is the
        one tier that still needs a per-call confirmation gate (see
        tools/mcp_registry.py's dispatch()) — read-only and plain-write
        tools run autonomously the moment they're approved here."""
        import time
        conn = self._data.setdefault("mcp_connections", {}).setdefault(server, {"tools": {}})
        conn.setdefault("tools", {})[tool_name] = {
            "hash": tool_hash, "description": description, "input_schema": input_schema,
            "read_only": read_only, "destructive": destructive, "approved_at": time.time(),
        }
        self._save()

    def get_mcp_tool_baseline(self, server: str, tool_name: str) -> dict | None:
        return self._data.get("mcp_connections", {}).get(server, {}).get("tools", {}).get(tool_name)

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


def _migrate_dispatcher_chat_add_fallback():
    """
    One-time addition (2026-08-29) for instances whose dispatcher_chat
    role was already persisted (via Filen backup) before the "fallback"
    key existed on this role at all — editing DEFAULTS alone only affects
    a brand-new store. Adds LLM7's "gpt-oss" as a same-model-family
    fallback after a real "Groq rate limited" hard failure in production
    (ordinary free-tier rate-limiting per Groq's own status page, not an
    outage — see the dispatcher_chat comment above). Guarded so a manual
    reassignment later isn't fought by this.
    """
    if config.get("migrated_dispatcher_chat_add_fallback"):
        return
    role = config.get_role("dispatcher_chat") or {}
    if not role.get("fallback"):
        config.set_role(
            "dispatcher_chat", role.get("provider", "groq"), role.get("model", "openai/gpt-oss-120b"),
            fallback=[{"provider": "llm7", "model": "gpt-oss"}],
        )
    config.set("migrated_dispatcher_chat_add_fallback", True)


_migrate_dispatcher_chat_add_fallback()


def _migrate_dispatcher_chat_to_normal_chat():
    """
    One-time rename (2026-09-01) for instances that already had
    dispatcher_chat persisted under its old name — editing DEFAULTS alone
    only affects a brand-new store. Copies the fully-formed role
    (including whatever provider/model/fallback it currently has — this
    runs after the three migrations above, so it picks up the real
    current state, not a stale default) to "normal_chat" and leaves the
    old "dispatcher_chat" key in place rather than deleting it — harmless
    dead data, and safer than risking a delete bug on a live server's
    only copy of this config. Guarded so a manual reassignment to
    "normal_chat" later isn't fought by this.
    """
    if config.get("migrated_dispatcher_chat_to_normal_chat"):
        return
    old_role = config.get_role("dispatcher_chat")
    if old_role and not config.get_role("normal_chat"):
        config.set_role(
            "normal_chat", old_role.get("provider", "groq"), old_role.get("model", "openai/gpt-oss-120b"),
            fallback=old_role.get("fallback"),
        )
    config.set("migrated_dispatcher_chat_to_normal_chat", True)


_migrate_dispatcher_chat_to_normal_chat()


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


def _migrate_add_dev_slate_chat_role():
    """One-time addition (2026-09-01) for instances whose config.json
    already existed before dev_slate_chat was added to DEFAULTS — editing
    DEFAULTS alone only materializes for a brand-new store, same reasoning
    as every migration above."""
    if config.get("migrated_add_dev_slate_chat_role"):
        return
    if not config.get_role("dev_slate_chat"):
        config.set_role(
            "dev_slate_chat", "cloudflare", "@cf/qwen/qwen2.5-coder-32b-instruct",
            fallback=[{"provider": "mistral", "model": "codestral-latest"}],
        )
    config.set("migrated_add_dev_slate_chat_role", True)


_migrate_add_dev_slate_chat_role()


def _migrate_add_agent_work_role():
    """One-time addition (2026-09-01) for instances whose config.json
    already existed before agent_work was added to DEFAULTS — same
    reasoning as _migrate_add_dev_slate_chat_role above."""
    if config.get("migrated_add_agent_work_role"):
        return
    if not config.get_role("agent_work"):
        config.set_role(
            "agent_work", "groq", "openai/gpt-oss-120b",
            fallback=[{"provider": "llm7", "model": "gpt-oss"}],
        )
    config.set("migrated_add_agent_work_role", True)


_migrate_add_agent_work_role()


def _migrate_chat_roles_off_groq():
    """One-time correction (2026-09-01): normal_chat and agent_work both
    switched primary+fallback away from Groq/OpenRouter in DEFAULTS
    above, but a role that already exists in a saved config.json isn't
    touched by a DEFAULTS change — same reasoning as every other
    migration here. Unlike those, this one force-overwrites an existing
    value rather than only filling in a missing one, since the whole
    point is correcting roles that already exist. Guarded by the usual
    one-time flag so it doesn't stomp a deliberate manual pick made
    after this migration already ran once."""
    if config.get("migrated_chat_roles_off_groq"):
        return
    config.set_role("normal_chat", "llm7", "gpt-oss", fallback=[{"provider": "mistral", "model": "mistral-small-latest"}])
    config.set_role("agent_work", "llm7", "gpt-oss", fallback=[{"provider": "mistral", "model": "mistral-small-latest"}])
    config.set("migrated_chat_roles_off_groq", True)


_migrate_chat_roles_off_groq()


def _migrate_remove_retired_commands():
    """One-time cleanup (2026-09-04): /code, /tailor, /create-image, and
    /design-read are retired — /code is redundant with dev_slate_chat
    (same model, real conversational chat instead of a one-shot command);
    the other three were JuanJo's call for a commercial-harness MVP
    ("this will be a commercial harness... no placeholders for a MVP" —
    design-read was already shipped disabled in the PWA). Editing
    DEFAULTS alone only affects a brand-new store — an existing
    config.json (this server included, live on Lightsail) keeps whatever
    it already had until this runs once. Guarded so a manual re-add later
    isn't fought by this."""
    if config.get("migrated_remove_retired_commands"):
        return
    for command in ("code", "tailor", "create-image", "design-read"):
        config.remove_task_routing(command)
    config.set("migrated_remove_retired_commands", True)


_migrate_remove_retired_commands()


def _migrate_widen_chat_fallbacks_2026_09_04():
    """One-time correction (2026-09-04) for an already-materialized
    config.json (this server's live one included) — DEFAULTS above now
    has normal_chat/agent_work on their new fallback chains, but that
    only affects a brand-new store. Force-overwrites both roles (not a
    fill-in-if-missing migration) since the whole point is correcting
    values that already exist, same pattern as
    _migrate_chat_roles_off_groq. agent_work also moves its primary back
    to Groq's gpt-oss-120b — see that role's DEFAULTS comment for why
    that's safe now (agent_work went stateless 2026-09-03)."""
    if config.get("migrated_widen_chat_fallbacks_2026_09_04"):
        return
    config.set_role(
        "normal_chat", "llm7", "gpt-oss",
        fallback=[
            {"provider": "cloudflare", "model": "@cf/meta/llama-3.1-8b-instruct-fp8-fast"},
            {"provider": "mistral", "model": "mistral-small-latest"},
        ],
    )
    config.set_role(
        "agent_work", "groq", "openai/gpt-oss-120b",
        fallback=[
            {"provider": "cloudflare", "model": "@cf/meta/llama-3.1-8b-instruct-fp8-fast"},
            {"provider": "mistral", "model": "ministral-8b-latest"},
        ],
    )
    config.set("migrated_widen_chat_fallbacks_2026_09_04", True)


_migrate_widen_chat_fallbacks_2026_09_04()
