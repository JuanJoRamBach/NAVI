"""
dispatcher/compaction.py

compact_conversation — the shared schema-constrained "read a conversation,
produce a structured result" primitive designed in how_to_handle_context.md
(2026-08-30/2026-09-01 design). First real caller: Research mode's
plan-drafting synthesis step (dispatcher/research.py, 2026-09-12) — the
Agent_Work_Context.md handoff and weekly chat-canvas compaction are the
other two designed-but-not-yet-built callers for this same function, same
shape ("N messages in, one schema-constrained JSON object out"), different
instruction/schema per caller.

Deliberately reuses the "chat" role (the same one normal_chat/research/
brainstorm already share) rather than a dedicated compaction role — this
fires rarely (once per research plan, not once per turn), so there's no
repeated-cost concern to optimize the way normal_chat's own routing has to.
"""

import asyncio
import json

from config.store import config
from providers.base import ChatMessage, ProviderError
from providers.registry import ProviderNotConfigured, get_dispatcher_role, get_provider


def strip_code_fence(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = text[3:]
        if text.lower().startswith("json"):
            text = text[4:]
        if text.endswith("```"):
            text = text[:-3]
    return text.strip()


async def compact_conversation(messages: list[ChatMessage], instruction: str) -> dict | None:
    """Runs one schema-constrained extraction call over `messages` — the
    caller's job to pass the FULL relevant history, not a windowed slice;
    that's the entire point of this primitive over just replaying the
    normal recency window (see how_to_handle_context.md's "Is 20 messages
    actually enough?" section). `instruction` is the system prompt
    describing the desired JSON shape and any drafting guidance.

    Returns the parsed JSON object, or None if every attempt in the
    fallback chain failed, returned empty, or produced something that
    doesn't parse as JSON — callers MUST handle the None case explicitly
    (e.g. telling the user to try again), this never raises to signal
    failure, matching every other provider-call site in this codebase."""
    try:
        role = get_dispatcher_role(context="chat")
    except ProviderNotConfigured:
        return None

    call_messages = [ChatMessage(role="system", content=instruction)] + messages
    attempts = config.get_attempts([{"provider": role["provider"], "model": role["model"]}] + role.get("fallback", []))
    for attempt in attempts:
        try:
            provider = get_provider(attempt["provider"])
        except Exception:
            continue
        try:
            response = await asyncio.to_thread(provider.chat, model=attempt["model"], messages=call_messages)
        except ProviderError as e:
            if e.is_rate_limit:
                config.mark_rate_limited(attempt["provider"], attempt["model"])
            continue
        text = strip_code_fence(response.text or "")
        if not text:
            continue
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None
