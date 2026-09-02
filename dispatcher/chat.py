"""
dispatcher/chat.py

Free-form chat — a message that isn't a typed /command. Replaces the old
fixed reply from the role now named normal_chat (renamed 2026-09-01 from
dispatcher_chat — see config/store.py's migration): loads the active
mode's brief (system prompt
+ allowed tools from dispatcher/modes/), then lets the model decide
whether to use any of those tools, resolving calls via the same loop
/research's command chain uses (run_tool_loop, capped at
MAX_TOOL_ITERATIONS).

run_mode_chat (below) stays fully stateless — kept as-is for any caller
that still wants single-message behavior. run_stored_mode_chat is the
new persisted sibling (2026-09-01): first real multi-turn memory for
Normal/Research/Brainstorm/Agent Work, previously only Dev Slate had
this. Deliberately the dumbest version that could work — a flat recency
window (RECENT_MESSAGE_WINDOW, same first-cut approach as Dev Slate's
own), no topic classifier, no compaction — see how_to_handle_context.md:
the goal right now is to find out empirically where plain context
actually breaks, not to pre-solve failures nobody's hit yet.
"""

import asyncio
from datetime import datetime, timezone

from dispatcher.executor import CITATION_STYLE_PROMPT, run_tool_loop
from dispatcher.mode_briefs import get_mode_brief
from providers.base import ChatMessage, ProviderError
from providers.registry import ProviderNotConfigured, get_dispatcher_role, get_provider
from storage.conversations import append_message, get_messages
from tools.registry import schemas_for

RECENT_MESSAGE_WINDOW = 20


def run_mode_chat(mode: str, text: str) -> str:
    brief = get_mode_brief(mode)

    # Every existing mode (normal/research/brainstorm) shares normal_chat's
    # role — only Agent Work gets its own (agent_work role, same reasoning
    # as Dev Slate having dev_slate_chat: it needs real tool-calling
    # reliability, not whatever's cheapest for everyday chat). Context
    # string intentionally differs from `mode` itself for every other mode
    # (stays "chat") so this doesn't silently change normal_chat/research/
    # brainstorm's existing behavior.
    role_context = "agent_work" if mode == "agent_work" else "chat"
    try:
        role = get_dispatcher_role(context=role_context)
    except ProviderNotConfigured as e:
        return f"⚠️ Can't reply right now — {role_context} isn't configured: {e}"

    # Combined into one system message, not two — see run_stored_mode_chat
    # below for why (Cloudflare rejects more than one system-role entry).
    tools = schemas_for(brief.tools) if brief.tools else None
    system_content = f"{brief.system_prompt}\n\n{CITATION_STYLE_PROMPT}" if tools else brief.system_prompt
    messages = [ChatMessage(role="system", content=system_content), ChatMessage(role="user", content=text)]

    # Same-model-family fallback chain (added 2026-08-29 after a real
    # "Groq rate limited" hard failure) — try the primary, then each
    # configured fallback in order, same pattern as /research's gathering
    # phase (executor.py). Any one succeeding returns immediately.
    attempts = [{"provider": role["provider"], "model": role["model"]}] + role.get("fallback", [])
    last_error = None
    for i, attempt in enumerate(attempts):
        try:
            provider = get_provider(attempt["provider"])
        except Exception as e:
            last_error = str(e)
            continue
        try:
            response = provider.chat(model=attempt["model"], messages=messages, tools=tools)
            if tools and response.tool_calls:
                # Free-form chat has no StepResult to attach an attempt
                # count to (that's a /research-command-chain concept) —
                # discard it here, not silently drop it by accident.
                response, _messages, _iterations = run_tool_loop(
                    provider, attempt["model"], messages, response,
                    context={"command": f"chat-{mode}", "topic_slug": "chat"},
                    tools=tools,
                )
            reply = response.text or "(empty reply)"
            if i > 0:
                reply += f"\n\n⚡ (Groq was busy, answered via {attempt['provider']}/{attempt['model']} instead)"
            elif response.usage_note:
                reply += f"\n\n⚡ {response.usage_note}"
            return reply
        except ProviderError as e:
            last_error = str(e)
            continue

    return f"⚠️ normal_chat failed on every configured provider: {last_error}"


# Agent Work's "review changes" mode (2026-09-01) — same review-vs-
# auto-accept *concept* as Dev Slate's EditModeSelector, but a genuinely
# different mechanism underneath: Dev Slate can literally pause mid-tool-
# call and await the browser's Accept/Reject over its live WebSocket
# (see dispatcher/devslate_chat.py's requestWriteReview). Agent Work's
# chat is plain stateless REST — there's no live connection to pause on,
# so there's nothing to await mid-request. This achieves the same
# *behavior* (nothing gets created/run without the user seeing it first)
# through prompt instruction instead: the model is told to describe what
# it would do and wait for explicit confirmation on a LATER turn before
# actually calling create_workflow/run_workflow. Less airtight than a
# real gate (a model could still ignore the instruction), but the only
# option available without adding a live connection just for this.
AGENT_WORK_REVIEW_INSTRUCTION = (
    "Before calling create_workflow or run_workflow, first describe in "
    "plain text what you're about to create or run and ask the user to "
    "confirm. Only call the tool after they've explicitly confirmed in "
    "a later message — never on the same turn you proposed it."
)


async def run_stored_mode_chat(mode: str, conversation_id: str, text: str, auto_accept: bool = True) -> dict:
    """Persisted sibling of run_mode_chat above — appends the user's
    message, replays a windowed slice of REAL history (not just this one
    message) alongside the mode's brief, calls the model, persists the
    reply. Returns {text, provider, model} (provider/model reflect
    whichever fallback actually answered, mirroring
    dispatcher/devslate_chat.py's run_devslate_turn, which this is
    deliberately modeled on — same role-selection/fallback/tool-loop
    shape as run_mode_chat above, just with storage/conversations.py
    wrapped around it instead of nothing.

    auto_accept only means anything for mode == "agent_work" (see
    AGENT_WORK_REVIEW_INSTRUCTION above) — defaults True so every other
    mode's behavior is completely unaffected by this parameter existing."""
    await append_message(conversation_id, "user", text)

    brief = get_mode_brief(mode)
    history = await get_messages(conversation_id, limit=RECENT_MESSAGE_WINDOW)

    role_context = "agent_work" if mode == "agent_work" else "chat"
    try:
        role = get_dispatcher_role(context=role_context)
    except ProviderNotConfigured as e:
        error_text = f"⚠️ Can't reply right now — {role_context} isn't configured: {e}"
        await append_message(conversation_id, "navi", error_text)
        return {"text": error_text, "provider": None, "model": None}

    # Ordering here is a real invariant, not incidental (JuanJo,
    # 2026-09-01): static content first (brief/citation prompt — byte-
    # identical every call), then growing context (history), with the
    # new message naturally landing last since it's already the final
    # row `history` returns. This is what lets a provider's prefix-based
    # prompt caching (Groq/OpenRouter automatic, Cloudflare automatic
    # baseline, Mistral manual via prompt_cache_key though not wired up
    # yet — see providers/*.py's own docstrings) actually hit: the
    # unchanging prefix (static + already-seen history) stays identical
    # turn to turn, only the newest message is genuinely new. Don't
    # insert anything that changes between calls (a live timestamp, a
    # per-turn-computed block) ahead of the growing-but-stable part, or
    # it breaks the prefix match for every provider's cache at once.
    # 2026-09-02: these three were previously separate ChatMessage entries
    # (all consecutively first, before any history) — correct per the
    # usual "system messages must lead" convention, but the Cloudflare 400
    # ("System message must be at the beginning") persisted even after the
    # trailing UTC-time message was removed. agent_work is the one mode
    # that reliably stacks all three at once (brief + citation format,
    # since it has tools + the review instruction, since auto_accept
    # defaults off) — strong circumstantial evidence Cloudflare's real
    # rule is "at most one system message, and it must be message[0]," not
    # just "system messages must be consecutively first." Combining into
    # one message satisfies either reading and can't regress anything.
    tools = schemas_for(brief.tools) if brief.tools else None
    system_parts = [brief.system_prompt]
    if tools:
        system_parts.append(CITATION_STYLE_PROMPT)
    if mode == "agent_work" and not auto_accept:
        system_parts.append(AGENT_WORK_REVIEW_INSTRUCTION)
    messages = [ChatMessage(role="system", content="\n\n".join(system_parts))]
    # The just-appended user message is already the last row `history`
    # returns (get_messages reads it back from storage) — not double
    # counted. "navi" -> "assistant" matches storage's own role
    # convention (see storage/conversations.py / devslate_chat.py).
    for m in history:
        messages.append(ChatMessage(role="assistant" if m["role"] == "navi" else m["role"], content=m["content"]))
    # UTC time grounding (JuanJo, 2026-09-01: "if it asks for something
    # close to 'do it in X time', must send the messages with a UTC
    # signal") — the model has no inherent sense of "now," so a request
    # like "in 5 minutes" or "every hour starting now" is unresolvable
    # without this.
    #
    # 2026-09-02: originally a separate trailing system message, which
    # broke outright on Cloudflare (400: "System message must be at the
    # beginning") — Cloudflare's OpenAI-compatible endpoint rejects any
    # system-role message that isn't the very first one, and every other
    # system message here (brief/citation/review instruction) already IS
    # first, consecutively. Rather than move the time signal to the front
    # (which would poison the whole history block's cache-prefix stability
    # with a value that changes every call), it's appended directly onto
    # the final user message's own content instead — that message was
    # already guaranteed unique this turn, so this costs nothing
    # additional for prefix-based caching while keeping every system
    # message genuinely first. The stored copy (already persisted above,
    # via append_message) is untouched — only this outgoing copy changes.
    messages[-1].content = f"{messages[-1].content}\n\n[Current UTC time: {datetime.now(timezone.utc).isoformat()}]"

    attempts = [{"provider": role["provider"], "model": role["model"]}] + role.get("fallback", [])
    last_error = None
    for i, attempt in enumerate(attempts):
        try:
            provider = get_provider(attempt["provider"])
        except Exception as e:
            last_error = str(e)
            continue
        try:
            response = await asyncio.to_thread(provider.chat, model=attempt["model"], messages=messages, tools=tools)
            if tools and response.tool_calls:
                response, _messages, _iterations = await asyncio.to_thread(
                    run_tool_loop, provider, attempt["model"], messages, response,
                    context={"command": f"chat-{mode}", "topic_slug": "chat"}, tools=tools,
                )
            if not response.text and not response.tool_calls:
                # A real failure mode, not a valid (if terse) answer — a
                # model that returns neither text nor a tool call did
                # nothing at all (2026-09-02: gpt-oss-20b via Cloudflare,
                # asked to create a workflow, returned a completely blank
                # response — no create_workflow call, nothing). Treat it
                # the same as a ProviderError so the next attempt in the
                # fallback chain actually gets tried, instead of silently
                # "succeeding" with an unhelpful "(empty reply)" placeholder
                # and nothing having happened.
                last_error = f"{attempt['provider']}/{attempt['model']} returned neither text nor a tool call"
                continue
            reply = response.text or "(empty reply)"
            if i > 0:
                reply += f"\n\n⚡ (primary was unavailable, answered via {attempt['provider']}/{attempt['model']} instead)"
            await append_message(conversation_id, "navi", reply, provider=attempt["provider"], model=attempt["model"])
            return {
                "text": reply, "provider": attempt["provider"], "model": attempt["model"],
                "usage_note": response.usage_note,
            }
        except ProviderError as e:
            last_error = str(e)
            continue

    error_text = f"⚠️ {role_context} failed on every configured provider: {last_error}"
    await append_message(conversation_id, "navi", error_text)
    return {"text": error_text, "provider": None, "model": None}
