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
import re
from datetime import datetime, timezone

from dispatcher.executor import CITATION_STYLE_PROMPT, _parse_tool_args, run_tool_loop
from dispatcher.mode_briefs import get_mode_brief
from dispatcher.provider_debug import save_failed_exchange
from providers.base import ChatMessage, ProviderError
from providers.registry import ProviderNotConfigured, get_dispatcher_role, get_provider
from storage.conversations import append_message, get_messages
from tools.registry import schemas_for

RECENT_MESSAGE_WINDOW = 20

_CREATE_WORKFLOW_ID_RE = re.compile(r"Created workflow ([0-9a-fA-F-]{36})\.")


def _extract_created_workflow_id(sent_messages: list[ChatMessage]) -> str | None:
    """Agent Work Chat's whole point is that the model, not the user,
    builds the graph — but the frontend still needs to know WHICH
    workflow just got created so it can load it onto the canvas as real
    nodes (2026-09-03, JuanJo: a workflow the chat created should show up
    as nodes, not just an entry in the Workflows list). tools/registry.py's
    create_workflow branch returns a fixed "Created workflow <id>." string
    as its tool-result content; that's the only place the id exists once
    run_tool_loop has finished, so pulling it out of the transcript here
    is simpler than widening dispatch()'s return contract for every tool.
    Returns the LAST match if create_workflow was somehow called more
    than once in a single turn."""
    found = None
    for m in sent_messages:
        if m.role == "tool" and m.name == "create_workflow" and isinstance(m.content, str):
            match = _CREATE_WORKFLOW_ID_RE.search(m.content)
            if match:
                found = match.group(1)
    return found


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


# Agent Work's old "review changes" mode (2026-09-01 - 2026-09-03) lived
# here as a prompt instruction telling the model to describe its plan and
# wait for an explicit "yes" on a LATER turn before actually calling
# create_workflow/run_workflow — the only option available without a live
# connection to pause on, mirroring Dev Slate's EditModeSelector concept.
# Removed 2026-09-03 once agent_work went stateless (JuanJo: "no context.md
# for the chat... it should just create"): a confirm-on-a-later-turn design
# cannot work once the model no longer sees its own earlier turns.
# auto_accept is kept as a parameter (harmless no-op for agent_work now,
# still meaningful for nothing else) purely so no caller needs updating.


async def run_stored_mode_chat(mode: str, conversation_id: str, text: str, auto_accept: bool = True) -> dict:
    """Persisted sibling of run_mode_chat above — appends the user's
    message, replays a windowed slice of REAL history (not just this one
    message, except for agent_work — see below) alongside the mode's
    brief, calls the model, persists the reply. Returns {text, provider,
    model} (provider/model reflect whichever fallback actually answered,
    mirroring dispatcher/devslate_chat.py's run_devslate_turn, which this
    is deliberately modeled on — same role-selection/fallback/tool-loop
    shape as run_mode_chat above, just with storage/conversations.py
    wrapped around it instead of nothing."""
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
    # AGENT_WORK_REVIEW_INSTRUCTION's "confirm on a LATER message" design
    # requires the model to remember its own proposal on a future turn —
    # incompatible with agent_work now being stateless (2026-09-03,
    # JuanJo: "no context.md for the chat... just needs the LLM to create
    # the json schema with the steps"). auto_accept stays a harmless no-op
    # parameter for every other mode.
    messages = [ChatMessage(role="system", content="\n\n".join(system_parts))]
    # The just-appended user message is already the last row `history`
    # returns (get_messages reads it back from storage) — not double
    # counted. "navi" -> "assistant" matches storage's own role
    # convention (see storage/conversations.py / devslate_chat.py).
    #
    # A past failure's own error text (always prefixed "⚠️" — see every
    # error_text/return above) is skipped here, not replayed as if it were
    # a real prior reply (2026-09-02, JuanJo: "are we giving the errors as
    # context in the chat? that might be [messing] us too"). A retry in
    # the same conversation would otherwise feed the model its own past
    # failure's raw error text as supposed prior context on every
    # subsequent attempt — noise at best, actively confusing at worst.
    # Still shown to the user in the UI (this only filters what's SENT to
    # the model, not what's persisted/displayed) — see get_messages calls
    # elsewhere, unaffected by this.
    #
    # agent_work is deliberately stateless (2026-09-03, JuanJo: "we
    # actually make Agent Work Chat have no context. it should just
    # create, it doesn't need any context") — each message is a
    # standalone "build/run this" instruction, not a turn in an ongoing
    # conversation. Replaying old turns was also part of what let a
    # flaky model's confusion compound across turns (a stale tool result
    # or an earlier vague reply sitting in context, nudging a later
    # attempt toward repeating work). Still fully persisted via
    # append_message above/below for the frontend's own display
    # history — this only changes what's SENT to the model.
    if mode == "agent_work":
        messages.append(ChatMessage(role="user", content=text))
    else:
        for m in history:
            if m["role"] == "navi" and m["content"].startswith("⚠️"):
                continue
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
    attempt_labels = [f"{a['provider']}/{a['model']}" for a in attempts]
    print(f"[run_stored_mode_chat] mode={mode} conversation={conversation_id} attempts={attempt_labels}")
    last_error = None
    for i, attempt in enumerate(attempts):
        print(f"[run_stored_mode_chat] attempt {i}: {attempt['provider']}/{attempt['model']}")
        try:
            provider = get_provider(attempt["provider"])
        except Exception as e:
            last_error = str(e)
            print(f"[run_stored_mode_chat] attempt {i} get_provider failed: {e}")
            continue
        try:
            sent_messages = messages
            response = await asyncio.to_thread(provider.chat, model=attempt["model"], messages=messages, tools=tools)
            print(
                f"[run_stored_mode_chat] attempt {i} FIRST reply: "
                f"text={(response.text or '')[:200]!r} tool_calls={[tc.name for tc in response.tool_calls]}"
            )
            choice_call = next((tc for tc in response.tool_calls if tc.name == "ask_user_choice"), None)
            if choice_call:
                # Intercepted BEFORE run_tool_loop, not executed through it
                # — this tool has no server-side action; calling it IS the
                # model handing a question back to the user. question is
                # persisted/returned as the reply text (so it reads
                # naturally without the tool call), options ride alongside
                # for the frontend to render as clickable buttons. Doesn't
                # survive a page refresh (only the question text is
                # persisted) — same known limit as usage_note.
                args = _parse_tool_args(choice_call.arguments)
                question = args.get("question", "")
                options = args.get("options") or []
                await append_message(conversation_id, "navi", question, provider=attempt["provider"], model=attempt["model"])
                return {
                    "text": question, "provider": attempt["provider"], "model": attempt["model"],
                    "usage_note": response.usage_note, "choices": options,
                }
            if tools and response.tool_calls:
                print(f"[run_stored_mode_chat] attempt {i}: entering run_tool_loop")
                response, sent_messages, iterations = await asyncio.to_thread(
                    run_tool_loop, provider, attempt["model"], messages, response,
                    context={"command": f"chat-{mode}", "topic_slug": "chat"}, tools=tools,
                )
                created_workflow_id = _extract_created_workflow_id(sent_messages)
                print(
                    f"[run_stored_mode_chat] attempt {i}: run_tool_loop returned iterations={iterations} "
                    f"text={(response.text or '')[:200]!r} tool_calls={[tc.name for tc in response.tool_calls]} "
                    f"created_workflow_id={created_workflow_id}"
                )
                if iterations > 0 and not response.text and not response.tool_calls:
                    # run_tool_loop actually executed a real tool call here
                    # (e.g. create_workflow persisted a row, send_to_telegram
                    # sent a message) — falling through to `continue` below
                    # would retry the NEXT fallback provider from scratch,
                    # replaying the same request and re-running that same
                    # side effect again. 2026-09-03, JuanJo: one "send me a
                    # Telegram message" request produced 5 duplicate
                    # workflows this way, one per fallback provider that
                    # also came back with an empty wrap-up. Once execution
                    # already happened, an empty wrap-up is a done-but-
                    # unsummarized outcome, not a failure to retry.
                    reply = "Done — the action completed, but I didn't get a summary back. Check Workflows / Run History for the result."
                    print(f"[run_stored_mode_chat] attempt {i}: DECISION = stop here (real tool call already ran, empty wrap-up) — NOT retrying fallback")
                    await append_message(conversation_id, "navi", reply, provider=attempt["provider"], model=attempt["model"])
                    return {
                        "text": reply, "provider": attempt["provider"], "model": attempt["model"],
                        "usage_note": response.usage_note,
                        **({"created_workflow_id": created_workflow_id} if created_workflow_id else {}),
                    }
            else:
                created_workflow_id = None
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
                print(f"[run_stored_mode_chat] attempt {i}: DECISION = retry next fallback ({last_error})")
                await asyncio.to_thread(
                    save_failed_exchange, role_context, attempt["provider"], attempt["model"],
                    sent_messages, last_error, response.raw,
                )
                continue
            reply = response.text or "(empty reply)"
            if i > 0:
                reply += f"\n\n⚡ (primary was unavailable, answered via {attempt['provider']}/{attempt['model']} instead)"
            print(f"[run_stored_mode_chat] attempt {i}: DECISION = success, returning (created_workflow_id={created_workflow_id})")
            await append_message(conversation_id, "navi", reply, provider=attempt["provider"], model=attempt["model"])
            return {
                "text": reply, "provider": attempt["provider"], "model": attempt["model"],
                "usage_note": response.usage_note,
                **({"created_workflow_id": created_workflow_id} if created_workflow_id else {}),
            }
        except ProviderError as e:
            last_error = str(e)
            print(f"[run_stored_mode_chat] attempt {i}: DECISION = retry next fallback (ProviderError: {last_error})")
            await asyncio.to_thread(
                save_failed_exchange, role_context, attempt["provider"], attempt["model"], messages, last_error,
            )
            continue

    error_text = f"⚠️ {role_context} failed on every configured provider: {last_error}"
    await append_message(conversation_id, "navi", error_text)
    return {"text": error_text, "provider": None, "model": None}
