"""
dispatcher/devslate_chat.py

Dev Slate's own chat loop — deliberately separate from dispatcher/chat.py's
run_mode_chat rather than folded into it, for two real reasons:

1. It has real server-side conversation memory (storage/conversations.py) —
   run_mode_chat still takes one bare `text` string with no history, since
   only Dev Slate is wired to the new SQLite store this pass (Normal/
   Research/Brainstorm chat is deliberately untouched — see JuanJo's own
   sequencing call, 2026-09-01).
2. Its tool calls split into "runs on this server" (update_task_state) vs.
   "has to be relayed to the user's browser and awaited" (read_file/
   write_file/grep — the user's project files live on their own machine,
   never on this server). run_tool_loop in executor.py is synchronous and
   shared by /research and other live commands; retrofitting it to
   sometimes await a network round-trip mid-loop would risk that shared,
   already-in-production code path. A separate async loop here carries
   zero risk to anything else.

Every provider transport (providers/*.py) is synchronous (built on
`requests`) — calling one directly inside an `async def` would block this
whole process's event loop for the duration of every model call, stalling
every other concurrent WebSocket/HTTP connection, not just the one that
issued it. asyncio.to_thread() runs each blocking call in a thread pool
instead, so the transports themselves need no changes.
"""

import asyncio
import json
from typing import Awaitable, Callable

from dispatcher.mode_briefs import get_mode_brief
from providers.base import ChatMessage, ProviderError
from providers.registry import ProviderNotConfigured, get_dispatcher_role, get_provider
from storage.conversations import append_message, get_messages, get_task_state
from tools.devslate_tools import LOCAL_TOOL_NAMES, TOOL_SCHEMAS, dispatch_local, format_task_state_for_prompt

MAX_TOOL_ITERATIONS = 5

# Layer 4 (short-term execution loop) as a first cut: cap how many of the
# most recently STORED messages get replayed into the model's context each
# turn. This is not the full pruning design (raw-tool-output collapse,
# diff-collapse-after-build-passes, error-signal-first log truncation) —
# those need real tool output to prune, which doesn't exist until
# read_file/write_file/grep are live end-to-end. This is the one piece
# that's real today: an unbounded conversation doesn't unboundedly grow
# the prompt. Full-fidelity history always stays in SQLite regardless.
RECENT_MESSAGE_WINDOW = 20

# ToolRelay: given a tool name and its arguments, executes it on the
# user's browser (over the open WebSocket) and returns the result text.
# Owned entirely by the caller (server.py's WebSocket handler), which is
# the one thing that actually holds the live connection and the
# pending-tool-call bookkeeping — this module stays transport-agnostic,
# same separation every other dispatcher module already keeps.
ToolRelay = Callable[[str, dict], Awaitable[str]]


class DevSlateChatError(Exception):
    pass


def _parse_tool_args(raw: str | dict) -> dict:
    if isinstance(raw, dict):
        return raw
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {}


async def run_devslate_turn(conversation_id: str, user_text: str, relay: ToolRelay) -> dict:
    """Runs one full turn: persists the user's message, builds Layer 1+3
    context plus a windowed Layer-4 slice of history, calls the model
    (walking dev_slate_chat's fallback chain — no Groq, ever, see
    config/store.py's role comment), resolves any tool calls (local or
    relayed), persists the reply, and returns {text, provider, model} —
    the provider/model actually used, not just what was pinned, since a
    fallback may have fired."""
    await append_message(conversation_id, "user", user_text)

    brief = get_mode_brief("devslate")
    task_state = await get_task_state(conversation_id)
    history = await get_messages(conversation_id, limit=RECENT_MESSAGE_WINDOW)

    # One combined system message, not two — Cloudflare (dev_slate_chat's
    # own default provider) rejects any system-role message that isn't
    # both the first AND only one (2026-09-02, discovered via the same bug
    # in dispatcher/chat.py's run_stored_mode_chat).
    state_block = format_task_state_for_prompt(task_state)
    system_content = f"{brief.system_prompt}\n\nCurrent Slate task state:\n{state_block}" if state_block else brief.system_prompt
    messages = [ChatMessage(role="system", content=system_content)]
    # The just-appended user message is already the last row `history`
    # returns (get_messages reads it back from storage), so this isn't
    # double-counted.
    for m in history:
        messages.append(ChatMessage(role="assistant" if m["role"] == "navi" else m["role"], content=m["content"]))

    try:
        role = get_dispatcher_role(context="devslate")
    except ProviderNotConfigured as e:
        text = f"⚠️ Can't reply right now — dev_slate_chat isn't configured: {e}"
        await append_message(conversation_id, "navi", text)
        return {"text": text, "provider": None, "model": None}

    attempts = [{"provider": role["provider"], "model": role["model"]}] + role.get("fallback", [])
    last_error = None

    for i, attempt in enumerate(attempts):
        try:
            provider = get_provider(attempt["provider"])
        except Exception as e:
            last_error = str(e)
            continue

        try:
            response = await asyncio.to_thread(
                provider.chat, model=attempt["model"], messages=messages, tools=TOOL_SCHEMAS,
            )

            iterations = 0
            while response.tool_calls and iterations < MAX_TOOL_ITERATIONS:
                raw_choice = ((response.raw or {}).get("choices") or [{}])[0].get("message", {})
                messages = messages + [ChatMessage(
                    role="assistant", content=response.text or "", tool_calls=raw_choice.get("tool_calls"),
                )]

                for tc in response.tool_calls:
                    args = _parse_tool_args(tc.arguments)
                    if tc.name in LOCAL_TOOL_NAMES:
                        result_text = await dispatch_local(tc.name, args, conversation_id)
                    else:
                        result_text = await relay(tc.name, args)
                    messages = messages + [ChatMessage(
                        role="tool", content=result_text, tool_call_id=tc.id, name=tc.name,
                    )]

                response = await asyncio.to_thread(
                    provider.chat, model=attempt["model"], messages=messages, tools=TOOL_SCHEMAS,
                )
                iterations += 1

            reply = response.text or "(empty reply)"
            if i > 0:
                reply += f"\n\n⚡ (primary was unavailable, answered via {attempt['provider']}/{attempt['model']} instead)"
            await append_message(conversation_id, "navi", reply, provider=attempt["provider"], model=attempt["model"])
            return {"text": reply, "provider": attempt["provider"], "model": attempt["model"]}

        except ProviderError as e:
            last_error = str(e)
            continue

    error_text = f"⚠️ dev_slate_chat failed on every configured provider: {last_error}"
    await append_message(conversation_id, "navi", error_text)
    return {"text": error_text, "provider": None, "model": None}
