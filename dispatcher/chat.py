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
"""

from dispatcher.executor import CITATION_STYLE_PROMPT, run_tool_loop
from dispatcher.mode_briefs import get_mode_brief
from providers.base import ChatMessage, ProviderError
from providers.registry import ProviderNotConfigured, get_dispatcher_role, get_provider
from tools.registry import schemas_for


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

    tools = schemas_for(brief.tools) if brief.tools else None
    messages = [ChatMessage(role="system", content=brief.system_prompt)]
    if tools:
        messages.append(ChatMessage(role="system", content=CITATION_STYLE_PROMPT))
    messages.append(ChatMessage(role="user", content=text))

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
