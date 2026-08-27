"""
dispatcher/chat.py

Free-form chat — a message that isn't a typed /command. Replaces the old
fixed dispatcher_chat reply: loads the active mode's brief (system prompt
+ allowed tools from dispatcher/modes/), then lets the model decide
whether to use any of those tools, resolving calls via the same loop
/research's command chain uses (run_tool_loop, capped at
MAX_TOOL_ITERATIONS).
"""

from dispatcher.executor import CITATION_STYLE_PROMPT, run_tool_loop
from dispatcher.mode_briefs import get_mode_brief
from providers.base import ChatMessage, ProviderError
from providers.registry import ProviderNotConfigured, get_dispatcher
from tools.registry import schemas_for


def run_mode_chat(mode: str, text: str) -> str:
    brief = get_mode_brief(mode)

    try:
        provider, model = get_dispatcher(context="chat")
    except ProviderNotConfigured as e:
        return f"⚠️ Can't reply right now — dispatcher_chat isn't configured: {e}"

    tools = schemas_for(brief.tools) if brief.tools else None
    messages = [ChatMessage(role="system", content=brief.system_prompt)]
    if tools:
        messages.append(ChatMessage(role="system", content=CITATION_STYLE_PROMPT))
    messages.append(ChatMessage(role="user", content=text))

    try:
        response = provider.chat(model=model, messages=messages, tools=tools)
        if tools and response.tool_calls:
            response, _messages = run_tool_loop(
                provider, model, messages, response,
                context={"command": f"chat-{mode}", "topic_slug": "chat"},
                tools=tools,
            )
    except ProviderError as e:
        return f"⚠️ dispatcher_chat failed and has no fallback by design: {e}"

    reply = response.text or "(empty reply)"
    if response.usage_note:
        reply += f"\n\n⚡ {response.usage_note}"
    return reply
