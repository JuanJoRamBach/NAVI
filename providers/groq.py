"""
providers/groq.py

Groq transport. This provider fills the "dispatcher" role — every message
that isn't caught by the free deterministic parser comes through here first,
using the model pinned in config/store.py DEFAULTS (currently
llama-3.1-8b-instant — see the comment there for why, despite it being
officially deprecated). Consistency matters more than raw capability for
this role.

Prompt caching (2026-09-01 research pass, cross-provider): fully automatic
here — Groq's own docs: "works automatically on all your API requests with
no code changes required and no additional fees." 50% off cached portions,
cache expires after 2 hours idle. Nothing to wire on NAVI's side to benefit
from it; `usage_note` below already surfaces cached_tokens when a hit
occurs. Compare providers/openrouter.py (mostly automatic, a few model
families need an explicit marker) and providers/cloudflare.py (automatic
baseline, optional header to improve hit rate) — caching behavior is NOT
uniform across NAVI's providers, check the specific transport before
assuming.
"""

import requests

from providers.base import ChatMessage, ChatResponse, Provider, ProviderError, ToolCall

BASE_URL = "https://api.groq.com/openai/v1/chat/completions"


def _serialize_message(m: ChatMessage) -> dict:
    entry = {"role": m.role, "content": m.content}
    if m.name:
        entry["name"] = m.name
    if m.tool_call_id:
        entry["tool_call_id"] = m.tool_call_id
    if m.tool_calls:
        entry["tool_calls"] = m.tool_calls
    return entry


class GroqProvider(Provider):
    name = "groq"

    def _do_chat(
        self,
        model: str,
        messages: list[ChatMessage],
        tools: list[dict] | None = None,
        tool_choice: str | dict | None = None,
    ) -> ChatResponse:
        payload = {
            "model": model,
            "messages": [_serialize_message(m) for m in messages],
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = tool_choice or "auto"

        try:
            resp = requests.post(
                BASE_URL,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=30,  # Groq is fast — no need for a long timeout
            )
        except requests.RequestException as e:
            raise ProviderError(f"Groq request failed: {e}")

        if resp.status_code == 429:
            raise ProviderError("Groq rate limited", is_rate_limit=True)
        if resp.status_code >= 400:
            raise ProviderError(f"Groq error {resp.status_code}: {resp.text[:300]}")

        data = resp.json()
        choices = data.get("choices") or []
        if not choices or not choices[0].get("message"):
            raise ProviderError(f"Groq returned a malformed response (no message): {str(data)[:300]}")
        choice = choices[0]["message"]

        tool_calls = []
        for tc in choice.get("tool_calls") or []:
            tool_calls.append(ToolCall(
                id=tc["id"],
                name=tc["function"]["name"],
                arguments=tc["function"]["arguments"],
            ))

        # Every OpenAI-compatible response includes a "usage" object — the
        # provider needs it for their own billing, so it's returned
        # unconditionally, not gated behind any particular tier. Was
        # already arriving in `data` on every call, just never extracted.
        # cached_tokens (when present) is Groq's automatic prompt-caching
        # hit count — confirms whether the 50%-off static-prefix reuse
        # (system prompt + tool schemas, sent unchanged on every call) is
        # actually firing, rather than just assuming it is.
        usage = data.get("usage") or {}
        cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens")
        usage_note = None
        if usage.get("total_tokens") is not None:
            cached_part = f" ({cached} cached)" if cached else ""
            usage_note = (
                f"{usage.get('prompt_tokens', '?')} in{cached_part} / "
                f"{usage.get('completion_tokens', '?')} out / {usage['total_tokens']} total tokens"
            )

        return ChatResponse(
            text=choice.get("content"),
            tool_calls=tool_calls,
            model_used=data.get("model", model),
            raw=data,
            usage_note=usage_note,
        )
