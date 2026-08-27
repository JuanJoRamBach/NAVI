"""
providers/groq.py

Groq transport. This provider fills the "dispatcher" role — every message
that isn't caught by the free deterministic parser comes through here first,
using the model pinned in config/store.py DEFAULTS (currently
llama-3.1-8b-instant — see the comment there for why, despite it being
officially deprecated). Consistency matters more than raw capability for
this role.
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

    def chat(
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

        return ChatResponse(
            text=choice.get("content"),
            tool_calls=tool_calls,
            model_used=data.get("model", model),
            raw=data,
        )
