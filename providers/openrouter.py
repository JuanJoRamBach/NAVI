"""
providers/openrouter.py

OpenRouter transport. This is the "executor" in the dispatcher/executor
split — the dispatcher (Groq) decides what needs doing, this actually does
the heavy-lifting task work, using whichever model the daily-ranked list
picked for that task type.
"""

import requests

from providers.base import ChatMessage, ChatResponse, Provider, ProviderError, ToolCall

BASE_URL = "https://openrouter.ai/api/v1/chat/completions"


def _serialize_message(m: ChatMessage) -> dict:
    entry = {"role": m.role, "content": m.content}
    if m.name:
        entry["name"] = m.name
    if m.tool_call_id:
        entry["tool_call_id"] = m.tool_call_id
    if m.tool_calls:
        entry["tool_calls"] = m.tool_calls
    return entry


class OpenRouterProvider(Provider):
    name = "openrouter"

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
                    # OpenRouter asks for these for attribution — harmless to include
                    "HTTP-Referer": "https://github.com/juanjorambach",
                    "X-Title": "juanjo-agent",
                },
                json=payload,
                timeout=60,
            )
        except requests.RequestException as e:
            raise ProviderError(f"OpenRouter request failed: {e}")

        if resp.status_code == 429:
            raise ProviderError("OpenRouter rate limited", is_rate_limit=True)
        if resp.status_code >= 400:
            raise ProviderError(f"OpenRouter error {resp.status_code}: {resp.text[:300]}")

        data = resp.json()
        choices = data.get("choices") or []
        if not choices or not choices[0].get("message"):
            raise ProviderError(f"OpenRouter returned a malformed response (no message): {str(data)[:300]}")
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
        usage = data.get("usage") or {}
        usage_note = (
            f"{usage.get('prompt_tokens', '?')} in / {usage.get('completion_tokens', '?')} out / {usage['total_tokens']} total tokens"
            if usage.get("total_tokens") is not None else None
        )

        return ChatResponse(
            text=choice.get("content"),
            tool_calls=tool_calls,
            model_used=data.get("model", model),
            raw=data,
            usage_note=usage_note,
        )
