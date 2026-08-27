"""
providers/llm7.py

LLM7 transport (api.llm7.io) — OpenAI-compatible endpoint. Free "turbo"
tier gives ~1M tokens/day; DeepSeek-V4-Flash-0731 is the strongest model on
that tier (400k context, tool-calling, reasoning) and independently showed
up as free on Hugging Face's warm-models list too.
"""

import requests

from providers.base import ChatMessage, ChatResponse, Provider, ProviderError, ToolCall

BASE_URL = "https://api.llm7.io/v1/chat/completions"


def _serialize_message(m: ChatMessage) -> dict:
    entry = {"role": m.role, "content": m.content}
    if m.name:
        entry["name"] = m.name
    if m.tool_call_id:
        entry["tool_call_id"] = m.tool_call_id
    if m.tool_calls:
        entry["tool_calls"] = m.tool_calls
    return entry


class LLM7Provider(Provider):
    name = "llm7"

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
                timeout=60,  # unproven latency vs. Groq — generous timeout until observed
            )
        except requests.RequestException as e:
            raise ProviderError(f"LLM7 request failed: {e}")

        if resp.status_code == 429:
            raise ProviderError("LLM7 rate limited", is_rate_limit=True)
        if resp.status_code >= 400:
            raise ProviderError(f"LLM7 error {resp.status_code}: {resp.text[:300]}")

        data = resp.json()
        choices = data.get("choices") or []
        if not choices or not choices[0].get("message"):
            raise ProviderError(f"LLM7 returned a malformed response (no message): {str(data)[:300]}")
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
