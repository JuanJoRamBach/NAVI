"""
providers/gmi.py

GMI Cloud transport — OpenAI-compatible endpoint (api.gmi-serving.com),
confirmed via docs.gmicloud.ai/quickstart 2026-09-01: base URL
https://api.gmi-serving.com/v1, `Authorization: Bearer <key>`, standard
choices[0].message response shape.

Added specifically for MiniMax M3, which GMI is running free as a
promotion through 2026-09-06 — a 5-day window from when this was added.
Not hardcoding the model slug anywhere: the daily model-fetch job pulls
it live from GET /v1/models like every other provider here, so if the
promo's exact slug or its free status changes (or the promo ends and the
model comes back paid/removed), the fetch reflects that automatically
instead of this transport silently keeping a stale assumption.

Prompt caching: UNKNOWN — no caching documentation found during the
2026-09-01 research pass, same gap as llm7.py. Don't assume either way.
"""

import requests

from providers.base import ChatMessage, ChatResponse, Provider, ProviderError, ToolCall

BASE_URL = "https://api.gmi-serving.com/v1/chat/completions"


def _serialize_message(m: ChatMessage) -> dict:
    entry = {"role": m.role, "content": m.content}
    if m.name:
        entry["name"] = m.name
    if m.tool_call_id:
        entry["tool_call_id"] = m.tool_call_id
    if m.tool_calls:
        entry["tool_calls"] = m.tool_calls
    return entry


class GMIProvider(Provider):
    name = "gmi"

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
                timeout=60,  # unproven latency — generous timeout until observed
            )
        except requests.RequestException as e:
            raise ProviderError(f"GMI request failed: {e}")

        if resp.status_code == 429:
            raise ProviderError("GMI rate limited", is_rate_limit=True)
        if resp.status_code >= 400:
            raise ProviderError(f"GMI error {resp.status_code}: {resp.text[:300]}")

        data = resp.json()
        choices = data.get("choices") or []
        if not choices or not choices[0].get("message"):
            raise ProviderError(f"GMI returned a malformed response (no message): {str(data)[:300]}")
        choice = choices[0]["message"]

        tool_calls = []
        for tc in choice.get("tool_calls") or []:
            tool_calls.append(ToolCall(
                id=tc["id"],
                name=tc["function"]["name"],
                arguments=tc["function"]["arguments"],
            ))

        # Same best-effort extraction as mistral.py/llm7.py — OpenAI-
        # compatible enough that this should work, not yet confirmed
        # against a real populated response.
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
