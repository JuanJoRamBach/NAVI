"""
providers/ollama_cloud.py

Ollama Cloud transport, via its OpenAI-compatible endpoint (not the SDK,
not the native /api/chat shape — same "plain requests" pattern as every
other provider here). Currently backing /research: a staged trial across
deepseek-v4-flash, minimax-m3, and gemma4:31b to see which one actually
holds up under real use before picking one permanently — see the
navi-v2-escalation-design memory / project notes for the fuller reasoning.

Free tier here isn't token-metered like Groq/OpenRouter — it's GPU-time
metered with session (5h) and weekly windows, and Ollama doesn't publish
a hard number. We find the real ceiling by using it, not by guessing.
"""

import requests

from providers.base import ChatMessage, ChatResponse, Provider, ProviderError, ToolCall

BASE_URL = "https://ollama.com/v1/chat/completions"


def _serialize_message(m: ChatMessage) -> dict:
    entry = {"role": m.role, "content": m.content}
    if m.name:
        entry["name"] = m.name
    if m.tool_call_id:
        entry["tool_call_id"] = m.tool_call_id
    if m.tool_calls:
        entry["tool_calls"] = m.tool_calls
    return entry


class OllamaCloudProvider(Provider):
    name = "ollama_cloud"

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
                timeout=60,  # cloud-hosted large models — give it more room than Groq
            )
        except requests.RequestException as e:
            raise ProviderError(f"Ollama Cloud request failed: {e}")

        if resp.status_code == 429:
            raise ProviderError("Ollama Cloud rate/quota limited", is_rate_limit=True)
        if resp.status_code >= 400:
            raise ProviderError(f"Ollama Cloud error {resp.status_code}: {resp.text[:300]}")

        data = resp.json()
        choice = data["choices"][0]["message"]

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
