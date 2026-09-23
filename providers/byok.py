"""
providers/byok.py

Bring-your-own-key transports (2026-09-23) — providers NAVI never routes
to by default, only when an Owner/Admin pastes their own key into
Settings and then deliberately picks one of that provider's models.

Both speak OpenAI's chat-completions wire format, so they share one
transport class here rather than getting a near-identical file each the
way the free-tier providers did:

- DeepSeek: OpenAI format natively (api-docs.deepseek.com, checked
  2026-09-23). Model names have already churned once (deepseek-chat /
  deepseek-reasoner became deepseek-flash / deepseek-v4-pro), which is why
  the model list is fetched live from GET /models when the key is saved
  instead of hardcoded here.
- Anthropic (Claude): through Anthropic's OpenAI SDK compatibility layer
  (platform.claude.com/docs/en/api/openai-sdk, checked 2026-09-23).
  Anthropic's own docs say this layer is "primarily intended to test and
  compare model capabilities, and is not considered a long-term or
  production-ready solution" — which is exactly the job it has here:
  seeing how Claude performs inside NAVI. Known, accepted limits of that
  layer: no prompt caching, `reasoning_effort` and `strict` are ignored,
  and system messages are hoisted into one (harmless — NAVI already sends
  exactly one). If Claude ever becomes a real routed provider rather than
  a test bench, it deserves a native /v1/messages transport instead.
"""

import requests

from providers.base import (
    ChatMessage, ChatResponse, Provider, ProviderError, ToolCall, consume_openai_stream,
)

ANTHROPIC_VERSION = "2023-06-01"


def _serialize_message(m: ChatMessage) -> dict:
    entry = {"role": m.role, "content": m.content}
    if m.name:
        entry["name"] = m.name
    if m.tool_call_id:
        entry["tool_call_id"] = m.tool_call_id
    if m.tool_calls:
        entry["tool_calls"] = m.tool_calls
    return entry


class _OpenAICompatibleBYOK(Provider):
    """Shared transport. Subclasses set `name`, `label` and `chat_url`,
    and implement `list_models`."""

    label = ""
    chat_url = ""
    supports_streaming = True
    # Sent only when the caller didn't set one. Anthropic's native API
    # requires max_tokens; the compatibility layer fills a default, but a
    # silent default could truncate a long research answer, so it's set
    # explicitly rather than trusted.
    default_max_tokens: int | None = None

    def _headers(self) -> dict:
        return {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}

    def _do_chat(
        self,
        model: str,
        messages: list[ChatMessage],
        tools: list[dict] | None = None,
        tool_choice: str | dict | None = None,
        extra_params: dict | None = None,
        on_token=None,
    ) -> ChatResponse:
        payload = {
            "model": model,
            "messages": [_serialize_message(m) for m in messages],
        }
        if self.default_max_tokens:
            payload["max_tokens"] = self.default_max_tokens
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = tool_choice or "auto"
        if extra_params:
            payload.update(extra_params)
        streaming = on_token is not None
        if streaming:
            payload["stream"] = True
            payload["stream_options"] = {"include_usage": True}

        try:
            resp = requests.post(
                self.chat_url, headers=self._headers(), json=payload, timeout=60, stream=streaming,
            )
        except requests.RequestException as e:
            raise ProviderError(f"{self.label} request failed: {e}")

        if resp.status_code == 429:
            raise ProviderError(f"{self.label} rate limited: {resp.text[:300]}", is_rate_limit=True)
        if resp.status_code >= 400:
            raise ProviderError(
                f"{self.label} error {resp.status_code}: {resp.text[:300]}",
                is_overloaded=resp.status_code >= 500,
            )

        if streaming:
            text, raw_tool_calls, usage = consume_openai_stream(resp, on_token)
            data = {"choices": [{"message": {"content": text, "tool_calls": raw_tool_calls}}], "usage": usage}
            choice = data["choices"][0]["message"]
        else:
            data = resp.json()
            choices = data.get("choices") or []
            if not choices or not choices[0].get("message"):
                raise ProviderError(f"{self.label} returned a malformed response (no message): {str(data)[:300]}")
            choice = choices[0]["message"]

        tool_calls = [
            ToolCall(id=tc["id"], name=tc["function"]["name"], arguments=tc["function"]["arguments"])
            for tc in choice.get("tool_calls") or []
        ]

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

    @classmethod
    def list_models(cls, api_key: str) -> list[str]:
        """Model ids this key can use. Raises ProviderError with the
        provider's own message when the key is rejected — that call doubles
        as the key check, so a typo'd key is refused at save time instead of
        on the first chat message."""
        raise NotImplementedError

    @classmethod
    def _get_models(cls, url: str, headers: dict) -> list[dict]:
        try:
            resp = requests.get(url, headers=headers, timeout=15)
        except requests.RequestException as e:
            raise ProviderError(f"Couldn't reach {cls.label}: {e}")
        if resp.status_code in (401, 403):
            raise ProviderError(f"{cls.label} rejected this key ({resp.status_code}).")
        if resp.status_code >= 400:
            raise ProviderError(f"{cls.label} error {resp.status_code}: {resp.text[:200]}")
        return (resp.json() or {}).get("data") or []


class DeepSeekProvider(_OpenAICompatibleBYOK):
    name = "deepseek"
    label = "DeepSeek"
    chat_url = "https://api.deepseek.com/chat/completions"

    @classmethod
    def list_models(cls, api_key: str) -> list[str]:
        data = cls._get_models("https://api.deepseek.com/models", {"Authorization": f"Bearer {api_key}"})
        return [m["id"] for m in data if m.get("id")]


class AnthropicProvider(_OpenAICompatibleBYOK):
    name = "anthropic"
    label = "Claude"
    chat_url = "https://api.anthropic.com/v1/chat/completions"
    default_max_tokens = 8192

    @classmethod
    def list_models(cls, api_key: str) -> list[str]:
        # The native models endpoint, not the compatibility layer: it is the
        # documented one, and it needs Anthropic's own headers.
        data = cls._get_models(
            "https://api.anthropic.com/v1/models?limit=100",
            {"x-api-key": api_key, "anthropic-version": ANTHROPIC_VERSION},
        )
        return [m["id"] for m in data if m.get("id")]


BYOK_TRANSPORTS: dict[str, type[_OpenAICompatibleBYOK]] = {
    DeepSeekProvider.name: DeepSeekProvider,
    AnthropicProvider.name: AnthropicProvider,
}
