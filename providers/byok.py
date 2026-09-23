"""
providers/byok.py

Bring-your-own-key transports (2026-09-23) — providers NAVI never routes
to by default, only when an Owner/Admin pastes their own key into
Settings and then deliberately picks one of that provider's models.

All of them speak OpenAI's chat-completions wire format, so they share one
transport class here rather than getting a near-identical file each the
way the free-tier providers did:

- OpenAI, xAI (Grok), Moonshot (Kimi): OpenAI format natively.
- DeepSeek: OpenAI format natively (api-docs.deepseek.com, checked
  2026-09-23). Model names have already churned once (deepseek-chat /
  deepseek-reasoner became deepseek-flash / deepseek-v4-pro), which is why
  every provider's model list is fetched live from GET /models when the
  key is saved instead of hardcoded here.
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
- "Other": any OpenAI-compatible API at an address the person types in
  (CustomOpenAIProvider). Stored as provider "custom-<slug>".
"""

import ipaddress
import re
import socket
from urllib.parse import urlparse

import requests

from providers.base import (
    ChatMessage, ChatResponse, Provider, ProviderError, ToolCall, consume_openai_stream,
)

ANTHROPIC_VERSION = "2023-06-01"
CUSTOM_PREFIX = "custom-"

# Model ids that are clearly not chat models. OpenAI's /models in particular
# lists embeddings, speech, image and moderation models next to chat ones,
# and none of them can answer a chat turn — offering them in the picker
# would just be a list of ways to break a conversation.
_NON_CHAT_MARKERS = (
    "embed", "whisper", "tts", "dall-e", "davinci", "babbage", "moderation",
    "transcribe", "realtime", "audio", "image", "sora", "search", "computer-use",
)


def _chat_models(ids: list[str]) -> list[str]:
    return [i for i in ids if not any(m in i.lower() for m in _NON_CHAT_MARKERS)]


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
    """Shared transport. Subclasses set `name`, `label` and `base_url`
    (the part before /chat/completions and /models)."""

    label = ""
    base_url = ""
    supports_streaming = True
    # Sent only when the caller didn't set one. Anthropic's native API
    # requires max_tokens; the compatibility layer fills a default, but a
    # silent default could truncate a long research answer, so it's set
    # explicitly rather than trusted.
    default_max_tokens: int | None = None

    @property
    def chat_url(self) -> str:
        return f"{self.base_url}/chat/completions"

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
        """Chat model ids this key can use. Raises ProviderError with the
        provider's own message when the key is rejected — that call doubles
        as the key check, so a typo'd key is refused at save time instead of
        on the first chat message."""
        data = _get_model_list(cls.label, f"{cls.base_url}/models", {"Authorization": f"Bearer {api_key}"})
        return _chat_models([m["id"] for m in data if m.get("id")])


def _get_model_list(label: str, url: str, headers: dict) -> list[dict]:
    try:
        resp = requests.get(url, headers=headers, timeout=15)
    except requests.RequestException as e:
        raise ProviderError(f"Couldn't reach {label}: {e}")
    if resp.status_code in (401, 403):
        raise ProviderError(f"{label} rejected this key ({resp.status_code}).")
    if resp.status_code == 404:
        raise ProviderError(
            f"{label} has no model list at {url}. NAVI needs an OpenAI-compatible API, "
            "usually an address ending in /v1."
        )
    if resp.status_code >= 400:
        raise ProviderError(f"{label} error {resp.status_code}: {resp.text[:200]}")
    try:
        return (resp.json() or {}).get("data") or []
    except ValueError:
        raise ProviderError(f"{label} answered, but not with an OpenAI-style model list.")


class OpenAIProvider(_OpenAICompatibleBYOK):
    name = "openai"
    label = "OpenAI"
    base_url = "https://api.openai.com/v1"


class AnthropicProvider(_OpenAICompatibleBYOK):
    name = "anthropic"
    label = "Claude"
    base_url = "https://api.anthropic.com/v1"
    default_max_tokens = 8192

    @classmethod
    def list_models(cls, api_key: str) -> list[str]:
        # The native models endpoint, with Anthropic's own headers: it is the
        # documented one, not part of the compatibility layer.
        data = _get_model_list(
            cls.label, f"{cls.base_url}/models?limit=100",
            {"x-api-key": api_key, "anthropic-version": ANTHROPIC_VERSION},
        )
        return _chat_models([m["id"] for m in data if m.get("id")])


class DeepSeekProvider(_OpenAICompatibleBYOK):
    name = "deepseek"
    label = "DeepSeek"
    base_url = "https://api.deepseek.com"


class XAIProvider(_OpenAICompatibleBYOK):
    name = "xai"
    label = "xAI (Grok)"
    base_url = "https://api.x.ai/v1"


class MoonshotProvider(_OpenAICompatibleBYOK):
    name = "moonshot"
    label = "Moonshot (Kimi)"
    base_url = "https://api.moonshot.ai/v1"


BYOK_TRANSPORTS: dict[str, type[_OpenAICompatibleBYOK]] = {
    cls.name: cls
    for cls in (OpenAIProvider, AnthropicProvider, DeepSeekProvider, XAIProvider, MoonshotProvider)
}


# ---- "Other": any OpenAI-compatible API --------------------------------

class CustomOpenAIProvider(_OpenAICompatibleBYOK):
    """One instance per saved "Other" provider. Unlike the classes above,
    name, label and address come from the config store, not the class."""

    def __init__(self, api_key: str, name: str, label: str, base_url: str):
        super().__init__(api_key=api_key)
        self.name = name
        self.label = label
        self.base_url = base_url

    @staticmethod
    def list_models_at(label: str, base_url: str, api_key: str) -> list[str]:
        data = _get_model_list(label, f"{base_url}/models", {"Authorization": f"Bearer {api_key}"})
        return _chat_models([m["id"] for m in data if m.get("id")])


def custom_provider_id(label: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", label.lower()).strip("-")[:40]
    return f"{CUSTOM_PREFIX}{slug}" if slug else ""


def is_custom_provider(name: str) -> bool:
    return name.startswith(CUSTOM_PREFIX)


def normalize_base_url(raw: str) -> str:
    """Turns whatever address was pasted into the base the transport needs,
    and refuses addresses the server must never be pointed at.

    Accepts the common shapes people copy from provider docs:
    "https://api.example.com/v1", ".../v1/", ".../v1/chat/completions".

    HTTPS only, and never a private, loopback or link-local address. The
    server makes this request itself, so an address like 169.254.169.254
    (the cloud metadata service) or localhost would turn "Other" into a
    way to read things inside the server's own network. Owner/Admin only
    already, but a Settings field should not be able to do that at all.
    Known remaining gap: a hostname that resolves publicly now and
    privately later (DNS rebinding) is only checked at save time."""
    url = (raw or "").strip().rstrip("/")
    for suffix in ("/chat/completions", "/models"):
        if url.endswith(suffix):
            url = url[: -len(suffix)]
    parsed = urlparse(url)
    if parsed.scheme != "https" or not parsed.hostname:
        raise ProviderError("The API address must start with https://")
    try:
        infos = socket.getaddrinfo(parsed.hostname, parsed.port or 443)
    except socket.gaierror:
        raise ProviderError(f"Couldn't find {parsed.hostname}. Check the address.")
    for info in infos:
        ip = ipaddress.ip_address(info[4][0])
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved or ip.is_multicast:
            raise ProviderError("That address points inside a private network, which NAVI won't call.")
    return url
