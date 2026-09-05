"""
providers/openrouter.py

OpenRouter transport. This is the "executor" in the dispatcher/executor
split — the dispatcher (Groq) decides what needs doing, this actually does
the heavy-lifting task work, using whichever model the daily-ranked list
picked for that task type.

Prompt caching (2026-09-01 research pass, cross-provider): automatic
(implicit) for most routed models — OpenAI-style, DeepSeek, Gemini 2.5-class
— but Anthropic and Alibaba/Qwen specifically require an explicit
cache_control marker per message, which NAVI doesn't send (not relevant to
the nvidia/nemotron models actually routed here today, but would matter if
routing ever picks an Anthropic/Qwen model on OpenRouter). OpenRouter also
does provider-sticky routing to keep cache hits warm across requests —
10-minute idle expiry, tracked by conversation by default (hashes the first
system + first user message), or explicitly via a session_id field if
NAVI ever wants tighter control than the default hashing gives. See
providers/groq.py for how caching differs there (fully automatic, no
exceptions) — behavior is NOT uniform across NAVI's providers.
"""

import time

import requests

from providers.base import ChatMessage, ChatResponse, Provider, ProviderError, ToolCall

BASE_URL = "https://openrouter.ai/api/v1/chat/completions"
KEY_INFO_URL = "https://openrouter.ai/api/v1/key"

# Short server-side cache for the Usage counters panel's "requests left"
# card — OpenRouter's own /api/v1/key endpoint is the real source (rate
# limit + spend tied to the actual key), so there's no local counting to
# do here at all, just don't hammer it on every panel open.
_key_info_cache: dict = {"data": None, "fetched_at": 0.0}
_KEY_INFO_TTL_SECONDS = 60


def _serialize_message(m: ChatMessage) -> dict:
    entry = {"role": m.role, "content": m.content}
    if m.name:
        entry["name"] = m.name
    if m.tool_call_id:
        entry["tool_call_id"] = m.tool_call_id
    if m.tool_calls:
        entry["tool_calls"] = m.tool_calls
    return entry


def get_key_info(api_key: str) -> dict | None:
    """Real, authoritative usage/limit info for this API key, straight
    from OpenRouter — sidesteps needing to know their exact reset-time
    behavior (unconfirmed from a primary source, see storage/usage.py's
    docstring) by just asking OpenRouter what it currently thinks.
    Returns None on any failure rather than raising — this backs a
    display-only panel, not a chat request. Cached briefly since this is
    an account-info endpoint, not meant for high-frequency polling."""
    now = time.time()
    if _key_info_cache["data"] is not None and (now - _key_info_cache["fetched_at"]) < _KEY_INFO_TTL_SECONDS:
        return _key_info_cache["data"]
    try:
        resp = requests.get(KEY_INFO_URL, headers={"Authorization": f"Bearer {api_key}"}, timeout=10)
        if resp.status_code >= 400:
            return _key_info_cache["data"]  # stale-but-present beats nothing
        data = resp.json().get("data")
        _key_info_cache["data"] = data
        _key_info_cache["fetched_at"] = now
        return data
    except requests.RequestException:
        return _key_info_cache["data"]


class OpenRouterProvider(Provider):
    name = "openrouter"

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
        # cached_tokens (when present) confirms a prompt-cache hit —
        # OpenRouter caches automatically for most routed models, though
        # Anthropic/Qwen specifically need explicit cache_control markers
        # NAVI doesn't send (not relevant to the nvidia/nemotron models
        # actually routed here today).
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
