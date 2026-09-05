"""
providers/mistral.py

Mistral AI transport (La Plateforme), via the plain REST API — same
"plain requests" rule as every other provider here. Genuinely OpenAI-
compatible response shape (choices[0].message, standard usage object),
confirmed directly against docs.mistral.ai/api rather than assumed.

Free "Experiment" tier: no card required, ~1B tokens/month, ~1 RPS /
500K TPM. Restricted to non-commercial use — not enforced in code here,
NAVI isn't commercial today so this isn't blocking, but flag it if that
ever changes (deliberately not built into the ranking system yet, see
the 2026-09-01 model-ranking design conversation).

Prompt caching (2026-09-01 research pass, cross-provider): manual/opt-in
— unlike Groq (fully automatic) or OpenRouter (automatic for most),
Mistral requires an explicit `prompt_cache_key` in the request body,
reused across requests that share a prompt prefix (multi-turn
conversations, repeated system prompts). Cached tokens bill at 10% of
standard input price — confirmed on Ministral 8B: $0.15/M standard ->
$0.015/M cached. NOT wired into `chat()` below yet — no caller threads a
stable per-conversation key through yet, so caching won't fire until one
does. Add a `prompt_cache_key` param here once something actually needs
it, rather than exposing an unused parameter speculatively.

Usage/cost tracking: unlike every other provider here, Mistral exposes a
real admin cost-reporting endpoint — GET /v1/admin/usage (params: month,
year, workspace_id, all optional), header x-api-key not Authorization —
returning consumption broken down by category (chat, completion, ocr,
audio, connectors, libraries_api, fine_tuning, vibe_usage) with period and
currency. Confirmed working with the same key used for chat completions
(2026-09-01) — no separate admin-scoped key needed in practice, at least
for this workspace. If NAVI ever builds real per-provider usage counters
(see the model-ranking design conversation), Mistral doesn't need the
sum-per-call-usage-field approach the other providers would need — pull
the authoritative number straight from this endpoint instead.
"""

import time

import requests

from providers.base import ChatMessage, ChatResponse, Provider, ProviderError, ToolCall

BASE_URL = "https://api.mistral.ai/v1/chat/completions"
ADMIN_USAGE_URL = "https://api.mistral.ai/v1/admin/usage"

# Fetched lazily (only when the Usage counters panel's Mistral card is
# opened, not on every chat call) and cached — this is a monthly-
# granularity billing endpoint, not a per-request counter, so polling it
# after every Mistral chat call would just be wasted admin-API traffic
# without even guaranteeing fresher data.
_admin_usage_cache: dict = {"data": None, "fetched_at": 0.0}
_ADMIN_USAGE_TTL_SECONDS = 600


def _serialize_message(m: ChatMessage) -> dict:
    entry = {"role": m.role, "content": m.content}
    if m.name:
        entry["name"] = m.name
    if m.tool_call_id:
        entry["tool_call_id"] = m.tool_call_id
    if m.tool_calls:
        entry["tool_calls"] = m.tool_calls
    return entry


def get_admin_usage(api_key: str) -> dict | None:
    """Real billed-dollar consumption for the current month, straight from
    Mistral's own admin endpoint — deliberately NOT computed locally from
    per-model $/token pricing, since that would need knowing which tokens
    hit the (opt-in, not currently wired) prompt-cache discount to be
    accurate. Mistral already knows the real answer; this just asks.
    Returns None on failure — display-only, never blocks a chat request."""
    now = time.time()
    if _admin_usage_cache["data"] is not None and (now - _admin_usage_cache["fetched_at"]) < _ADMIN_USAGE_TTL_SECONDS:
        return _admin_usage_cache["data"]
    try:
        resp = requests.get(ADMIN_USAGE_URL, headers={"x-api-key": api_key}, timeout=10)
        if resp.status_code >= 400:
            return _admin_usage_cache["data"]
        data = resp.json()
        _admin_usage_cache["data"] = data
        _admin_usage_cache["fetched_at"] = now
        return data
    except requests.RequestException:
        return _admin_usage_cache["data"]


class MistralProvider(Provider):
    name = "mistral"

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
                timeout=30,
            )
        except requests.RequestException as e:
            raise ProviderError(f"Mistral request failed: {e}")

        if resp.status_code == 429:
            raise ProviderError("Mistral rate limited", is_rate_limit=True)
        if resp.status_code >= 400:
            raise ProviderError(f"Mistral error {resp.status_code}: {resp.text[:300]}")

        data = resp.json()
        choices = data.get("choices") or []
        if not choices or not choices[0].get("message"):
            raise ProviderError(f"Mistral returned a malformed response (no message): {str(data)[:300]}")
        choice = choices[0]["message"]

        tool_calls = []
        for tc in choice.get("tool_calls") or []:
            tool_calls.append(ToolCall(
                id=tc["id"],
                name=tc["function"]["name"],
                arguments=tc["function"]["arguments"],
            ))

        # Same shape as groq.py/openrouter.py's extraction — Mistral's
        # response is OpenAI-compatible enough that this should work
        # unchanged, but unlike those two this hasn't been confirmed
        # against a real populated response yet. Best-effort until a real
        # call verifies the cached_tokens field name matches.
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
