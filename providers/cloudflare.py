"""
providers/cloudflare.py

Cloudflare Workers AI transport, via the plain REST API (no SDK — same
"plain requests" rule as every other provider here).

Migrated 2026-09-06 from Cloudflare's native /ai/run/{model} endpoint to
their OpenAI-compatible /ai/v1/chat/completions one — a real bug, not a
model problem: JuanJo hit gpt-oss-120b (the SAME failure already
documented for gemma-4-26b) calling web_search/fetch_page in a loop,
never synthesizing an answer, hitting MAX_TOOL_ITERATIONS every time.
Cloudflare's own changelog explains why: the native endpoint used to
regenerate tool_call ids instead of preserving the model's real ones,
"which broke multi-turn tool calling because clients could not match
tool results to their original calls" — exactly this symptom, and not
specific to any one model, since it's the endpoint doing it. Their
OpenAI-compatible endpoint preserves real ids. This also lets response
parsing follow the exact same shape as every other OpenAI-compatible
provider here (see providers/groq.py) instead of guessing between two
different envelope shapes, which the old native-endpoint code had to do.

UNVERIFIED, disclosed rather than assumed: whether this endpoint's
`usage` object still reports a Cloudflare-specific `neurons` field the
way the native endpoint's response did (needed for the Usage Counters
panel's real per-call cost). No Cloudflare credentials are available to
test this from here — falls back to a token-count usage_note (same
shape as Groq/Mistral) if `neurons` isn't present, so a real call either
way still gets a real usage_note, just possibly not Neuron-denominated
until this is confirmed live.
"""

import os

import requests

from providers.base import ChatMessage, ChatResponse, Provider, ProviderError, ToolCall

BASE_URL = "https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/v1/chat/completions"


def _serialize_message(m: ChatMessage) -> dict:
    entry = {"role": m.role, "content": m.content}
    if m.name:
        entry["name"] = m.name
    if m.tool_call_id:
        entry["tool_call_id"] = m.tool_call_id
    if m.tool_calls:
        entry["tool_calls"] = m.tool_calls
    return entry


class CloudflareProvider(Provider):
    name = "cloudflare"

    def _do_chat(
        self,
        model: str,
        messages: list[ChatMessage],
        tools: list[dict] | None = None,
        tool_choice: str | dict | None = None,
    ) -> ChatResponse:
        account_id = os.environ.get("CLOUDFLARE_ACCOUNT_ID")
        if not account_id:
            raise ProviderError("CLOUDFLARE_ACCOUNT_ID not set")

        payload = {
            "model": model,
            "messages": [_serialize_message(m) for m in messages],
            # Cloudflare's own default max_tokens is 256 (their changelog) —
            # far too little for a "thinking mode" model (qwen3.8-27b,
            # gpt-oss's harmony reasoning channel, etc.), which spends
            # tokens on hidden reasoning BEFORE any visible answer or tool
            # call. Real incident (2026-09-02): three unrelated reasoning-
            # capable models all returned a "successful" response with both
            # text and tool_calls completely empty — the model was cut off
            # mid-thought before ever reaching visible output. 8192 gives
            # real headroom; billing is by tokens actually generated, not
            # this ceiling, so raising it costs nothing unless a call
            # genuinely needs it. Still true on this endpoint — nothing
            # about the max_tokens-starves-reasoning-models issue was
            # specific to the old native endpoint.
            "max_tokens": 8192,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = tool_choice or "auto"

        try:
            resp = requests.post(
                BASE_URL.format(account_id=account_id),
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=60,
            )
        except requests.RequestException as e:
            raise ProviderError(f"Cloudflare request failed: {e}")

        if resp.status_code == 429:
            raise ProviderError("Cloudflare rate/quota limited", is_rate_limit=True)
        if resp.status_code >= 400:
            raise ProviderError(f"Cloudflare error {resp.status_code}: {resp.text[:300]}")

        data = resp.json()
        choices = data.get("choices") or []
        if not choices or not choices[0].get("message"):
            raise ProviderError(f"Cloudflare returned a malformed response (no message): {str(data)[:300]}")
        choice = choices[0]["message"]

        tool_calls = []
        for tc in choice.get("tool_calls") or []:
            tool_calls.append(ToolCall(
                id=tc["id"],
                name=tc["function"]["name"],
                arguments=tc["function"]["arguments"],
            ))

        usage = data.get("usage") or {}
        # See module docstring — real Neuron cost if this OpenAI-compatible
        # endpoint still exposes it (unverified), a token-count usage_note
        # (same shape as Groq/Mistral) otherwise, just for the human-
        # readable note below. Real token PERSISTENCE now happens once,
        # centrally, in providers/base.py's Provider.chat() (2026-09-10) —
        # this only still records `neurons` directly, Cloudflare's own
        # extra signal base.py has no way to know about.
        neurons = usage.get("neurons")
        total_tokens = usage.get("total_tokens")
        if neurons is not None:
            usage_note = f"{neurons:.2f} Neurons"
        elif total_tokens is not None:
            usage_note = f"{usage.get('prompt_tokens', '?')} in / {usage.get('completion_tokens', '?')} out / {total_tokens} total tokens"
        else:
            usage_note = None
        if neurons is not None:
            try:
                from storage.usage import record_usage
                record_usage("cloudflare", model, neurons=neurons)
            except Exception:
                pass

        return ChatResponse(
            text=choice.get("content"),
            tool_calls=tool_calls,
            model_used=data.get("model", model),
            raw=data,
            usage_note=usage_note,
        )
