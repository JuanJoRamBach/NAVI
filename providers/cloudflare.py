"""
providers/cloudflare.py

Cloudflare Workers AI transport, via the plain REST API (no SDK — same
"plain requests" rule as every other provider here). Currently backing
/code via @cf/qwen/qwen2.5-coder-32b-instruct, verified directly against
the real account before wiring in (real call cost 2.7 Neurons out of the
10,000/day free allowance).

Prompt caching (2026-09-01 research pass, cross-provider): automatic
baseline — Cloudflare's own docs say prefix caching is "enabled by default
for select models," no code change required to get some benefit. Hit rate
improves further by sending an x-session-affinity header with a stable
per-session identifier, routing repeat requests to the same model instance
— not wired here yet, worth adding if this transport sees enough repeat-
prefix traffic to matter. See providers/groq.py (fully automatic, no lever
needed) and providers/openrouter.py (mostly automatic, a few exceptions)
for how this differs elsewhere — behavior is NOT uniform across providers.

Unlike every other provider, the endpoint URL itself needs a Cloudflare
account ID, not just the API token. Account ID isn't a secret the way an
API key is, so rather than extend config/store.py's schema for one
provider-specific value, it's read straight from CLOUDFLARE_ACCOUNT_ID at
call time — a deliberate, small inconsistency, not an oversight.

Response shape is Cloudflare's own, not OpenAI's: {"result": {"response":
..., "tool_calls": [...]}, "success": bool, "errors": [...]} — so this
provider has its own parsing rather than reusing the choices[0].message
shape the OpenAI-compatible providers share. Tool-call parsing here is
best-effort and untested against a real populated example (the
verification call didn't trigger one) — /code doesn't currently use the
tool belt, so this isn't load-bearing yet.
"""

import os

import requests

from providers.base import ChatMessage, ChatResponse, Provider, ProviderError, ToolCall

BASE_URL = "https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/run/{model}"


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

        # Cloudflare's own default max_tokens is 256 (their changelog) — far
        # too little for a "thinking mode" model (qwen3.8-27b, gpt-oss's
        # harmony reasoning channel, etc.), which spends tokens on hidden
        # reasoning BEFORE any visible answer or tool call. Real incident
        # (2026-09-02): three unrelated reasoning-capable models all
        # returned a "successful" response with both text and tool_calls
        # completely empty — the model was cut off mid-thought before ever
        # reaching visible output. 8192 gives real headroom; billing is by
        # tokens actually generated, not this ceiling, so raising it costs
        # nothing unless a call genuinely needs it.
        payload = {"messages": [_serialize_message(m) for m in messages], "max_tokens": 8192}
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = tool_choice or "auto"

        try:
            resp = requests.post(
                BASE_URL.format(account_id=account_id, model=model),
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
        if not data.get("success"):
            raise ProviderError(f"Cloudflare error: {data.get('errors')}")

        result = data.get("result", {})

        # Two response shapes exist depending on the model (2026-09-02,
        # found via a real Filen-logged "empty response" failure — see
        # dispatcher/provider_debug.py). Older/simpler Workers AI models
        # return {"result": {"response": "...", "tool_calls": [...]}}, but
        # several newer chat-completions-compatible models (gemma-4-26b,
        # gpt-oss, qwen3.8-27b) return an OpenAI-style {"result": {
        # "choices": [{"message": {"content": ..., "tool_calls": [...]}}]}}
        # envelope instead. Every "the model returned nothing" incident
        # today across three different models was actually this: a real,
        # complete answer sitting in choices[0].message.content, silently
        # read as empty because only the first shape was ever checked.
        choices_message = (result.get("choices") or [{}])[0].get("message", {})
        text = result.get("response")
        if text is None:
            text = choices_message.get("content")

        raw_tool_calls = result.get("tool_calls") or choices_message.get("tool_calls") or []
        tool_calls = []
        for tc in raw_tool_calls:
            tool_calls.append(ToolCall(
                id=tc.get("id", ""),
                name=tc.get("name") or tc.get("function", {}).get("name", ""),
                arguments=tc.get("arguments") or tc.get("function", {}).get("arguments", {}),
            ))

        neurons = result.get("usage", {}).get("neurons")
        usage_note = f"{neurons:.2f} Neurons" if neurons is not None else None
        if neurons is not None:
            # Real per-call Neuron cost, straight from Cloudflare's own
            # response — no estimation. Usage counters panel sums today's
            # (UTC) total against the confirmed 10,000/day free allowance.
            try:
                from storage.usage import record_usage
                record_usage("cloudflare", model, neurons=neurons)
            except Exception:
                pass

        return ChatResponse(
            text=text,
            tool_calls=tool_calls,
            model_used=model,
            raw=data,
            usage_note=usage_note,
        )
