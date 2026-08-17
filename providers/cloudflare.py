"""
providers/cloudflare.py

Cloudflare Workers AI transport, via the plain REST API (no SDK — same
"plain requests" rule as every other provider here). Currently backing
/code via @cf/qwen/qwen2.5-coder-32b-instruct, verified directly against
the real account before wiring in (real call cost 2.7 Neurons out of the
10,000/day free allowance).

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

    def chat(
        self,
        model: str,
        messages: list[ChatMessage],
        tools: list[dict] | None = None,
        tool_choice: str | dict | None = None,
    ) -> ChatResponse:
        account_id = os.environ.get("CLOUDFLARE_ACCOUNT_ID")
        if not account_id:
            raise ProviderError("CLOUDFLARE_ACCOUNT_ID not set")

        payload = {"messages": [_serialize_message(m) for m in messages]}
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

        tool_calls = []
        for tc in result.get("tool_calls") or []:
            tool_calls.append(ToolCall(
                id=tc.get("id", ""),
                name=tc.get("name") or tc.get("function", {}).get("name", ""),
                arguments=tc.get("arguments") or tc.get("function", {}).get("arguments", {}),
            ))

        neurons = result.get("usage", {}).get("neurons")
        usage_note = f"{neurons:.2f} Neurons" if neurons is not None else None

        return ChatResponse(
            text=result.get("response"),
            tool_calls=tool_calls,
            model_used=model,
            raw=data,
            usage_note=usage_note,
        )
