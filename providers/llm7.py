"""
providers/llm7.py

LLM7 transport (api.llm7.io) — OpenAI-compatible endpoint. Free "turbo"
tier gives ~1M tokens/day with a key (2026-09-01: confirmed the real 5
turbo-tier models live via GET /v1/models, which works fully keyless —
codestral-latest, gemma4:31b, gpt-oss, minimax-m2.7, mistral-Nemo-
Instruct-2407. Correcting an earlier wrong claim in this docstring:
DeepSeek-V4-Flash-0731 is NOT on the free turbo tier — it's confirmed
`pro` (paid) in the live catalog. Don't trust that claim if it resurfaces
elsewhere; this is the corrected version.

Also confirmed 2026-09-01: anonymous (keyless) requests get their OWN
separate 500K-tokens/24h pool, tracked by IP not account — genuinely
additive on top of a keyed account's 1M/24h, not double-counted from the
same budget. Worth deliberately splitting traffic between keyed and
keyless calls rather than always using the key.

Prompt caching (2026-09-01 research pass, cross-provider): UNKNOWN — no
published caching documentation found for this provider. Don't assume
either way; if it matters, test empirically (send two requests sharing a
prefix, check whether `usage` reports a cached-token discount) rather than
guessing. Every other NAVI provider's caching behavior is documented in its
own transport file (providers/groq.py, openrouter.py, cloudflare.py) —
this is the one gap.
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
        # LLM7 has no documented prompt-caching support at all (checked
        # docs.llm7.io/limits directly, 2026-08-27) — cached_tokens will
        # almost certainly never appear here, but this extracts it
        # defensively in case that changes rather than assuming it never
        # will.
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
