"""
providers/gemini.py

Google AI Studio (Gemini) transport, via Google's OpenAI-COMPATIBLE
endpoint rather than the `google-genai` SDK.

That's a deliberate choice, not an oversight — Google's own docs lead
with the SDK. Reasoning: every other transport in this package is a plain
`requests` POST to an OpenAI-shaped endpoint sharing one `_do_chat`
signature from providers/base.py. Going native would mean a new
dependency PLUS real translation layers in three places — Gemini's native
API uses `contents`/`parts` instead of `messages`, and
`functionDeclarations` instead of OpenAI's `tools` array — all of which
can drift out of sync with the rest of the codebase. The compat endpoint
gives all of that for free, and if it ever becomes a problem the swap is
contained to this one file (the Provider base class means nothing else
knows how a transport talks to its API).

Verified against Google's own compatibility doc (ai.google.dev/gemini-api/
docs/openai, 2026-09-13) before committing to this path: function/tool
calling is FULLY supported here, with documented `tool_choice: "auto"`
examples — that was the one capability NAVI actually depends on and it
would have been the dealbreaker. Streaming and structured output are
supported too (neither used here yet).

Two documented caveats, both real:
- Google labels the compat layer "still in beta while we extend feature
  support."
- Unsupported parameters are "silently ignored" rather than rejected.
  That's actually the SAFE failure mode for NAVI's extra_params
  passthrough (dispatcher/prompt_family.py sends reasoning_effort on
  gpt-oss attempts; a Gemini attempt silently no-ops instead of 400ing) —
  but it does mean a parameter could quietly stop working without any
  error to notice it by.

Free-tier shape, read from JuanJo's own AI Studio dashboard 2026-09-13
(NOT from published docs — Google stopped publishing a static table and
defers to the per-account dashboard, and the third-party figures floating
around were wrong):
  - RPD is metered PER MODEL, not per project. This matters a lot.
  - Flash-Lite models (3.1, 3.5): 15 RPM / 250K TPM / **500 RPD each**
  - Full Flash models (3, 3.5, 3.6, 3.7, 3.8): 5 RPM / 250K TPM / 20 RPD
  - Any Pro model: 0 — paid tier only since April/May 2026
So two different Flash-Lite models are two independent 500/day pools,
which is where NAVI's real volume should sit; the full Flash models are
scarce (20/day) and worth reserving for genuinely hard turns.

DATA POLICY — the thing that actually gates using this for client work:
Google's free tier normally trains on inputs AND outputs, with human
reviewers able to read them, which would be disqualifying under NAVI's
GDPR client-data constraint. The exception is that customers in the
EEA/Switzerland/UK get the PAID-tier data policy applied even on free
tiers. NAVI operates from Spain, so this should apply — but it is
account-specific and worth confirming against Google's own terms for the
account in use before routing real client conversation content here.
"""

import os

import requests

from providers.base import ChatMessage, ChatResponse, Provider, ProviderError, ToolCall

BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"

# Google AI Studio's own naming for the credential, matching what's set in
# .env on the server. The provider is registered as "gemini" everywhere
# else in NAVI (models are all gemini-*), so this mapping is the one place
# the two names meet — see config/store.py's PROVIDER_KEY_ENV.
API_KEY_ENV = "GOOGLE_AI_API_KEY"


def _serialize_message(m: ChatMessage) -> dict:
    entry = {"role": m.role, "content": m.content}
    if m.name:
        entry["name"] = m.name
    if m.tool_call_id:
        entry["tool_call_id"] = m.tool_call_id
    if m.tool_calls:
        entry["tool_calls"] = m.tool_calls
    return entry


class GeminiProvider(Provider):
    name = "gemini"

    def _do_chat(
        self,
        model: str,
        messages: list[ChatMessage],
        tools: list[dict] | None = None,
        tool_choice: str | dict | None = None,
        extra_params: dict | None = None,
    ) -> ChatResponse:
        payload = {
            "model": model,
            "messages": [_serialize_message(m) for m in messages],
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = tool_choice or "auto"
        if extra_params:
            payload.update(extra_params)

        try:
            resp = requests.post(
                BASE_URL,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=60,
            )
        except requests.RequestException as e:
            raise ProviderError(f"Gemini request failed: {e}")

        # 429 is the free tier's real, expected failure mode here — RPD is
        # only 20/day on the full Flash models, so this will fire in normal
        # operation, not just under abuse. is_rate_limit=True lets
        # config.mark_rate_limited demote it and the fallback chain carry
        # on, which is exactly the intended behaviour.
        if resp.status_code == 429:
            raise ProviderError("Gemini rate limited / daily quota exhausted", is_rate_limit=True)
        if resp.status_code >= 400:
            raise ProviderError(f"Gemini error {resp.status_code}: {resp.text[:300]}")

        data = resp.json()
        choices = data.get("choices") or []
        if not choices or not choices[0].get("message"):
            raise ProviderError(f"Gemini returned a malformed response (no message): {str(data)[:300]}")
        choice = choices[0]["message"]

        tool_calls = []
        for tc in choice.get("tool_calls") or []:
            tool_calls.append(ToolCall(
                id=tc["id"],
                name=tc["function"]["name"],
                arguments=tc["function"]["arguments"],
            ))

        # Gemini genuinely does implicit context caching on its own (no
        # opt-in parameter), and the compat layer reports it in the
        # standard OpenAI shape when it happens — extracted the same way
        # every other transport here does.
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
