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
                # 180s, not 60. Ollama Cloud is used for exactly one role,
                # context_synthesis, and every job on it is the same shape:
                # read a whole document and generate a long structured
                # result. Compaction, branch specs and Source distillation
                # are all minutes-of-work calls, not chat replies.
                #
                # 60s was inherited from chat-shaped assumptions and it
                # genuinely bit: distilling a 4,500-token article through
                # nemotron-3-super (120B MoE, free GPU pool, possible cold
                # start) read-timed-out every time, and the user saw
                # "couldn't distil this page" for what was only impatience.
                # 190s, and the number is not ours to choose.
                #
                # Ollama Cloud enforces a HARD 182-second server-side
                # timeout that kills a generation mid-progress
                # (github.com/ollama/ollama/issues/15973). It is
                # undocumented, applies to paying subscribers too, and
                # cannot be extended. Waiting longer than that on our side
                # buys nothing except a slower failure - the previous 300s
                # spent two extra minutes on a request Ollama had already
                # abandoned.
                #
                # What we measured before learning that (2026-09-13/14,
                # nemotron-3-super) explains why this bites so often:
                #
                #   cold start ...................... ~80s
                #   trivial prompt once warm ........  ~11s
                #   distil an 18,700-token page .....  ~82s warm
                #
                # A first call therefore lands around 160s - inside 20
                # seconds of a cap it cannot see coming. Two of five
                # benchmark reps exceeded it. This provider is structurally
                # unsuited to long generations, and no timeout value here
                # changes that; 190s simply fails promptly so the fallback
                # starts sooner.
                timeout=190,
            )
        except requests.RequestException as e:
            raise ProviderError(f"Ollama Cloud request failed: {e}")

        if resp.status_code == 429:
            raise ProviderError("Ollama Cloud rate/quota limited", is_rate_limit=True)
        if resp.status_code >= 400:
            raise ProviderError(
                f"Ollama Cloud error {resp.status_code}: {resp.text[:300]}",
                # Any 5xx is the provider failing, not us. Cools the
                # endpoint down like a 429 does, but stays visible in
                # friction where a 429 deliberately does not.
                is_overloaded=resp.status_code >= 500,
            )

        data = resp.json()
        choices = data.get("choices") or []
        if not choices or not choices[0].get("message"):
            raise ProviderError(f"Ollama Cloud returned a malformed response (no message): {str(data)[:300]}")
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
        # Ollama Cloud's prompt-caching support is unconfirmed — an open
        # GitHub feature request (ollama/ollama#15600, #16714) with real
        # reports of cache_read_tokens showing 0 even on workloads that
        # should hit cache. Extracted defensively anyway in case it
        # starts working, not because it's expected to right now.
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
