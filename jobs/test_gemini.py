"""
jobs/test_gemini.py

Live smoke test for the Gemini transport (providers/gemini.py). Run on the
server, where GOOGLE_AI_API_KEY actually exists:

    cd ~/navi && set -a && source .env && set +a && venv/bin/python -m jobs.test_gemini

Three things, in order of how likely they are to be the problem:

1. What models this key can really see, with real context limits — Google
   stopped publishing a static table and defers to the per-account
   dashboard, so this is the only trustworthy source for THIS account.
2. A plain chat round-trip through the OpenAI-compat endpoint.
3. A real TOOL-CALLING round-trip. This is the one that matters: NAVI's
   idle tier has to call flag_key_insight and propose_research_mode, and
   Flash-Lite variants are exactly where tool calling tends to get flaky.
   Google's compat docs say function calling is fully supported; this
   checks whether that's true for the specific small models NAVI wants to
   put the most traffic through.

Makes real API calls against the free tier's real quota — the full Flash
models only allow 20 requests/day, so don't run this in a loop.
"""

import sys

from config.store import config
from jobs.model_ranking import fetch_gemini_models
from providers.base import ChatMessage, ProviderError
from providers.registry import get_provider

# Deliberately a Lite model: it's where NAVI's volume is meant to sit
# (500 RPD vs 20 on full Flash) and therefore where tool-calling
# reliability actually needs proving.
DEFAULT_MODEL = "gemini-3.5-flash-lite"

WEATHER_TOOL = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string", "description": "City name"}},
            "required": ["city"],
        },
    },
}]


def main() -> None:
    model = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_MODEL

    if not config.get_provider_key("gemini"):
        print("FAIL: no Gemini key. Set GOOGLE_AI_API_KEY in .env and `set -a; source .env`.")
        sys.exit(1)

    print("=== 1. Catalog visible to this key ===")
    models = fetch_gemini_models(config.get_provider_key("gemini"))
    if not models:
        print("FAIL: catalog fetch returned nothing — key rejected, or the endpoint changed shape.")
        sys.exit(1)
    free = [m for m in models if m["free"]]
    print(f"{len(models)} chat models visible, {len(free)} classified free\n")
    for m in sorted(models, key=lambda x: x["id"]):
        tag = "free" if m["free"] else "PAID"
        print(f"  {m['id']:44s} ctx={str(m['context_length']):>9s}  {tag}")

    print(f"\n=== 2. Plain chat round-trip ({model}) ===")
    provider = get_provider("gemini")
    try:
        r = provider.chat(model=model, messages=[
            ChatMessage(role="user", content="Reply with exactly the word: pong"),
        ])
    except ProviderError as e:
        print(f"FAIL: {e}")
        sys.exit(1)
    print(f"  text     : {(r.text or '').strip()[:120]!r}")
    print(f"  usage    : {r.usage_note}")
    if not r.text:
        print("FAIL: empty reply")
        sys.exit(1)
    print("  PASS")

    print(f"\n=== 3. Tool calling ({model}) — the one that actually matters ===")
    try:
        r2 = provider.chat(
            model=model,
            messages=[ChatMessage(role="user", content="What's the weather in Valencia? Use the tool.")],
            tools=WEATHER_TOOL,
        )
    except ProviderError as e:
        print(f"FAIL: {e}")
        sys.exit(1)
    names = [tc.name for tc in r2.tool_calls]
    print(f"  tool_calls: {names}")
    print(f"  args      : {[tc.arguments for tc in r2.tool_calls]}")
    print(f"  text      : {(r2.text or '').strip()[:120]!r}")
    if "get_weather" in names:
        print("  PASS — tool calling works on this model, safe for the idle tier")
    else:
        print("  FAIL — no tool call. This model can NOT host the idle tier")
        print("         (flag_key_insight / propose_research_mode would silently never fire).")
        sys.exit(1)

    print("\nALL PASS")


if __name__ == "__main__":
    main()
