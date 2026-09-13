"""
jobs/test_tool_calling.py

"Can this (provider, model) actually do tool calling?" — provider-agnostic.

    venv/bin/python -m jobs.test_tool_calling <provider> <model>
    venv/bin/python -m jobs.test_tool_calling cloudflare @cf/nvidia/nemotron-3-120b-a12b

This keeps coming up and it's never been answerable without writing a
throwaway script. A catalog's own `tools` flag is not enough on its own —
NAVI already has a documented case of a model advertising tool support
and then looping instead of ever answering (@cf/google/gemma-4-26b-a4b-it,
see BANNED_FOR_TOOLS in jobs/model_ranking.py), which is exactly the kind
of thing only a real call reveals.

Matters because a model that silently doesn't call tools doesn't error —
it just quietly never fires flag_key_insight, propose_research_mode, or
web_search, and the feature looks broken rather than misrouted.

Real API call against real quota. One request per run.
"""

import sys

from providers.base import ChatMessage, ProviderError
from providers.registry import ProviderNotConfigured, get_provider

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


def check(provider_name: str, model: str) -> bool:
    try:
        provider = get_provider(provider_name)
    except (ProviderNotConfigured, ValueError) as e:
        print(f"FAIL: {e}")
        return False

    print(f"=== {provider_name} / {model} ===")
    try:
        r = provider.chat(
            model=model,
            messages=[ChatMessage(role="user", content="What's the weather in Valencia? Use the tool.")],
            tools=WEATHER_TOOL,
        )
    except ProviderError as e:
        print(f"FAIL: {e}")
        return False

    names = [tc.name for tc in r.tool_calls]
    print(f"  tool_calls : {names}")
    print(f"  args       : {[tc.arguments for tc in r.tool_calls]}")
    print(f"  text       : {(r.text or '').strip()[:150]!r}")
    print(f"  usage      : {r.usage_note}")

    if "get_weather" in names:
        print("  PASS — real tool call with arguments")
        return True
    if r.text:
        # The dangerous case: it answered in prose instead of calling the
        # tool. No error, no tool call — this is what "the feature is
        # silently broken" actually looks like in production.
        print("  FAIL — answered in prose instead of calling the tool")
    else:
        print("  FAIL — neither a tool call nor text")
    return False


def main() -> None:
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(2)
    sys.exit(0 if check(sys.argv[1], sys.argv[2]) else 1)


if __name__ == "__main__":
    main()
