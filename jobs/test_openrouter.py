"""
jobs/test_openrouter.py

Standalone diagnostic — not wired into any workflow. Sends one trivial
prompt through the real OpenRouterProvider transport to confirm the key
actually works end to end, independent of which task/model routing is
configured. Run with the real key in the environment:

    OPENROUTER_API_KEY=sk-or-v1-... python jobs/test_openrouter.py
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.store import config  # noqa: E402
from providers.base import ChatMessage, ProviderError  # noqa: E402
from providers.registry import ProviderNotConfigured, get_provider  # noqa: E402

MODEL = "openai/gpt-oss-120b:free"  # same free model already used for the brainstorm role


def main() -> None:
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise SystemExit("Set OPENROUTER_API_KEY in the environment first.")
    config.set_provider_key("openrouter", key)

    try:
        provider = get_provider("openrouter")
    except ProviderNotConfigured as e:
        raise SystemExit(f"Provider not configured: {e}")

    try:
        response = provider.chat(
            model=MODEL,
            messages=[ChatMessage(role="user", content="What's the weather like in Estepona, Spain right now?")],
        )
    except ProviderError as e:
        raise SystemExit(f"OpenRouter call failed: {e}")

    print(f"Model: {response.model_used}")
    print(f"Usage: {response.usage_note}")
    print(f"Reply:\n{response.text}")


if __name__ == "__main__":
    main()
