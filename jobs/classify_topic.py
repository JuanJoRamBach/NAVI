"""
jobs/classify_topic.py

Standalone diagnostic — NOT wired into any real flow. Give it one message
at a time; it tells you which established topic (if any) it matches, per
dispatcher/policies/TOPIC_CONTINUITY.md and openai/gpt-oss-safeguard-20b.
Established topics persist across separate runs in a local state file
(topic_classifier_state.json, gitignored — scratch, not app data), so
you can just call this repeatedly as a running conversation instead of
scripting a fixed scenario like jobs/test_topic_classifier.py does.

Usage:
    GROQ_API_KEY=gsk_... python jobs/classify_topic.py "your message here"
    GROQ_API_KEY=gsk_... python jobs/classify_topic.py --reset
"""

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.store import config  # noqa: E402
from dispatcher.topic_classifier import EstablishedTopic, classify, load_policy  # noqa: E402
from providers.base import ProviderError  # noqa: E402
from providers.registry import ProviderNotConfigured, get_provider  # noqa: E402

STATE_PATH = Path(__file__).resolve().parent.parent / "topic_classifier_state.json"


def _load_state() -> list[EstablishedTopic]:
    if not STATE_PATH.exists():
        return []
    raw = json.loads(STATE_PATH.read_text(encoding="utf-8"))
    return [EstablishedTopic(**t) for t in raw]


def _save_state(topics: list[EstablishedTopic]) -> None:
    STATE_PATH.write_text(
        json.dumps([{"number": t.number, "label": t.label, "description": t.description} for t in topics], indent=2),
        encoding="utf-8",
    )


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit('Usage: python jobs/classify_topic.py "your message" (or --reset)')

    if sys.argv[1] == "--reset":
        STATE_PATH.unlink(missing_ok=True)
        print("State cleared — next message starts a fresh topic 1.")
        return

    message = " ".join(sys.argv[1:])

    key = os.environ.get("GROQ_API_KEY")
    if not key:
        raise SystemExit("Set GROQ_API_KEY in the environment first.")
    config.set_provider_key("groq", key)

    try:
        provider = get_provider("groq")
    except ProviderNotConfigured as e:
        raise SystemExit(f"Provider not configured: {e}")

    topics = _load_state()

    if not topics:
        # First message ever (or first since --reset) — nothing to
        # classify against yet, it just establishes topic 1.
        topics.append(EstablishedTopic(1, "topic 1", message[:60]))
        _save_state(topics)
        print("Established topic 1 (first message, nothing to compare against yet).")
        return

    policy = load_policy()
    try:
        result = classify(provider, policy, topics, message)
    except ProviderError as e:
        raise SystemExit(f"Classification failed: {e}")

    print(result)

    if "NEW" in result.upper() and "TOPIC:" in result.upper():
        new_num = len(topics) + 1
        topics.append(EstablishedTopic(new_num, f"topic {new_num}", message[:60]))
        _save_state(topics)


if __name__ == "__main__":
    main()
