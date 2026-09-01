"""
jobs/test_topic_classifier.py

Standalone diagnostic — NOT wired into any real flow. Tests whether
openai/gpt-oss-safeguard-20b (a safety-classification-tuned model, bring-
your-own-policy at inference time) can be repurposed for topic-boundary
detection, per dispatcher/policies/TOPIC_CONTINUITY.md.

Test conversation, exactly as specified: 3 messages on one topic (a
friend's birthday party), then 2 messages each on a DIFFERENT topic (car
noise, then a dinner recipe), then a 6th message that returns to the
FIRST topic. The real question this tests: can the model correctly
re-match message 6 back to topic 1, having seen two unrelated topics in
between — not just "did the topic change," but "which established topic,
if any, does this actually belong to."

Run with the real key in the environment:

    GROQ_API_KEY=gsk_... python jobs/test_topic_classifier.py

For ad-hoc single-message testing instead of this fixed scenario, see
jobs/classify_topic.py.
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.store import config  # noqa: E402
from dispatcher.topic_classifier import MODEL as DEFAULT_MODEL  # noqa: E402
from dispatcher.topic_classifier import EstablishedTopic, classify, load_policy  # noqa: E402
from providers.base import ProviderError  # noqa: E402
from providers.registry import ProviderNotConfigured, get_provider  # noqa: E402

# 3 messages on topic 1, then one message each on two DIFFERENT other
# topics, then a 6th message returning to topic 1. Expected classification
# printed alongside each result, so a mismatch is obvious at a glance.
CONVERSATION = [
    ("I'm planning a surprise birthday party for my friend Maria next month, thinking of doing it at a rooftop bar.", None),
    ("Do you think 25 guests is too many for a rooftop venue that size?", "topic 1 (party)"),
    ("What flavor cake do you think goes best for a summer evening party?", "topic 1 (party)"),
    ("My car has been making a weird clicking noise when I turn left, any idea what that could be?", "NEW (topic 2, car)"),
    ("What's a quick vegetarian pasta recipe I could make tonight in under 30 minutes?", "NEW (topic 3, recipe)"),
    ("Actually, what time should we tell people to arrive at the party?", "topic 1 (party) — the hard case, resuming after 2 unrelated topics"),
]


def main() -> None:
    # Optional: python jobs/test_topic_classifier.py openai/gpt-oss-20b
    # to re-run this exact validated scenario against a fallback
    # candidate instead of the default gpt-oss-safeguard-20b — same test,
    # same scoring, so a fallback gets real evidence, not an assumption.
    model = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_MODEL

    key = os.environ.get("GROQ_API_KEY")
    if not key:
        raise SystemExit("Set GROQ_API_KEY in the environment first.")
    config.set_provider_key("groq", key)

    try:
        provider = get_provider("groq")
    except ProviderNotConfigured as e:
        raise SystemExit(f"Provider not configured: {e}")

    print(f"Testing model: {model}")
    policy = load_policy()
    topics: list[EstablishedTopic] = []

    for i, (message, expected) in enumerate(CONVERSATION, start=1):
        print(f"\n--- Message {i} ---")
        print(f"Text: {message}")
        print(f"Expected: {expected or '(establishes topic 1)'}")

        if i == 1:
            # Nothing to classify against yet — this message *creates*
            # topic 1. No model call needed for message 1 itself.
            topics.append(EstablishedTopic(1, "party", "Planning Maria's surprise birthday party"))
            print("Result: (n/a — this message establishes topic 1)")
            continue

        try:
            result = classify(provider, policy, topics, message, model=model)
        except ProviderError as e:
            print(f"Result: ERROR — {e}")
            continue

        print(f"Result:\n{result}")

        if "NEW" in result.upper() and "TOPIC:" in result.upper():
            new_num = len(topics) + 1
            topics.append(EstablishedTopic(new_num, f"topic {new_num}", message[:60]))


if __name__ == "__main__":
    main()
