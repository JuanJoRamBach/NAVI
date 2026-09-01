"""
jobs/test_topic_classifier_batch.py

Standalone diagnostic — NOT wired into any real flow. Batch variant of
jobs/test_topic_classifier.py: classifies 10 messages across 6 topics in
ONE call instead of one call per message, per
dispatcher/policies/TOPIC_CONTINUITY_BATCH.md.

Deliberately jumbled, not neat blocks like the single-message test:
topic 1 (party) returns twice, far apart (messages 4 and 10); topic 2
(car noise) appears ONCE and never returns; topics 3 and 5 each return
once, with a deliberate confusability trap — message 9 mentions "rental
car" right after topic 5 (Porto trip), which should NOT get pulled into
topic 2 (car noise) just because both mention cars. This tests whether
the model tracks actual subject matter or gets fooled by surface-word
overlap.

Run with the real key in the environment:

    GROQ_API_KEY=gsk_... python jobs/test_topic_classifier_batch.py
"""

import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.store import config  # noqa: E402
from providers.base import ChatMessage, ProviderError  # noqa: E402
from providers.registry import ProviderNotConfigured, get_provider  # noqa: E402

DEFAULT_MODEL = "openai/gpt-oss-safeguard-20b"
POLICY_PATH = Path(__file__).resolve().parent.parent / "dispatcher" / "policies" / "TOPIC_CONTINUITY_BATCH.md"

# (message, expected establishing message number) — expected is what a
# correct read of the subject matter should produce, not what's "nearby."
MESSAGES = [
    ("I'm planning a surprise birthday party for my friend Maria next month at a rooftop bar.", 1),
    ("My car's been making a weird clicking noise when I turn left.", 2),
    ("What should I wear to a job interview at a design agency?", 3),
    ("Do you think 25 guests is too many for the rooftop venue?", 1),
    ("We're thinking of visiting Porto for a week in October, any recommendations?", 5),
    ("What's a quick vegetarian pasta recipe I could make tonight?", 6),
    ("My parents' 30th anniversary is coming up, what's a good gift idea?", 7),
    ("Should I bring a printed portfolio or just show it on an iPad for the interview?", 3),
    ("Is a rental car necessary in Porto or is public transport enough?", 5),  # trap: mentions "car", must NOT match topic 2
    ("What cake flavor works best for a summer evening party?", 1),
]


def _load_policy() -> str:
    text = POLICY_PATH.read_text(encoding="utf-8")
    return re.sub(r"^---\n.*?\n---\n", "", text, flags=re.DOTALL).strip()


def main() -> None:
    # Optional: python jobs/test_topic_classifier_batch.py openai/gpt-oss-20b
    # to re-run this exact validated scenario against a fallback candidate.
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
    policy = _load_policy()
    numbered = "\n".join(f"{i}. {msg}" for i, (msg, _) in enumerate(MESSAGES, start=1))

    try:
        response = provider.chat(
            model=model,
            messages=[
                ChatMessage(role="system", content=policy),
                ChatMessage(role="user", content=numbered),
            ],
        )
    except ProviderError as e:
        raise SystemExit(f"Classification failed: {e}")

    result = (response.text or "").strip()
    print("--- Raw model output ---")
    print(result)
    print()

    # Parse "N: TOPIC=M" lines and compare against expected.
    got = {}
    for line in result.splitlines():
        m = re.match(r"\s*(\d+)\s*:\s*TOPIC\s*=\s*(\d+)", line)
        if m:
            got[int(m.group(1))] = int(m.group(2))

    print("--- Scored ---")
    correct = 0
    for i, (msg, expected) in enumerate(MESSAGES, start=1):
        actual = got.get(i)
        ok = actual == expected
        correct += ok
        mark = "OK" if ok else "MISMATCH"
        print(f"{i}. [{mark}] expected TOPIC={expected}, got TOPIC={actual}  —  {msg[:60]}")

    print(f"\n{correct}/{len(MESSAGES)} correct")


if __name__ == "__main__":
    main()
