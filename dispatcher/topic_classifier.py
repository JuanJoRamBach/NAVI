"""
dispatcher/topic_classifier.py

Experimental — NOT wired into any real chat flow yet. Shared classify
logic behind dispatcher/policies/TOPIC_CONTINUITY.md, used by both
jobs/test_topic_classifier.py (the fixed 6-message scripted test) and
jobs/classify_topic.py (the interactive one-message-at-a-time CLI).
Extracted here instead of duplicated so there's one place that actually
talks to the model.
"""

import re
from dataclasses import dataclass
from pathlib import Path

from providers.base import ChatMessage, Provider

MODEL = "openai/gpt-oss-safeguard-20b"
POLICY_PATH = Path(__file__).resolve().parent / "policies" / "TOPIC_CONTINUITY.md"


@dataclass
class EstablishedTopic:
    number: int
    label: str
    description: str


def load_policy() -> str:
    text = POLICY_PATH.read_text(encoding="utf-8")
    # Strip the YAML frontmatter — that's metadata for humans/tooling,
    # not part of what gets sent to the model.
    return re.sub(r"^---\n.*?\n---\n", "", text, flags=re.DOTALL).strip()


def build_topics_block(topics: list[EstablishedTopic]) -> str:
    if not topics:
        return "(no established topics yet — this is the first message)"
    return "\n".join(f"{t.number}. {t.label}: {t.description}" for t in topics)


def classify(
    provider: Provider, policy: str, topics: list[EstablishedTopic], message: str, model: str = MODEL,
) -> str:
    """Returns the model's raw TOPIC:/REASON: response text. Caller
    decides what to do with it (register a new topic, print it, etc.) —
    this function only ever talks to the model, no state mutation.

    `model` defaults to the validated gpt-oss-safeguard-20b, but can be
    overridden — e.g. to test openai/gpt-oss-20b as a fallback candidate
    against the exact same scenarios, rather than assuming a fallback
    performs the same without evidence."""
    prompt = (
        f"Established topics:\n{build_topics_block(topics)}\n\n"
        f"New message to classify:\n{message}"
    )
    response = provider.chat(
        model=model,
        messages=[
            ChatMessage(role="system", content=policy),
            ChatMessage(role="user", content=prompt),
        ],
    )
    return (response.text or "").strip()
