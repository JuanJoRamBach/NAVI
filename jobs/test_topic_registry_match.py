"""
jobs/test_topic_registry_match.py

Standalone diagnostic — NOT wired into any real flow. Harder, larger-
scale variant of the earlier topic-classifier tests: instead of matching
against topics established fresh within one short conversation (6-10
messages, 6-7 topics), this matches against a REGISTRY of 15 pre-existing
NAMED topics — the scale this would actually need to operate at after
real accumulated usage, not just a short test window.

Registry topics are drawn from JuanJo's actual real projects (job
hunting, portfolio, NAVI's own sub-areas, Bando/Pulso, Umbral) rather
than generic filler, since realistic topic CLOSENESS is what actually
stresses this — two vaguely-related generic topics are an easy test,
two genuinely close real topics (e.g. "NAVI provider routing" vs. "NAVI
model-ranking design", which overlap but are distinct) are the hard one.

8 test messages, each targeting a specific stress case:
  - 2 close-pair tests, both directions (job hunting vs. interview prep;
    NAVI routing vs. NAVI model-ranking)
  - 1 genuinely cross-topic/ambiguous message (mentions two real topics
    at once — which one is it MORE centrally about)
  - 1 message with zero registry match at all (must say NEW, not force
    a fit)
  - 3 straightforward reinforcement matches, including one topic
    (birthday party) referenced a third time to test durability

Run with the real key in the environment:

    GROQ_API_KEY=gsk_... python jobs/test_topic_registry_match.py
    GROQ_API_KEY=gsk_... python jobs/test_topic_registry_match.py openai/gpt-oss-20b
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
POLICY_PATH = Path(__file__).resolve().parent.parent / "dispatcher" / "policies" / "TOPIC_REGISTRY_MATCH.md"

# 15 named topics, mimicking weeks of accumulated real usage — not
# invented generic filler. Several deliberately close pairs:
#   - "Job hunting" vs "Interview prep" (related, but distinct focus)
#   - "NAVI provider routing" vs "NAVI model-ranking design" (same
#     project, different sub-area)
REGISTRY = {
    "Job hunting": "Searching and applying for junior UX/UI design roles across the EU and US.",
    "Portfolio site": "Building and updating the personal portfolio website (Vite + React, GitHub Pages).",
    "NAVI provider routing": "Backend work on NAVI's LLM provider integrations, transports, and task routing config.",
    "NAVI chat UI": "Frontend work on NAVI's PWA chat interface — navi-pwa, sidebar, message rendering, streaming.",
    "Bando/Pulso app": "The hyperlocal Spain news and civic-safety app, renamed Bando to Pulso, diary study.",
    "Umbral platform": "The reading/writing platform project — writer tool, bounded AI assistance, rights/monetization.",
    "Car maintenance": "Dealing with car issues, noises, and repairs.",
    "Birthday party planning": "Planning a surprise birthday party for a friend, Maria, at a rooftop bar.",
    "Cooking and recipes": "Meal ideas and recipes for everyday cooking.",
    "Porto trip": "Planning a week-long trip to Porto in October.",
    "Gift ideas": "Finding a gift for a parents' 30th wedding anniversary.",
    "Interview prep": "Preparing specifically for an upcoming job interview — what to wear, how to present a portfolio.",
    "NAVI model-ranking design": "Designing NAVI's daily model-ranking/tiering system — benchmarks, fetch strategy, selection logic.",
    "Spanish token usage": "A note about non-English text tokenizing less efficiently, relevant to rate-limit budgeting.",
    "Home and apartment": "General household topics — repairs, neighbors, apartment logistics.",
}

# (message, expected topic name or "NEW") — 15 cases: the original 8
# plus 7 more covering topics that weren't touched yet (Portfolio site,
# Umbral, car maintenance, cooking, Porto trip, Spanish token usage) and
# more hard/ambiguous cases.
TEST_MESSAGES = [
    ("Any good junior UX roles posted this week in Barcelona?", "Job hunting"),
    ("What should I wear to the interview, and should I bring a printed portfolio or show it on an iPad?", "Interview prep"),
    ("Is the OpenRouter fallback chain still pointing at the nemotron models?", "NAVI provider routing"),
    ("Should I pull the Intelligence Index scores for gpt-oss models before finalizing the ranking rubric?", "NAVI model-ranking design"),
    ("Should I mention the Bando case study in my job applications, or keep it separate?", "Job hunting"),  # ambiguous: touches Bando AND job hunting — more centrally about the job-hunting decision
    ("What's a good book to read on a rainy weekend?", "NEW"),  # zero registry match
    ("What time should we tell people to arrive at the party?", "Birthday party planning"),
    ("Should we let Maria pick the cake flavor herself when she gets there?", "Birthday party planning"),  # 3rd reference to same topic
    ("Should I add the Pulso case study to my portfolio site, with screenshots of the diary study?", "Portfolio site"),  # same Bando/Pulso project, but a DIFFERENT resolution than case 5 — this one's about the site, not job applications
    ("What's the current stance on how much AI should assist inside Umbral's writer tool versus the human author?", "Umbral platform"),
    ("The clicking noise from the car is still happening even after the mechanic looked at it.", "Car maintenance"),
    ("What other quick dinner ideas do you have besides pasta, something under 30 minutes?", "Cooking and recipes"),
    ("I think I need new running shoes, mine are falling apart.", "NEW"),  # second zero-match case, unrelated to anything in the registry
    ("Should we book the Porto flights now or wait to see if there's a sale?", "Porto trip"),
    ("If I test NAVI in Spanish will that eat into the Groq rate limits faster than English testing would?", "Spanish token usage"),  # hard: phrased as a question, also touches NAVI/rate-limit topics, but centrally about the Spanish-tokenization note itself
]


def _load_policy() -> str:
    text = POLICY_PATH.read_text(encoding="utf-8")
    return re.sub(r"^---\n.*?\n---\n", "", text, flags=re.DOTALL).strip()


def _build_registry_block() -> str:
    return "\n".join(f"- {name}: {desc}" for name, desc in REGISTRY.items())


def main() -> None:
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
    print(f"Registry size: {len(REGISTRY)} topics\n")

    policy = _load_policy()
    numbered_messages = "\n".join(f"{i}. {msg}" for i, (msg, _) in enumerate(TEST_MESSAGES, start=1))
    prompt = f"Registry:\n{_build_registry_block()}\n\nNew messages:\n{numbered_messages}"

    try:
        response = provider.chat(
            model=model,
            messages=[
                ChatMessage(role="system", content=policy),
                ChatMessage(role="user", content=prompt),
            ],
        )
    except ProviderError as e:
        raise SystemExit(f"Classification failed: {e}")

    result = (response.text or "").strip()
    print("--- Raw model output ---")
    print(result)
    print()

    got = {}
    for line in result.splitlines():
        m = re.match(r"\s*(\d+)\s*:\s*(.+)", line)
        if m:
            got[int(m.group(1))] = m.group(2).strip()

    print("--- Scored ---")
    correct = 0
    for i, (msg, expected) in enumerate(TEST_MESSAGES, start=1):
        actual = got.get(i, "(no answer)")
        is_new = expected == "NEW"
        ok = (is_new and actual.upper().startswith("NEW")) or (not is_new and actual == expected)
        correct += ok
        mark = "OK" if ok else "MISMATCH"
        print(f"{i}. [{mark}] expected {expected!r:30} got {actual!r:35} — {msg[:55]}")

    print(f"\n{correct}/{len(TEST_MESSAGES)} correct")


if __name__ == "__main__":
    main()
