"""
dispatcher/slugify.py

Turns a step's text into a short, filesystem-safe topic slug — deterministically,
no AI call. Keeps the whole pipeline predictable: the same request always
lands in the same folder, and generating a slug never costs a request against
any provider's daily quota.

The rule from earlier: the FIRST step in a chain sets the slug. Later steps
inherit it unless assign_slugs() decides they've clearly named something new
(see the naive "new topic" heuristic below).
"""

import re

_STOPWORDS = {
    "a", "an", "the", "of", "for", "to", "on", "in", "and", "or", "is", "are",
    "do", "does", "did", "i", "you", "it", "this", "that", "then", "please",
    "want", "need", "can", "could", "would", "should", "with", "about",
    "look", "looking", "find", "make", "create", "get", "give", "some",
    "into", "so", "if", "using", "use", "from", "will", "be",
}

MAX_WORDS = 5
MAX_LEN = 60


def slugify(text: str) -> str:
    words = re.findall(r"[a-zA-Z0-9]+", text.lower())
    meaningful = [w for w in words if w not in _STOPWORDS and len(w) > 1]
    if not meaningful:
        meaningful = words[:MAX_WORDS] or ["untitled"]
    slug = "-".join(meaningful[:MAX_WORDS])
    return slug[:MAX_LEN].rstrip("-") or "untitled"


def assign_slugs(steps: list) -> None:
    """
    Mutates each Step in place, setting .topic_slug.
    First step always gets a fresh slug from its own text.
    Later steps inherit it, UNLESS their text shares zero meaningful words
    with the first step's text — a naive but deterministic "this is probably
    a different topic" signal (e.g. two unrelated /research calls chained
    in one message).
    """
    if not steps:
        return

    first_words = set(re.findall(r"[a-zA-Z0-9]+", steps[0].text.lower())) - _STOPWORDS
    base_slug = slugify(steps[0].text)
    steps[0].topic_slug = base_slug

    for step in steps[1:]:
        step_words = set(re.findall(r"[a-zA-Z0-9]+", step.text.lower())) - _STOPWORDS
        overlap = first_words & step_words
        if overlap:
            step.topic_slug = base_slug
        else:
            step.topic_slug = slugify(step.text)
