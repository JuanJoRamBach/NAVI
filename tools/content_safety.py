"""
tools/content_safety.py

Screens externally-fetched content (web_search/fetch_page results)
before it ever reaches a model, using Groq-hosted Llama Prompt Guard 2
(meta-llama/llama-prompt-guard-2-86m, primary) — a small (86M-
parameter), purpose-built classifier for exactly this: does this text
contain an explicit attempt to override prior instructions. Falls back
to the 22M variant on a primary failure (see PROMPT_GUARD_FALLBACK_MODEL
below) — each model has its OWN separate 14.4K-requests/day free-tier
quota (verified 2026-09-12 against Groq's own rate-limits page: 30 RPM
/ 14.4K RPD / 15K TPM / 500K TPD, identical for both variants), so
using both gives ~28.8K requests/day of combined headroom before this
ever has to fail open — comfortably enough for NAVI's real traffic
either way.

This is NOT the same check as the (separate, not-yet-built) context.md
save-time overlap check described in how_to_handle_context.md — that
one asks "is this flagged memory substantially copied from what was
just fetched," a different question this model structurally can't
answer, since it only ever classifies ONE text in isolation, not a
comparison between two. This module answers a narrower, earlier
question: "is the fetched content itself an attack," screened BEFORE
the content is ever handed to a model as a tool result — the "taint at
ingestion" half of the design, not the "taint reaching a memory sink"
half.

Real trade-off, deliberate: on a Groq error (rate-limited, timeout,
outage) this fails OPEN — content is allowed through, logged, not
blocked — matching this codebase's standing pattern of never letting an
auxiliary safety/tracking call break the primary flow (see providers/
groq.py's own "usage tracking must never break a real chat request").
The residual risk this accepts is narrow (needs Groq's classifier
specifically to be down or rate-limited AND the fetched content to
actually be malicious, at the same moment) and the alternative —
blocking every fetch whenever the classifier has a bad moment — would
make the tools measurably less useful against a far more common failure
(a transient API hiccup) than the risk it's defending against.

Output format, live-verified 2026-09-12 (Groq's own model page didn't
document it, so this was confirmed with 5 real API calls rather than
assumed): the model returns a raw probability string, not a text label
— e.g. "0.00035461020888760686" for benign text, "0.999582827091217"
for a classic injection attempt. Real, cleanly separated scores from
the actual test: benign prompts scored ~0.0003-0.0004, three different
injection styles (classic override, an override embedded mid-paragraph,
a roleplay-jailbreak framing) all scored 0.9994+. 0.5 as the cutoff
below sits with wide margin on both sides of everything actually
observed — not tuned against edge cases yet, since none showed up in
this first real test, but there's room to move it if real traffic ever
produces a genuinely borderline score.
"""

from providers.base import ChatMessage
from providers.registry import get_provider

# 86M primary, 22M fallback — verified live (2026-09-12) to return the
# identical output shape (a raw probability string) and comparably
# clean separation, so falling back doesn't trade away real accuracy.
# Each model gets ITS OWN separate 14.4K-requests/day quota on Groq's
# free tier (confirmed against Groq's own rate-limits page) — using
# both means a rate-limited 86M doesn't have to fail the check open,
# it just tries 22M next, ~28.8K requests/day combined before this
# ever needs to give up and let content through unscreened.
PROMPT_GUARD_MODEL = "meta-llama/llama-prompt-guard-2-86m"
PROMPT_GUARD_FALLBACK_MODEL = "meta-llama/llama-prompt-guard-2-22m"

# Prompt Guard 2 has a 512-TOKEN context window (Meta's own model card).
#
# This used to cap at 2,000 CHARACTERS, which is a character budget
# defending a token limit — and the ratio between them is not a constant.
# Measured against the live endpoint 2026-09-13, all at exactly 2,000
# characters:
#
#   plain English ...... 445 tokens  -> 200 OK
#   accented Spanish ... 492 tokens  -> 200 OK
#   code ............... 529 tokens  -> 400 context_length_exceeded
#   markdown + links ... 693 tokens  -> 400 context_length_exceeded
#
# Both 400s then failed the 22m fallback the same way, and screen_content
# fails OPEN — so screening silently did nothing for exactly the content
# most likely to be scraped and structured. Worse, tools/fetch.py started
# returning Markdown with inline links that same day, which moved ordinary
# fetched pages into the failing band.
#
# Budgeted in tokens now, with real headroom: cl100k_base is not Llama's
# tokenizer, so the count here is an estimate of a different tokenizer's
# answer and must not sit near the ceiling.
_MAX_SCREEN_TOKENS = 380

# Characters per token, used only when tiktoken is unavailable. 2.5 is
# deliberately pessimistic — denser than any of the samples above —
# because under-estimating here means a 400 and a silent fail-open.
_FALLBACK_CHARS_PER_TOKEN = 2.5


def _truncate_for_screening(text: str, max_tokens: int = _MAX_SCREEN_TOKENS) -> str:
    """Cuts `text` to something Prompt Guard will actually accept.

    Only the START of a long page is screened, which is a real limitation
    and not a new one: an injection buried on page nine goes unscreened.
    Chunking the whole document is the honest fix and is not built. What
    IS fixed is the case where nothing was screened at all.
    """
    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        tokens = enc.encode(text)
        if len(tokens) <= max_tokens:
            return text
        return enc.decode(tokens[:max_tokens])
    except Exception:  # noqa: BLE001 - tiktoken missing or failing must not stop screening
        return text[:int(max_tokens * _FALLBACK_CHARS_PER_TOKEN)]

# Live-verified 2026-09-12 — see module docstring for the real scores
# this sits between.
_MALICIOUS_THRESHOLD = 0.5

UNSAFE_CONTENT_NOTICE = (
    "[Content withheld: this page was flagged by an automated safety check "
    "as containing an embedded attempt to override instructions, and was not "
    "passed through. Try a different source if this page was expected to be safe.]"
)


def _is_flagged(raw_output: str) -> bool:
    try:
        score = float(raw_output.strip())
    except (ValueError, TypeError):
        # Genuinely unparseable — can't confirm the model said "safe",
        # so don't trust it silently. See module docstring.
        return True
    return score >= _MALICIOUS_THRESHOLD


def _run_prompt_guard(model: str, text: str) -> str:
    """One screening call, with a single shrink-and-retry.

    The retry exists because the token budget above is an estimate made
    with the wrong tokenizer. If it is ever slightly optimistic the result
    is a 400 and, through screen_content's fail-open, no screening at all
    — so it is worth one cheap halving before giving up rather than
    letting a rounding error disable a safety check.
    """
    provider = get_provider("groq")
    budget = _MAX_SCREEN_TOKENS
    for attempt in range(2):
        try:
            response = provider.chat(
                model=model,
                messages=[ChatMessage(role="user", content=_truncate_for_screening(text, budget))],
            )
            return (response.text or "").strip()
        except Exception as e:  # noqa: BLE001 - re-raised below if the retry also fails
            if attempt == 0 and "context_length" in str(e).lower():
                budget //= 2
                print(f"[content_safety] {model} rejected the length; retrying at {budget} tokens")
                continue
            raise
    return ""


def screen_content(text: str) -> tuple[bool, str | None]:
    """Returns (is_safe, raw_model_output). is_safe=True with
    raw_model_output=None means neither model answered (both attempts
    failed) — fails open per the module's documented trade-off, NOT the
    same thing as a confirmed "benign" verdict."""
    if not text or not text.strip():
        return True, None
    for model in (PROMPT_GUARD_MODEL, PROMPT_GUARD_FALLBACK_MODEL):
        try:
            raw_output = _run_prompt_guard(model, text)
            return (not _is_flagged(raw_output)), raw_output
        except Exception as e:
            print(f"[content_safety] {model} check failed: {e}")
            continue
    print("[content_safety] Both Prompt Guard models failed, failing open")
    return True, None


def screened(text: str) -> str:
    """The actual call sites (tools/registry.py's web_search/fetch_page
    dispatch) want a drop-in replacement string, not a tuple — this is
    that convenience wrapper."""
    is_safe, _ = screen_content(text)
    return text if is_safe else UNSAFE_CONTENT_NOTICE
