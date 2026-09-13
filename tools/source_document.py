"""
tools/source_document.py

Turns a fetched page into a Source document a model can actually read,
and grounds every claim in it back to the page it came from.

The problem this replaces: `save_source` asked the model for a free-text
`content` field, mid-tool-loop, on the idle-tier model, while it was also
running searches and judging relevance. It copied a chunk of text. That
is the predictable outcome of asking for extraction as a side task with
no schema and no verification.

The shape is borrowed from what actually makes NotebookLM work — not the
model, the discipline around it:

  - answers grounded ONLY in the source, never the model's own knowledge
  - every claim cites the exact passage it came from
  - a persistent, curated set rather than one-off fetches
  - measured payoff: ~13% hallucination vs ~40% ungrounded on the same
    material

So a Source document here is a schema-constrained structure whose every
key point carries a verbatim quote, and those quotes are CHECKED.

THE DIVISION OF LABOUR, which is the whole point:

  The dispatcher decides what is CHECKABLE. Whether a string appears in a
  document is a fact, and asking a model to judge it is strictly worse
  than looking — a model can be confidently wrong about something we can
  simply know. Independent RAG-evaluation guidance lands in the same
  place: "do not ask an LLM judge to decide exact quotation when a string
  or token comparison can do so more reliably."

  A model decides what genuinely requires judgement — is this paraphrase
  faithful, is this claim supported by a section rather than a sentence.
  No amount of string comparison touches those.

  When both have an opinion, the deterministic one wins. The model's
  verdict is evidence for the human reviewing the document, never a
  verdict that overwrites a measurement.

NOTHING IS EVER DROPPED. A quote that fails to verify is a PROVENANCE
failure, not a content failure — the point may be true, important, and
the best thing in the document; what failed is our ability to say where
it came from. Unverified points are kept and labelled, and the cleaned
Markdown is stored alongside the structured document permanently, so the
structure is a VIEW of the source and never a replacement for it.
"""

from __future__ import annotations

import difflib
import re
import unicodedata

# ---- Thresholds -------------------------------------------------------
#
# Researched rather than invented, but still provisional until they have
# been run against real documents — same standing rule as every other
# threshold in this codebase.

# At or above this, a quote is treated as verbatim. Not 1.0: normalization
# can leave harmless residue (a stray double space, a soft hyphen), and
# failing a genuine quote over that would be the checker's bug, not the
# model's.
VERBATIM_RATIO = 0.95

# Below this, there is no locatable match. 0.60 is difflib's own
# documented rule of thumb for "the sequences are close matches" — the
# standard threshold, not a number picked for this project.
FUZZY_MIN = 0.60

# Proportion of a document's points that must be grounded for the document
# itself to read as grounded. 0.9 follows the faithfulness convention in
# RAG evaluation, where a claims-entailed ratio above 0.9 is what gets
# called grounded.
GROUNDED_DOC_RATIO = 0.9

# How many candidate passages to hand the model when a quote can't be
# located at all. Bounded on purpose: sending the whole page is what blows
# a per-minute token budget, and the point of locating first is that we
# rarely need to.
UNLOCATED_CANDIDATES = 3

STATUS_VERIFIED = "verified"
STATUS_PARAPHRASE = "paraphrase"
STATUS_UNLOCATED = "unlocated"


# ---- Normalization ----------------------------------------------------
#
# Most quote-check failures would otherwise be OUR bug, not the model's:
# smart quotes, non-breaking spaces, em dashes and line wrapping all break
# naive string matching on text that is genuinely present. Normalizing
# both sides first is what makes the deterministic check trustworthy
# enough to outrank a model on.

_QUOTES = {
    "‘": "'", "’": "'", "‚": "'", "‛": "'",
    "“": '"', "”": '"', "„": '"', "‟": '"',
    "«": '"', "»": '"',
}
_DASHES = {"‐": "-", "‑": "-", "‒": "-", "–": "-", "—": "-", "―": "-"}
_WS_RE = re.compile(r"\s+")
_MD_NOISE_RE = re.compile(r"[*_`#>]|\[([^\]]*)\]\([^)]*\)")


def normalize(text: str) -> str:
    """Both sides of every comparison go through this. Deliberately
    aggressive: the question being asked is "is this the same sentence",
    not "is this byte-identical"."""
    t = unicodedata.normalize("NFKC", text or "")
    for src, dst in {**_QUOTES, **_DASHES}.items():
        t = t.replace(src, dst)
    # Markdown emphasis around a quoted span shouldn't decide whether the
    # quote is found; a link becomes its visible text, which is what a
    # reader would have quoted.
    t = _MD_NOISE_RE.sub(lambda m: m.group(1) or "", t)
    return _WS_RE.sub(" ", t).strip().casefold()


# ---- Locating a quote in its source -----------------------------------

def locate_quote(quote: str, source: str) -> dict:
    """Finds where `quote` came from in `source`.

    Returns {status, ratio, matched_text, candidates}. `matched_text` is
    the SOURCE's own wording, not the model's — on a paraphrase that is
    the material difference, and it is what should be displayed, because
    the source outranks the summary of it.

    Never raises and never judges the CLAIM — only whether we can point
    at where it came from.
    """
    nq, ns = normalize(quote), normalize(source)
    if not nq or not ns:
        return {"status": STATUS_UNLOCATED, "ratio": 0.0, "matched_text": None, "candidates": []}

    # Exact first: cheapest, and certain when it fires.
    idx = ns.find(nq)
    if idx != -1:
        return {
            "status": STATUS_VERIFIED, "ratio": 1.0,
            "matched_text": _original_span(source, ns, idx, len(nq)),
            "candidates": [],
        }

    # Otherwise find the longest block the two genuinely share and score a
    # window around it. One pass, rather than sliding a window across the
    # whole document and scoring every position.
    matcher = difflib.SequenceMatcher(None, nq, ns, autojunk=False)
    block = matcher.find_longest_match(0, len(nq), 0, len(ns))
    best = {"ratio": 0.0, "start": 0, "length": len(nq)}
    if block.size:
        # The window has to be about as long as the quote. difflib's ratio
        # is 2*matched/(len(a)+len(b)), so a window much longer than the
        # quote caps the score no matter how well it matches — a 41-char
        # quote against a 121-char window tops out at 0.5 even on a
        # perfect hit. That mis-scored a real paraphrase as unlocated.
        #
        # Anchor where the quote would START in the source (the longest
        # shared block, offset by where that block sits inside the quote),
        # then try a few nearby shifts because the anchor is approximate.
        anchor = max(0, block.b - block.a)
        span = int(len(nq) * 1.15) + 8
        shift = max(len(nq) // 4, 12)
        for start in {max(0, anchor - shift), anchor, min(max(0, len(ns) - span), anchor + shift)}:
            window = ns[start:start + span]
            if not window:
                continue
            ratio = difflib.SequenceMatcher(None, nq, window, autojunk=False).ratio()
            if ratio > best["ratio"]:
                best = {"ratio": ratio, "start": start, "length": len(window)}

    if best["ratio"] >= VERBATIM_RATIO:
        status = STATUS_VERIFIED
    elif best["ratio"] >= FUZZY_MIN:
        status = STATUS_PARAPHRASE
    else:
        status = STATUS_UNLOCATED

    matched = _original_span(source, ns, best["start"], best["length"]) if block.size else None
    return {
        "status": status,
        "ratio": round(best["ratio"], 3),
        "matched_text": matched if status != STATUS_UNLOCATED else None,
        # Only an unlocated quote needs candidates — everywhere else we
        # already know the passage, and shipping more would just spend
        # tokens re-deciding something settled.
        "candidates": _candidate_passages(source) if status == STATUS_UNLOCATED else [],
    }


def _original_span(source: str, normalized: str, start: int, length: int) -> str:
    """Maps a span in normalized space back to readable source text.

    Approximate by construction — normalization collapses whitespace, so
    offsets shift. Good enough for showing a reader the passage, and
    deliberately NOT used for any decision: every verdict above is made on
    the normalized strings, where the arithmetic is sound.
    """
    if not source:
        return ""
    scale = len(source) / max(len(normalized), 1)
    a = max(0, int(start * scale) - 40)
    b = min(len(source), int((start + length) * scale) + 40)
    return source[a:b].strip()


_PARA_RE = re.compile(r"\n\s*\n")


def _candidate_passages(source: str, limit: int = UNLOCATED_CANDIDATES) -> list[str]:
    """The passages most worth showing a model when a quote can't be
    located — the longest real paragraphs, as a proxy for substance.

    Honest about what this is: a heuristic. A claim supported by material
    spread across several sections could still look unsupported here,
    which is exactly why an unlocated point is FLAGGED FOR REVIEW rather
    than deleted.
    """
    paras = [p.strip() for p in _PARA_RE.split(source) if len(p.strip()) > 120]
    return sorted(paras, key=len, reverse=True)[:limit]


# ---- Grading a whole document -----------------------------------------

def verify_document(doc: dict, source_markdown: str) -> dict:
    """Annotates every key point with where it came from, and the document
    with how grounded it is overall.

    Returns a new dict; the input is not mutated. Points are never
    removed, reordered, or rewritten — only annotated. `needs_review` is
    the signal that matters: it is what turns the existing pending_review
    gate from rubber-stamping into a real check, by saying WHICH points
    to look at and why.
    """
    out = dict(doc)
    points = []
    for point in (doc.get("key_points") or []):
        if not isinstance(point, dict):
            point = {"claim": str(point), "quote": ""}
        located = locate_quote(point.get("quote") or "", source_markdown)
        points.append({
            **point,
            "grounding": {
                "status": located["status"],
                "ratio": located["ratio"],
                "source_text": located["matched_text"],
                "candidates": located["candidates"],
                # Filled in later, only for the points the deterministic
                # check couldn't settle. None here means "not asked",
                # which is different from "asked and unsupported".
                "verdict": None,
            },
        })
    out["key_points"] = points

    total = len(points)
    grounded = sum(1 for p in points if p["grounding"]["status"] == STATUS_VERIFIED)
    ratio = (grounded / total) if total else 0.0
    out["grounding_summary"] = {
        "points": total,
        "verified": grounded,
        "paraphrase": sum(1 for p in points if p["grounding"]["status"] == STATUS_PARAPHRASE),
        "unlocated": sum(1 for p in points if p["grounding"]["status"] == STATUS_UNLOCATED),
        "grounded_ratio": round(ratio, 3),
        "needs_review": total == 0 or ratio < GROUNDED_DOC_RATIO,
    }
    return out


def undecided_points(doc: dict) -> list[int]:
    """Indexes of the points a model still needs to rule on — the
    paraphrases and the unlocated ones.

    An exact match is already proven by string comparison; sending it to a
    model would spend tokens re-confirming a certainty and invite an
    opinion to contradict something we actually know.
    """
    return [
        i for i, p in enumerate(doc.get("key_points") or [])
        if p.get("grounding", {}).get("status") in (STATUS_PARAPHRASE, STATUS_UNLOCATED)
    ]


# ---- The model's half: adjudicating what the checker couldn't ---------
#
# Real finding from testing this, worth keeping because it justifies the
# whole two-layer design and would otherwise get re-litigated: the ratio
# ALONE cannot separate a heavy paraphrase from a fabrication. A true
# claim reworded past recognition ("there are five distinct layers in a
# good harness" against "a production-grade harness contains five
# layers") scored 0.27, while an invented statistic about the same
# document scored 0.44. The number is excellent at confirming presence
# and useless at judging meaning — which is exactly the boundary where a
# model has to take over.

import re as _re
from pathlib import Path as _Path

POLICY_PATH = _Path(__file__).resolve().parent.parent / "dispatcher" / "policies" / "QUOTE_GROUNDING.md"

VERDICT_RE = _re.compile(r"VERDICT:\s*(ENTAILED|PARTIAL|NOT_FOUND|CONTRADICTED)", _re.IGNORECASE)
REASON_RE = _re.compile(r"REASON:\s*(.+)", _re.IGNORECASE)


def load_grounding_policy() -> str:
    text = POLICY_PATH.read_text(encoding="utf-8")
    return _re.sub(r"^---\n.*?\n---\n", "", text, flags=_re.DOTALL).strip()


def build_grounding_prompt(claim: str, passages: list[str]) -> str:
    """One claim against its located passages. Deliberately NOT the whole
    page: the deterministic pass already found where this came from, and
    sending the document instead is what blows a per-minute token budget
    on a question that only needed a paragraph."""
    body = "\n\n---\n\n".join(p.strip() for p in passages if p and p.strip())
    return f"PASSAGES:\n{body}\n\nCLAIM:\n{claim.strip()}"


def parse_grounding_reply(text: str) -> dict | None:
    """Returns {verdict, reason}, or None if the reply doesn't match the
    contract. None means UNPARSEABLE, which must not be read as a verdict
    — the caller keeps the deterministic label instead."""
    v = VERDICT_RE.search(text or "")
    if not v:
        return None
    r = REASON_RE.search(text or "")
    return {"verdict": v.group(1).upper(), "reason": (r.group(1).strip() if r else "")}


def apply_verdicts(doc: dict, verdicts: dict[int, dict | None]) -> dict:
    """Records the model's judgement ALONGSIDE the measurement, never over
    it. A point's `status` — the deterministic result — is not touched
    here, because that is the half we can be certain about; the verdict is
    evidence for whoever reviews the document.

    A point the model couldn't be asked about, or answered unparseably,
    keeps a null verdict. That reads as "not independently checked",
    which is honestly different from "checked and unsupported".
    """
    out = dict(doc)
    points = [dict(p) for p in (doc.get("key_points") or [])]
    for idx, verdict in verdicts.items():
        if 0 <= idx < len(points):
            grounding = dict(points[idx].get("grounding") or {})
            grounding["verdict"] = verdict
            points[idx]["grounding"] = grounding
    out["key_points"] = points

    summary = dict(out.get("grounding_summary") or {})
    disputed = sum(
        1 for p in points
        if (p.get("grounding") or {}).get("verdict", {}) and
        (p["grounding"]["verdict"] or {}).get("verdict") in ("NOT_FOUND", "CONTRADICTED")
    )
    summary["disputed"] = disputed
    unchecked = sum(
        1 for p in points
        if (p.get("grounding") or {}).get("status") in (STATUS_PARAPHRASE, STATUS_UNLOCATED)
        and not (p.get("grounding") or {}).get("verdict")
    )
    summary["unchecked"] = unchecked
    # Any disputed point drags the whole document back to review, whatever
    # the ratio said — one contradicted claim is worse than several merely
    # unlocated ones, and a ratio can't express that.
    summary["needs_review"] = bool(summary.get("needs_review")) or disputed > 0 or unchecked > 0
    out["grounding_summary"] = summary
    return out

# ---- Structuring: turning a page into a document ----------------------

SOURCE_DOCUMENT_INSTRUCTION = """You are turning ONE fetched web page into a structured Source document. Someone researching "{term}" will read your output instead of the page, and an AI assistant will later answer questions from it. Everything you leave out is lost to both of them.

Work ONLY from the page text given below. Never add anything you know about this subject from elsewhere — not context, not correction, not background. If the page is wrong, that is a fact ABOUT the page and belongs in caveats, not something for you to fix.

Reply with ONLY a single JSON object, no prose, no markdown fence:
{{
  "summary": "string — what this page actually says, in three or four sentences, for someone who will not read it",
  "key_points": [
    {{"claim": "string — one self-contained point the page makes",
      "quote": "string — the exact words from the page that carry this point"}}
  ],
  "relevance": "string — what this page contributes to \"{term}\" specifically, and what it does not",
  "caveats": ["string", "..."]
}}

On "quote": copy the page's words EXACTLY, character for character. Do not tidy, shorten, join separated sentences, or fix the grammar. These are checked against the page automatically, and a quote you improved is a quote that no longer matches. Quote the smallest span that genuinely carries the claim.

If a point is real but no single passage states it — it runs across several paragraphs, or it is about the page as a whole — keep the point and quote the single most representative passage you can. Do NOT invent a sentence to quote, and do NOT drop the point: an unquotable observation is often the most valuable one in the document.

On "caveats": what a careful reader needs to know before trusting this. Is it vendor marketing for a product? Are its numbers stated without attribution? Is it opinion, a first-person report, out of date, paywalled, or contradicting itself? An empty list is a real answer if the page is straightforwardly sound — do not invent doubts.

6 to 10 key points for a substantial article, fewer for a short one. Never pad."""


def build_structuring_messages(term, title, url, markdown):
    """The content goes in as a normal message, not folded into the
    instruction — keeps the instruction byte-identical across every
    document, which is what lets a provider's prefix cache hit."""
    from providers.base import ChatMessage
    header = f"PAGE: {title or '(untitled)'}\nURL: {url}\n\n"
    return [ChatMessage(role="user", content=header + (markdown or ""))]
