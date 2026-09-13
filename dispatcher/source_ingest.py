"""
dispatcher/source_ingest.py

Sources, rebuilt 2026-09-13 around pasted URLs instead of search terms.

WHY THE TERM WENT AWAY. The old flow took search terms, let an 8B model
run searches against a trusted-site list, judge relevance, fetch pages,
and distil them — all inside one shared tool loop. That conflates two
genuinely different jobs: DISCOVERY (given a question, find pages) and
CURATION (given a page, understand it). NotebookLM only does the second,
which is why it never asks for a search term, and it is the half that
actually produces something worth keeping.

Dropping the term also fixes a subtler problem. A document written
against a term varies by WHY it happened to be fetched — the same page
found under two terms would produce two different documents. A source
should be understood on its own terms; relevance to a question belongs at
retrieval time, not ingestion.

THE PIPELINE, one URL at a time:

    fetch + extract (dispatcher)
      -> distil into a structured document (one model call, no tool loop)
      -> locate every quote (dispatcher, deterministic)
      -> adjudicate what it couldn't settle (safeguard-20b)
      -> store, flagged for review

Two things about that are deliberate.

FETCHING IS THE DISPATCHER'S JOB, not a tool the model calls. That single
change removes the tool loop entirely: the model is handed clean Markdown
and does exactly one thing with it. No MAX_TOOL_ITERATIONS budget shared
between documents, no model deciding it has fetched enough.

ONE CALL PER DOCUMENT, not one loop per batch. Each page gets full
attention, a failure on one URL cannot derail the others, progress is
visible per source, and rate-limit pacing falls out naturally.

WHICH MODEL DOES WHAT — three jobs, three sizes:
  - distillation: context_synthesis (whole-document judgement, and on a
    GPU-time-metered pool rather than the daily token budget chat spends)
  - adjudication: safeguard-20b (pick-one-label, validated)
  - nothing else: relevance judging is gone with the terms
"""

from __future__ import annotations

import asyncio
import json
import re
import threading
import time

from config.store import config
from dispatcher.compaction import compact_conversation
from providers.base import ChatMessage, ProviderError
from providers.registry import ProviderNotConfigured, get_provider
from storage.sources import create_batch, create_document, find_by_url, finish_batch
from tools.fetch import SOURCE_MAX_CHARS, FetchError, fetch_document
from tools.source_document import (
    SOURCE_DOCUMENT_INSTRUCTION,
    apply_verdicts,
    build_grounding_prompt,
    build_structuring_messages,
    flatten_for_pdf,
    load_grounding_policy,
    parse_grounding_reply,
    render_source_markdown,
    undecided_points,
    verify_document,
)

GROUNDING_MODEL = "openai/gpt-oss-safeguard-20b"
# Provider-diverse, per the standing rule that a fallback exists to
# survive a provider failing: a second Groq model dies with the first.
GROUNDING_ATTEMPTS = [
    {"provider": "groq", "model": GROUNDING_MODEL},
    {"provider": "cloudflare", "model": "@cf/openai/gpt-oss-20b"},
]

# Pacing between adjudication calls. Groq's per-minute token budget is the
# real constraint, and because the deterministic pass already localised
# each claim to a passage, one call is a few hundred tokens rather than a
# whole page — so this is ordinary politeness, not a workaround for
# oversized requests.
GROUNDING_PAUSE_S = 2.0


def ingest_urls(urls: list[str]) -> str:
    """Runs the whole pipeline synchronously. Call from a background
    thread — a batch of real fetches plus a model call each is far too
    slow to hold an HTTP request open for."""
    batch_id = create_batch()
    failures: list[str] = []
    for url in urls:
        url = (url or "").strip()
        if not url:
            continue
        try:
            _ingest_one(batch_id, url)
        except Exception as e:  # noqa: BLE001 - see below
            # PER URL, not per batch. _ingest_one already handles the
            # failures it can anticipate (dead link, empty extraction,
            # failed distillation) by writing a row that explains itself.
            # This catches the ones it cannot — and the whole point of
            # one-call-per-document is that a single bad page does not take
            # the others down with it. A batch-level try/except silently
            # abandoned every URL after the first failure, which is exactly
            # what happened live when a NULL term hit a NOT NULL column.
            print(f"[sources] unexpected failure on {url}: {e}")
            failures.append(f"{url}: {e}")
            try:
                create_document(
                    batch_id=batch_id, term=None, title=url, url=url, filen_path=None,
                    status="failed", reason=f"Something went wrong reading this: {e}",
                    fetch_error=str(e),
                )
            except Exception as inner:  # noqa: BLE001 - reporting must not raise either
                print(f"[sources] couldn't even record that failure: {inner}")
    # The batch itself only reports an error if EVERY url failed. A mixed
    # run is a success with visible failures, not a failed run — the rows
    # say which is which.
    finish_batch(batch_id, error="; ".join(failures) if failures and len(failures) == len([u for u in urls if u.strip()]) else None)
    return batch_id


def _ingest_one(batch_id: str, url: str) -> None:
    # Already read? Record the hit and move on. The model call is by far
    # the dominant cost here and it is entirely avoidable when the answer
    # is already on disk.
    existing = find_by_url(url)
    if existing:
        print(f"[sources] already inspected, skipping: {url}")
        create_document(
            batch_id=batch_id, term=None, title=existing.get("title") or url, url=url,
            filen_path=None, status="duplicate",
            reason="Already inspected — reusing the document read earlier.",
            document=existing.get("document"), markdown=existing.get("markdown"),
            extractor=existing.get("extractor"), grounding=existing.get("grounding"),
        )
        return

    try:
        page = fetch_document(url, max_chars=SOURCE_MAX_CHARS)
    except FetchError as e:
        # A dead link still gets a row. Pasting five URLs and getting three
        # cards with no explanation reads as a bug, not as a bad link.
        create_document(
            batch_id=batch_id, term=None, title=url, url=url, filen_path=None,
            status="failed", reason=str(e), fetch_error=str(e),
        )
        return

    markdown = page.get("markdown") or ""
    if not markdown.strip():
        create_document(
            batch_id=batch_id, term=None, title=page.get("title") or url, url=url,
            filen_path=None, status="failed",
            reason="Nothing readable could be extracted from this page.",
            fetch_error="empty extraction", extractor=page.get("extractor"),
        )
        return

    doc = _distil(page)
    if doc is None:
        # The page was read fine; distillation failed. Keep the markdown —
        # it is the material, and it is what a re-run would use.
        create_document(
            batch_id=batch_id, term=None, title=page.get("title") or url, url=url,
            filen_path=None, status="failed",
            reason="Couldn't distil this page into a document — the raw text is kept below.",
            markdown=markdown, extractor=page.get("extractor"),
            truncated=bool(page.get("truncated")),
        )
        return

    doc = verify_document(doc, markdown)
    doc = _adjudicate(doc)

    doc["source"] = {
        "url": page.get("url") or url,
        "title": page.get("title"),
        "author": page.get("author"),
        "published": page.get("published"),
    }
    grounding = doc.get("grounding_summary") or {}
    filen_path = _save_to_knowledge(doc, page)
    create_document(
        batch_id=batch_id, term=None,
        title=page.get("title") or url, url=page.get("url") or url,
        filen_path=filen_path,
        status="pending_review",
        document=json.dumps(doc, ensure_ascii=False),
        markdown=markdown,
        extractor=page.get("extractor"),
        truncated=bool(page.get("truncated")),
        grounding=json.dumps(grounding, ensure_ascii=False),
    )
    print(f"[sources] ingested {url} — {grounding.get('verified', 0)}/{grounding.get('points', 0)} claims grounded")


def _distil(page: dict) -> dict | None:
    """One schema-constrained call on context_synthesis. No tools, no
    loop: the page is already fetched and cleaned, so there is exactly one
    thing left to do with it."""
    messages = build_structuring_messages(
        page.get("title"), page.get("url"), page.get("markdown"),
    )
    # compact_conversation is async and this whole call chain is not.
    # asyncio.run is correct HERE specifically — ingest_urls runs in its
    # own dedicated thread with no event loop of its own, which is exactly
    # the case asyncio.run exists for. It would be wrong inside a request
    # handler or a running loop, and isn't used in either.
    doc = asyncio.run(compact_conversation(messages, SOURCE_DOCUMENT_INSTRUCTION))
    if not isinstance(doc, dict) or not (doc.get("summary") or "").strip():
        return None
    return doc


def _adjudicate(doc: dict) -> dict:
    """Sends only the claims the deterministic pass couldn't settle.

    An exact match is already proven by string comparison; asking a model
    about it would spend tokens re-confirming a certainty and invite an
    opinion to contradict something we know.

    Every failure path here keeps the deterministic label and records NO
    verdict, which reads as "not independently checked" — honestly
    different from "checked and unsupported". Verification being
    unavailable must never quietly upgrade or condemn a claim.
    """
    pending = undecided_points(doc)
    if not pending:
        return doc

    try:
        policy = load_grounding_policy()
    except OSError as e:
        print(f"[sources] grounding policy unreadable, skipping adjudication: {e}")
        return doc

    points = doc.get("key_points") or []
    verdicts: dict[int, dict | None] = {}
    for n, idx in enumerate(pending):
        point = points[idx]
        grounding = point.get("grounding") or {}
        passages = [p for p in ([grounding.get("source_text")] + (grounding.get("candidates") or [])) if p]
        if not passages:
            continue
        verdicts[idx] = _ask_grounding(policy, point.get("claim") or "", passages)
        if n < len(pending) - 1:
            time.sleep(GROUNDING_PAUSE_S)
    return apply_verdicts(doc, verdicts)


def _ask_grounding(policy: str, claim: str, passages: list[str]) -> dict | None:
    prompt = build_grounding_prompt(claim, passages)
    for attempt in config.get_attempts(GROUNDING_ATTEMPTS):
        try:
            provider = get_provider(attempt["provider"])
        except (ProviderNotConfigured, Exception):  # noqa: BLE001
            continue
        try:
            response = provider.chat(
                model=attempt["model"],
                messages=[
                    ChatMessage(role="system", content=policy),
                    ChatMessage(role="user", content=prompt),
                ],
            )
        except ProviderError as e:
            if e.is_rate_limit:
                config.mark_rate_limited(attempt["provider"], attempt["model"])
            print(f"[sources] grounding attempt failed on {attempt['provider']}: {e}")
            continue
        parsed = parse_grounding_reply(response.text or "")
        if parsed:
            return parsed
        # A reply that doesn't match the contract is UNPARSEABLE, not a
        # verdict — fall through to the next provider rather than guess.
        print(f"[sources] unparseable grounding reply from {attempt['provider']}")
    return None


def start_ingest(urls: list[str]) -> None:
    """Fire-and-forget, same threading pattern every other dispatcher
    background job here uses (a real OS thread, not asyncio — this call
    chain is synchronous top to bottom)."""
    threading.Thread(target=ingest_urls, args=(urls,), daemon=True).start()


def _save_to_knowledge(doc: dict, page: dict) -> str | None:
    """Renders the document to PDF and files it in Knowledge.

    Worth being clear about what this is FOR, because the format cuts
    against the rest of this module: a PDF is for a person to read, keep
    or send on. It is NOT how NAVI will read this back — the structured
    document and the cleaned Markdown are both stored in the database, and
    those are strictly better for that, since a PDF would have to be
    parsed back into the structure we already have.

    Best-effort by design. A rendering or upload failure loses a
    convenience copy, never the document itself, so it must not fail the
    ingest — same rule as every other optional output here.
    """
    try:
        from tools.documents import DocumentRenderError, render_pdf
        from storage.filen import save_bytes
    except ImportError as e:
        print(f"[sources] no PDF export available: {e}")
        return None

    title = page.get("title") or page.get("url") or "Source"
    markdown = render_source_markdown(
        doc, extractor=page.get("extractor"), truncated=bool(page.get("truncated")),
    )
    try:
        pdf = render_pdf(title, flatten_for_pdf(markdown))
    except DocumentRenderError as e:
        print(f"[sources] PDF render failed (document still saved): {e}")
        return None
    except Exception as e:  # noqa: BLE001 - a convenience copy must never fail an ingest
        print(f"[sources] PDF render failed unexpectedly (document still saved): {e}")
        return None

    slug = re.sub(r"[^a-z0-9]+", "-", (title or "source").lower()).strip("-")[:60] or "source"
    try:
        return save_bytes("sources", slug, f"{slug}.pdf", pdf)
    except Exception as e:  # noqa: BLE001 - same reason
        print(f"[sources] Knowledge upload failed (document still saved): {e}")
        return None
