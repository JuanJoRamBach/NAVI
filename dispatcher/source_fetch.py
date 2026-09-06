"""
dispatcher/source_fetch.py

Runs the Sources tab's "Batch Dispatch" — one background job that works
through every search term the user typed in, judges relevance against
their Trusted Sites registry, and saves a document per real find. Same
shape as dispatcher/executor.py's _run_research_gather_phase: try the
configured primary/fallback chain in order, run the tool-calling loop,
report what happened.

Key difference from every other tool-calling caller in this codebase:
save_source's side effect (writing a document + DB row) happens INSIDE
tools/registry.py's dispatch() as tool calls are executed, not extracted
from the transcript afterward. That means a mid-batch provider failure
(primary dies partway through, rotates to fallback) does NOT lose
documents already saved for earlier terms — only work not yet reached.
The fallback attempt just continues on whatever terms are left implied
by the conversation so far... actually it doesn't know what's "left" on
its own, which is why each attempt gets the FULL term list again (see
run_source_fetch_batch below) — a fallback re-doing an already-completed
term calls save_source again for it, which is a real, accepted
duplicate-document cost, not a bug: safe (rejected/duplicate reviews are
cheap for the user to dismiss) is chosen over complex resume-from-where-
it-left-off tracking for what is, so far, a first real slice of this
feature.
"""

import threading

from config.store import config
from dispatcher.executor import run_tool_loop
from providers.base import ChatMessage, ProviderError
from providers.registry import get_provider
from storage.sources import create_batch, finish_batch
from tools.registry import schemas_for

_SOURCE_FETCH_TOOLS = ["web_search", "fetch_page", "save_source"]

_SYSTEM_PROMPT = (
    "You are finding real, relevant source documents for a research collection. "
    "You are given a list of search terms and must process EACH ONE:\n\n"
    "For every term:\n"
    "1. Call web_search with that term as the query.\n"
    "2. Look at the results. They are already restricted to the user's trusted sites "
    "— but that only means the SITE is trusted, not that any given result is actually "
    "about the term. Judge relevance yourself.\n"
    "3. For each result that's genuinely relevant, call fetch_page to read it.\n"
    "4. If the fetched content really is useful for the term, call save_source with the "
    "term, the page's real title, its exact url, and the relevant extracted content. "
    "Do NOT call save_source for a page you judged irrelevant, and never invent a url "
    "or content you didn't actually fetch.\n"
    "5. Move to the next term.\n\n"
    "Batch your tool calls per step where you can (e.g. multiple web_search calls in "
    "one turn if you're confident about several terms at once) rather than one at a "
    "time, since you have a limited number of turns to get through the whole list. "
    "When every term has been handled, reply with a one-line plain-text summary of how "
    "many documents you saved — do not call any more tools after that."
)


def run_source_fetch_batch(terms: list[str], trusted_sites: list[str]) -> str:
    """The actual work, run synchronously — call this from a background
    thread (see start_source_fetch_batch below), never from a request
    handler directly; a multi-term batch with real web fetches is too
    slow to hold an HTTP request open for, same reasoning as /research.
    Returns the batch_id immediately... no wait, this IS the blocking
    part; start_source_fetch_batch is what returns immediately."""
    batch_id = create_batch()
    routing = config.get_task_routing("source_fetch")
    if not routing:
        finish_batch(batch_id, error="No routing configured for source_fetch.")
        return batch_id

    tools = schemas_for(_SOURCE_FETCH_TOOLS)
    attempts = [routing["primary"]] + routing.get("fallback", [])
    term_list = "\n".join(f"- {t}" for t in terms)
    last_error = None

    for attempt in attempts:
        model = attempt.get("model")
        if not model:
            continue
        try:
            provider = get_provider(attempt["provider"])
        except Exception as e:
            last_error = str(e)
            continue

        messages = [
            ChatMessage(role="system", content=_SYSTEM_PROMPT),
            ChatMessage(role="user", content=f"Search terms:\n{term_list}"),
        ]
        try:
            response = provider.chat(model=model, messages=messages, tools=tools)
            run_tool_loop(
                provider, model, messages, response,
                context={"batch_id": batch_id, "trusted_sites": trusted_sites},
                tools=tools,
            )
            finish_batch(batch_id)
            return batch_id
        except ProviderError as e:
            last_error = str(e)
            continue

    finish_batch(batch_id, error=last_error or "Every configured provider failed.")
    return batch_id


def start_source_fetch_batch(terms: list[str], trusted_sites: list[str]) -> None:
    """Fire-and-forget — the batch_id itself isn't returned to the caller
    here because the caller (server.py's /sources/batch route) needs to
    respond to the HTTP request immediately, before create_batch() has
    even run on the background thread. The PWA instead polls GET
    /sources (storage.sources.list_documents with no batch filter) and
    just watches the review queue grow, rather than tracking one
    specific batch_id end to end — simpler, and this is single-user
    software where "is anything new to review" is the only question
    that actually matters to the UI."""
    threading.Thread(target=run_source_fetch_batch, args=(terms, trusted_sites), daemon=True).start()
