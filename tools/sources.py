"""
tools/sources.py

The save_source tool — the ONE thing the source-fetch batch loop
(dispatcher/source_fetch.py) actually writes to disk on the model's
behalf. Matches the rest of this codebase's split: the model judges
(is this result relevant to the term I'm searching for?), the dispatcher
executes deterministically (writing the file, recording the DB row).
The model never touches storage directly — same reasoning as
create_workflow, save_note, etc. all being dispatcher-executed tools
rather than the model having any real filesystem/DB access.

Every call needs a batch_id (which batch this document belongs to) —
threaded in via dispatch()'s `context`, same pattern save_note already
uses for command/topic_slug, not something the model supplies itself.
"""

from storage.filen import StorageError, save_result
from storage.sources import create_document


class SourceSaveError(Exception):
    pass


def save_source_document(
    batch_id: str, term: str, title: str, url: str, content: str,
    status: str = "pending_review", reason: str | None = None,
) -> str:
    """Saves `content` to Filen (same storage every other saved artifact
    in this codebase uses) and records a source_documents row — default
    status='pending_review', nothing usable as real chat context until a
    human reviews and accepts it (see server.py's /sources/{id}/review
    route). `status`/`reason` are overridden by the caller only for a
    document the dispatcher already auto-rejected (tools/registry.py's
    content-quality guard) — still saved to Filen and still visible in
    the UI, just pre-marked 'rejected' with a reason so the human can see
    exactly what was extracted and why it wasn't good enough, rather than
    the term just silently having nothing. Returns the new document id."""
    filen_path = None
    try:
        filen_path = save_result(command="sources", topic_slug=term, filename=f"{title[:60]}.md", content=content)
    except StorageError as e:
        # A save failure shouldn't lose the document from the review
        # queue entirely — it just won't have a backing file. The DB row
        # (with filen_path=None) is still real and still reviewable.
        # Logged now (2026-09-06) — this used to swallow the real rclone
        # error completely, so a whole batch failing to save left zero
        # diagnostic trail (JuanJo found every document in a real batch
        # came back "content unavailable" with no way to see why).
        print(f"[save_source_document] Filen save failed for {url!r} (term={term!r}): {e}")
    return create_document(batch_id=batch_id, term=term, title=title, url=url, filen_path=filen_path, status=status, reason=reason)
