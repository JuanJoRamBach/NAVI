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


def save_source_document(batch_id: str, term: str, title: str, url: str, content: str) -> str:
    """Saves `content` to Filen (same storage every other saved artifact
    in this codebase uses) and records a source_documents row with
    status='pending_review' — nothing saved through this tool is usable
    as real chat context until a human reviews and accepts it (see
    server.py's /sources/{id}/review route). Returns the new document id."""
    filen_path = None
    try:
        filen_path = save_result(command="sources", topic_slug=term, filename=f"{title[:60]}.md", content=content)
    except StorageError:
        # A save failure shouldn't lose the document from the review
        # queue entirely — it just won't have a backing file. The DB row
        # (with filen_path=None) is still real and still reviewable.
        pass
    return create_document(batch_id=batch_id, term=term, title=title, url=url, filen_path=filen_path)
