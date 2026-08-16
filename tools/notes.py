"""
tools/notes.py

Lets a model save an intermediate note mid-step (e.g. "here's a source I
found, worth keeping separately from the final result") via the same
Filen-backed storage the executor already uses for final step output.
Thin wrapper around storage.filen.save_result — same folder convention,
just callable as a tool.
"""

from storage.filen import StorageError, save_result


class NoteError(Exception):
    pass


def save_note(command: str, topic_slug: str, filename: str, content: str) -> str:
    """Returns the Filen path the note was saved to. Raises NoteError on
    failure — same disclosure principle as the executor's own saves."""
    try:
        return save_result(command=command, topic_slug=topic_slug, filename=filename, content=content)
    except StorageError as e:
        raise NoteError(str(e))
