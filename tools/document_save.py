"""
tools/document_save.py

Lets a model save a finished, user-facing document (something asked for as
a real file — a write-up, a report, a list to keep) and hand back a real
download link. Distinct from tools/notes.py's save_note: a note is an
internal working artifact with no link back to the user; a document is the
actual deliverable. Also distinct from tools/documents.py, an unrelated,
pre-existing PDF/DOCX/PPTX renderer used by /research's opt-in file
formats — this module never touches that one, plain text/markdown/HTML
content saved as-is.

Folder grouping uses dispatcher/slugify.py on the document's own title
(deterministic, no extra model call) rather than the generic "chat" slug
run_tool_loop's context carries for chat-mode tool calls — otherwise every
document ever created from chat would collapse into one undifferentiated
folder.
"""

from dispatcher.slugify import slugify
from storage.filen import StorageError, file_download_url, save_result


class DocumentError(Exception):
    pass


def create_document(command: str, title: str, filename: str, content: str) -> str:
    """Saves `content` to Filen and returns a download URL for it. Raises
    DocumentError on a save failure, or if no download link could be built
    (NAVI_FILES_TOKEN not configured) — never claim success without a real,
    usable link, same disclosure principle as the executor's own saves."""
    topic_slug = slugify(title)
    try:
        saved_path = save_result(command=command, topic_slug=topic_slug, filename=filename, content=content)
    except StorageError as e:
        raise DocumentError(str(e))

    render = filename.lower().endswith(".html")
    url = file_download_url(saved_path, render=render)
    if not url:
        raise DocumentError("saved, but no download link could be generated (NAVI_FILES_TOKEN not configured)")
    return url
