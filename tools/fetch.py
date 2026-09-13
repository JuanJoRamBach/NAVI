"""
tools/fetch.py

Fetches a URL and extracts readable content from the HTML.

Two extractors, in order of preference:

1. **trafilatura** — real boilerplate removal plus MARKDOWN output, so
   headings, lists, emphasis, tables and links survive into what the
   model reads. Also returns real metadata (title, author, date).
2. The original regex stripper, kept as a fallback.

The regex path exists because it has to — a missing optional dependency
must never break fetching (see tools/report_render.py for what happens
when one does) — but it is genuinely bad, and knowing WHY matters for
anyone tempted to rely on it: it replaces every tag with a newline, so a
document's entire structure collapses into undifferentiated lines. A
model handed that output cannot recover headings or list boundaries
because they no longer exist. That is a real, observed failure: a Sources
batch produced documents that were "unreadable, no format, just copied a
bunch" (JuanJo, 2026-09-13), and this was why.

It is also expensive. A typical page is 70-90% markup, navigation, ads
and scripts; the regex path strips tags but keeps far more of the
surviving cruft than real boilerplate detection does, so it burns context
budget on material that was never content.
"""

import html
import re

import requests

# Soft import — same rule as every other optional dependency here after
# an uninstalled docxtpl crash-looped the whole API (see
# tools/report_render.py). A missing trafilatura degrades extraction
# quality; it must not stop a page being fetched at all.
try:
    import trafilatura
except ImportError:
    trafilatura = None

MAX_CHARS = 8000  # keep tool output small enough to not blow the model's context

# Sources reads whole pages, and 8,000 was sized for a different job.
# That number exists so a fetch_page result dropped into a CHAT turn
# doesn't swallow the context window — a real constraint there, where the
# fetch is one of several things competing for room alongside history, a
# brief and tool schemas.
#
# A Source distillation has none of that competition: one call, one page,
# an instruction of about 600 tokens, on a role whose models carry 128K
# (nemotron-3-super) and 256K (mistral-small-latest). ~60,000 characters
# is roughly 15,000 tokens, which reads most articles end to end and
# still leaves the context overwhelmingly empty.
#
# This matters for correctness, not comfort. The Faros article hit the
# 8,000 cap, so its document was an honest reading of the first third of
# the page presented as a reading of the page. Truncation is still
# reported when it happens — it just now happens far less.
SOURCE_MAX_CHARS = 60_000

_SCRIPT_STYLE_RE = re.compile(r"<(script|style)[^>]*>.*?</\1>", re.DOTALL | re.IGNORECASE)
# Whole-element strip for the containers that are reliably navigation/
# boilerplate, not article content — <nav>/<header>/<footer>/<aside>/
# <form> and their entire subtree, same reasoning as script/style above.
_BOILERPLATE_RE = re.compile(
    r"<(nav|header|footer|aside|form)[^>]*>.*?</\1>", re.DOTALL | re.IGNORECASE
)
_TAG_RE = re.compile(r"<[^>]+>")
_WHITESPACE_RE = re.compile(r"[ \t]+")


class FetchError(Exception):
    pass


def fetch_document(url: str, max_chars: int = MAX_CHARS) -> dict:
    """Fetches a page and returns its readable content plus what we know
    about where it came from:

        {"url", "title", "author", "published", "text", "markdown",
         "extractor", "truncated"}

    `markdown` is the structure-preserving version and is what anything
    building a Source document should use. `text` is the same content
    flattened, for callers that only want prose. `extractor` names which
    path produced it, so a downstream consumer can tell real extraction
    from the degraded fallback rather than guessing from the shape.

    Raises FetchError on request failure or a non-HTML/non-text response.
    """
    raw, resp_url = _get(url)
    title = author = published = None
    markdown = ""

    if trafilatura is not None:
        # include_links keeps citations intact; include_tables/formatting
        # keep the structure that makes a document readable at all.
        markdown = trafilatura.extract(
            raw, output_format="markdown", include_links=True,
            include_tables=True, include_formatting=True,
            favor_precision=True, url=resp_url,
        ) or ""
        try:
            meta = trafilatura.extract_metadata(raw, default_url=resp_url)
            if meta:
                title = meta.title or None
                author = meta.author or None
                published = meta.date or None
        except Exception:  # noqa: BLE001 - metadata is a bonus, never a reason to fail
            pass

    markdown = _strip_template_placeholders(markdown)

    extractor = "trafilatura"
    if not markdown.strip():
        # Either trafilatura isn't installed, or it found nothing it was
        # confident about (favor_precision means it would rather return
        # nothing than return navigation). Fall back rather than return an
        # empty document.
        markdown = _extract_text(raw)
        extractor = "regex-fallback"
        if title is None:
            title = _title_from_html(raw)

    truncated = len(markdown) > max_chars
    markdown = markdown[:max_chars]
    return {
        "url": resp_url,
        "title": title,
        "author": author,
        "published": published,
        "markdown": markdown,
        "text": _markdown_to_plain(markdown),
        "extractor": extractor,
        "truncated": truncated,
    }


# CMS template tokens that survive extraction because they were never
# HTML — a publishing platform's own placeholder for a block it renders
# later ({{cta}}, {% newsletter %}, [[related]]). Real, observed: the
# Faros article carries two bare {{cta}} lines. They are pure noise to a
# reader and to a model, and a Source document quoting one would be
# quoting the CMS rather than the author.
#
# Deliberately conservative on two counts, because over-stripping here
# destroys real content:
#
#  - WHOLE-LINE ONLY. A token alone on its own line is a rendered block.
#    The same token inline is very likely the subject matter — an article
#    about Jinja, Liquid or Handlebars is full of them, and so is
#    anything written about Agent Work's own {{state.node.field}} syntax.
#  - NEVER INSIDE A FENCE. Code blocks are exactly where these appear
#    legitimately, so fenced regions are passed through untouched.
_PLACEHOLDER_LINE_RE = re.compile(r"^\s*(\{\{[^}]*\}\}|\{%[^%]*%\}|\[\[[^\]]*\]\])\s*$")
_FENCE_RE = re.compile(r"^\s*(```|~~~)")


def _strip_template_placeholders(md: str) -> str:
    out: list[str] = []
    in_fence = False
    for line in md.split("\n"):
        if _FENCE_RE.match(line):
            in_fence = not in_fence
            out.append(line)
            continue
        if not in_fence and _PLACEHOLDER_LINE_RE.match(line):
            continue
        out.append(line)
    # A dropped block can leave three blank lines where there were two.
    return re.sub(r"\n{3,}", "\n\n", "\n".join(out)).strip()


_TITLE_RE = re.compile(r"<title[^>]*>(.*?)</title>", re.DOTALL | re.IGNORECASE)
_MD_SYNTAX_RE = re.compile(r"^[#>\s]*|[*_`]+")
_MD_LINK_RE = re.compile(r"\[([^\]]*)\]\([^)]*\)")


def _title_from_html(raw_html: str) -> str | None:
    m = _TITLE_RE.search(raw_html)
    return html.unescape(m.group(1)).strip() if m else None


def _markdown_to_plain(md: str) -> str:
    """Flattens markdown for callers that want prose. Deliberately lossy —
    the markdown itself is kept alongside, so nothing is destroyed by
    this, only set aside."""
    lines = [_MD_SYNTAX_RE.sub("", _MD_LINK_RE.sub(r"\1", line)).strip() for line in md.split("\n")]
    return "\n".join(l for l in lines if l)


def fetch_page(url: str, max_chars: int = MAX_CHARS) -> str:
    """Plain-text extraction, unchanged contract — every existing caller
    (the web_search/fetch_page tool path, source_fetch, research) still
    gets a string. Now backed by the better extractor when it's available,
    so those callers improve without changing.
    """
    return fetch_document(url, max_chars=max_chars)["text"]


def _get(url: str) -> tuple[str, str]:
    try:
        resp = requests.get(
            url,
            headers={"User-Agent": "Mozilla/5.0 (compatible; NAVI/1.0)"},
            timeout=20,
        )
    except requests.RequestException as e:
        raise FetchError(f"Fetch failed for {url}: {e}")

    if resp.status_code >= 400:
        raise FetchError(f"Fetch error {resp.status_code} for {url}")

    content_type = resp.headers.get("Content-Type", "")
    if "text" not in content_type and "html" not in content_type:
        raise FetchError(f"Unsupported content type '{content_type}' for {url}")

    # The RESOLVED url, not the requested one — a redirect means the
    # content came from somewhere else, and a Source document that cites
    # the pre-redirect address points at the wrong place.
    return resp.text, str(resp.url or url)


def _extract_text(raw_html: str) -> str:
    no_script = _SCRIPT_STYLE_RE.sub(" ", raw_html)
    no_boilerplate = _BOILERPLATE_RE.sub(" ", no_script)
    # Every tag becomes its own newline — on a real page that's thousands
    # of them (one per <li>/<a>/<div> in whatever nav survived the strip
    # above), so most "lines" at this point are empty or just whitespace
    # from a text node between two tags. A blank-line-collapse regex
    # doesn't fix this (those aren't multiple consecutive newlines, just
    # one newline per near-empty line) — filtering out empty lines after
    # stripping each one is what actually removes the noise. This was a
    # real bug: /research's synthesis budget was getting eaten by exactly
    # this (hundreds of blank/whitespace lines per fetched page).
    no_tags = _TAG_RE.sub("\n", no_boilerplate)
    unescaped = html.unescape(no_tags)
    collapsed = _WHITESPACE_RE.sub(" ", unescaped)
    lines = [line.strip() for line in collapsed.split("\n")]
    return "\n".join(line for line in lines if line).strip()
