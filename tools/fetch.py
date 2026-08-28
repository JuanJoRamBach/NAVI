"""
tools/fetch.py

Fetches a URL and extracts readable text from the HTML — no new
dependency (no BeautifulSoup/readability), just enough regex-based
stripping to hand a model plain text instead of markup soup. Good enough
for research-tool use; not meant to be a general-purpose scraper.
"""

import html
import re

import requests

MAX_CHARS = 8000  # keep tool output small enough to not blow the model's context

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


def fetch_page(url: str, max_chars: int = MAX_CHARS) -> str:
    """Returns extracted plain text from the page, truncated to max_chars.
    Raises FetchError on request failure or a non-HTML/non-text response."""
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

    text = _extract_text(resp.text)
    return text[:max_chars]


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
