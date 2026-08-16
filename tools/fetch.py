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
_TAG_RE = re.compile(r"<[^>]+>")
_WHITESPACE_RE = re.compile(r"[ \t]+")
_BLANK_LINES_RE = re.compile(r"\n{3,}")


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
    no_tags = _TAG_RE.sub("\n", no_script)
    unescaped = html.unescape(no_tags)
    collapsed = _WHITESPACE_RE.sub(" ", unescaped)
    return _BLANK_LINES_RE.sub("\n\n", collapsed).strip()
