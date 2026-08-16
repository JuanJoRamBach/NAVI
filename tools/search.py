"""
tools/search.py

No-key web search via DuckDuckGo's HTML endpoint (html.duckduckgo.com) —
no API key, no signup, matches the free-tier-only constraint. Parsed with
plain regex rather than a new HTML-parsing dependency (BeautifulSoup etc.)
since the result markup is small and stable enough not to need one.
"""

import html
import re

import requests

SEARCH_URL = "https://html.duckduckgo.com/html/"

_RESULT_RE = re.compile(
    r'<a[^>]+class="result__a"[^>]+href="(?P<href>[^"]+)"[^>]*>(?P<title>.*?)</a>',
    re.DOTALL,
)
_SNIPPET_RE = re.compile(
    r'<a[^>]+class="result__snippet"[^>]*>(?P<snippet>.*?)</a>',
    re.DOTALL,
)
_TAG_RE = re.compile(r"<[^>]+>")


class SearchError(Exception):
    pass


def _strip_tags(s: str) -> str:
    return html.unescape(_TAG_RE.sub("", s)).strip()


def web_search(query: str, max_results: int = 5) -> list[dict]:
    """
    Returns up to max_results dicts: {"title": str, "url": str, "snippet": str}.
    Raises SearchError on request failure — callers decide whether that's
    fatal for the step or just means an empty result set gets reported.
    """
    try:
        resp = requests.post(
            SEARCH_URL,
            data={"q": query},
            headers={"User-Agent": "Mozilla/5.0 (compatible; NAVI/1.0)"},
            timeout=20,
        )
    except requests.RequestException as e:
        raise SearchError(f"DuckDuckGo search failed: {e}")

    if resp.status_code >= 400:
        raise SearchError(f"DuckDuckGo search error {resp.status_code}")

    titles_links = _RESULT_RE.findall(resp.text)
    snippets = _SNIPPET_RE.findall(resp.text)

    results = []
    for i, (href, title) in enumerate(titles_links):
        if len(results) >= max_results:
            break
        url = _clean_ddg_redirect(href)
        if "duckduckgo.com/y.js" in url or "bing.com/aclick" in url:
            continue  # sponsored result, not an organic hit
        snippet = _strip_tags(snippets[i]) if i < len(snippets) else ""
        results.append({
            "title": _strip_tags(title),
            "url": url,
            "snippet": snippet,
        })
    return results


def _clean_ddg_redirect(href: str) -> str:
    """DuckDuckGo's HTML results wrap URLs in a //duckduckgo.com/l/?uddg=...
    redirect — unwrap it so callers (and fetch_page) get the real URL."""
    href = html.unescape(href)
    match = re.search(r"uddg=([^&]+)", href)
    if match:
        from urllib.parse import unquote
        return unquote(match.group(1))
    return href if href.startswith("http") else f"https:{href}"
