"""
tools/mcp_marketplace.py

Thin read-only proxy over the official MCP Registry
(registry.modelcontextprotocol.io) — lets ConnectionsOverlay show a real,
searchable list of publicly available MCP servers instead of only the
fixed SERVICE_CATALOG, and pre-fill the existing connect form from a
result instead of the user typing a command/URL by hand.

Called server-side (not directly from the browser) for the same reason
every other external call in this codebase is server-side: keeps API
keys/tokens (none needed here, but matches the pattern) and CORS off the
frontend's plate, and leaves room for a future rate-limit/cache layer in
one place.

Registry note (2026-09-04 research pass): the official docs say host
apps should consume a DOWNSTREAM aggregator (Smithery, Glama, mcp.so)
rather than the official registry directly — those exist mainly to add
curation/ratings on top of the same underlying metadata. The official
registry's own public API works fine with no key for a plain search
(Glama's, by contrast, requires one), so this hits it directly for a v1;
swapping to an aggregator later is a small change (same response shape
family, not a rewrite) if curation ever becomes worth the extra
dependency.
"""

import requests

REGISTRY_URL = "https://registry.modelcontextprotocol.io/v0/servers"


class MCPMarketplaceError(Exception):
    pass


def _simplify(entry: dict) -> dict:
    """Reduces one registry entry down to exactly what ConnectionsOverlay
    needs to show a result and pre-fill the existing connect form —
    real transport info if there's something NAVI's own MCP client can
    actually use (a remote streamable-http URL, or an npm package run
    via `npx`), `transport: None` otherwise (a docker-only or other
    package type this v1 doesn't support connecting to yet)."""
    server = entry.get("server", {})
    remotes = server.get("remotes") or []
    packages = server.get("packages") or []
    remote = remotes[0] if remotes else None
    npm_package = next((p for p in packages if p.get("registryType") == "npm"), None)

    result = {
        "name": server.get("name", ""),
        "title": server.get("title") or server.get("name", "").rsplit("/", 1)[-1],
        "description": server.get("description", ""),
        "repository_url": (server.get("repository") or {}).get("url"),
        "transport": None,
        "requires_auth": False,
    }
    if remote:
        headers = remote.get("headers") or []
        result["transport"] = "http"
        result["url"] = remote.get("url")
        result["requires_auth"] = any(h.get("isRequired") for h in headers)
    elif npm_package:
        env_vars = npm_package.get("environmentVariables") or []
        result["transport"] = "stdio"
        result["command"] = "npx"
        result["args"] = ["-y", npm_package.get("identifier", "")]
        result["requires_auth"] = any(v.get("isRequired") for v in env_vars)
    return result


def search(query: str, limit: int = 20) -> list[dict]:
    """Real names, no key needed. `query` searches server names/
    descriptions server-side (the registry's own `search` param);
    empty returns whatever's most recently published."""
    params: dict = {"limit": limit}
    if query:
        params["search"] = query
    try:
        resp = requests.get(REGISTRY_URL, params=params, timeout=15)
    except requests.RequestException as e:
        raise MCPMarketplaceError(f"couldn't reach the MCP registry: {e}")
    if resp.status_code >= 400:
        raise MCPMarketplaceError(f"MCP registry error {resp.status_code}: {resp.text[:200]}")

    data = resp.json()
    results = []
    for entry in data.get("servers", []):
        # The registry returns every published version of a server as
        # its own row (confirmed: searching "github" returned the same
        # server twice, different versions back to back) — only the
        # latest is useful to show.
        meta = entry.get("_meta", {}).get("io.modelcontextprotocol.registry/official", {})
        if not meta.get("isLatest", True):
            continue
        simplified = _simplify(entry)
        # http-only (2026-09-04): stdio servers are disabled entirely — no
        # sandbox exists for arbitrary local process execution (deferred,
        # unfunded; see dispatcher/mcp_client.py). Surfacing a stdio-only
        # result here would just be a dead end at connect time, so it's
        # filtered before the user ever sees it rather than after.
        if simplified["transport"] == "http":
            results.append(simplified)
    return results
