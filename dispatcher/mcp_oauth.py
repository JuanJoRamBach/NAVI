"""
dispatcher/mcp_oauth.py

Real MCP-spec OAuth 2.1 + PKCE authorization flow for remote MCP
connections (2026-09-04) — the actual answer to "pasting a token is a
pain," not a GitHub-specific shortcut. Same mechanism Claude's own remote
MCP connectors use, so a server that already supports it just works.

How discovery actually works — verified by hand against GitHub's live
server before writing a line of this, not assumed from the spec text:

1. An unauthenticated request to the MCP endpoint gets back a 401 with a
   WWW-Authenticate header naming a resource_metadata URL (RFC9728).
   Confirmed: `curl -X POST https://api.githubcopilot.com/mcp/ ...` (no
   auth) returns exactly this.
2. That URL's JSON names `authorization_servers` — the issuer to trust
   for this resource. GitHub's is https://github.com/login/oauth.
3. The issuer's OWN metadata (RFC8414) isn't always at the plain
   `<issuer>/.well-known/oauth-authorization-server` suffix — GitHub's
   only answers at the path-insertion form,
   `<origin>/.well-known/oauth-authorization-server<path>`. Both forms
   are tried, in that order, since a future server might implement
   either one.
4. If that metadata has a `registration_endpoint`, a client can register
   itself dynamically (RFC7591) — zero manual setup, ever, for that
   server. GitHub has none (confirmed: not present in its response), so
   it falls back to a manually-registered OAuth App's client_id/secret,
   read from `<SERVER_NAME>_OAUTH_CLIENT_ID`/`_OAUTH_CLIENT_SECRET` env
   vars — the one real gap in "fully automatic," and it's GitHub's
   limitation, not something this code chose to skip.

PKCE (RFC7636) is used unconditionally, dynamic-registration or not —
cheap, strictly more secure, and GitHub's own metadata advertises S256
support.

Scope handling is deliberately conservative, not "request everything the
server supports": GitHub's resource metadata alone lists scopes like
`delete_repo` and `admin:enterprise` — requesting those by default would
violate the same least-privilege principle this session's other security
work has been built around. KNOWN_MINIMAL_SCOPES below is the only place
scope gets set; anything not listed there gets no `scope` param at all
(most authorization servers fall back to a base/default scope when it's
omitted, rather than granting everything).
"""

import base64
import hashlib
import os
import re
import secrets
import time
from urllib.parse import urlencode, urlsplit

import requests

_HTTP_TIMEOUT_SECONDS = 15

# Minimal, real scopes per known service — extend this list by hand only
# when a specific NAVI feature actually needs a broader scope, never
# speculatively. Unlisted servers get no `scope` param (see module
# docstring).
KNOWN_MINIMAL_SCOPES: dict[str, str] = {
    "github": "repo read:org read:user",
    # Gmail/Calendar/Drive real scopes confirmed 2026-09-06 by fetching
    # each server's own resource metadata directly (gmailmcp/calendarmcp/
    # drivemcp.googleapis.com's .well-known/oauth-protected-resource/
    # <tool> endpoints), not copied from generic Google API docs.
    # gmail.compose covers draft creation (confirmed tool: "Creates a new
    # draft email") — NOT gmail.send, which isn't in this server's own
    # scopes_supported list at all, so real sending may need widening to
    # gmail.modify once a real send-capable tool is confirmed to need it.
    "gmail": "https://www.googleapis.com/auth/gmail.readonly https://www.googleapis.com/auth/gmail.compose",
    # calendar.events (not the broader bare "calendar" scope) — covers
    # read/create/modify events without full calendar-list admin rights.
    "calendar": "https://www.googleapis.com/auth/calendar.events",
    # drive.file (Google's own documented least-privilege pattern for a
    # third-party app) covers files NAVI creates/opens — NOT arbitrary
    # pre-existing files the user never opened through NAVI. Real
    # tradeoff, not an oversight: widen to add drive.readonly if reading
    # arbitrary existing Drive files turns out to be a real need.
    "drive": "https://www.googleapis.com/auth/drive.file",
}


class MCPOAuthError(Exception):
    pass


def _env_name(server_name: str, suffix: str) -> str:
    return f"{re.sub(r'[^A-Za-z0-9]', '_', server_name).upper()}_OAUTH_{suffix}"


def _fetch_json(url: str) -> dict:
    try:
        resp = requests.get(url, timeout=_HTTP_TIMEOUT_SECONDS)
    except requests.RequestException as e:
        raise MCPOAuthError(f"couldn't fetch {url}: {e}")
    if resp.status_code >= 400:
        raise MCPOAuthError(f"{url} returned {resp.status_code}")
    try:
        return resp.json()
    except ValueError:
        raise MCPOAuthError(f"{url} didn't return JSON")


def _mcp_call(mcp_url: str, method: str, params: dict) -> requests.Response:
    return requests.post(
        mcp_url,
        json={"jsonrpc": "2.0", "id": 1, "method": method, "params": params},
        headers={"Content-Type": "application/json", "Accept": "application/json, text/event-stream"},
        timeout=_HTTP_TIMEOUT_SECONDS,
    )


def _extract_resource_metadata_url(resp: requests.Response) -> str | None:
    match = re.search(r'resource_metadata="([^"]+)"', resp.headers.get("WWW-Authenticate", ""))
    return match.group(1) if match else None


def _discover_resource_metadata_url(mcp_url: str) -> str:
    """Finds the server's OAuth-protected-resource metadata URL (RFC9728)
    by provoking a 401. Two real, DIFFERENT challenge points exist in the
    wild, both verified by hand (2026-09-06) rather than assumed from the
    spec text — GitHub's own docstring note above only covered the first:

    1. GitHub: the bare `initialize` handshake itself is gated — a plain
       unauthenticated initialize call gets 401 immediately.
    2. Google's Gmail/Calendar/Drive/etc. MCP servers: `initialize` AND
       `tools/list` both answer 200 with real, full data, completely
       unauthenticated — only an actual `tools/call` invocation is
       gated. Confirmed live against gmailmcp/calendarmcp/drivemcp.
       googleapis.com: 200 on initialize and tools/list, 401 (with a
       real resource_metadata) only once called with a REAL tool name
       from that server's own tools/list response — an invented/wrong
       tool name still returns 200 (looks like an unauthenticated pass
       but is actually just "no such tool"), so the probe below reuses
       a name the server itself just told us about, not a guess."""
    try:
        resp = _mcp_call(mcp_url, "initialize", {
            "protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": {"name": "navi", "version": "1"},
        })
    except requests.RequestException as e:
        raise MCPOAuthError(f"couldn't reach the server: {e}")

    if resp.status_code == 401:
        url = _extract_resource_metadata_url(resp)
        if url:
            return url
        raise MCPOAuthError("server returned 401 but no resource_metadata in WWW-Authenticate — can't discover its OAuth setup")

    # Not gated at the handshake (Google-style) — find a real tool name
    # and provoke the challenge on an actual tool call instead.
    try:
        tools_resp = _mcp_call(mcp_url, "tools/list", {})
        tools = (tools_resp.json().get("result") or {}).get("tools") or []
    except (requests.RequestException, ValueError):
        tools = []
    if not tools:
        raise MCPOAuthError(
            f"server didn't challenge for auth on initialize (got {resp.status_code}) and tools/list "
            "returned no tools to probe with — it may not support OAuth here"
        )
    probe_name = tools[0]["name"]
    try:
        call_resp = _mcp_call(mcp_url, "tools/call", {"name": probe_name, "arguments": {}})
    except requests.RequestException as e:
        raise MCPOAuthError(f"couldn't reach the server: {e}")
    if call_resp.status_code != 401:
        raise MCPOAuthError(
            f"server didn't challenge for auth on a real tool call (got {call_resp.status_code}) — it may not support OAuth here"
        )
    url = _extract_resource_metadata_url(call_resp)
    if not url:
        raise MCPOAuthError("server returned 401 but no resource_metadata in WWW-Authenticate — can't discover its OAuth setup")
    return url


def _discover_authorization_server_metadata(issuer: str) -> dict:
    parts = urlsplit(issuer)
    candidates = [
        f"{parts.scheme}://{parts.netloc}{parts.path.rstrip('/')}/.well-known/oauth-authorization-server",
        f"{parts.scheme}://{parts.netloc}/.well-known/oauth-authorization-server{parts.path}",
    ]
    last_error: Exception | None = None
    for url in candidates:
        try:
            return _fetch_json(url)
        except MCPOAuthError as e:
            last_error = e
    raise MCPOAuthError(f"couldn't discover the authorization server's metadata: {last_error}")


def _register_dynamic_client(registration_endpoint: str, redirect_uri: str) -> tuple[str, str | None]:
    try:
        resp = requests.post(
            registration_endpoint,
            json={
                "client_name": "NAVI", "redirect_uris": [redirect_uri],
                "grant_types": ["authorization_code"], "response_types": ["code"],
                "token_endpoint_auth_method": "none",
            },
            timeout=_HTTP_TIMEOUT_SECONDS,
        )
    except requests.RequestException as e:
        raise MCPOAuthError(f"dynamic client registration failed: {e}")
    if resp.status_code >= 400:
        raise MCPOAuthError(f"dynamic client registration returned {resp.status_code}: {resp.text[:200]}")
    data = resp.json()
    client_id = data.get("client_id")
    if not client_id:
        raise MCPOAuthError("dynamic client registration response had no client_id")
    return client_id, data.get("client_secret")


def _generate_pkce() -> tuple[str, str]:
    verifier = base64.urlsafe_b64encode(secrets.token_bytes(32)).rstrip(b"=").decode()
    challenge = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).rstrip(b"=").decode()
    return verifier, challenge


def start_authorization(server_name: str, mcp_url: str, redirect_uri: str) -> dict:
    """Runs the full discovery chain and returns everything server.py
    needs: `authorize_url` to redirect the browser to, plus the pending-
    flow fields (state, code_verifier, token_endpoint, client_id,
    client_secret) it should stash server-side keyed by `state` until the
    callback arrives."""
    resource_metadata_url = _discover_resource_metadata_url(mcp_url)
    resource_metadata = _fetch_json(resource_metadata_url)
    authorization_servers = resource_metadata.get("authorization_servers") or []
    if not authorization_servers:
        raise MCPOAuthError("server's resource metadata named no authorization server")
    issuer = authorization_servers[0]
    as_metadata = _discover_authorization_server_metadata(issuer)

    authorization_endpoint = as_metadata.get("authorization_endpoint")
    token_endpoint = as_metadata.get("token_endpoint")
    if not authorization_endpoint or not token_endpoint:
        raise MCPOAuthError("authorization server metadata is missing authorization_endpoint/token_endpoint")

    registration_endpoint = as_metadata.get("registration_endpoint")
    if registration_endpoint:
        client_id, client_secret = _register_dynamic_client(registration_endpoint, redirect_uri)
    else:
        client_id = os.environ.get(_env_name(server_name, "CLIENT_ID"))
        client_secret = os.environ.get(_env_name(server_name, "CLIENT_SECRET"))
        if not client_id:
            raise MCPOAuthError(
                f"'{server_name}' has no dynamic client registration and no "
                f"{_env_name(server_name, 'CLIENT_ID')} configured — register an OAuth "
                f"App with this service and set that env var first."
            )

    verifier, challenge = _generate_pkce()
    state = secrets.token_urlsafe(24)

    params = {
        "response_type": "code", "client_id": client_id, "redirect_uri": redirect_uri,
        "state": state, "code_challenge": challenge, "code_challenge_method": "S256",
        # RFC8707 resource indicator — binds the issued token to THIS mcp
        # server specifically, not a blanket credential for the issuer.
        "resource": mcp_url,
    }
    scope = KNOWN_MINIMAL_SCOPES.get(server_name)
    if scope:
        params["scope"] = scope
    # Real gap found and fixed 2026-09-06: without this, Google issues a
    # plain ~1-hour access token and NO refresh token at all, silently
    # breaking the connection an hour after every single reconnect —
    # confirmed this is Google-specific (their own docs: "access_type=
    # offline" is required to get a refresh token; "prompt=consent"
    # ensures one is issued even on a repeat authorization, since Google
    # otherwise only grants it the very first time a user ever consents).
    # GitHub's own OAuth app flow neither needs nor documents these
    # params — scoped to Google's issuer specifically rather than sent
    # unconditionally to every provider, since an unfamiliar authorization
    # server's handling of unrecognized params isn't something to assume.
    if "accounts.google.com" in issuer:
        params["access_type"] = "offline"
        params["prompt"] = "consent"

    return {
        "authorize_url": f"{authorization_endpoint}?{urlencode(params)}",
        "state": state, "code_verifier": verifier, "token_endpoint": token_endpoint,
        "client_id": client_id, "client_secret": client_secret, "redirect_uri": redirect_uri,
    }


def exchange_code_for_token(
    token_endpoint: str, code: str, code_verifier: str,
    client_id: str, client_secret: str | None, redirect_uri: str,
) -> dict:
    """Returns {"access_token", "refresh_token" (None if the server didn't
    issue one), "expires_in" (seconds, None if the server didn't say)} —
    widened 2026-09-06 from a bare access_token string so a caller can
    actually persist enough to refresh later (see refresh_access_token
    below and config.set_mcp_oauth_tokens)."""
    data = {
        "grant_type": "authorization_code", "code": code, "redirect_uri": redirect_uri,
        "client_id": client_id, "code_verifier": code_verifier,
    }
    if client_secret:
        data["client_secret"] = client_secret
    try:
        resp = requests.post(token_endpoint, data=data, headers={"Accept": "application/json"}, timeout=_HTTP_TIMEOUT_SECONDS)
    except requests.RequestException as e:
        raise MCPOAuthError(f"token exchange failed: {e}")
    if resp.status_code >= 400:
        raise MCPOAuthError(f"token exchange returned {resp.status_code}: {resp.text[:200]}")
    try:
        payload = resp.json()
    except ValueError:
        raise MCPOAuthError("token exchange response wasn't JSON")
    access_token = payload.get("access_token")
    if not access_token:
        raise MCPOAuthError(f"token exchange response had no access_token: {payload}")
    return {
        "access_token": access_token,
        "refresh_token": payload.get("refresh_token"),
        "expires_in": payload.get("expires_in"),
    }


def refresh_access_token(token_endpoint: str, refresh_token: str, client_id: str, client_secret: str | None) -> dict:
    """Returns {"access_token", "expires_in"} — a refresh grant, unlike
    the initial authorization_code exchange, never returns a NEW refresh
    token from Google (the original one stays valid and reusable), so
    there's nothing else to persist here."""
    data = {"grant_type": "refresh_token", "refresh_token": refresh_token, "client_id": client_id}
    if client_secret:
        data["client_secret"] = client_secret
    try:
        resp = requests.post(token_endpoint, data=data, headers={"Accept": "application/json"}, timeout=_HTTP_TIMEOUT_SECONDS)
    except requests.RequestException as e:
        raise MCPOAuthError(f"token refresh failed: {e}")
    if resp.status_code >= 400:
        raise MCPOAuthError(f"token refresh returned {resp.status_code}: {resp.text[:200]}")
    try:
        payload = resp.json()
    except ValueError:
        raise MCPOAuthError("token refresh response wasn't JSON")
    access_token = payload.get("access_token")
    if not access_token:
        raise MCPOAuthError(f"token refresh response had no access_token: {payload}")
    return {"access_token": access_token, "expires_in": payload.get("expires_in")}


def ensure_fresh_access_token(server_name: str) -> str | None:
    """The one place anything that's about to actually USE a connection's
    token should call first — proactively refreshes if the stored token
    is expired or about to be (60s buffer for clock skew/request latency)
    and a refresh_token is on file, persists the new token, and returns
    the real bearer auth_header to use. Returns the connection's existing
    auth_header unchanged if there's nothing to refresh (no expiry known,
    not close to expiring, or no refresh_token stored — e.g. GitHub's
    connection, which never gets one at all). Never raises — a refresh
    failure here shouldn't crash the caller; it just returns the
    (possibly stale) existing auth_header and lets the real API call
    fail with its own real 401 if the token truly is dead."""
    from config.store import config
    conn = config.get_mcp_connection(server_name)
    if conn is None:
        return None
    expires_at = conn.get("oauth_expires_at")
    refresh_token = conn.get("oauth_refresh_token")
    if not refresh_token or not expires_at or time.time() < expires_at - 60:
        return conn.get("auth_header")
    try:
        result = refresh_access_token(
            conn["oauth_token_endpoint"], refresh_token, conn["oauth_client_id"], conn.get("oauth_client_secret"),
        )
    except MCPOAuthError as e:
        print(f"[ensure_fresh_access_token] refresh failed for '{server_name}': {e}")
        return conn.get("auth_header")
    new_expires_at = time.time() + result["expires_in"] if result.get("expires_in") else None
    config.set_mcp_oauth_tokens(
        server_name, access_token=result["access_token"], refresh_token=refresh_token,
        expires_at=new_expires_at, token_endpoint=conn["oauth_token_endpoint"],
        client_id=conn["oauth_client_id"], client_secret=conn.get("oauth_client_secret"),
    )
    return f"Bearer {result['access_token']}"
