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
from urllib.parse import urlencode, urlsplit

import requests

_HTTP_TIMEOUT_SECONDS = 15

# Minimal, real scopes per known service — extend this list by hand only
# when a specific NAVI feature actually needs a broader scope, never
# speculatively. Unlisted servers get no `scope` param (see module
# docstring).
KNOWN_MINIMAL_SCOPES: dict[str, str] = {
    "github": "repo read:org read:user",
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


def _discover_resource_metadata_url(mcp_url: str) -> str:
    """A minimal, unauthenticated MCP initialize call — a compliant
    OAuth-protected server answers 401 with a WWW-Authenticate header
    naming its protected-resource metadata (RFC9728)."""
    try:
        resp = requests.post(
            mcp_url,
            json={
                "jsonrpc": "2.0", "id": 1, "method": "initialize",
                "params": {"protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": {"name": "navi", "version": "1"}},
            },
            headers={"Content-Type": "application/json", "Accept": "application/json, text/event-stream"},
            timeout=_HTTP_TIMEOUT_SECONDS,
        )
    except requests.RequestException as e:
        raise MCPOAuthError(f"couldn't reach the server: {e}")
    if resp.status_code != 401:
        raise MCPOAuthError(
            f"server didn't challenge for auth (got {resp.status_code}) — it may not support OAuth here"
        )
    match = re.search(r'resource_metadata="([^"]+)"', resp.headers.get("WWW-Authenticate", ""))
    if not match:
        raise MCPOAuthError("server returned 401 but no resource_metadata in WWW-Authenticate — can't discover its OAuth setup")
    return match.group(1)


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

    return {
        "authorize_url": f"{authorization_endpoint}?{urlencode(params)}",
        "state": state, "code_verifier": verifier, "token_endpoint": token_endpoint,
        "client_id": client_id, "client_secret": client_secret, "redirect_uri": redirect_uri,
    }


def exchange_code_for_token(
    token_endpoint: str, code: str, code_verifier: str,
    client_id: str, client_secret: str | None, redirect_uri: str,
) -> str:
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
    return access_token
