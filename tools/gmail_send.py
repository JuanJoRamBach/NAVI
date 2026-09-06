"""
tools/gmail_send.py

Real Gmail send — bypasses the connected Gmail MCP server entirely.
Confirmed by hand (2026-09-06): Google's official gmailmcp.googleapis.com
has NO send capability at all — only create_draft, list_drafts, and
thread/label management. This calls Gmail's own REST API directly
instead, reusing the same OAuth access token the Gmail connection
already holds (config.get_mcp_connection("gmail")'s auth_header) — no
separate credential, no second connection needed.

Known gap, not fixed here: the stored token is access-token-only. The
OAuth flow (dispatcher/mcp_oauth.py) doesn't request offline access, so
there's no refresh token — this will start failing with a real 401 once
the access token expires (~1 hour after connecting). Reconnecting Gmail
gets a fresh token in the meantime; real refresh-token handling is a
separate, deliberate follow-up, not silently worked around here.
"""

import base64
from email.mime.text import MIMEText

import requests

from dispatcher.mcp_oauth import ensure_fresh_access_token

GMAIL_SEND_URL = "https://gmail.googleapis.com/gmail/v1/users/me/messages/send"


class GmailSendError(Exception):
    pass


def send_gmail_message(to: str, subject: str, body: str, html: bool = True) -> str:
    """Sends a real email via Gmail's REST API, using the connected
    Gmail account's own OAuth token — refreshed first if it's expired or
    close to it (ensure_fresh_access_token), so this never surfaces the
    "reconnect Gmail" error just because an hour passed since connecting
    (2026-09-06: "having to refresh the token all the time is a huge pain
    point that we must not make the users go through"). `to` accepts a
    single address or a comma-separated list (standard RFC 2822 —
    MIMEText's own "to" header handles either form). Raises
    GmailSendError on any failure — a failed send must never be reported
    back as if it succeeded."""
    auth_header = ensure_fresh_access_token("gmail")
    if not auth_header:
        raise GmailSendError("Gmail isn't connected — connect it first in the Connections panel.")

    mime = MIMEText(body, "html" if html else "plain")
    mime["to"] = to
    mime["subject"] = subject
    raw = base64.urlsafe_b64encode(mime.as_bytes()).decode()

    try:
        resp = requests.post(
            GMAIL_SEND_URL,
            headers={"Authorization": auth_header, "Content-Type": "application/json"},
            json={"raw": raw},
            timeout=20,
        )
    except requests.RequestException as e:
        raise GmailSendError(f"Gmail send request failed: {e}")
    if resp.status_code == 401:
        raise GmailSendError("Gmail token expired or invalid — reconnect Gmail in the Connections panel.")
    if resp.status_code >= 400:
        raise GmailSendError(f"Gmail send failed ({resp.status_code}): {resp.text[:300]}")
    try:
        return resp.json().get("id", "sent")
    except ValueError:
        return "sent"
