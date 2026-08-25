"""
push/sender.py

Sends a Web Push notification to every subscribed browser via
pywebpush, using this deploy's VAPID identity (env vars —
VAPID_PRIVATE_KEY_B64, VAPID_SUBJECT). VAPID_PRIVATE_KEY_B64 is the
raw 32-byte EC private key, base64url-encoded with no padding — NOT a
PEM string. pywebpush's Vapid.from_string() expects exactly this raw
format and will silently produce a mismatched key pair if handed a PEM
blob instead, so don't "helpfully" reformat it.

Subscriptions live in config/store.py's generic key-value store (key:
"push_subscriptions") — same Filen-backed persistence as everything
else in config, no separate storage layer needed for what's just a
list of a few JSON objects.
"""

import json
import os

from pywebpush import webpush, WebPushException

from config.store import config


class PushError(Exception):
    pass


def add_subscription(subscription: dict) -> None:
    """Called from POST /push/subscribe. Subscriptions are keyed by their
    unique endpoint URL, so re-subscribing (e.g. after a browser clears
    its push registration) replaces the old entry instead of piling up
    duplicates."""
    subs = config.get("push_subscriptions", [])
    endpoint = subscription.get("endpoint")
    subs = [s for s in subs if s.get("endpoint") != endpoint]
    subs.append(subscription)
    config.set("push_subscriptions", subs)


def subscription_count() -> int:
    return len(config.get("push_subscriptions", []))


def send_push(title: str, body: str, url: str | None = None) -> list[str]:
    """Sends to every stored subscription. A subscription that comes back
    404/410 (browser unsubscribed, or the endpoint expired) is pruned
    automatically instead of failing the same way on every future send.
    Returns a list of human-readable errors for subscriptions that failed
    for some other reason, so /push/test can actually report what went
    wrong instead of a bare 200.
    """
    subs = config.get("push_subscriptions", [])
    if not subs:
        raise PushError("no push subscriptions registered")

    private_key = os.environ.get("VAPID_PRIVATE_KEY_B64")
    if not private_key:
        raise PushError("VAPID_PRIVATE_KEY_B64 not set")
    subject = os.environ.get("VAPID_SUBJECT", "mailto:navi-app@localhost")

    payload = {"title": title, "body": body}
    if url:
        payload["url"] = url

    errors: list[str] = []
    still_valid: list[dict] = []
    for sub in subs:
        try:
            webpush(
                subscription_info=sub,
                data=json.dumps(payload),
                vapid_private_key=private_key,
                vapid_claims={"sub": subject},
            )
            still_valid.append(sub)
        except WebPushException as e:
            status = getattr(e.response, "status_code", None)
            if status in (404, 410):
                continue  # expired/unsubscribed — drop it, don't re-add
            still_valid.append(sub)  # transient failure — keep it, report the error
            errors.append(str(e))

    if len(still_valid) != len(subs):
        config.set("push_subscriptions", still_valid)

    return errors
