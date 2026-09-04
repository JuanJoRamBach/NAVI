"""
scripts/set_telegram_webhook.py

One-time setup: tells Telegram where to send updates. Run this locally
once after the first Render deploy (and again if the Render URL ever
changes).

Usage:
    TELEGRAM_BOT_TOKEN=... RENDER_URL=https://navi-xxxx.onrender.com python scripts/set_telegram_webhook.py

TELEGRAM_WEBHOOK_SECRET is optional but recommended (2026-09-04) — set it
here AND as an env var on the server (server.py reads it under the same
name to verify incoming updates). Without it, POST /webhook/telegram has
no way to tell a real Telegram update from anyone who guesses the URL and
POSTs a fake one.
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from messaging.base import MessagingError  # noqa: E402
from messaging.telegram import TelegramAdapter  # noqa: E402


def main() -> None:
    bot_token = os.environ.get("TELEGRAM_BOT_TOKEN")
    render_url = os.environ.get("RENDER_URL")
    if not bot_token or not render_url:
        raise SystemExit("Set TELEGRAM_BOT_TOKEN and RENDER_URL env vars first.")

    secret = os.environ.get("TELEGRAM_WEBHOOK_SECRET")
    webhook_url = render_url.rstrip("/") + "/webhook/telegram"
    try:
        TelegramAdapter(bot_token).set_webhook(webhook_url, secret_token=secret)
    except MessagingError as e:
        raise SystemExit(f"Failed to set webhook: {e}")

    print(f"Webhook set to {webhook_url}" + (" (with a secret token)" if secret else " (NO secret token set — set TELEGRAM_WEBHOOK_SECRET and re-run)"))


if __name__ == "__main__":
    main()
