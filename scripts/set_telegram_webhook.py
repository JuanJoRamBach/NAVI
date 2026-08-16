"""
scripts/set_telegram_webhook.py

One-time setup: tells Telegram where to send updates. Run this locally
once after the first Render deploy (and again if the Render URL ever
changes).

Usage:
    TELEGRAM_BOT_TOKEN=... RENDER_URL=https://navi-xxxx.onrender.com python scripts/set_telegram_webhook.py
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

    webhook_url = render_url.rstrip("/") + "/webhook/telegram"
    try:
        TelegramAdapter(bot_token).set_webhook(webhook_url)
    except MessagingError as e:
        raise SystemExit(f"Failed to set webhook: {e}")

    print(f"Webhook set to {webhook_url}")


if __name__ == "__main__":
    main()
