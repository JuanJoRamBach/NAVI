"""
tools/telegram_send.py

Lets the model push a message to JuanJo's own Telegram from any chat
channel — not just as a reply within an existing Telegram conversation.
Uses the same TELEGRAM_BOT_TOKEN as the webhook adapter, targeting the
fixed TELEGRAM_CHAT_ID env var (the same one the autonomous jobs already
report to) since NAVI is single-user — there's no "which chat" to ask.
"""

import os

from messaging.base import MessagingError
from messaging.telegram import TelegramAdapter


class TelegramSendError(Exception):
    pass


def send_to_telegram(text: str) -> str:
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        raise TelegramSendError("TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID not configured")

    try:
        TelegramAdapter(token).send_message(chat_id, text)
    except MessagingError as e:
        raise TelegramSendError(str(e))

    return "Sent to Telegram."
