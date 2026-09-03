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


def send_file_to_telegram(file_path: str, filename: str, caption: str = "") -> str:
    """Same shape as send_to_telegram, for an actual file attachment
    (2026-09-03) — used when a workflow's Output node rendered a real
    PDF (see dispatcher/agent_work.py's FILE_OUTPUT_PREFIX convention)
    that a following send_to_telegram step needs sent as a document,
    not text. TelegramAdapter.send_file already existed for this; it
    was just never reachable from a workflow step before."""
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        raise TelegramSendError("TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID not configured")

    try:
        with open(file_path, "rb") as f:
            file_bytes = f.read()
    except OSError as e:
        raise TelegramSendError(f"couldn't read {file_path}: {e}")

    try:
        TelegramAdapter(token).send_file(chat_id, file_bytes, filename, caption)
    except MessagingError as e:
        raise TelegramSendError(str(e))

    return "Sent file to Telegram."
