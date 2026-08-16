"""
messaging/discord.py

Discord adapter — outbound-only for this phase (see decision in chat:
receiving needs either a persistent Gateway websocket, which fights
Render free tier's spin-down, or Interactions-webhook signature
verification, which needs a new dependency (PyNaCl). Both deferred.
Discord is priority #2 behind Telegram anyway.

What this DOES do: send plain messages and files to a channel via the bot
REST API, so the daily digest / opportunity scan jobs (or executor
results) can post to Discord even before receiving is wired up.

parse_incoming() is a stub that always returns None — server.py can still
register a Discord webhook route without crashing, it just won't act on
anything yet. Wiring real receiving is a follow-up (see README).
"""

import requests

from messaging.base import IncomingMessage, MessagingAdapter, MessagingError

API_ROOT = "https://discord.com/api/v10"


class DiscordAdapter(MessagingAdapter):
    name = "discord"

    def __init__(self, bot_token: str):
        self.bot_token = bot_token

    def _headers(self) -> dict:
        return {"Authorization": f"Bot {self.bot_token}"}

    def parse_incoming(self, payload: dict) -> IncomingMessage | None:
        # Not wired up yet — outbound-only phase. See module docstring.
        return None

    def send_message(self, chat_id: str, text: str) -> None:
        # Discord caps messages at 2000 chars — split rather than truncate.
        for chunk in _chunk_text(text, 2000):
            try:
                resp = requests.post(
                    f"{API_ROOT}/channels/{chat_id}/messages",
                    headers=self._headers(),
                    json={"content": chunk},
                    timeout=30,
                )
            except requests.RequestException as e:
                raise MessagingError(f"Discord send message failed: {e}")
            if resp.status_code >= 400:
                raise MessagingError(f"Discord send message error {resp.status_code}: {resp.text[:300]}")

    def send_file(self, chat_id: str, file_bytes: bytes, filename: str, caption: str = "") -> None:
        try:
            resp = requests.post(
                f"{API_ROOT}/channels/{chat_id}/messages",
                headers=self._headers(),
                data={"content": caption},
                files={"file": (filename, file_bytes)},
                timeout=60,
            )
        except requests.RequestException as e:
            raise MessagingError(f"Discord send file failed: {e}")
        if resp.status_code >= 400:
            raise MessagingError(f"Discord send file error {resp.status_code}: {resp.text[:300]}")


def _chunk_text(text: str, size: int) -> list[str]:
    if len(text) <= size:
        return [text]
    return [text[i:i + size] for i in range(0, len(text), size)]
