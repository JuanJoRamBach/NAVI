"""
messaging/telegram.py

Telegram Bot API adapter. Priority #1 — must work end to end.

Uses webhooks (not long-polling): Telegram POSTs each update straight to
server.py, which is the right fit for Render since there's no always-on
process to run a polling loop against anyway (free tier spins down on
idle). set_webhook() below is a one-time setup call, not something that
runs on every message.
"""

import base64
import html
import re

import requests

from messaging.base import IncomingMessage, MessagingAdapter, MessagingError

API_ROOT = "https://api.telegram.org/bot{token}/{method}"
FILE_ROOT = "https://api.telegram.org/file/bot{token}/{file_path}"

_MARKDOWN_LINK_RE = re.compile(r"\[([^\]]+)\]\((https?://[^\s)]+)\)")


class TelegramAdapter(MessagingAdapter):
    name = "telegram"

    def __init__(self, bot_token: str):
        self.bot_token = bot_token

    def _url(self, method: str) -> str:
        return API_ROOT.format(token=self.bot_token, method=method)

    def parse_incoming(self, payload: dict) -> IncomingMessage | None:
        message = payload.get("message") or payload.get("edited_message")
        if not message:
            return None  # e.g. a channel_post, callback_query, etc. — not handled yet

        photos = message.get("photo")
        text = message.get("text")
        image_data_url = None

        if photos:
            # A photo message routes straight to /design-read regardless
            # of caption wording — sending a screenshot to the bot has no
            # other purpose in this app, so requiring the user to also
            # type the command every time would just be friction. The
            # caption (if any) becomes extra instruction text for the step.
            caption = (message.get("caption") or "").strip()
            text = f"/design-read {caption}".strip()
            try:
                image_data_url = self._download_photo(photos[-1]["file_id"])
            except MessagingError:
                image_data_url = None  # step will surface "no image attached" itself
        elif not text:
            return None  # sticker/etc. with no text — nothing for the parser to act on

        chat = message.get("chat", {})
        sender = message.get("from", {})
        return IncomingMessage(
            chat_id=str(chat.get("id")),
            text=text,
            sender_id=str(sender.get("id", "")),
            sender_name=sender.get("username") or sender.get("first_name", ""),
            raw=payload,
            image_data_url=image_data_url,
        )

    def _download_photo(self, file_id: str) -> str:
        """Resolves a Telegram file_id to bytes via getFile, then encodes
        as a data: URL ready to drop into a vision-model ChatMessage."""
        try:
            resp = requests.get(self._url("getFile"), params={"file_id": file_id}, timeout=30)
        except requests.RequestException as e:
            raise MessagingError(f"Telegram getFile failed: {e}")
        if resp.status_code >= 400:
            raise MessagingError(f"Telegram getFile error {resp.status_code}: {resp.text[:300]}")

        file_path = resp.json().get("result", {}).get("file_path", "")
        try:
            file_resp = requests.get(
                FILE_ROOT.format(token=self.bot_token, file_path=file_path), timeout=30,
            )
        except requests.RequestException as e:
            raise MessagingError(f"Telegram file download failed: {e}")
        if file_resp.status_code >= 400:
            raise MessagingError(f"Telegram file download error {file_resp.status_code}")

        mime = "image/jpeg" if file_path.lower().endswith((".jpg", ".jpeg")) else "image/png"
        encoded = base64.b64encode(file_resp.content).decode("ascii")
        return f"data:{mime};base64,{encoded}"

    def send_message(self, chat_id: str, text: str) -> None:
        # Telegram caps messages at 4096 chars — split rather than truncate,
        # since a truncated research result silently loses the end of it.
        # Chunk the raw Markdown-ish text first, then convert each chunk to
        # HTML — safer than converting once and risking a split landing
        # inside an <a> tag.
        for chunk in _chunk_text(text, 4096):
            try:
                resp = requests.post(
                    self._url("sendMessage"),
                    json={
                        "chat_id": chat_id,
                        "text": _markdown_links_to_html(chunk),
                        "parse_mode": "HTML",
                    },
                    timeout=30,
                )
            except requests.RequestException as e:
                raise MessagingError(f"Telegram sendMessage failed: {e}")
            if resp.status_code >= 400:
                raise MessagingError(f"Telegram sendMessage error {resp.status_code}: {resp.text[:300]}")

    def send_file(self, chat_id: str, file_bytes: bytes, filename: str, caption: str = "") -> None:
        try:
            resp = requests.post(
                self._url("sendDocument"),
                data={"chat_id": chat_id, "caption": caption},
                files={"document": (filename, file_bytes)},
                timeout=60,
            )
        except requests.RequestException as e:
            raise MessagingError(f"Telegram sendDocument failed: {e}")
        if resp.status_code >= 400:
            raise MessagingError(f"Telegram sendDocument error {resp.status_code}: {resp.text[:300]}")

    def set_webhook(self, webhook_url: str) -> None:
        """One-time setup: tells Telegram where to POST updates. Call this
        once after deploying (or whenever the Render URL changes)."""
        try:
            resp = requests.post(self._url("setWebhook"), json={"url": webhook_url}, timeout=30)
        except requests.RequestException as e:
            raise MessagingError(f"Telegram setWebhook failed: {e}")
        if resp.status_code >= 400:
            raise MessagingError(f"Telegram setWebhook error {resp.status_code}: {resp.text[:300]}")


def _chunk_text(text: str, size: int) -> list[str]:
    if len(text) <= size:
        return [text]
    return [text[i:i + size] for i in range(0, len(text), size)]


def _markdown_links_to_html(text: str) -> str:
    """
    Converts [title](url) links to Telegram-HTML <a> tags and escapes
    everything else, so parse_mode="HTML" can render clickable citations
    without choking on ordinary prose containing _, *, or ` — the strict
    Markdown/MarkdownV2 modes require escaping those everywhere, which is
    fragile for free-form AI output; HTML only needs &, <, > escaped.
    """
    escaped = html.escape(text, quote=False)

    def repl(match: re.Match) -> str:
        link_text, url = match.group(1), match.group(2)
        return f'<a href="{url.replace(chr(34), "&quot;")}">{link_text}</a>'

    return _MARKDOWN_LINK_RE.sub(repl, escaped)
