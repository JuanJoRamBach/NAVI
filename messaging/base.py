"""
messaging/base.py

The shared contract every messaging adapter (Telegram, Discord, whatever
gets added later) implements. Same transport-interface pattern as
providers/base.py: one shape in, one shape out, so server.py never has to
know which platform a message came from.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class IncomingMessage:
    chat_id: str  # platform-native id to reply to (Telegram chat id, Discord channel id)
    text: str
    sender_id: str = ""
    sender_name: str = ""
    raw: dict | None = None
    # Set when the incoming message carried an image (e.g. a Telegram
    # photo) — a data: URL, ready to drop straight into a vision-model
    # ChatMessage's content. None for a plain text message.
    image_data_url: str | None = None


class MessagingError(Exception):
    """Raised for any adapter failure sending or parsing a message."""


class MessagingAdapter(ABC):
    """Base class every concrete messaging transport implements."""

    name: str = "base"

    @abstractmethod
    def parse_incoming(self, payload: dict) -> IncomingMessage | None:
        """Turns a raw webhook payload into an IncomingMessage, or None if
        the payload isn't a message worth acting on (e.g. a non-text update)."""
        raise NotImplementedError

    @abstractmethod
    def send_message(self, chat_id: str, text: str) -> None:
        """Sends a plain text reply. Raises MessagingError on failure."""
        raise NotImplementedError

    def send_file(self, chat_id: str, file_bytes: bytes, filename: str, caption: str = "") -> None:
        """Sends a file attachment. Optional — not every adapter needs it
        wired up on day one, so the base class provides a clear failure
        instead of forcing every subclass to implement it immediately."""
        raise NotImplementedError(f"{self.name} adapter does not support send_file yet")
