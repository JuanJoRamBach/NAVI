"""
server.py

The deployable entrypoint Render runs. A small stdlib-only web service
(no Flask/FastAPI — keeps the "pure Python, requests for HTTP, no heavy
SDKs" rule intact for the one piece of infra that has to be always-on)
that:

  1. Receives Telegram webhook POSTs at /webhook/telegram.
  2. Feeds the message text through dispatcher/parser.py.
  3. Plain chat -> answered directly by dispatcher_chat (Groq), no routing.
     Commands -> dispatcher/executor.py runs the chain, real model calls,
     real Filen saves.
     Near-miss typo -> asks for confirmation before doing either.
  4. Sends the reply back through the messaging adapter it came in on.

Discord has a /webhook/discord route wired up for symmetry, but
messaging/discord.py's parse_incoming() is a stub (outbound-only phase,
see that file's docstring) so it always 200s without acting on anything.

Processing happens on a background thread per request so the webhook
POST gets an immediate 200 — Telegram doesn't need the reply in the HTTP
response body (sendMessage is called directly), and this avoids Telegram
retrying/duplicating an update because our model calls took a while.
"""

import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from dispatcher.executor import format_summary, run_chain
from dispatcher.parser import ParseResult, parse_message
from messaging.base import IncomingMessage, MessagingAdapter, MessagingError
from messaging.discord import DiscordAdapter
from messaging.telegram import TelegramAdapter
from config.store import config
from providers.base import ChatMessage, ProviderError
from providers.registry import ProviderNotConfigured, get_dispatcher

PORT = int(os.environ.get("PORT", "10000"))

CHAT_SYSTEM_PROMPT = (
    "You are NAVI, a personal AI agent for JuanJo (a UX/game designer job-hunting "
    "in the EU/US). Reply directly and concisely — this is a live chat message, "
    "not a research task."
)

# In-memory only, deliberately not persisted: if a near-miss confirmation
# is still pending across a Render restart, the worst case is the user
# just gets treated as plain chat and can re-type the command. Not worth
# the complexity of persisting through Filen alongside the real config.
_pending_confirmations: dict[str, ParseResult] = {}
_pending_lock = threading.Lock()


def _telegram_adapter() -> TelegramAdapter | None:
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    return TelegramAdapter(token) if token else None


def _discord_adapter() -> DiscordAdapter | None:
    token = os.environ.get("DISCORD_BOT_TOKEN")
    return DiscordAdapter(token) if token else None


def _dispatcher_chat_reply(text: str) -> str:
    try:
        provider, model = get_dispatcher(context="chat")
    except ProviderNotConfigured as e:
        return f"⚠️ Can't reply right now — dispatcher_chat isn't configured: {e}"

    try:
        response = provider.chat(model=model, messages=[
            ChatMessage(role="system", content=CHAT_SYSTEM_PROMPT),
            ChatMessage(role="user", content=text),
        ])
    except ProviderError as e:
        return f"⚠️ dispatcher_chat failed and has no fallback by design: {e}"

    return response.text or "(empty reply)"


def _reconstruct_confirmed_text(pending: ParseResult) -> str:
    """Turns a confirmed near-miss back into a real command by swapping
    the mistyped word (slash or no slash) for the real one, so it parses
    as a command on the second pass."""
    word = pending.near_miss_word or ""
    return pending.raw_text.replace(word, f"/{pending.near_miss_suggestion}", 1)


def handle_message(adapter: MessagingAdapter, msg: IncomingMessage) -> None:
    with _pending_lock:
        pending = _pending_confirmations.pop(msg.chat_id, None)

    normalized = msg.text.strip().lower()

    if pending and normalized in ("yes", "y", "yeah", "confirm"):
        corrected_text = _reconstruct_confirmed_text(pending)
        result = parse_message(corrected_text)
    elif pending and normalized in ("no", "n", "cancel"):
        result = ParseResult(kind="plain_chat", raw_text=pending.raw_text)
    else:
        result = parse_message(msg.text)

    reply_text = _handle_parse_result(result, msg.chat_id)

    try:
        adapter.send_message(msg.chat_id, reply_text)
    except MessagingError:
        pass  # nothing more we can do if the reply itself fails to send


def _handle_parse_result(result: ParseResult, chat_id: str) -> str:
    if result.kind == "commands":
        results = run_chain(result.steps)
        return format_summary(results)

    if result.kind == "near_miss":
        with _pending_lock:
            _pending_confirmations[chat_id] = result
        return (
            f"Did you mean /{result.near_miss_suggestion}? "
            f"(you typed \"{result.near_miss_word}\") — reply yes/no."
        )

    return _dispatcher_chat_reply(result.raw_text)


class WebhookHandler(BaseHTTPRequestHandler):
    def _read_json(self) -> dict:
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length) if length else b"{}"
        try:
            return json.loads(raw or b"{}")
        except json.JSONDecodeError:
            return {}

    def _respond(self, status: int, body: str = "ok") -> None:
        self.send_response(status)
        self.send_header("Content-Type", "text/plain")
        self.end_headers()
        self.wfile.write(body.encode())

    def do_GET(self):
        if self.path == "/":
            self._respond(200, "NAVI is running")
        else:
            self._respond(404, "not found")

    def do_POST(self):
        if self.path == "/webhook/telegram":
            payload = self._read_json()
            self._respond(200)  # ack immediately, process in background
            adapter = _telegram_adapter()
            if not adapter:
                return
            msg = adapter.parse_incoming(payload)
            if msg:
                threading.Thread(target=handle_message, args=(adapter, msg), daemon=True).start()
            return

        if self.path == "/webhook/discord":
            self._read_json()
            self._respond(200)  # outbound-only phase — nothing to act on yet
            return

        self._respond(404, "not found")

    def log_message(self, fmt, *args):
        # Default BaseHTTPRequestHandler logging is noisy for a webhook
        # endpoint that gets hit constantly — keep it to stdout via print
        # so Render's log viewer still shows something, just quieter.
        print(f"{self.address_string()} - {fmt % args}")


def _seed_keys_from_env() -> None:
    """
    Config store is normally meant to be filled in via chat ("here's my
    Groq key") — see config/store.py's docstring — so it survives without
    a redeploy. That chat-side key-intake isn't built yet (out of scope
    for this pass), so for the FIRST boot on a fresh Render instance with
    nothing restored yet from Filen, fall back to env vars. Never
    overwrites a key that's already configured (e.g. restored from Filen,
    or set via chat once that lands), so env vars only matter until the
    real config takes over.
    """
    for provider_name, env_var in (("groq", "GROQ_API_KEY"), ("openrouter", "OPENROUTER_API_KEY")):
        if config.get_provider_key(provider_name):
            continue
        value = os.environ.get(env_var)
        if value:
            config.set_provider_key(provider_name, value)


def main() -> None:
    _seed_keys_from_env()
    server = ThreadingHTTPServer(("0.0.0.0", PORT), WebhookHandler)
    print(f"NAVI listening on 0.0.0.0:{PORT}")
    server.serve_forever()


if __name__ == "__main__":
    main()
