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

from dispatcher.chat import run_mode_chat
from dispatcher.executor import format_summary, run_chain
from dispatcher.parser import COMMANDS, ParseResult, parse_message
from messaging.base import IncomingMessage, MessagingAdapter, MessagingError
from messaging.discord import DiscordAdapter
from messaging.telegram import TelegramAdapter
from config.store import config
from push.sender import PushError, add_subscription, send_push, subscription_count

PORT = int(os.environ.get("PORT", "10000"))

# The PWA (navi-ui, on GitHub Pages) calls /push/* from a different
# origin than this server — browsers block that without an explicit
# CORS allow. Scoped to the one real frontend origin rather than "*",
# since this endpoint accepts push subscription data.
PWA_ORIGIN = "https://juanjorambach.github.io"

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

    reply_text, attachments = _handle_parse_result(result, msg.chat_id)

    try:
        adapter.send_message(msg.chat_id, reply_text)
        for image_bytes, filename, caption in attachments:
            adapter.send_file(msg.chat_id, image_bytes, filename, caption=caption)
    except MessagingError:
        pass  # nothing more we can do if the reply itself fails to send


def _handle_parse_result(
    result: ParseResult, chat_id: str, mode: str = "normal",
) -> tuple[str, list[tuple[bytes, str, str]]]:
    if result.kind == "commands":
        results = run_chain(result.steps)
        attachments = [
            (r.image_bytes, r.image_filename or "chart.png", r.text)
            for r in results if r.image_bytes
        ]
        return format_summary(results), attachments

    if result.kind == "near_miss":
        with _pending_lock:
            _pending_confirmations[chat_id] = result
        return (
            f"Did you mean /{result.near_miss_suggestion}? "
            f"(you typed \"{result.near_miss_word}\") — reply yes/no.",
            [],
        )

    # Plain chat — not a typed /command. `mode` selects which brief in
    # dispatcher/modes/ governs the system prompt and which tools (if
    # any) the model can reach for. Telegram/Discord have no mode concept
    # and always pass the default ("normal"); the PWA sends whichever
    # mode the user has selected.
    return run_mode_chat(mode, result.raw_text), []


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

    def _respond_json(self, status: int, payload: dict) -> None:
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", PWA_ORIGIN)
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/":
            self._respond(200, "NAVI is running")
        elif self.path == "/config/routing":
            # Real provider/model roster for the PWA's "Today's models" and
            # "Routing & fallbacks" panels — no API keys in here, just
            # provider + model names, so it's safe to expose publicly.
            self._respond_json(200, {
                "roles": {
                    "dispatcher_chat": config.get_role("dispatcher_chat"),
                    "dispatcher_autonomous": config.get_role("dispatcher_autonomous"),
                },
                "task_routing": {cmd: config.get_task_routing(cmd) for cmd in COMMANDS},
                "enabled_providers": config.enabled_providers(),
            })
        else:
            self._respond(404, "not found")

    def do_OPTIONS(self):
        # CORS preflight — the browser sends this before the actual POST
        # to /push/* because it's a cross-origin request with a JSON
        # content type. Only the push routes need it; nothing else on
        # this server is called cross-origin.
        if self.path in ("/push/subscribe", "/push/test", "/chat/send"):
            self.send_response(204)
            self.send_header("Access-Control-Allow-Origin", PWA_ORIGIN)
            self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            self.end_headers()
            return
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

        if self.path == "/chat/send":
            # The PWA's own chat surface — synchronous unlike the Telegram/
            # Discord webhooks above (no adapter to hand off to; the
            # frontend just awaits this response directly).
            payload = self._read_json()
            text = (payload.get("text") or "").strip()
            mode = payload.get("mode") or "normal"
            if not text:
                self._respond_json(400, {"error": "missing 'text'"})
                return
            result = parse_message(text)
            # Attachments (e.g. a /graph-data chart image) aren't supported
            # over this channel yet — the PWA has no attachment UI built,
            # so they're dropped rather than silently mismatched.
            reply_text, _attachments = _handle_parse_result(result, "pwa", mode)
            self._respond_json(200, {"reply": reply_text})
            return

        if self.path == "/push/subscribe":
            payload = self._read_json()
            if not payload.get("endpoint"):
                self._respond_json(400, {"error": "missing 'endpoint'"})
                return
            add_subscription(payload)
            self._respond_json(200, {"ok": True, "subscriptions": subscription_count()})
            return

        if self.path == "/push/test":
            try:
                errors = send_push(
                    "NAVI",
                    "Test notification — if you see this, push is wired up correctly.",
                )
            except PushError as e:
                self._respond_json(400, {"ok": False, "error": str(e)})
                return
            self._respond_json(200, {"ok": True, "errors": errors})
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
    for provider_name, env_var in (
        ("groq", "GROQ_API_KEY"),
        ("openrouter", "OPENROUTER_API_KEY"),
        ("ollama_cloud", "OLLAMA_API_KEY"),
        ("cloudflare", "CLOUDFLARE_API_KEY"),
        ("llm7", "LLM7_API_KEY"),
    ):
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
