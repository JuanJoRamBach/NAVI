"""
server.py

The deployable entrypoint Render runs. A small stdlib-only web service
(no Flask/FastAPI — keeps the "pure Python, requests for HTTP, no heavy
SDKs" rule intact for the one piece of infra that has to be always-on)
that:

  1. Receives Telegram webhook POSTs at /webhook/telegram.
  2. Feeds the message text through dispatcher/parser.py.
  3. Plain chat -> answered directly by normal_chat (Groq), no routing.
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
import mimetypes
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import quote, unquote, urlparse, parse_qs

from dispatcher.chat import run_mode_chat
from dispatcher.executor import format_summary, run_chain
from dispatcher.parser import COMMANDS, ParseResult, parse_message
from dispatcher.reminders import due_reminders, mark_delivered
from dispatcher.research_status import get_status, set_status
from tools.telegram_send import TelegramSendError, send_to_telegram
from messaging.base import IncomingMessage, MessagingAdapter, MessagingError
from messaging.discord import DiscordAdapter
from messaging.telegram import TelegramAdapter
from config.store import config
from push.sender import PushError, add_subscription, send_push, subscription_count
from storage.filen import StorageError, download_for_reply

PORT = int(os.environ.get("PORT", "10000"))
NAVI_BASE_URL = "https://api.getnavi.online"

# The PWA (navi-ui, on GitHub Pages, custom domain getnavi.online) calls
# /push/* from a different origin than this server — browsers block that
# without an explicit CORS allow. Scoped to the one real frontend origin
# rather than "*", since this endpoint accepts push subscription data.
PWA_ORIGIN = "https://getnavi.online"

# Gates GET /files/<path> — unlike Telegram (which gets real file
# attachments via sendDocument) the PWA has no attachment channel of its
# own, so a saved artifact reaches it as a plain download URL embedded
# in the reply text. That endpoint serves real document content (research,
# recaps, tailored CVs), not just reminder text like the other
# unauthenticated routes — worth a real credential, not just an
# unguessable path. Fails closed: if this isn't set, every request 403s
# rather than silently serving without a check.
NAVI_FILES_TOKEN = os.environ.get("NAVI_FILES_TOKEN")

# Conservative — Web Push payloads are capped around 4KB total by the
# push service itself (title + body + JSON overhead + encryption), not
# something we control. A long /research report gets split across
# several pushes rather than silently failing to deliver — same idea as
# TelegramAdapter's own 4096-char chunking, just a smaller ceiling.
PUSH_CHUNK_SIZE = 3000

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


def _file_download_url(saved_path: str | None, render: bool = False) -> str | None:
    """saved_path is a full 'filen:...' path as returned by save_result/
    save_bytes. Returns None (rather than a link that would just 403)
    if NAVI_FILES_TOKEN isn't configured or the path is missing.

    render=True asks /files/ to serve the content inline (renders as a
    real page in a browser tab) instead of forcing a download — only
    meaningful for /code's HTML output (see CodeFile.viewable in
    dispatcher/executor.py); every other saved artifact just wants the
    plain download behavior."""
    if not NAVI_FILES_TOKEN or not saved_path or not saved_path.startswith("filen:"):
        return None
    relative = saved_path[len("filen:"):]
    url = f"{NAVI_BASE_URL}/files/{quote(relative)}?token={NAVI_FILES_TOKEN}"
    return f"{url}&render=1" if render else url


def _pwa_download_links(results: list) -> str:
    """The PWA has no file-attachment channel (unlike Telegram's real
    sendDocument) — a saved artifact reaches it as plain URLs appended
    to the reply text instead, which the frontend detects and renders
    as clickable chips. Skips image results (graph-data/create-image)
    since those aren't meant to be re-downloaded as a separate file —
    they're the image. A viewable /code result (bundled HTML) gets
    BOTH a download line and a separate view line, same file, two
    different Content-Disposition modes."""
    lines = []
    for r in results:
        if r.code_saved:
            # /code — one chip per saved file (bundled HTML is still just
            # one entry here; separate-files mode is several).
            viewable_by_name = {cf.filename: cf.viewable for cf in r.code_files}
            for filename, saved_path in r.code_saved:
                download_url = _file_download_url(saved_path)
                if download_url:
                    lines.append(f"📎 {filename}: {download_url}")
                if viewable_by_name.get(filename):
                    view_url = _file_download_url(saved_path, render=True)
                    if view_url:
                        lines.append(f"🌐 {filename}: {view_url}")
        elif r.rendered_file_saved_path and r.rendered_file_name:
            url = _file_download_url(r.rendered_file_saved_path)
            if url:
                lines.append(f"📎 {r.rendered_file_name}: {url}")
        elif r.saved_path and not r.image_bytes:
            filename = r.saved_path.rsplit("/", 1)[-1]
            download_url = _file_download_url(r.saved_path)
            if download_url:
                lines.append(f"📎 {filename}: {download_url}")
    return ("\n\n" + "\n".join(lines)) if lines else ""


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

    if msg.image_data_url and result.kind == "commands":
        for s in result.steps:
            if s.command == "design-read":
                s.image_data_url = msg.image_data_url

    reply_text, attachments = _handle_parse_result(result, msg.chat_id)

    try:
        adapter.send_message(msg.chat_id, reply_text)
        for image_bytes, filename, caption in attachments:
            adapter.send_file(msg.chat_id, image_bytes, filename, caption=caption)
    except MessagingError:
        pass  # nothing more we can do if the reply itself fails to send


def _handle_parse_result(
    result: ParseResult, chat_id: str, mode: str = "normal", channel: str = "generic",
) -> tuple[str, list[tuple[bytes, str, str]]]:
    if result.kind == "commands":
        results = run_chain(result.steps)
        attachments = [
            (r.image_bytes, r.image_filename or "chart.png", r.text)
            for r in results if r.image_bytes
        ]
        # A step with a snippet (currently only /research) is long-form —
        # the full report goes as an attached document, not chunked
        # inline text; format_summary() already uses the snippet for the
        # message body itself.
        attachments += [
            (r.text.encode("utf-8"), f"{r.step.command}.md", "")
            for r in results if r.snippet and r.text
        ]
        # An opt-in file request ("--file pdf" on /summarize, /recap,
        # /note) — additive alongside the plain-text reply, same as
        # /research's attachment above.
        attachments += [
            (r.rendered_file_bytes, r.rendered_file_name, "")
            for r in results if r.rendered_file_bytes and r.rendered_file_name
        ]
        reply_text = format_summary(results)
        if channel == "pwa":
            # The PWA never receives the `attachments` list above (see
            # /chat/send) — a download link embedded in the text is the
            # only way a saved artifact reaches it at all.
            reply_text += _pwa_download_links(results)
        return reply_text, attachments

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


def _chunk_text_at_boundary(text: str, size: int) -> list[str]:
    """Like a fixed-size split, but tries to break at a paragraph, then
    a sentence, then a word boundary before falling back to a hard cut —
    a blind text[i:i+size] slice reads badly when it lands mid-word."""
    chunks = []
    remaining = text
    while len(remaining) > size:
        window = remaining[:size]
        cut = window.rfind("\n\n")
        if cut < size // 2:
            sentence_cut = window.rfind(". ")
            cut = sentence_cut + 1 if sentence_cut >= size // 2 else cut
        if cut < size // 2:
            word_cut = window.rfind(" ")
            cut = word_cut if word_cut >= size // 2 else size
        chunks.append(remaining[:cut].rstrip())
        remaining = remaining[cut:].lstrip()
    if remaining:
        chunks.append(remaining)
    return chunks or [""]


def _deliver_via_push(title: str, text: str) -> None:
    """Splits `text` across multiple pushes if it's too big for one
    payload — each chunk becomes its own message bubble in the PWA via
    the existing service-worker push handler, no frontend changes
    needed. Best-effort: a failed push here has no user-facing fallback,
    since the HTTP request that triggered the work is long gone."""
    chunks = _chunk_text_at_boundary(text, PUSH_CHUNK_SIZE)
    for i, chunk in enumerate(chunks):
        chunk_title = title if len(chunks) == 1 else f"{title} ({i + 1}/{len(chunks)})"
        try:
            send_push(chunk_title, chunk)
        except PushError:
            pass


def _run_research_async(result: ParseResult) -> None:
    """Runs a /research command chain in the background — see /chat/send,
    which acks immediately rather than blocking on this (gathering plus
    up to 3 minutes of synthesis retries is too long to hold an HTTP
    request open). Delivers the finished result via push; dispatcher/
    research_status.py carries live progress in the meantime."""
    try:
        results = run_chain(result.steps)
        # Only reachable from /chat/send (see the call site below) — a
        # PWA-bound delivery, so the download-link treatment always
        # applies here, unlike _handle_parse_result which needs the
        # explicit channel="pwa" flag to tell it apart from Telegram.
        text = format_summary(results) + _pwa_download_links(results)
    except Exception as e:
        set_status(None)
        _deliver_via_push("NAVI — research failed", f"Something went wrong: {e}")
        return
    set_status(None)
    _deliver_via_push("NAVI — research ready", text)


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
        # Needed since the PWA's server-status indicator (see App.tsx)
        # does a plain cross-origin GET to "/" to check if NAVI is awake —
        # without this, the browser blocks the response as a CORS
        # violation before the frontend ever sees it, same reasoning as
        # _respond_json already applies to every JSON route.
        self.send_header("Access-Control-Allow-Origin", PWA_ORIGIN)
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
                    "normal_chat": config.get_role("normal_chat"),
                    "dispatcher_autonomous": config.get_role("dispatcher_autonomous"),
                },
                "task_routing": {cmd: config.get_task_routing(cmd) for cmd in COMMANDS},
                "enabled_providers": config.enabled_providers(),
            })
        elif self.path == "/research/status":
            # Polled by the PWA while an async /research job runs in the
            # background — see dispatcher/research_status.py. `status` is
            # null when nothing's in flight.
            self._respond_json(200, {"status": get_status()})
        elif self.path == "/reminders/check":
            # Hit periodically by a GitHub Actions cron (see
            # .github/workflows/check_reminders.yml) rather than run as a
            # standalone job — this way delivery reads the config
            # singleton this live process already has loaded, no separate
            # Filen round-trip needed just to check reminders. Delivers
            # via both push (PWA bubble) and Telegram (instant, no PWA
            # dependency) — best-effort each, one channel failing doesn't
            # block the other or leave the reminder stuck.
            delivered = 0
            for r in due_reminders():
                text = f"⏰ Reminder: {r['message']}"
                try:
                    send_push("NAVI", text)
                except PushError:
                    pass
                try:
                    send_to_telegram(text)
                except TelegramSendError:
                    pass
                mark_delivered(r["id"])
                delivered += 1
            self._respond_json(200, {"delivered": delivered})
        elif self.path.startswith("/files/"):
            self._handle_file_download()
        else:
            self._respond(404, "not found")

    def _handle_file_download(self) -> None:
        """Serves a saved Filen artifact back down — the only way a
        rendered document or /research report reaches the PWA, which has
        no real attachment channel the way Telegram's sendDocument does.
        Fails closed on a missing/wrong token rather than falling back to
        open access, since this serves real document content.

        ?render=1 serves the content inline (renders as a real page in a
        browser tab) instead of forcing a download — restricted to
        .html specifically, not any file type, so this stays predictable:
        it's for /code's bundled HTML preview, not a general "display
        anything inline" switch. Rendering a static HTML document this
        way is inherently as safe as visiting any website — nothing
        executes server-side, the browser's normal page sandbox applies
        — consistent with the original design rule that this was never
        meant to run arbitrary code, only render browser-safe output."""
        parsed = urlparse(self.path)
        query = parse_qs(parsed.query)
        token = (query.get("token") or [None])[0]
        if not NAVI_FILES_TOKEN or token != NAVI_FILES_TOKEN:
            self._respond(403, "forbidden")
            return

        relative_path = unquote(parsed.path[len("/files/"):])
        if not relative_path or ".." in relative_path:
            self._respond(400, "bad path")
            return

        try:
            content = download_for_reply(f"filen:{relative_path}")
        except StorageError as e:
            self._respond(404, f"not found: {e}")
            return

        filename = relative_path.rsplit("/", 1)[-1]
        content_type, _ = mimetypes.guess_type(filename)
        wants_render = (query.get("render") or [None])[0] == "1"
        inline = wants_render and filename.lower().endswith(".html")

        self.send_response(200)
        self.send_header("Content-Type", content_type or "application/octet-stream")
        disposition = "inline" if inline else "attachment"
        self.send_header("Content-Disposition", f'{disposition}; filename="{filename}"')
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Access-Control-Allow-Origin", PWA_ORIGIN)
        self.end_headers()
        self.wfile.write(content)

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

            if result.kind == "commands" and any(s.command == "research" for s in result.steps):
                # /research can take minutes (gathering, plus up to 3min
                # of synthesis retries) — too long to hold this request
                # open. Ack immediately, run in the background, deliver
                # the finished report via push (see _run_research_async).
                threading.Thread(target=_run_research_async, args=(result,), daemon=True).start()
                self._respond_json(200, {
                    "reply": "Researching — I'll ping you when it's ready. Feel free to keep chatting.",
                    "async": True,
                })
                return

            # Image attachments (e.g. a /graph-data chart) still aren't
            # supported over this channel — the PWA has no image-render
            # UI built, so those are dropped. Saved-file attachments
            # (rendered documents, /research's report, etc.) reach the
            # PWA as a download link embedded in reply_text instead —
            # see _pwa_download_links via channel="pwa" below.
            reply_text, _attachments = _handle_parse_result(result, "pwa", mode, channel="pwa")
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
        ("mistral", "MISTRAL_API_KEY"),
        ("gmi", "GMI_API_KEY"),
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
