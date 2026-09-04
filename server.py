"""
server.py

The deployable entrypoint (now via uvicorn, see the bottom of this file) —
FastAPI, migrated 2026-09-01 from the original stdlib-only BaseHTTPRequestHandler
(see this file's git history for that version). The "pure Python, no heavy
SDKs" rule that motivated stdlib-only was explicitly lifted once NAVI moved
onto Lightsail with real headroom — a proper ASGI stack gives native
WebSocket support (needed for Dev Slate's live chat + relayed file tools)
instead of hand-rolling a second server on a second port, and is more
"professional and reliable" per JuanJo's own framing for this migration.

Every existing route's BEHAVIOR is preserved as-is — this is a transport-
layer migration, not a redesign of what any endpoint does. New in this
pass: the Dev Slate pieces (GET /config/models, POST /config/role,
/devslate/conversations, WS /ws/devslate/{id}) — see dispatcher/devslate_chat.py
and storage/conversations.py for what actually backs those.

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

Telegram/Discord webhook processing happens on a background thread per
request so the webhook POST gets an immediate 200 — Telegram doesn't need
the reply in the HTTP response body (sendMessage is called directly), and
this avoids Telegram retrying/duplicating an update because our model
calls took a while. Plain `threading.Thread`, not `asyncio.create_task` —
these call into synchronous code (requests-based provider transports,
rclone subprocess calls) that would block the event loop exactly the same
way dispatcher/devslate_chat.py's asyncio.to_thread() call exists to avoid;
a real background OS thread sidesteps that without needing every synchronous
call site rewritten as async.
"""

import asyncio
import mimetypes
import os
import threading
import time
from urllib.parse import quote

from fastapi import FastAPI, HTTPException, Query, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse, RedirectResponse, Response

from dispatcher.agent_work import WorkflowError, check_due_workflows, start_workflow_run
from dispatcher.mcp_client import MCPError, approve_tools, discover_tools
from dispatcher.mcp_oauth import MCPOAuthError, exchange_code_for_token, start_authorization
from tools.mcp_marketplace import MCPMarketplaceError, search as search_mcp_marketplace
from dispatcher.scheduler import register_job, start_scheduler
from dispatcher.chat import run_mode_chat, run_stored_mode_chat
from dispatcher.devslate_chat import run_devslate_turn
from dispatcher.executor import format_summary, run_chain
from dispatcher.parser import COMMANDS, ParseResult, parse_message
from dispatcher.reminders import due_reminders, mark_delivered
from dispatcher.research_status import get_status, set_status
from tools.telegram_send import TelegramSendError, send_to_telegram
from messaging.base import IncomingMessage, MessagingAdapter, MessagingError
from messaging.discord import DiscordAdapter
from messaging.telegram import TelegramAdapter
from config.store import config
from jobs.model_ranking import fetch_aa_benchmarks, list_candidates, load_snapshot, refresh_snapshot
from push.sender import PushError, add_subscription, send_push, subscription_count
from storage.filen import StorageError, download_for_reply
from storage.conversations import (
    create_conversation, get_conversation, get_messages, get_task_state,
)
from storage.agent_work import (
    create_workflow as create_workflow_definition,
    delete_all_runs, delete_run,
    delete_workflow as delete_workflow_definition,
    get_run, get_run_steps, get_workflow, list_runs, list_workflows,
)
from storage.agents import create_agent, delete_agent, get_agent, get_agent_by_workflow_id, list_agents, update_agent
from tools.devslate_tools import new_tool_call_id

PORT = int(os.environ.get("PORT", "10000"))
NAVI_BASE_URL = "https://api.getnavi.online"

# The PWA (navi-ui, on GitHub Pages, custom domain getnavi.online) calls
# this server from a different origin — browsers block that without an
# explicit CORS allow. Scoped to the one real frontend origin rather than
# "*", since several of these endpoints accept real data (push
# subscriptions, chat text, file writes relayed from Dev Slate).
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
# is still pending across a restart, the worst case is the user just gets
# treated as plain chat and can re-type the command. Not worth the
# complexity of persisting through Filen alongside the real config.
_pending_confirmations: dict[str, ParseResult] = {}
_pending_lock = threading.Lock()

# In-memory only, same reasoning as _pending_confirmations above — an
# OAuth flow abandoned mid-way (browser closed before the redirect back)
# just leaves a small unused entry until the process restarts; nothing
# worth persisting through Filen for. Keyed by `state`, the standard
# OAuth anti-CSRF token — see /mcp/oauth/callback's own comment.
_pending_oauth: dict[str, dict] = {}
_pending_oauth_lock = threading.Lock()

app = FastAPI(title="NAVI")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[PWA_ORIGIN],
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["Content-Type", "X-Navi-Api-Key"],
)

# Shared-secret gate for the whole API (2026-09-04) — real gap found and
# confirmed by hand: every route here had zero authentication. A bare
# `curl https://api.getnavi.online/mcp/connections` returned real data
# with no credentials at all; anyone who found the URL could also POST a
# new MCP connection, repoint /config/role to a different provider, or
# spend NAVI's own LLM provider quota for free via /chat/send. CORS
# (above) only restricts which *browser origins* can call this — it does
# nothing against a direct request, which is exactly how the check above
# was made.
#
# This is a stopgap for the testing phase, not real per-user auth — there
# is no user-account system anywhere in NAVI yet (see IDEAS.md's "Project
# as the real top-level container" gap). A single shared key raises the
# bar from "anyone who finds the URL" to "someone who was actually given
# the key"; it does not protect against a determined holder of that key
# misusing it. Real per-user auth is the eventual proper fix.
#
# Exempt paths, deliberately narrow:
#   - "/"                    health check, no sensitive data
#   - "/files/..."           already gated by its own NAVI_FILES_TOKEN
#                             query-param check (see files_get below) —
#                             reached via a plain link, can't carry a
#                             custom header
#   - "/webhook/telegram"    Telegram calls this directly, can't send our
#                            header — see TELEGRAM_WEBHOOK_SECRET below
#                            for its own real check instead
#   - "/webhook/discord"     currently a pure no-op (outbound-only phase,
#                            nothing to act on), so left alone rather than
#                            building real signature verification for a
#                            handler that does nothing yet
#   - "/mcp/oauth/callback"  the OAuth provider's own redirect (e.g.
#                            GitHub) lands here directly — it can't carry
#                            our custom header either. Real security here
#                            is the `state` param, checked against the
#                            pending flow stashed by /oauth/start below —
#                            that's the standard OAuth anti-CSRF
#                            mechanism, not something this gate adds.
NAVI_API_KEY = os.environ.get("NAVI_API_KEY")
_PUBLIC_PATHS = {"/", "/webhook/telegram", "/webhook/discord", "/mcp/oauth/callback"}


@app.middleware("http")
async def _require_api_key(request: Request, call_next):
    if (
        request.method == "OPTIONS"  # let CORSMiddleware answer preflights
        or request.url.path in _PUBLIC_PATHS
        or request.url.path.startswith("/files/")
    ):
        return await call_next(request)
    if not NAVI_API_KEY:
        # Fail closed: an unset key locks the whole API down rather than
        # silently running open, same principle as NAVI_FILES_TOKEN below.
        return JSONResponse({"error": "NAVI_API_KEY is not configured on the server"}, status_code=503)
    if request.headers.get("X-Navi-Api-Key") != NAVI_API_KEY:
        return JSONResponse({"error": "missing or invalid API key"}, status_code=401)
    return await call_next(request)


async def _refresh_model_ranking_snapshot() -> None:
    """Regenerates model_ranking_snapshot.json in-process, on the live
    server itself — the only place load_snapshot() (GET /config/models,
    the PWA's "Today's models" picker) actually reads it from. Real gap
    found 2026-09-04 (JuanJo: a model confirmed live on LLM7 wasn't
    selectable — the snapshot on disk was 3 days stale): nothing ever
    ran jobs/model_ranking.py on a schedule anywhere. A GitHub Actions
    cron would write to that runner's own ephemeral disk, never reaching
    this process, so this has to be an in-process job, not an external
    ping. build_ranking_snapshot() does real blocking HTTP calls to every
    provider — run off the event loop so it can't stall other requests."""
    await asyncio.to_thread(refresh_snapshot)


@app.on_event("startup")
async def _start_background_scheduler() -> None:
    """In-process cron (dispatcher/scheduler.py) — registered here so it
    starts exactly once, when uvicorn actually boots the app, not at
    import time (matters for tests/tooling that import server.py without
    running it, e.g. this file's own TestClient-based checks)."""
    register_job("check_due_agent_workflows", config.get("agent_work_due_check_cron", "*/5 * * * *"), check_due_workflows)
    register_job("refresh_model_ranking_snapshot", "0 6 * * *", _refresh_model_ranking_snapshot)
    start_scheduler()


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
    meaningful for a saved .html artifact; every other saved artifact
    just wants the plain download behavior. No current caller passes
    render=True (its one use, /code's bundled HTML output, was retired
    2026-09-04) — kept as generic /files/ capability rather than ripped
    out, since it isn't /code-specific machinery itself."""
    if not NAVI_FILES_TOKEN or not saved_path or not saved_path.startswith("filen:"):
        return None
    relative = saved_path[len("filen:"):]
    url = f"{NAVI_BASE_URL}/files/{quote(relative)}?token={NAVI_FILES_TOKEN}"
    return f"{url}&render=1" if render else url


def _pwa_download_links(results: list) -> str:
    """The PWA has no file-attachment channel (unlike Telegram's real
    sendDocument) — a saved artifact reaches it as plain URLs appended
    to the reply text instead, which the frontend detects and renders
    as clickable chips. Skips image results (graph-data)
    since those aren't meant to be re-downloaded as a separate file —
    they're the image."""
    lines = []
    for r in results:
        if r.rendered_file_saved_path and r.rendered_file_name:
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
    """Real gap found and fixed 2026-09-04: nothing anywhere in this
    inbound path checked WHO was messaging the bot — chat_id/sender_id
    were only ever used to know where to send the reply, never compared
    against JuanJo's own TELEGRAM_CHAT_ID (already set, already used
    elsewhere to know where to SEND the daily digest — same identifier,
    now also used to gate who's allowed to send TO the bot). Any Telegram
    user who found the bot's @username could already run commands and
    spend NAVI's own LLM quota, same shape as the API-auth gap fixed
    earlier this session, just a different front door. Fails closed like
    everything else touched this session (NAVI_API_KEY, NAVI_FILES_TOKEN):
    an unset TELEGRAM_CHAT_ID means nobody is allowed through, not
    "everybody is.\""""
    owner_chat_id = os.environ.get("TELEGRAM_CHAT_ID")
    if not owner_chat_id or msg.chat_id != owner_chat_id:
        return
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
    result: ParseResult, chat_id: str, mode: str = "normal", channel: str = "generic",
) -> tuple[str, list[tuple[bytes, str, str]]]:
    if result.kind == "commands":
        results = run_chain(result.steps)
        attachments = [
            (r.image_bytes, r.image_filename or "chart.png", r.text)
            for r in results if r.image_bytes
        ]
        attachments += [
            (r.text.encode("utf-8"), f"{r.step.command}.md", "")
            for r in results if r.snippet and r.text
        ]
        attachments += [
            (r.rendered_file_bytes, r.rendered_file_name, "")
            for r in results if r.rendered_file_bytes and r.rendered_file_name
        ]
        reply_text = format_summary(results)
        if channel == "pwa":
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
        text = format_summary(results) + _pwa_download_links(results)
    except Exception as e:
        set_status(None)
        _deliver_via_push("NAVI — research failed", f"Something went wrong: {e}")
        return
    set_status(None)
    _deliver_via_push("NAVI — research ready", text)


# ---- Plain routes (behavior identical to the pre-migration stdlib version) ----

@app.get("/")
def health() -> PlainTextResponse:
    return PlainTextResponse("NAVI is running")


@app.get("/config/routing")
def config_routing() -> dict:
    """Real provider/model roster for the PWA's "Today's models" and
    "Routing & fallbacks" panels — no API keys in here, just provider +
    model names, so it's safe to expose publicly."""
    return {
        "roles": {
            "normal_chat": config.get_role("normal_chat"),
            "dispatcher_autonomous": config.get_role("dispatcher_autonomous"),
            "dev_slate_chat": config.get_role("dev_slate_chat"),
            "agent_work": config.get_role("agent_work"),
        },
        "task_routing": {cmd: config.get_task_routing(cmd) for cmd in COMMANDS},
        "enabled_providers": config.enabled_providers(),
    }


@app.get("/research/status")
def research_status() -> dict:
    """Polled by the PWA while an async /research job runs in the
    background — see dispatcher/research_status.py. `status` is null
    when nothing's in flight."""
    return {"status": get_status()}


@app.get("/reminders/check")
def reminders_check() -> dict:
    """Hit periodically by a GitHub Actions cron (see
    .github/workflows/check_reminders.yml) rather than run as a
    standalone job — this way delivery reads the config singleton this
    live process already has loaded, no separate Filen round-trip
    needed just to check reminders. Delivers via both push (PWA bubble)
    and Telegram (instant, no PWA dependency) — best-effort each, one
    channel failing doesn't block the other or leave the reminder
    stuck."""
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
    return {"delivered": delivered}


@app.get("/files/{relative_path:path}")
def file_download(relative_path: str, token: str | None = None, render: str | None = None) -> Response:
    """Serves a saved Filen artifact back down — the only way a rendered
    document or /research report reaches the PWA, which has no real
    attachment channel the way Telegram's sendDocument does. Fails
    closed on a missing/wrong token rather than falling back to open
    access, since this serves real document content.

    ?render=1 serves the content inline (renders as a real page in a
    browser tab) instead of forcing a download — restricted to .html
    specifically, not any file type, so this stays predictable: it's for
    /code's bundled HTML preview, not a general "display anything
    inline" switch."""
    if not NAVI_FILES_TOKEN or token != NAVI_FILES_TOKEN:
        raise HTTPException(status_code=403, detail="forbidden")
    if not relative_path or ".." in relative_path:
        raise HTTPException(status_code=400, detail="bad path")

    try:
        content = download_for_reply(f"filen:{relative_path}")
    except StorageError as e:
        raise HTTPException(status_code=404, detail=f"not found: {e}")

    filename = relative_path.rsplit("/", 1)[-1]
    content_type, _ = mimetypes.guess_type(filename)
    wants_render = render == "1"
    inline = wants_render and filename.lower().endswith(".html")
    disposition = "inline" if inline else "attachment"
    return Response(
        content=content,
        media_type=content_type or "application/octet-stream",
        headers={"Content-Disposition": f'{disposition}; filename="{filename}"'},
    )


TELEGRAM_WEBHOOK_SECRET = os.environ.get("TELEGRAM_WEBHOOK_SECRET")


@app.post("/webhook/telegram")
async def webhook_telegram(request: Request) -> PlainTextResponse:
    """Exempt from the shared NAVI_API_KEY gate above (Telegram can't send
    our custom header) — verified instead via the secret_token Telegram
    echoes back on every update once set_telegram_webhook.py registers
    one (see TELEGRAM_WEBHOOK_SECRET's own docstring there). Optional, not
    enforced, if the operator hasn't set one up yet — same reasoning as
    NAVI_FILES_TOKEN's own "unset = feature just doesn't check" precedent
    elsewhere in this file, since forcing this closed would silently break
    an already-working bot for anyone who deployed before this existed."""
    if TELEGRAM_WEBHOOK_SECRET and request.headers.get("X-Telegram-Bot-Api-Secret-Token") != TELEGRAM_WEBHOOK_SECRET:
        return PlainTextResponse("forbidden", status_code=403)
    payload = await request.json()
    adapter = _telegram_adapter()
    if adapter:
        msg = adapter.parse_incoming(payload)
        if msg:
            threading.Thread(target=handle_message, args=(adapter, msg), daemon=True).start()
    return PlainTextResponse("ok")  # ack immediately, process in background


@app.post("/webhook/discord")
async def webhook_discord() -> PlainTextResponse:
    return PlainTextResponse("ok")  # outbound-only phase — nothing to act on yet


@app.post("/chat/send")
async def chat_send(request: Request) -> JSONResponse:
    """The PWA's own chat surface for Normal/Research/Brainstorm/Agent
    Work. Plain-chat turns (not a typed /command, not a near-miss
    confirmation) now get real server-side memory — see
    dispatcher/chat.py's run_stored_mode_chat and how_to_handle_context.md
    (2026-09-01: first real multi-turn memory outside Dev Slate, the
    deliberately dumbest version, to find out empirically where plain
    context actually breaks before building anything fancier). Typed
    /commands and near-miss confirmations are untouched — they already
    have their own separate save/output mechanisms and aren't part of
    what's being tested here."""
    payload = await request.json()
    text = (payload.get("text") or "").strip()
    mode = payload.get("mode") or "normal"
    conversation_id = payload.get("conversation_id")
    # Only meaningful for mode == "agent_work" (see
    # dispatcher/chat.py's AGENT_WORK_REVIEW_INSTRUCTION) — defaults True
    # so omitting it (every other mode's client) is a no-op.
    auto_accept = payload.get("auto_accept", True)
    if not text:
        return JSONResponse({"error": "missing 'text'"}, status_code=400)
    result = parse_message(text)

    if result.kind == "commands" and any(s.command == "research" for s in result.steps):
        threading.Thread(target=_run_research_async, args=(result,), daemon=True).start()
        return JSONResponse({
            "reply": "Researching — I'll ping you when it's ready. Feel free to keep chatting.",
            "async": True,
        })

    if result.kind == "plain_chat":
        if not conversation_id:
            conversation_id = await create_conversation(mode=mode)
        reply = await run_stored_mode_chat(mode, conversation_id, text, auto_accept=auto_accept)
        return JSONResponse({
            "reply": reply["text"], "conversation_id": conversation_id,
            "usage_note": reply.get("usage_note"), "choices": reply.get("choices"),
            # Set only when this turn's create_workflow call actually ran
            # (dispatcher/chat.py's _extract_created_workflow_id) — lets
            # AgentWorkChat.tsx load the real graph onto the canvas as
            # nodes right after the chat builds it, instead of the
            # workflow only ever showing up in the Workflows list.
            "created_workflow_id": reply.get("created_workflow_id"),
        })

    reply_text, _attachments = _handle_parse_result(result, "pwa", mode, channel="pwa")
    return JSONResponse({"reply": reply_text})


@app.post("/push/subscribe")
async def push_subscribe(request: Request) -> JSONResponse:
    payload = await request.json()
    if not payload.get("endpoint"):
        return JSONResponse({"error": "missing 'endpoint'"}, status_code=400)
    add_subscription(payload)
    return JSONResponse({"ok": True, "subscriptions": subscription_count()})


@app.post("/push/test")
async def push_test() -> JSONResponse:
    try:
        errors = send_push("NAVI", "Test notification — if you see this, push is wired up correctly.")
    except PushError as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=400)
    return JSONResponse({"ok": True, "errors": errors})


# ---- Dev Slate: model catalog + manual pin (generic — Agent Work can reuse these later) ----

@app.get("/config/models")
def config_models(task: str = Query(...)) -> dict:
    """Qualifying free models for a given task, sorted best-first, off
    the cached daily ranking snapshot (jobs/model_ranking.py) — never
    triggers a live fetch (that's the daily job's concern, not a GET
    request's). Falls back to just the task's currently-pinned model if
    the snapshot has never been generated on this instance yet, so the
    picker isn't empty on a fresh deploy."""
    snapshot = load_snapshot()
    role_name = {"devslate": "dev_slate_chat", "agent_work": "agent_work", "normal_chat": "normal_chat"}.get(task)
    current_role = config.get_role(role_name) if role_name else None
    current = {"provider": current_role["provider"], "model": current_role["model"]} if current_role else None

    if not snapshot:
        return {"task": task, "current": current, "candidates": [current] if current else []}

    aa_index = fetch_aa_benchmarks(None)  # cache-only — no live fetch from a GET handler
    candidates = list_candidates(task, snapshot.get("catalog", []), aa_index)
    return {
        "task": task,
        "current": current,
        "candidates": [{"provider": c["provider"], "model": c["id"], "context_length": c.get("context_length")} for c in candidates],
    }


@app.post("/config/role")
async def config_role(request: Request) -> JSONResponse:
    """Manually pins a role to a specific provider/model — e.g. Dev
    Slate's model switcher. Clears the auto-fallback chain on purpose:
    a manual pick is an explicit choice, a silent fallback to something
    the user didn't pick would undercut the point of picking it."""
    payload = await request.json()
    role = payload.get("role")
    provider = payload.get("provider")
    model = payload.get("model")
    if not role or not provider or not model:
        return JSONResponse({"error": "missing 'role', 'provider', or 'model'"}, status_code=400)
    if not config.get_provider_key(provider):
        return JSONResponse({"error": f"provider '{provider}' has no API key configured"}, status_code=400)
    config.set_role(role, provider, model, fallback=[])
    return JSONResponse({"ok": True, "role": config.get_role(role)})


# ---- Dev Slate: Slate (conversation) management ----

@app.post("/devslate/conversations")
async def devslate_create_conversation(request: Request) -> JSONResponse:
    """Creates a new Slate — a Root Slate if no parent_id, a sub-Slate
    otherwise. Sub-Slates are an experienced-user opt-in (see the
    dispatcher/devslate_chat.py module docstring); the default flow
    never sets parent_id."""
    payload = await request.json() if await request.body() else {}
    conversation_id = await create_conversation(
        mode="devslate", project_id=payload.get("project_id"), parent_id=payload.get("parent_id"),
    )
    conversation = await get_conversation(conversation_id)
    return JSONResponse(conversation)


@app.get("/devslate/conversations/{conversation_id}")
async def devslate_get_conversation(conversation_id: str) -> dict:
    conversation = await get_conversation(conversation_id)
    if not conversation:
        raise HTTPException(status_code=404, detail="not found")
    return conversation


@app.get("/devslate/conversations/{conversation_id}/messages")
async def devslate_get_messages(conversation_id: str) -> dict:
    messages = await get_messages(conversation_id)
    task_state = await get_task_state(conversation_id)
    return {"messages": messages, "task_state": task_state}


# ---- Agent Work: workflow definitions + runs ----
# Native (not third-party-embedded) multi-step agent execution — see
# storage/agent_work.py and dispatcher/agent_work.py for the data model
# and executor. Mirrors the Dev Slate conversations routes above (create/
# get/list) plus /reminders/check's externally-pinged scheduling pattern,
# since no in-process scheduler exists anywhere in this codebase.

@app.post("/agent/workflows")
async def agent_create_workflow(request: Request) -> JSONResponse:
    payload = await request.json()
    name = payload.get("name")
    graph = payload.get("graph")
    if not name or not graph:
        return JSONResponse({"error": "missing 'name' or 'graph'"}, status_code=400)
    trigger = payload.get("trigger") or {"type": "manual"}
    workflow_id = await create_workflow_definition(name, payload.get("description"), graph, trigger)
    return JSONResponse(await get_workflow(workflow_id))


@app.get("/agent/workflows")
async def agent_list_workflows() -> list[dict]:
    return await list_workflows()


@app.get("/agent/workflows/due")
async def agent_due_workflows() -> dict:
    """Manual poke / health-check — the real trigger is now
    dispatcher/scheduler.py's in-process cron, which calls
    check_due_workflows() directly on its own schedule. This route just
    exposes the same check over HTTP, e.g. to poke it by hand or verify
    it's wired up, without waiting for the next scheduled fire.

    Registered BEFORE /agent/workflows/{workflow_id} below on purpose —
    FastAPI matches routes in registration order, so a static path like
    this one must come before a dynamic path that would otherwise treat
    "due" as a workflow_id and swallow every request here (caught before
    ever deploying this: a quick route-table dump during dev testing
    showed the dynamic route matching first)."""
    started = await check_due_workflows()
    return {"started": started}


@app.get("/agent/workflows/{workflow_id}")
async def agent_get_workflow(workflow_id: str) -> dict:
    workflow = await get_workflow(workflow_id)
    if not workflow:
        raise HTTPException(status_code=404, detail="not found")
    return workflow


@app.delete("/agent/workflows/{workflow_id}")
async def agent_delete_workflow(workflow_id: str) -> JSONResponse:
    """Deletes the workflow definition. This is also the entire "cancel its
    schedule" operation — see delete_workflow's docstring in
    storage/agent_work.py for why nothing else needs to be touched. Past
    runs/steps for it are kept, not cascade-deleted (real audit trail)."""
    deleted = await delete_workflow_definition(workflow_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="not found")
    return JSONResponse({"deleted": True})


@app.post("/agent/workflows/{workflow_id}/run")
async def agent_run_workflow(workflow_id: str) -> JSONResponse:
    """Manual trigger — returns the new run's id immediately, execution
    continues on a background thread (see dispatcher/agent_work.py)."""
    try:
        run_id = await start_workflow_run(workflow_id, trigger_source="manual")
    except WorkflowError as e:
        return JSONResponse({"error": str(e)}, status_code=404)
    return JSONResponse({"run_id": run_id})


@app.get("/agent/runs")
async def agent_list_runs(workflow_id: str | None = None, status: str | None = None) -> list[dict]:
    return await list_runs(workflow_id=workflow_id, status=status)


@app.delete("/agent/runs")
async def agent_delete_all_runs(workflow_id: str | None = None) -> JSONResponse:
    """Bulk "clear history" (2026-09-04, JuanJo: "I don't actually wanna
    know which runs were done so long ago") — every run and its steps,
    optionally scoped to one workflow via ?workflow_id=. The workflow
    definitions themselves are completely untouched; this only clears
    what already ran."""
    deleted = await delete_all_runs(workflow_id=workflow_id)
    return JSONResponse({"deleted": deleted})


@app.get("/agent/runs/{run_id}")
async def agent_get_run(run_id: str) -> dict:
    run = await get_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="not found")
    return run


@app.delete("/agent/runs/{run_id}")
async def agent_delete_run(run_id: str) -> JSONResponse:
    deleted = await delete_run(run_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="not found")
    return JSONResponse({"deleted": True})


@app.get("/agent/runs/{run_id}/steps")
async def agent_get_run_steps(run_id: str) -> list[dict]:
    return await get_run_steps(run_id)


# ---- Star a workflow into the Agent Vault (2026-09-03) — a reference, not
# a copy: the Vault entry's `workflow_id` points back here so running it
# always executes the live graph, and its "Tools/Nodes" list is derived
# from the graph client-side rather than duplicated into storage. ----

@app.post("/agent/workflows/{workflow_id}/star")
async def agent_star_workflow(workflow_id: str) -> dict:
    existing = await get_agent_by_workflow_id(workflow_id)
    if existing:
        return existing
    workflow = await get_workflow(workflow_id)
    if not workflow:
        raise HTTPException(status_code=404, detail="not found")
    instructions = workflow.get("creation_transcript") or workflow.get("description") or ""
    agent_id = await create_agent(
        workflow["name"], instructions, [], None, None, workflow_id=workflow_id,
    )
    return await get_agent(agent_id)


@app.delete("/agent/workflows/{workflow_id}/star")
async def agent_unstar_workflow(workflow_id: str) -> JSONResponse:
    existing = await get_agent_by_workflow_id(workflow_id)
    if not existing:
        return JSONResponse({"deleted": False})
    await delete_agent(existing["id"])
    return JSONResponse({"deleted": True})


# ---- MCP connections (see dispatcher/mcp_client.py for the actual
# client/security model — this is just the REST surface over it) ----

@app.get("/mcp/marketplace/search")
async def mcp_marketplace_search(q: str = "", limit: int = 20) -> JSONResponse:
    """Real-time proxy over the official MCP Registry (tools/
    mcp_marketplace.py) — lets ConnectionsOverlay show actual, searchable
    MCP servers instead of only the fixed SERVICE_CATALOG. Read-only, no
    connection is made here; a result just pre-fills the existing
    /mcp/connections + /connect flow below with real transport info."""
    try:
        results = await asyncio.to_thread(search_mcp_marketplace, q, min(limit, 50))
    except MCPMarketplaceError as e:
        return JSONResponse({"error": str(e)}, status_code=502)
    return JSONResponse({"results": results})


@app.post("/mcp/connections")
async def mcp_create_connection(request: Request) -> JSONResponse:
    """Registers a connection's transport config — does NOT connect yet,
    that's the separate /connect call below, since a live handshake can
    be slow or fail and shouldn't block "just save what I typed".

    http-only (2026-09-04): stdio (local subprocess) servers are refused
    here — no sandbox exists yet for arbitrary local process execution,
    see dispatcher/mcp_client.py's own docstring on why this is deferred
    by cost rather than fixed."""
    payload = await request.json()
    name = payload.get("name")
    transport = payload.get("transport")
    if not name or transport != "http":
        return JSONResponse(
            {"error": "name is required and transport must be 'http' — local (stdio) MCP "
                      "servers are disabled until NAVI has a sandbox to run them in"},
            status_code=400,
        )
    config.set_mcp_connection(name, transport, url=payload.get("url"), auth_header=payload.get("auth_header"))
    return JSONResponse({"ok": True})


@app.get("/mcp/connections")
async def mcp_list_connections() -> list[dict]:
    # Never echoes auth_header back — a saved token is write-only from
    # the client's perspective from here on, same principle as a
    # password field never round-tripping its own value.
    return [
        {
            "name": name, "transport": conn.get("transport"), "connected": conn.get("connected", False),
            "tools": [
                {
                    "name": tool_name, "read_only": t["read_only"],
                    "destructive": t.get("destructive", not t["read_only"]),
                    "approved_at": t["approved_at"],
                }
                for tool_name, t in conn.get("tools", {}).items()
            ],
        }
        for name, conn in config.list_mcp_connections().items()
    ]


@app.delete("/mcp/connections/{name}")
async def mcp_delete_connection(name: str) -> JSONResponse:
    config.remove_mcp_connection(name)
    return JSONResponse({"ok": True})


@app.post("/mcp/connections/{name}/connect")
async def mcp_connect(name: str) -> JSONResponse:
    """Real handshake + tool discovery. Tools the server has never shown
    before ("new") are auto-approved — the user just configured this
    connection themselves, so first trust is reasonable. Tools whose
    definition changed since a prior approval ("changed" — the rug-pull
    case) are deliberately left unapproved for /approve below to handle;
    silently re-trusting a changed tool here would defeat the entire
    point of pinning it in the first place."""
    try:
        discovered = await asyncio.to_thread(discover_tools, name)
    except MCPError as e:
        return JSONResponse({"error": str(e)}, status_code=502)
    new_tools = [t for t in discovered if t["status"] == "new"]
    if new_tools:
        approve_tools(name, new_tools)
    config.set_mcp_connected(name, True)
    return JSONResponse({"tools": discovered})


MCP_OAUTH_REDIRECT_URI = f"{NAVI_BASE_URL}/mcp/oauth/callback"


@app.post("/mcp/connections/{name}/oauth/start")
async def mcp_oauth_start(name: str) -> JSONResponse:
    """Runs the real MCP-spec OAuth discovery chain (dispatcher/
    mcp_oauth.py) against this connection's URL and returns an
    authorize_url for the frontend to redirect the whole browser to (not
    an XHR — the user needs to actually see and approve the provider's
    consent screen). The resulting pending flow is stashed here, keyed by
    state, for /mcp/oauth/callback below to pick back up once the
    provider redirects the browser back."""
    conn = config.get_mcp_connection(name)
    if conn is None or not conn.get("url"):
        return JSONResponse({"error": "connection has no URL configured yet — save the connection first"}, status_code=400)
    try:
        flow = await asyncio.to_thread(start_authorization, name, conn["url"], MCP_OAUTH_REDIRECT_URI)
    except MCPOAuthError as e:
        return JSONResponse({"error": str(e)}, status_code=502)
    with _pending_oauth_lock:
        _pending_oauth[flow["state"]] = {
            "server_name": name, "code_verifier": flow["code_verifier"], "token_endpoint": flow["token_endpoint"],
            "client_id": flow["client_id"], "client_secret": flow["client_secret"], "redirect_uri": flow["redirect_uri"],
        }
    return JSONResponse({"authorize_url": flow["authorize_url"]})


@app.get("/mcp/oauth/callback")
async def mcp_oauth_callback(request: Request) -> RedirectResponse:
    """The provider's own redirect lands here (see _PUBLIC_PATHS above
    for why this route is exempt from the shared API key). `state` is
    looked up against the pending flow /oauth/start stashed — a request
    with no match (forged, replayed, or expired-by-restart) is refused
    before anything else happens, same anti-CSRF role `state` always
    plays in OAuth, not something extra this route adds."""
    code = request.query_params.get("code")
    state = request.query_params.get("state")
    with _pending_oauth_lock:
        pending = _pending_oauth.pop(state, None) if state else None
    if request.query_params.get("error") or not pending or not code:
        return RedirectResponse(f"{PWA_ORIGIN}/?mcp_oauth=error")

    try:
        access_token = await asyncio.to_thread(
            exchange_code_for_token, pending["token_endpoint"], code, pending["code_verifier"],
            pending["client_id"], pending["client_secret"], pending["redirect_uri"],
        )
    except MCPOAuthError:
        return RedirectResponse(f"{PWA_ORIGIN}/?mcp_oauth=error")

    server_name = pending["server_name"]
    conn = config.get_mcp_connection(server_name)
    if conn is None:
        return RedirectResponse(f"{PWA_ORIGIN}/?mcp_oauth=error")
    # Stores the token exactly like a pasted one — encrypted at rest via
    # config.set_mcp_connection's existing auth_header handling, same
    # code path a manual paste already goes through.
    config.set_mcp_connection(server_name, conn["transport"], url=conn["url"], auth_header=f"Bearer {access_token}")

    try:
        discovered = await asyncio.to_thread(discover_tools, server_name)
        new_tools = [t for t in discovered if t["status"] == "new"]
        if new_tools:
            approve_tools(server_name, new_tools)
        config.set_mcp_connected(server_name, True)
    except MCPError:
        return RedirectResponse(f"{PWA_ORIGIN}/?mcp_oauth=partial&connection={server_name}")

    return RedirectResponse(f"{PWA_ORIGIN}/?mcp_oauth=success&connection={server_name}")


@app.post("/mcp/connections/{name}/approve")
async def mcp_approve_tools(name: str, request: Request) -> JSONResponse:
    """Explicitly approves specific tools by name — the path for a
    "changed" (rug-pull-flagged) tool the human has actually reviewed, or
    a "new" tool that wasn't auto-approved. Re-discovers first rather
    than trusting whatever the client last saw, so approval always pins
    the server's CURRENT definition, not a stale one from an earlier
    response."""
    payload = await request.json()
    tool_names = set(payload.get("tool_names", []))
    try:
        discovered = await asyncio.to_thread(discover_tools, name)
    except MCPError as e:
        return JSONResponse({"error": str(e)}, status_code=502)
    to_approve = [t for t in discovered if t["name"] in tool_names]
    approve_tools(name, to_approve)
    return JSONResponse({"approved": [t["name"] for t in to_approve]})


# ---- Agent Vault (2026-09-03) — saved, reusable agent configs. Separate
# from /agent/workflows on purpose: a saved agent isn't a workflow (see
# storage/agents.py's own docstring) — "Open in canvas" on the frontend
# is the one-way fork that turns one into the other, not a live link. ----

@app.post("/agents")
async def agents_create(request: Request) -> JSONResponse:
    payload = await request.json()
    name = payload.get("name")
    instructions = payload.get("instructions")
    if not name or not instructions:
        return JSONResponse({"error": "name and instructions are required"}, status_code=400)
    agent_id = await create_agent(
        name, instructions, payload.get("tools", []), payload.get("model"), payload.get("output_type"),
    )
    return JSONResponse({"id": agent_id})


@app.get("/agents")
async def agents_list() -> list[dict]:
    return await list_agents()


@app.get("/agents/{agent_id}")
async def agents_get(agent_id: str) -> dict:
    agent = await get_agent(agent_id)
    if not agent:
        raise HTTPException(status_code=404, detail="not found")
    return agent


@app.put("/agents/{agent_id}")
async def agents_update(agent_id: str, request: Request) -> JSONResponse:
    payload = await request.json()
    name = payload.get("name")
    instructions = payload.get("instructions")
    if not name or not instructions:
        return JSONResponse({"error": "name and instructions are required"}, status_code=400)
    await update_agent(
        agent_id, name, instructions, payload.get("tools", []), payload.get("model"), payload.get("output_type"),
    )
    return JSONResponse({"ok": True})


@app.delete("/agents/{agent_id}")
async def agents_delete(agent_id: str) -> JSONResponse:
    deleted = await delete_agent(agent_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="not found")
    return JSONResponse({"ok": True})


# ---- Dev Slate: the live chat WebSocket ----

@app.websocket("/ws/devslate/{conversation_id}")
async def devslate_ws(websocket: WebSocket, conversation_id: str) -> None:
    """One connection per open Dev Slate Slate. Two things happen over
    this socket:

    1. Ordinary chat: client sends {"type": "user_message", "text": ...},
       server eventually replies {"type": "assistant_message", ...}.
    2. Tool relay: when the model calls read_file/write_file/grep, the
       server can't run it (the files are on the user's machine) — it
       sends {"type": "tool_request", "id", "name", "arguments"} and
       waits for the browser to execute it locally and reply with
       {"type": "tool_result", "id", "result"}.

    This being a genuine open connection is also what satisfies "chat
    can send messages even when not asked" — every message down this
    socket is a push, not a poll response, by construction. Nothing
    currently triggers a truly unprompted message (no background job
    calls into an open Slate yet), but the mechanism is real and ready:
    anything holding this websocket can send an assistant_message frame
    at any time, not just in reply to a user_message.

    pending/tool-call futures are scoped to THIS connection (a local
    dict, not module-level) — two Slates open at once must never let one
    connection's disconnect resolve (or lose) the other's in-flight tool
    call.

    NOT covered by _require_api_key above — that's HTTP-only middleware,
    a WebSocket upgrade never passes through it (real gap, found and
    fixed 2026-09-04 in the same pass). Checked here instead, and via a
    ?key= query param rather than the X-Navi-Api-Key header, since the
    browser's native WebSocket API has no way to set custom headers on
    the handshake at all (see devslate.ts's wsUrl).
    """
    if not NAVI_API_KEY or websocket.query_params.get("key") != NAVI_API_KEY:
        await websocket.close(code=4401)
        return
    await websocket.accept()
    pending: dict[str, asyncio.Future] = {}
    # Serializes turns on THIS connection (so two rapid user_message
    # frames can't interleave writes to the same conversation's history)
    # without blocking the receive loop itself — see the deadlock note
    # on handle_user_message below for why that distinction matters.
    turn_lock = asyncio.Lock()

    async def relay(name: str, arguments: dict) -> str:
        call_id = new_tool_call_id()
        fut: asyncio.Future = asyncio.get_event_loop().create_future()
        pending[call_id] = fut
        await websocket.send_json({"type": "tool_request", "id": call_id, "name": name, "arguments": arguments})
        try:
            return await asyncio.wait_for(fut, timeout=60)
        except asyncio.TimeoutError:
            return f"Tool error: {name} timed out waiting for a response from the browser."
        finally:
            pending.pop(call_id, None)

    async def handle_user_message(text: str) -> None:
        """Runs as its own task (see the receive loop below), NOT awaited
        inline there — this is load-bearing, not a style choice. A tool
        call blocks here on `relay`'s pending future, which only the
        receive loop's own `tool_result` branch can resolve; if this
        coroutine were awaited directly inside that same loop, the loop
        could never reach the frame that unblocks it. Real deadlock,
        caught live (2026-09-01): the connection hung until uvicorn force-
        closed it with a 1011 after the client's keepalive ping timed out.
        Running this as a separate task lets the receive loop keep
        pulling frames — including the tool_result that unblocks it —
        the whole time this is in flight."""
        async with turn_lock:
            result = await run_devslate_turn(conversation_id, text, relay)
            await websocket.send_json({
                "type": "assistant_message",
                "text": result["text"],
                "provider": result["provider"],
                "model": result["model"],
                "choices": result.get("choices"),
            })

    try:
        while True:
            msg = await websocket.receive_json()
            msg_type = msg.get("type")

            if msg_type == "user_message":
                text = (msg.get("text") or "").strip()
                if text:
                    asyncio.create_task(handle_user_message(text))

            elif msg_type == "tool_result":
                call_id = msg.get("id")
                fut = pending.get(call_id)
                if fut and not fut.done():
                    fut.set_result(msg.get("result", ""))
            # Unknown/malformed frame types are ignored rather than
            # closing the connection — a stray message from a slightly
            # stale client shouldn't kill an otherwise-working session.

    except WebSocketDisconnect:
        for fut in pending.values():
            if not fut.done():
                fut.set_result("Tool error: connection closed before the browser responded.")


def _seed_keys_from_env() -> None:
    """
    Config store is normally meant to be filled in via chat ("here's my
    Groq key") — see config/store.py's docstring — so it survives without
    a redeploy. That chat-side key-intake isn't built yet (out of scope
    for this pass), so for the FIRST boot on a fresh instance with
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


_seed_keys_from_env()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT)
