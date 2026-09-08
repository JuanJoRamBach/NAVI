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
import json
import mimetypes
import os
import threading
import time

from fastapi import FastAPI, HTTPException, Query, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse, RedirectResponse, Response

from dispatcher.agent_work import (
    WEBHOOK_RESPONSE_TIMEOUT_SECONDS, WorkflowError, check_due_workflows, discard_webhook_waiter,
    peek_webhook_waiter, request_run_cancellation, set_webhook_trigger, start_webhook_run, start_workflow_run,
)
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
from storage.filen import StorageError, download_for_reply, file_download_url
from storage.conversations import (
    create_conversation, get_conversation, get_messages, get_task_state,
)
from storage.agent_work import (
    create_workflow as create_workflow_definition,
    delete_all_runs, delete_run,
    delete_workflow as delete_workflow_definition,
    get_latest_node_output, get_run, get_run_steps, get_workflow, get_workflow_by_webhook_token,
    list_runs, list_workflow_versions, list_workflows,
    purge_workflow, restore_workflow, revert_workflow_to_version,
    update_workflow as update_workflow_definition,
)
from storage.agents import create_agent, delete_agent, get_agent, get_agent_by_workflow_id, list_agents, update_agent
from storage.sources import delete_document as delete_source_document, get_document as get_source_document, latest_batch as latest_source_batch, list_documents as list_source_documents, set_document_status as set_source_document_status
from dispatcher.source_fetch import start_source_fetch_batch
from tools.devslate_tools import new_tool_call_id

PORT = int(os.environ.get("PORT", "10000"))
NAVI_BASE_URL = "https://api.getnavi.online"

# The PWA (navi-ui, on GitHub Pages, custom domain getnavi.online) calls
# this server from a different origin — browsers block that without an
# explicit CORS allow. Scoped to real frontend origins rather than "*",
# since several of these endpoints accept real data (push subscriptions,
# chat text, file writes relayed from Dev Slate).
#
# The real public site — used for OAuth redirects below (an external
# provider's consent screen sends the browser back here; a Tauri-only
# pseudo-origin isn't a navigable destination for that), and as the one
# real entry in PWA_CORS_ORIGINS just below.
PWA_ORIGIN = "https://getnavi.online"

# CORS allow-list — broader than PWA_ORIGIN alone. The Tauri desktop
# build (2026-09-06, v0.3.1) is the SAME built PWA code, just served
# from its own webview origin instead of GitHub Pages. Real gap found
# live: a fresh Windows install hit "Couldn't reach NAVI" on the
# access-key screen even after tauri://localhost/https://tauri.localhost
# were added — Tauri's own config schema (useHttpsScheme, defaults to
# false) clarifies the ACTUAL Windows default is http://tauri.localhost,
# not https. All four are listed (both schemes, since useHttpsScheme
# could change; tauri:// for macOS/Linux, a Mac build being planned too)
# rather than betting on one guess a second time.
PWA_CORS_ORIGINS = [
    PWA_ORIGIN,
    "tauri://localhost",
    "https://tauri.localhost",
    "http://tauri.localhost",
]

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
    allow_origins=PWA_CORS_ORIGINS,
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
        # Agent Work's webhook trigger (2026-09-07) — an external caller
        # (Stripe, GitHub, a cron service, your brother's always-connected
        # API) can't send NAVI's own header. The token itself, unguessable
        # in the path, IS the credential — same trust model as
        # TELEGRAM_WEBHOOK_SECRET, verified inside the route itself against
        # get_workflow_by_webhook_token, not here.
        or request.url.path.startswith("/agent/webhooks/")
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
    # 04:45 UTC = 6:45 Madrid (CEST) / 00:45 New York (EDT) — picked to run
    # shortly after providers' confirmed 00:00 UTC daily-quota resets
    # (Cloudflare docs) rather than an arbitrary hour. Drifts by up to an
    # hour during the ~1-week EU/US DST-changeover gap in late Oct/early
    # Nov since the two regions switch on different dates — known, not a bug.
    register_job("refresh_model_ranking_snapshot", "45 4 * * *", _refresh_model_ranking_snapshot)
    start_scheduler()


def _telegram_adapter() -> TelegramAdapter | None:
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    return TelegramAdapter(token) if token else None


def _discord_adapter() -> DiscordAdapter | None:
    token = os.environ.get("DISCORD_BOT_TOKEN")
    return DiscordAdapter(token) if token else None


def _pwa_download_links(results: list) -> str:
    """The PWA has no file-attachment channel (unlike Telegram's real
    sendDocument) — a saved artifact reaches it as plain URLs appended
    to the reply text instead, which the frontend detects and renders
    as clickable chips (App.tsx's parseAttachments/DOWNLOAD_LINE_RE).

    Real gap found and fixed 2026-09-06 (JuanJo: "/graph-data DOES not
    show the graph on the chat, it only says it was created in filen") —
    this used to skip image results (graph-data) entirely on the theory
    that "they're the image, not a file to re-download," but no OTHER
    delivery path for the web client was ever built to back that theory:
    the PWA's reply_text ended up with nothing but format_summary's own
    literal "Saved to filen:..." line, an internal path notation the
    browser can't do anything with. Image results now get a real
    clickable link the same as everything else here."""
    lines = []
    for r in results:
        if r.rendered_file_saved_path and r.rendered_file_name:
            url = file_download_url(r.rendered_file_saved_path)
            if url:
                lines.append(f"📎 {r.rendered_file_name}: {url}")
        elif r.saved_path:
            filename = r.saved_path.rsplit("/", 1)[-1]
            download_url = file_download_url(r.saved_path)
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



# Static caps for providers where "the cap" isn't something a live
# endpoint or per-response header can tell us (see storage/usage.py and
# providers/{groq,openrouter,mistral}.py for the providers that ARE
# tracked live instead of via a hardcoded number here). Sourced from each
# provider's own docs, checked directly, not guessed:
#   - Cloudflare: developers.cloudflare.com — "All limits reset daily at
#     00:00 UTC", 10,000 free Neurons/day, shared across all models.
#   - LLM7: this repo's own providers/llm7.py docstring, confirmed
#     2026-09-01 against docs.llm7.io — 1,000,000 tokens/24h keyed pool
#     (NAVI always sends a key), separate unused-by-NAVI 500,000/24h
#     anonymous pool shown for context only.
#   - Mistral: JuanJo's own figure — $10/month free "Experiment" tier
#     credit — compared against the real billed-dollar total from
#     providers.mistral.get_admin_usage(), not computed locally.
#   - OpenRouter: 50 free-model requests/day — JuanJo confirmed directly
#     (2026-09-05) his account has never crossed the $10-lifetime-
#     purchased threshold that bumps this to 1,000/day. This can't be
#     auto-detected from OpenRouter's own API: GET /api/v1/key's `usage`
#     field is all-time CONSUMED credits, but the real threshold that
#     decides 50 vs 1,000 is all-time PURCHASED credits (confirmed via
#     OpenRouter's own docs) — a field their API doesn't expose at all.
#     Also confirmed (2026-09-05): OpenRouter sends NO rate-limit headers
#     on successful responses (unlike Groq) — X-RateLimit-Remaining only
#     appears on the 429 itself, after the limit's already been hit — so
#     this compares NAVI's own local request count (same generic
#     per-provider tracking Cloudflare/LLM7 use) against this confirmed
#     number, an approximation that could drift if something outside NAVI
#     ever calls this same key, not an OpenRouter-reported live figure.
# Groq has NO entry here on purpose — its quota is per-model (JuanJo has
# had to correct this more than once, see the feedback-groq-per-model-
# quotas memory) and comes from storage.usage.get_groq_snapshots(),
# Groq's own live numbers, not a table NAVI maintains.
CLOUDFLARE_DAILY_NEURON_CAP = 10_000
LLM7_KEYED_DAILY_TOKEN_CAP = 1_000_000
LLM7_ANONYMOUS_DAILY_TOKEN_CAP = 500_000  # real, but unused by NAVI today
MISTRAL_MONTHLY_CREDIT_USD = 10.0
OPENROUTER_DAILY_REQUEST_CAP = 50


@app.get("/usage/counters")
def usage_counters() -> dict:
    """Real, server-side data for the PWA's Usage counters panel —
    replaces the old hardcoded USAGE_COUNTERS mock in App.tsx. Mistral is
    deliberately NOT included here (see /usage/mistral) — it's a monthly
    billing figure fetched lazily on-demand, not something to compute on
    every load of this route."""
    from storage.usage import get_groq_snapshots, get_usage_today

    groq_models = [
        {
            "model": row["model"],
            "used": (row["limit_requests"] - row["remaining_requests"]) if row["limit_requests"] is not None and row["remaining_requests"] is not None else None,
            "limit": row["limit_requests"],
            "reset_seconds": row["reset_requests_seconds"],
        }
        for row in get_groq_snapshots()
    ]

    cf_rows = get_usage_today("cloudflare")
    cloudflare_neurons = sum(r["neurons"] for r in cf_rows)

    llm7_rows = get_usage_today("llm7")
    llm7_tokens = sum(r["tokens"] for r in llm7_rows)

    gmi_rows = get_usage_today("gmi")
    gmi_requests = sum(r["requests"] for r in gmi_rows)

    ollama_rows = get_usage_today("ollama_cloud")
    ollama_requests = sum(r["requests"] for r in ollama_rows)
    ollama_tokens = sum(r["tokens"] for r in ollama_rows)

    or_rows = get_usage_today("openrouter")
    openrouter_requests = sum(r["requests"] for r in or_rows)
    key = config.get_provider_key("openrouter")
    openrouter_spend_info = None
    if key:
        from providers.openrouter import get_key_info
        openrouter_spend_info = get_key_info(key)

    return {
        "groq": {"models": groq_models},
        "cloudflare": {"neurons_used": cloudflare_neurons, "neurons_cap": CLOUDFLARE_DAILY_NEURON_CAP},
        "openrouter": {
            # Requests-left is NAVI's own local count against the
            # confirmed 50/day cap (see the constant's comment above) —
            # OpenRouter's API can't tell us this number directly.
            "requests_used": openrouter_requests,
            "requests_cap": OPENROUTER_DAILY_REQUEST_CAP,
            # Real, live spend data OpenRouter DOES report accurately —
            # shown alongside as context, not as the cap figure itself.
            "spend": openrouter_spend_info,
        },
        "llm7": {
            "tokens_used": llm7_tokens,
            "keyed_cap": LLM7_KEYED_DAILY_TOKEN_CAP,
            "anonymous_cap": LLM7_ANONYMOUS_DAILY_TOKEN_CAP,
        },
        "gmi": {"requests_today": gmi_requests, "status": "checking — promo status unconfirmed past 2026-09-06"},
        "ollama_cloud": {"requests_today": ollama_requests, "tokens_today": ollama_tokens, "cap": None},
    }


@app.get("/usage/mistral")
def usage_mistral() -> dict:
    """Separate route, fetched on-demand when the panel's Mistral card is
    actually opened — see providers/mistral.py's get_admin_usage() for why
    this isn't folded into /usage/counters above."""
    key = config.get_provider_key("mistral")
    if not key:
        return {"usage": None, "credit_usd": MISTRAL_MONTHLY_CREDIT_USD}
    from providers.mistral import get_admin_usage
    return {"usage": get_admin_usage(key), "credit_usd": MISTRAL_MONTHLY_CREDIT_USD}


@app.post("/sources/batch")
async def sources_batch_start(request: Request) -> JSONResponse:
    """Kicks off the Sources tab's Batch Dispatch in the background —
    acks immediately, same reasoning as /research (a multi-term batch of
    real web fetches is too slow to hold the HTTP request open for).
    `trusted_sites` is sent by the caller on every request rather than
    read from server-side state, because the registry itself currently
    only lives in the PWA's own localStorage (navi-pwa/src/
    trustedSources.ts) — there's no backend copy of it yet."""
    payload = await request.json()
    terms = payload.get("terms") or []
    trusted_sites = payload.get("trusted_sites") or []
    if not terms:
        return JSONResponse({"error": "'terms' must be a non-empty list"}, status_code=400)
    if not trusted_sites:
        return JSONResponse({"error": "No trusted sites configured — add at least one before dispatching."}, status_code=400)
    start_source_fetch_batch(terms, trusted_sites)
    return JSONResponse({"started": True})


@app.get("/sources/status")
def sources_status() -> dict:
    """Polled by the PWA while a batch runs. `batch` is None if nothing's
    ever been dispatched; otherwise the most recent one regardless of
    whether it's still running — see storage.sources.latest_batch."""
    return {"batch": latest_source_batch()}


@app.get("/sources")
def sources_list(status: str | None = None) -> list[dict]:
    """Every saved document, newest first, across all batches — Sources
    is app-wide, not scoped to one conversation (see storage/sources.py).
    `?status=accepted` is the query a future context-building consumer
    needs — everything a human has actually reviewed and approved."""
    return list_source_documents(status=status)


@app.get("/sources/{doc_id}")
def sources_get_one(doc_id: str) -> JSONResponse:
    """The actual review needs something to review AGAINST — until now
    /sources only ever returned title/url/domain/status, never the saved
    content itself (2026-09-06, JuanJo: 'I can't see the documents it
    created, so I can't review them'). Reads the real file back from
    Filen via filen_path, same mechanism download_for_reply already uses
    for Telegram/Discord attachments. `content` is None (not an error)
    when filen_path was never set — save_source_document already treats
    a Filen save failure as non-fatal, so the DB row can legitimately
    exist with no backing file; the frontend should show that as
    'content unavailable', not surface it as a fetch error."""
    doc = get_source_document(doc_id)
    if not doc:
        return JSONResponse({"error": "Document not found"}, status_code=404)
    content = None
    if doc.get("filen_path"):
        try:
            content = download_for_reply(doc["filen_path"]).decode("utf-8")
        except StorageError as e:
            # Logged now (2026-09-06) — same silent-swallow mistake as
            # save_source_document originally made, caught immediately
            # this time: every document read back as "content
            # unavailable" with nothing in the logs explaining why.
            print(f"[sources_get_one] Filen read-back failed for doc {doc_id} (path={doc['filen_path']!r}): {e}")
            content = None
    return JSONResponse({**doc, "content": content})


@app.post("/sources/{doc_id}/review")
async def sources_review(doc_id: str, request: Request) -> JSONResponse:
    """The actual review gate — a document is 'pending_review' until a
    human explicitly accepts or rejects it here. Nothing else in this
    codebase currently reads source_documents.status to decide what's
    usable as chat context — that's the next piece, not built yet;
    this route is what a future consumer would filter on."""
    payload = await request.json()
    status = payload.get("status")
    if status not in ("accepted", "rejected"):
        return JSONResponse({"error": "status must be 'accepted' or 'rejected'"}, status_code=400)
    ok = set_source_document_status(doc_id, status)
    if not ok:
        return JSONResponse({"error": "Document not found"}, status_code=404)
    return JSONResponse({"ok": True})


@app.delete("/sources/{doc_id}")
def sources_delete(doc_id: str) -> JSONResponse:
    """Permanent removal — 2026-09-06, JuanJo: 'I need to be able to
    eliminate rejected documents.' Doesn't touch the backing Filen file;
    the row disappearing from every list/review view is the actual ask."""
    ok = delete_source_document(doc_id)
    if not ok:
        return JSONResponse({"error": "Document not found"}, status_code=404)
    return JSONResponse({"ok": True})


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
    # Images default to inline regardless of ?render= (2026-09-06 — a
    # graph-data chart forced through Content-Disposition: attachment
    # never got a chance to display, only to download; an image has no
    # "meant to be saved as a document" case the way .html/.md do).
    # .html still needs the explicit ?render=1 opt-in — that one's the
    # existing /code bundled-preview behavior, unrelated to this fix.
    is_image = (content_type or "").startswith("image/")
    inline = is_image or (wants_render and filename.lower().endswith(".html"))
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
            # run_stored_mode_chat already computes these (which attempt in
            # its fallback chain actually answered) — this route just never
            # forwarded them to the client before (2026-09-06, JuanJo: "I
            # don't see which model was used... can't see how many
            # tokens"). The fallback notice itself was always appended
            # into reply["text"] as a "⚡ (primary was unavailable...)"
            # line; these two fields are what let the frontend show a
            # real model badge instead of relying on that string being
            # parsed back out of the message body.
            "provider": reply.get("provider"), "model": reply.get("model"),
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
        # quality/speed (2026-09-06) are the same real numbers
        # list_candidates ranks by, not new computation here — lets the
        # picker UI show/group by real signal instead of a flat list.
        # quality is 0 for a model with no matching AA benchmark entry
        # (no signal, not "bad") — the frontend should treat 0 as
        # "unranked," not the lowest real score.
        "candidates": [
            {"provider": c["provider"], "model": c["id"], "context_length": c.get("context_length"), "quality": c.get("_quality", 0), "speed": c.get("_speed", 0)}
            for c in candidates
        ],
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


@app.put("/agent/workflows/{workflow_id}")
async def agent_update_workflow(workflow_id: str, request: Request) -> JSONResponse:
    """Real update-in-place (2026-09-07) — "Save Edits" on an already-
    loaded workflow, as opposed to POST above which always creates a new
    one. See update_workflow's own docstring in storage/agent_work.py for
    why this exists: without it, editing a workflow silently forked a
    duplicate instead of changing the one you meant to, orphaning a
    webhook-triggered workflow's real URL in the process."""
    payload = await request.json()
    name = payload.get("name")
    graph = payload.get("graph")
    if not name or not graph:
        return JSONResponse({"error": "missing 'name' or 'graph'"}, status_code=400)
    trigger = payload.get("trigger") or {"type": "manual"}
    # edited_by (2026-09-08) — real audit trail field, but there are no
    # real user accounts anywhere in NAVI yet (one shared NAVI_API_KEY for
    # the whole API), so this stays whatever the caller sends, usually
    # None today. Not enforced/validated — the honest gap, not silently
    # faked. See storage/agent_work.py's update_workflow docstring.
    edited_by = payload.get("edited_by")
    updated = await update_workflow_definition(workflow_id, name, payload.get("description"), graph, trigger, edited_by)
    if not updated:
        return JSONResponse({"error": "not found"}, status_code=404)
    return JSONResponse(await get_workflow(workflow_id))


@app.get("/agent/workflows")
async def agent_list_workflows(include_deleted: bool = False) -> list[dict]:
    return await list_workflows(include_deleted=include_deleted)


@app.get("/agent/workflows/{workflow_id}/versions")
async def agent_list_workflow_versions(workflow_id: str) -> list[dict]:
    """Past versions of this workflow's graph/name/description, most
    recent first — archived automatically on every Save Edits or revert.
    See storage/agent_work.py's update_workflow docstring: a version is
    the row's content immediately BEFORE the update that just happened,
    not a snapshot of the update itself."""
    return await list_workflow_versions(workflow_id)


@app.post("/agent/workflows/{workflow_id}/versions/{version_id}/revert")
async def agent_revert_workflow(workflow_id: str, version_id: str, request: Request) -> JSONResponse:
    """Restores an old version's name/description/graph as the workflow's
    new live state — deliberately does NOT touch the workflow's current
    trigger (a webhook's token/URL, or a schedule), so reverting a graph
    edit can never silently break an already-configured integration. Like
    any other update, this itself archives what it's overwriting, so a
    revert is never a dead end — see revert_workflow_to_version's
    docstring in storage/agent_work.py."""
    payload = await request.json() if await request.body() else {}
    edited_by = payload.get("edited_by")
    reverted = await revert_workflow_to_version(workflow_id, version_id, edited_by)
    if not reverted:
        return JSONResponse({"error": "not found"}, status_code=404)
    return JSONResponse(await get_workflow(workflow_id))


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


@app.get("/agent/workflows/{workflow_id}/nodes/{node_id}/sample")
async def agent_node_sample_output(workflow_id: str, node_id: str) -> JSONResponse:
    """Real, read-only convenience for the canvas's "insert reference"
    picker (2026-09-07) — the most recent actual output this node
    produced, so a field referencing it (e.g. a Send Email node's "To")
    can browse real JSON keys instead of guessing a webhook payload's
    shape blind. `output` is None when this node has never completed a
    run yet — the picker falls back to a plain whole-value reference in
    that case, same "test it once, then map real fields" flow Zapier's
    own product requires."""
    output = await get_latest_node_output(workflow_id, node_id)
    return JSONResponse({"output": output})


@app.delete("/agent/workflows/{workflow_id}")
async def agent_delete_workflow(workflow_id: str) -> JSONResponse:
    """Soft-deletes the workflow definition (2026-09-08) — recoverable via
    the /restore route below, not gone. This is also the entire "cancel
    its schedule/webhook" operation — see delete_workflow's docstring in
    storage/agent_work.py for why nothing else needs to be touched. Past
    runs/steps/versions for it are kept, not cascade-deleted (real audit
    trail) — permanent removal is the separate, more severe /purge route."""
    deleted = await delete_workflow_definition(workflow_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="not found")
    return JSONResponse({"deleted": True})


@app.post("/agent/workflows/{workflow_id}/restore")
async def agent_restore_workflow(workflow_id: str) -> JSONResponse:
    """Undoes a soft-delete — the workflow reappears in the normal list
    and, if it has a schedule/webhook trigger, becomes eligible to fire
    again on the next poll."""
    restored = await restore_workflow(workflow_id)
    if not restored:
        raise HTTPException(status_code=404, detail="not found or not deleted")
    return JSONResponse({"restored": True})


@app.delete("/agent/workflows/{workflow_id}/purge")
async def agent_purge_workflow(workflow_id: str) -> JSONResponse:
    """Real, permanent erasure — unlike the soft DELETE above, this
    actually removes the row and its full history (runs, steps,
    versions). Meant for an Owner-tier admin view once real roles exist
    (today, gated by nothing beyond the single shared API key everyone
    already has — an honest, known stopgap, not real access control).
    Best-effort also removes any Agent Vault entry starred against this
    workflow — a separate DB file (agents.db), so this can't be one
    atomic transaction with the purge itself; a leftover saved_agents row
    pointing at a now-gone workflow_id is a harmless dangling reference
    the UI already has to treat as "instructions only," not a correctness
    bug, so this cleanup step failing silently is an acceptable trade-off
    here rather than blocking the purge on it."""
    agent = await get_agent_by_workflow_id(workflow_id)
    if agent:
        await delete_agent(agent["id"])
    purged = await purge_workflow(workflow_id)
    if not purged:
        raise HTTPException(status_code=404, detail="not found")
    return JSONResponse({"purged": True})


@app.post("/agent/workflows/{workflow_id}/run")
async def agent_run_workflow(workflow_id: str) -> JSONResponse:
    """Manual trigger — returns the new run's id immediately, execution
    continues on a background thread (see dispatcher/agent_work.py)."""
    try:
        run_id = await start_workflow_run(workflow_id, trigger_source="manual")
    except WorkflowError as e:
        return JSONResponse({"error": str(e)}, status_code=404)
    return JSONResponse({"run_id": run_id})


@app.post("/agent/workflows/{workflow_id}/webhook")
async def agent_set_webhook_trigger(workflow_id: str) -> JSONResponse:
    """Idempotently attaches (or returns the already-attached) webhook
    trigger for this workflow — an authenticated call, normal REST under
    the shared API key, unlike the public route below it actually
    triggers. See set_webhook_trigger's own docstring for why this never
    rotates an existing token."""
    try:
        token = await set_webhook_trigger(workflow_id)
    except WorkflowError as e:
        return JSONResponse({"error": str(e)}, status_code=404)
    return JSONResponse({"url": f"{NAVI_BASE_URL}/agent/webhooks/{token}"})


@app.post("/agent/webhooks/{token}")
async def agent_webhook_trigger(token: str, request: Request) -> JSONResponse:
    """The actual externally-called endpoint — exempt from the shared
    NAVI_API_KEY gate (see _PUBLIC_PATHS's startswith check above); the
    unguessable token in the path is the real credential here, checked
    against get_workflow_by_webhook_token before anything else happens.

    Acks immediately and runs the workflow async — Stripe-style, same
    pattern /research already uses — UNLESS the graph contains a Respond
    to Webhook node (2026-09-07), in which case this holds the HTTP
    connection open and waits for that node to actually run
    (start_webhook_run only registers the wait in that case — see its
    own call to register_response_waiter; a workflow with no such node
    is completely unaffected, same immediate-ack behavior as before).
    asyncio.wrap_future bridges the run's own background thread (a
    concurrent.futures.Future, not an asyncio one — see
    dispatcher/agent_work.py's _webhook_waiters docstring) back onto
    this request's event loop."""
    workflow = await get_workflow_by_webhook_token(token)
    if not workflow:
        return JSONResponse({"error": "unknown webhook"}, status_code=404)
    try:
        payload = await request.json()
    except Exception:
        payload = (await request.body()).decode("utf-8", errors="replace")
    run_id = await start_webhook_run(workflow, payload)

    waiter = peek_webhook_waiter(run_id)
    if waiter is None:
        return JSONResponse({"ok": True, "run_id": run_id})

    try:
        result = await asyncio.wait_for(asyncio.wrap_future(waiter), timeout=WEBHOOK_RESPONSE_TIMEOUT_SECONDS)
    except asyncio.TimeoutError:
        discard_webhook_waiter(run_id)
        return JSONResponse(
            {"ok": True, "run_id": run_id, "note": "still running past the response timeout — the workflow keeps executing"},
            status_code=202,
        )
    body = result["body"]
    try:
        content = json.loads(body)
    except (TypeError, ValueError):
        content = body
    return JSONResponse(content, status_code=result["status_code"])


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


@app.post("/agent/runs/{run_id}/cancel")
async def agent_cancel_run(run_id: str) -> JSONResponse:
    """Safe cancellation (2026-09-08) — flips an in-memory flag the run's
    own background thread checks between nodes (see dispatcher/agent_work.py's
    request_run_cancellation/_execute_run for why "between nodes," not
    mid-node). Returns immediately; the run stops at its own next safe
    checkpoint, not instantly — a currently-executing node (or a Delay
    node's own wait) finishes first. 404s only if the run genuinely
    doesn't exist; requesting cancellation on an already-finished run is a
    harmless no-op (the flag is simply never checked again)."""
    run = await get_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="not found")
    request_run_cancellation(run_id)
    return JSONResponse({"ok": True})


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
async def mcp_marketplace_search(q: str = "", limit: int = 20, cursor: str | None = None) -> JSONResponse:
    """Real-time proxy over the official MCP Registry (tools/
    mcp_marketplace.py) — lets ConnectionsOverlay show actual, searchable
    MCP servers instead of only the fixed CORE_SERVICES list. Read-only, no
    connection is made here; a result just pre-fills the existing
    /mcp/connections + /connect flow below with real transport info.
    `cursor` (from a previous response's `next_cursor`) pages through
    results — the registry's own empty-query listing is recency-only, not
    curated, so there's no natural "end" short of paging through it."""
    try:
        results, next_cursor = await asyncio.to_thread(search_mcp_marketplace, q, min(limit, 50), cursor)
    except MCPMarketplaceError as e:
        return JSONResponse({"error": str(e)}, status_code=502)
    return JSONResponse({"results": results, "next_cursor": next_cursor})


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
        print(f"[mcp_oauth_start] discovery/registration failed for '{name}': {e}")
        return JSONResponse({"error": str(e)}, status_code=502)
    print(f"[mcp_oauth_start] '{name}': redirecting to {flow['authorize_url'][:80]}…")
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
    if request.query_params.get("error"):
        print(f"[mcp_oauth_callback] provider returned an error: {dict(request.query_params)}")
        return RedirectResponse(f"{PWA_ORIGIN}/?mcp_oauth=error")
    if not pending:
        print(f"[mcp_oauth_callback] no pending flow for state={state!r} — expired, already used, or the server restarted mid-flow")
        return RedirectResponse(f"{PWA_ORIGIN}/?mcp_oauth=error")
    if not code:
        print(f"[mcp_oauth_callback] no code in callback params: {dict(request.query_params)}")
        return RedirectResponse(f"{PWA_ORIGIN}/?mcp_oauth=error")

    try:
        token = await asyncio.to_thread(
            exchange_code_for_token, pending["token_endpoint"], code, pending["code_verifier"],
            pending["client_id"], pending["client_secret"], pending["redirect_uri"],
        )
    except MCPOAuthError as e:
        print(f"[mcp_oauth_callback] token exchange failed for '{pending['server_name']}': {e}")
        return RedirectResponse(f"{PWA_ORIGIN}/?mcp_oauth=error")

    server_name = pending["server_name"]
    conn = config.get_mcp_connection(server_name)
    if conn is None:
        print(f"[mcp_oauth_callback] connection '{server_name}' vanished between /oauth/start and this callback")
        return RedirectResponse(f"{PWA_ORIGIN}/?mcp_oauth=error")
    # Persists everything needed to silently refresh later (2026-09-06) —
    # not just the access token, the way this used to work. A server that
    # never issues a refresh_token (GitHub) just stores None for it;
    # ensure_fresh_access_token treats that as "nothing to refresh,"
    # not an error.
    expires_at = time.time() + token["expires_in"] if token.get("expires_in") else None
    config.set_mcp_oauth_tokens(
        server_name, access_token=token["access_token"], refresh_token=token.get("refresh_token"),
        expires_at=expires_at, token_endpoint=pending["token_endpoint"],
        client_id=pending["client_id"], client_secret=pending["client_secret"],
    )
    print(f"[mcp_oauth_callback] token exchange succeeded for '{server_name}' (refresh_token={'yes' if token.get('refresh_token') else 'no'}), discovering tools next")

    try:
        discovered = await asyncio.to_thread(discover_tools, server_name)
        new_tools = [t for t in discovered if t["status"] == "new"]
        if new_tools:
            approve_tools(server_name, new_tools)
        config.set_mcp_connected(server_name, True)
        print(f"[mcp_oauth_callback] '{server_name}' fully connected, {len(discovered)} tool(s) discovered")
    except MCPError as e:
        print(f"[mcp_oauth_callback] token exchange succeeded but discover_tools failed for '{server_name}': {e}")
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
