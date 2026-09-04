"""
jobs/daily_model_digest.py

Runs from .github/workflows/daily_model_digest.yml on a schedule (and via
workflow_dispatch for manual testing). Uses dispatcher_autonomous — the
low-stakes Groq role, deliberately kept off the interactive chat quota —
to read today's free-model list and produce the digest message in the
exact agreed format.

Switched 2026-09-01 from ClawLabs' third-party GitHub tracker to
OpenRouter's own live `GET /api/v1/models` endpoint (per OpenRouter's own
quickstart docs — this is the documented way to list available slugs
programmatically, no auth required). The tracker was a proxy for
OpenRouter's actual catalog and could lag behind it — which is exactly
what happened to `openai/gpt-oss-120b:free` going 404 while still routed
in config/store.py's `brainstorm` entry: the tracker either never caught
the removal or nobody re-ran the digest against it in time. Querying
OpenRouter directly removes that lag entirely — the digest now reflects
exactly what OpenRouter will actually accept, not a second-hand snapshot
of it.

This does NOT write the result back into config/store.py's task_routing.
That "daily-refreshed live ranking" auto-wiring was explicitly marked
optional in the brief ("build it if time allows") and touches production
routing config from a separate ephemeral GitHub Actions process — that's
a bigger, riskier change than this pass covers. This job is read-only:
it informs JuanJo via Telegram, he updates routing by chatting with NAVI
if he wants to act on it. Flagging the gap here rather than half-wiring it.
"""

import sys
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from messaging.base import MessagingError  # noqa: E402
from messaging.telegram import TelegramAdapter  # noqa: E402
from providers.base import ChatMessage, ProviderError  # noqa: E402
from providers.registry import ProviderNotConfigured, get_dispatcher  # noqa: E402

OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models"

CAPABILITIES = [
    ("research", "/research"),
    ("graph-data", "/graph-data"),
    ("brainstorm", "/brainstorm"),
]

PROMPT_TEMPLATE = """You are given today's LIVE free-tier model list, pulled directly from \
OpenRouter's own API (not a third-party tracker — this is exactly what OpenRouter will \
actually accept right now). Each line is: model_id (context: N tokens, tools: yes/no).

Based ONLY on what's in the snapshot below, pick the single best current free model \
for each of these three capabilities, plus up to two fallback models each: \
research, graph-data (data analysis / structured output), \
brainstorm (general creative/conversational reasoning).

graph-data and brainstorm's "remind" sibling capability both require forced tool-calling \
(tool_choice) in NAVI's own routing — for graph-data specifically, only pick models marked \
"tools: yes". The other capabilities don't have that requirement.

If a capability genuinely has no free option listed in the snapshot, say so plainly \
instead of guessing or omitting it. Only use a model_id that appears verbatim in the \
snapshot below — never a model you recall from training, even if it sounds plausible.

Reply in EXACTLY this format, one line per capability, no extra commentary:
research: MODEL_NAME; fallbacks: MODEL_NAME, MODEL_NAME
graph-data: MODEL_NAME; fallbacks: MODEL_NAME
brainstorm: MODEL_NAME; fallbacks: MODEL_NAME

If none available for a line, write "none available today" in place of MODEL_NAME \
(no fallbacks clause needed on that line).

Snapshot:
{snapshot}
"""


def _fetch_snapshot() -> str | None:
    """Pulls OpenRouter's live model catalog and filters to genuinely free
    entries (pricing.prompt == pricing.completion == "0") — the same field
    OpenRouter's own pricing pages read from, so this can't drift out of
    sync with what a real request will actually be charged (or rejected
    for). Capped at 200 models in the snapshot to keep the prompt a
    sane size; OpenRouter's free tier has never come close to that."""
    try:
        resp = requests.get(OPENROUTER_MODELS_URL, timeout=30)
        if resp.status_code >= 400:
            return None
        models = (resp.json() or {}).get("data") or []
    except (requests.RequestException, ValueError):
        return None

    lines = []
    for m in models:
        pricing = m.get("pricing") or {}
        if pricing.get("prompt") != "0" or pricing.get("completion") != "0":
            continue  # not actually free — a "0.000000834"-style paid entry
        tools = "yes" if "tools" in (m.get("supported_parameters") or []) else "no"
        lines.append(f"{m.get('id')} (context: {m.get('context_length')} tokens, tools: {tools})")
        if len(lines) >= 200:
            break

    return "\n".join(lines) if lines else None


def build_digest() -> str:
    snapshot = _fetch_snapshot()
    if snapshot is None:
        return (
            "Today's AIs are:\n"
            "Could not reach OpenRouter's live model list (openrouter.ai/api/v1/models) "
            "today, or no free models were found in it — no digest generated. "
            "This is disclosed rather than guessed at."
        )

    try:
        provider, model = get_dispatcher(context="autonomous")
    except ProviderNotConfigured as e:
        return f"Today's AIs are:\nDigest generation failed (dispatcher not configured: {e})."

    prompt = PROMPT_TEMPLATE.format(snapshot=snapshot)

    try:
        response = provider.chat(model=model, messages=[ChatMessage(role="user", content=prompt)])
    except ProviderError as e:
        return f"Today's AIs are:\nDigest generation failed ({e}) — disclosed rather than silently skipped."

    body = (response.text or "").strip()
    return f"Today's AIs are:\n{body}"


def main() -> None:
    import os

    from config.store import config

    # CI runs as a fresh checkout with no agent_config.json and no rclone
    # installed — rather than depend on Filen being reachable just to run
    # a read-only digest job, seed the key straight from the GH Actions
    # secret. This still goes through the config store / registry (so the
    # provider stays swappable), it just sources the key differently here
    # than the chat-configured path does.
    groq_key = os.environ.get("GROQ_API_KEY")
    if groq_key:
        config.set_provider_key("groq", groq_key)

    digest = build_digest()

    bot_token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")
    if not bot_token or not chat_id:
        print(digest)
        raise SystemExit("TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID not set — printed digest instead of sending.")

    try:
        TelegramAdapter(bot_token).send_message(chat_id, digest)
    except MessagingError as e:
        print(digest)
        raise SystemExit(str(e))

    print("Digest sent.")


if __name__ == "__main__":
    main()
