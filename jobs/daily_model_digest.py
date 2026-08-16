"""
jobs/daily_model_digest.py

Runs from .github/workflows/daily_model_digest.yml on a schedule (and via
workflow_dispatch for manual testing). Uses dispatcher_autonomous — the
low-stakes Groq role, deliberately kept off the interactive chat quota —
to read today's free-model list from ClawLabs' tracker and produce the
digest message in the exact agreed format.

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

FREE_MODELS_SOURCE = "https://raw.githubusercontent.com/ClawLabsAI/free-ai-models/main/README.md"

CAPABILITIES = [
    ("research", "/research"),
    ("Image-generation", "/create-image"),
    ("graph-data", "/graph-data"),
    ("code", "/code"),
    ("brainstorm", "/brainstorm"),
]

PROMPT_TEMPLATE = """You are given today's snapshot of a free-tier AI model tracker. \
Based ONLY on what's in the snapshot below, pick the single best current free model \
for each of these five capabilities, plus up to two fallback models each: \
research, image-generation, graph-data (data analysis / structured output), code, \
brainstorm (general creative/conversational reasoning).

If a capability genuinely has no free option listed in the snapshot, say so plainly \
instead of guessing or omitting it.

Reply in EXACTLY this format, one line per capability, no extra commentary:
research: MODEL_NAME; fallbacks: MODEL_NAME, MODEL_NAME
Image-generation: MODEL_NAME; fallbacks: MODEL_NAME
graph-data: MODEL_NAME; fallbacks: MODEL_NAME
code: MODEL_NAME; fallbacks: MODEL_NAME
brainstorm: MODEL_NAME; fallbacks: MODEL_NAME

If none available for a line, write "none available today" in place of MODEL_NAME \
(no fallbacks clause needed on that line).

Snapshot:
{snapshot}
"""


def _fetch_snapshot() -> str | None:
    try:
        resp = requests.get(FREE_MODELS_SOURCE, timeout=30)
        if resp.status_code >= 400:
            return None
        return resp.text[:12000]  # keep prompt size sane
    except requests.RequestException:
        return None


def build_digest() -> str:
    snapshot = _fetch_snapshot()
    if snapshot is None:
        return (
            "Today's AIs are:\n"
            "Could not reach the free-model tracker (github.com/ClawLabsAI/free-ai-models) "
            "today — no digest generated. This is disclosed rather than guessed at."
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
