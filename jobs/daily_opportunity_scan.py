"""
jobs/daily_opportunity_scan.py

Runs from .github/workflows/daily_opportunity_scan.yml. Uses
dispatcher_autonomous (Groq) to scan two lanes for JuanJo:

  1. Games / systems / economy design opportunities
  2. UX-in-tech / domotics / robotics opportunities

Hard rule carried over from the design conversation: this reports
CONCRETE, VERIFIABLE signals only — job postings with links, open-source
issues with links, CFPs with dates. It must NEVER produce an inferred
"company X probably has problem Y" diagnosis — that framing was
explicitly rejected as presumptuous and unverifiable. The prompt below
enforces that rule directly; if the model can't find real signals, it
says so instead of inventing plausible-sounding ones.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from messaging.base import MessagingError  # noqa: E402
from messaging.telegram import TelegramAdapter  # noqa: E402
from providers.base import ChatMessage, ProviderError  # noqa: E402
from providers.registry import ProviderNotConfigured, get_dispatcher  # noqa: E402
from tools.search import SearchError, web_search  # noqa: E402

LANES = [
    (
        "Games / systems / economy design",
        [
            "game designer job posting remote",
            "systems designer game studio hiring",
            "game economy designer job",
        ],
    ),
    (
        "UX-in-tech / domotics / robotics",
        [
            "UX designer robotics job posting",
            "UX designer smart home domotics hiring",
            "product designer IoT job",
        ],
    ),
]

PROMPT_TEMPLATE = """You are scanning for concrete opportunities for a junior UX/game designer \
job-hunting across the EU and US. Below are raw web search results for the lane "{lane}".

STRICT RULE: report ONLY verifiable, concrete signals that are explicitly present in the \
search results below — a job posting with a link, an open-source issue with a link, or a \
CFP/event with a date. Do NOT infer or diagnose a company's problems ("Company X seems to \
struggle with Y") — that kind of unverifiable guess is explicitly forbidden. If a result \
isn't a concrete, linkable signal, leave it out entirely.

If there are no qualifying concrete signals at all, say exactly: "No concrete signals found \
today for this lane." Do not pad with speculation to avoid saying that.

Format each qualifying signal as one line: "- [type] title — link"

Search results:
{results}
"""


def _search_lane(queries: list[str]) -> str:
    blocks = []
    for q in queries:
        try:
            results = web_search(q, max_results=5)
        except SearchError as e:
            blocks.append(f"(search failed for '{q}': {e})")
            continue
        for r in results:
            blocks.append(f"- {r['title']} ({r['url']}): {r['snippet']}")
    return "\n".join(blocks) if blocks else "(no search results)"


def _scan_lane(provider, model, lane_name: str, queries: list[str]) -> str:
    results_text = _search_lane(queries)
    prompt = PROMPT_TEMPLATE.format(lane=lane_name, results=results_text)
    try:
        response = provider.chat(model=model, messages=[ChatMessage(role="user", content=prompt)])
    except ProviderError as e:
        return f"Scan failed for this lane ({e}) — disclosed rather than silently skipped."
    return (response.text or "").strip()


def build_report() -> str:
    try:
        provider, model = get_dispatcher(context="autonomous")
    except ProviderNotConfigured as e:
        return f"Opportunity scan failed (dispatcher not configured: {e})."

    sections = [f"Opportunity scan — {__import__('datetime').date.today().isoformat()}"]
    for lane_name, queries in LANES:
        sections.append(f"\n{lane_name}:")
        sections.append(_scan_lane(provider, model, lane_name, queries))
    return "\n".join(sections)


def main() -> None:
    import os

    from config.store import config

    groq_key = os.environ.get("GROQ_API_KEY")
    if groq_key:
        config.set_provider_key("groq", groq_key)

    report = build_report()

    bot_token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")
    if not bot_token or not chat_id:
        print(report)
        raise SystemExit("TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID not set — printed report instead of sending.")

    try:
        TelegramAdapter(bot_token).send_message(chat_id, report)
    except MessagingError as e:
        print(report)
        raise SystemExit(str(e))

    print("Opportunity scan sent.")


if __name__ == "__main__":
    main()
