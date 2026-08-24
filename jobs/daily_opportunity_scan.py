"""
jobs/daily_opportunity_scan.py

Runs from .github/workflows/daily_opportunity_scan.yml. Scans two lanes
for JuanJo, specifically at junior/entry level:

  1. Games / systems / economy design opportunities
  2. UX-in-tech / domotics / robotics opportunities

Uses groq/compound (fallback: groq/compound-mini) rather than the shared
dispatcher_autonomous role (gpt-oss-120b, still used by the model-digest
job). Compound has native, automatic web search built into the model
itself — Groq's own infrastructure does the searching server-side, no
tools/config needed — which sidesteps tools/search.py's DuckDuckGo
scraper entirely for this job. That scraper already broke once in
production (DuckDuckGo blocking Render's outbound IP), so this job no
longer depends on it. Primary/fallback is handled directly in this
script rather than the shared dispatcher-role system, which doesn't
support a fallback chain yet — scoped to just this one job rather than
restructuring that shared piece.

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
from providers.registry import ProviderNotConfigured, get_provider  # noqa: E402

# Primary first, fallback only on failure — both are Groq's search-capable
# "compound" models, just different sizes. See module docstring for why
# this bypasses the shared dispatcher-role fallback (which doesn't exist
# yet) rather than extending it for one job.
COMPOUND_MODELS = ["groq/compound", "groq/compound-mini"]

LANES = [
    (
        "Games / systems / economy design (junior/entry-level)",
        [
            "junior game designer job posting remote",
            "entry level systems designer game studio hiring",
            "associate game economy designer job",
            "junior game designer job linkedin OR hitmarker OR wellfound",
        ],
    ),
    (
        "UX-in-tech / domotics / robotics (junior/entry-level)",
        [
            "junior UX designer robotics job posting",
            "entry level UX designer smart home domotics hiring",
            "junior product designer IoT job",
            "associate UX designer job remote europe",
        ],
    ),
]

PROMPT_TEMPLATE = """You are scanning the live web for concrete opportunities for a JUNIOR/\
entry-level UX and game designer job-hunting across the EU and US. Lane: "{lane}".

Actually search the web (you have that capability) using queries like these as a starting \
point, and vary them as needed to find real, currently-open postings:
{queries}

STRICT RULES:
1. Report ONLY verifiable, concrete signals you actually found via search — a job posting \
with a real link, an open-source issue with a link, or a CFP/event with a date. Do NOT infer \
or diagnose a company's problems ("Company X seems to struggle with Y") — that kind of \
unverifiable guess is explicitly forbidden.
2. Prioritize roles that are explicitly junior/entry-level/associate, or that don't require \
years of professional experience. Skip senior/lead/staff-level postings entirely.
3. If a result isn't a concrete, linkable signal, or isn't actually junior-appropriate, leave \
it out entirely.

If there are no qualifying concrete signals at all, say exactly: "No concrete signals found \
today for this lane." Do not pad with speculation to avoid saying that.

Format each qualifying signal as one line: "- [type] title — link"
"""


def _scan_lane(lane_name: str, queries: list[str]) -> str:
    prompt = PROMPT_TEMPLATE.format(lane=lane_name, queries="\n".join(f"- {q}" for q in queries))

    last_error = None
    for model in COMPOUND_MODELS:
        try:
            provider = get_provider("groq")
            response = provider.chat(model=model, messages=[ChatMessage(role="user", content=prompt)])
            return (response.text or "").strip()
        except ProviderNotConfigured as e:
            return f"Scan failed for this lane (Groq not configured: {e})."
        except ProviderError as e:
            last_error = str(e)
            continue

    return f"Scan failed for this lane on both compound models ({last_error}) — disclosed rather than silently skipped."


def build_report() -> str:
    sections = [f"Opportunity scan — {__import__('datetime').date.today().isoformat()}"]
    for lane_name, queries in LANES:
        sections.append(f"\n{lane_name}:")
        sections.append(_scan_lane(lane_name, queries))
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
