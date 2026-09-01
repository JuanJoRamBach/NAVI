"""
jobs/daily_opportunity_scan.py

Runs from .github/workflows/daily_opportunity_scan.yml. Scans two lanes
for JuanJo, specifically at junior/entry level:

  1. Games / systems / economy design opportunities
  2. UX-in-tech / domotics / robotics opportunities

Redesigned 2026-09-01 to search first, ask the model second — this job's
queries were always hardcoded in LANES, never something the model needed
to decide, so routing search through an LLM tool-call loop (first via
groq/compound's built-in search, then via explicit web_search tool
calling) was solving a problem this job doesn't have. Two real problems
that design created, both gone now:

  1. groq/compound(-mini) returned 413 request_too_large on this job's
     own lanes — a known Groq platform issue where compound systems
     stitch fetched search results into the model's internal context
     before replying, and that internal payload (not the small prompt
     this job sends) is what tripped their size cap.
  2. A tool-call loop resends the ENTIRE growing message history on every
     iteration (see run_tool_loop in dispatcher/executor.py) — token cost
     compounds roughly with the square of iteration count, all within the
     same rate-limit minute. qwen3.6-27b's 8k-tokens/minute cap made that
     a real risk, not a hypothetical one.

Now: web_search() runs directly in Python for each lane's queries (real
results, no model involvement in *finding* them), filtered to the last
24 hours via web_search's `days` param — filtering at the source instead
of asking the model to guess freshness from a snippet that usually has no
clear date in it. The model gets ONE call per lane: real gathered
material in, a filtered/formatted report out. No loop, no resent history,
no compounding tokens.

Still reintroduces tools/search.py's DuckDuckGo scraper, which the
original compound-model design was chosen to avoid (it broke once in
production — DuckDuckGo blocking Render's outbound IP). This job runs on
GitHub Actions' shared runners, not Lightsail — a different but similarly
commonly-blocked range — so a SearchError here is a real risk, handled
per-query below rather than failing the whole lane.

Primary/fallback is handled directly in this script rather than the
shared dispatcher-role system, which doesn't support a fallback chain
yet — scoped to just this one job rather than restructuring that shared
piece.

Hard rule carried over from the design conversation: this reports
CONCRETE, VERIFIABLE signals only — job postings with links, open-source
issues with links, CFPs with dates. It must NEVER produce an inferred
"company X probably has problem Y" diagnosis — that framing was
explicitly rejected as presumptuous and unverifiable. Now structurally
enforced, not just prompted: the model only ever sees real search results
it's asked to filter/format, it has no path to inventing a signal that
was never actually found.
"""

import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from messaging.base import MessagingError  # noqa: E402
from messaging.telegram import TelegramAdapter  # noqa: E402
from providers.base import ChatMessage, ProviderError  # noqa: E402
from providers.registry import ProviderNotConfigured, get_provider  # noqa: E402
from tools.search import SearchError, web_search  # noqa: E402

# Fallback stays a plain second model, not another size of the same
# system — there's no "compound-mini" equivalent anymore since neither
# model does its own searching.
SCAN_MODELS = ["qwen/qwen3.6-27b", "openai/gpt-oss-120b"]

# DuckDuckGo's own "past day" filter — see tools/search.py's `days` param.
FRESHNESS_DAYS = 1

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

PROMPT_TEMPLATE = """You are filtering ALREADY-GATHERED search results down to concrete \
opportunities for a JUNIOR/entry-level UX and game designer job-hunting across the EU and US. \
Lane: "{lane}". Today's date: {today}.

Below is the real material found via search, filtered to roughly the last {days} day(s). You \
are NOT searching yourself — only select and format from what's actually listed here.

{material}

STRICT RULES:
1. Report ONLY signals that appear in the material above, with their real link. Do NOT infer \
or diagnose a company's problems ("Company X seems to struggle with Y") — that kind of \
unverifiable guess is explicitly forbidden. Do NOT invent or assume anything not present above.
2. Prioritize roles that are explicitly junior/entry-level/associate, or that don't require \
years of professional experience. Skip senior/lead/staff-level postings entirely.
3. If an item in the material isn't a concrete, linkable signal, or isn't actually \
junior-appropriate, leave it out entirely.

If nothing in the material qualifies, say exactly: "No concrete signals found today for this \
lane." Do not pad with speculation to avoid saying that.

Format each qualifying signal as one line: "- [type] title — link"
"""


def _gather_lane_material(queries: list[str]) -> str:
    """Runs every query for a lane through web_search directly (no model
    involved in finding results), filtered to FRESHNESS_DAYS. A query that
    fails (e.g. SearchError from a blocked scrape) is noted inline rather
    than failing the whole lane — the other queries' results still reach
    the model."""
    blocks = []
    for q in queries:
        try:
            results = web_search(q, max_results=5, days=FRESHNESS_DAYS)
        except SearchError as e:
            blocks.append(f'Query "{q}": search failed ({e}) — no results available for this query.')
            continue
        if not results:
            blocks.append(f'Query "{q}": no results.')
            continue
        lines = [f"- {r['title']} ({r['url']}): {r['snippet']}" for r in results]
        blocks.append(f'Query "{q}":\n' + "\n".join(lines))
    return "\n\n".join(blocks)


def _scan_lane(lane_name: str, queries: list[str]) -> str:
    material = _gather_lane_material(queries)
    if not material.strip():
        return "No concrete signals found today for this lane."

    prompt = PROMPT_TEMPLATE.format(
        lane=lane_name, today=date.today().isoformat(), days=FRESHNESS_DAYS, material=material,
    )

    last_error = None
    for model in SCAN_MODELS:
        try:
            provider = get_provider("groq")
            response = provider.chat(model=model, messages=[ChatMessage(role="user", content=prompt)])
            return (response.text or "").strip()
        except ProviderNotConfigured as e:
            return f"Scan failed for this lane (Groq not configured: {e})."
        except ProviderError as e:
            last_error = str(e)
            continue

    return f"Scan failed for this lane on all models ({last_error}) — disclosed rather than silently skipped."


LANE_COOLDOWN_SECONDS = 120  # let one lane's RPM window clear before starting the next


def build_report() -> str:
    import time

    sections = [f"Opportunity scan — {date.today().isoformat()}"]
    for i, (lane_name, queries) in enumerate(LANES):
        if i > 0:
            time.sleep(LANE_COOLDOWN_SECONDS)
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
