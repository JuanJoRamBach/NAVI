"""
jobs/benchmark.py

What every LLM call inside NAVI actually costs, measured rather than
assumed.

WHY THIS EXISTS. NAVI has never measured its own variance. Not once. A
7,393-token chat turn was found on 2026-09-13 because someone happened to
read a number off the screen, and a 60-second timeout failed silently for
a day before anyone reproduced it. The prompt was a finding from the
"Recursive by Design" essay: an open agent loop there burned between 732K
and 1.54M tokens on IDENTICAL runs — a 2.1x spread that was invisible
until somebody measured it.

A single run tells you almost nothing, which is the whole point. Each
scenario runs REPS times and the spread between them is the number worth
reading. A scenario whose best and worst runs differ by 2x is not
"roughly N tokens"; it is unpredictable, and that is a finding.

WHAT IT MEASURES. Real API calls against the live routing config. No
stubs anywhere — a stubbed benchmark measures the stub. That means this
spends real quota: REPS x scenarios calls, on whichever providers the
config actually routes to. Run it deliberately.

Every run records prompt / cached / completion / total tokens, wall
clock, and which provider+model actually answered — the last one matters
because a fallback firing mid-benchmark changes what is being measured,
and would otherwise look like variance in the primary.

CACHED TOKENS are captured here and nowhere else in NAVI. providers/
base.py records prompt/completion/total into storage/usage.py but drops
the cached count, even though most providers report it. Cached input is
the single biggest lever on real cost, so a cost benchmark that ignored
it would be measuring the wrong thing.

    python -m jobs.benchmark --check          # are the keys actually working?
    python -m jobs.benchmark --list           # what can be measured
    python -m jobs.benchmark                  # every scenario, 5 reps
    python -m jobs.benchmark chat_idle        # one scenario
    python -m jobs.benchmark --reps 3

Start with --check. It sends a one-word prompt to every provider the chat
tiers route to and reports what came back, because a key being PRESENT and
a key WORKING are different facts, and the gap between them is only
discovered at the first real call otherwise. --check --no-live checks
presence alone and spends nothing.
"""

from __future__ import annotations

import asyncio
import json
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.dotenv import ensure_env  # noqa: E402 - must follow the path insert

# Load the server's own .env before anything reads a key. systemd never
# exports into an interactive shell, so a job run by hand has none of the
# secrets the running service has — see config/dotenv.py for the run this
# cost us.
_ENV_FILE = ensure_env()

OUT_DIR = Path(__file__).resolve().parent.parent / "benchmarks"
DEFAULT_REPS = 5


def _isolate_storage() -> None:
    """Conversations and context go to a scratch database for the run.

    Routing config, provider keys and usage counters are deliberately NOT
    isolated: the point is to measure what the LIVE configuration actually
    does, and the tokens spent here are real tokens that should appear in
    the real counters.

    What should not happen is a benchmark leaving dozens of junk
    conversations in the user's own history.
    """
    import tempfile

    import storage.context_store as ctx
    import storage.conversations as conv

    scratch = Path(tempfile.gettempdir()) / "navi_benchmark.db"
    if scratch.exists():
        scratch.unlink()
    conv.DB_PATH = scratch
    ctx.DB_PATH = scratch


# ---- Usage extraction -------------------------------------------------

def extract_usage(raw: dict | None) -> dict:
    """Pulls token counts out of a provider's raw response.

    Tolerant by design: every transport here is OpenAI-compatible, but
    they disagree about where the CACHED count lives, and some omit it
    entirely. A missing cached figure is reported as None, not 0 — zero
    is a claim that nothing was cached, and we frequently cannot tell.
    """
    usage = (raw or {}).get("usage") or {}
    details = usage.get("prompt_tokens_details") or {}
    cached = details.get("cached_tokens")
    if cached is None:
        cached = usage.get("cached_tokens")
    if cached is None:
        cached = usage.get("cache_read_input_tokens")
    return {
        "prompt": usage.get("prompt_tokens"),
        "cached": cached,
        "completion": usage.get("completion_tokens"),
        "total": usage.get("total_tokens"),
    }


# ---- Scenario plumbing ------------------------------------------------

SCENARIOS: dict[str, dict] = {}


def scenario(name: str, description: str):
    def register(fn):
        SCENARIOS[name] = {"fn": fn, "description": description}
        return fn
    return register


class Probe:
    """Wraps a provider so a scenario's real call can be observed without
    the scenario having to report anything itself.

    Necessary because most call sites here are several layers deep
    (run_stored_mode_chat, compact_conversation) and none of them return
    token counts. Patching at the provider boundary catches every call a
    scenario makes, including fallbacks and tool-loop continuations, which
    is exactly what should be counted.
    """

    def __init__(self):
        self.calls: list[dict] = []

    def wrap(self, provider):
        real_chat = provider.chat
        probe = self

        def chat(model, messages, **kwargs):
            t0 = time.time()
            try:
                response = real_chat(model, messages, **kwargs)
            except Exception as e:
                probe.calls.append({
                    "provider": provider.name, "model": model, "ok": False,
                    "error": str(e)[:200], "seconds": round(time.time() - t0, 2),
                })
                raise
            probe.calls.append({
                "provider": provider.name, "model": model, "ok": True,
                "seconds": round(time.time() - t0, 2),
                **extract_usage(response.raw),
            })
            return response

        provider.chat = chat
        return provider

    def totals(self) -> dict:
        ok = [c for c in self.calls if c.get("ok")]
        return {
            "calls": len(self.calls),
            "failed_calls": len(self.calls) - len(ok),
            "prompt": sum(c.get("prompt") or 0 for c in ok),
            # None, not 0, when NO call reported a cached figure — see
            # extract_usage on why that distinction is kept.
            "cached": (sum(c["cached"] for c in ok if c.get("cached") is not None)
                       if any(c.get("cached") is not None for c in ok) else None),
            "completion": sum(c.get("completion") or 0 for c in ok),
            # Fall back to prompt+completion when a provider omits
            # total_tokens. Prompt Guard does exactly that — it is a
            # classifier returning a probability, not a chat completion —
            # and trusting `total` alone reported it as ALL RUNS FAILED
            # when it had in fact run fine three times.
            "total": sum((c.get("total") or ((c.get("prompt") or 0) + (c.get("completion") or 0))) for c in ok),
            "models": sorted({f"{c['provider']}/{c['model']}" for c in self.calls}),
        }


def _patched(probe: Probe):
    """Installs the probe across every module that resolves providers.

    Each of these holds its own reference to get_provider, so patching
    the registry alone would miss most of them.
    """
    import providers.registry as registry
    import dispatcher.chat as chat_mod
    import dispatcher.compaction as compaction_mod
    import dispatcher.executor as executor_mod
    import dispatcher.research as research_mod
    import dispatcher.source_ingest as ingest_mod
    import tools.content_safety as safety_mod

    real = registry.get_provider
    # Every module that did `from providers.registry import get_provider`
    # holds its OWN reference, so patching the registry alone misses them.
    # content_safety was measured at zero calls until it was added here —
    # a silent undercount, which is worse in a benchmark than a loud
    # failure. If a scenario reports 0 calls, suspect this list first.
    targets = [registry, chat_mod, compaction_mod, executor_mod, research_mod, ingest_mod, safety_mod]
    originals = [(m, getattr(m, "get_provider", None)) for m in targets]

    def patched(name):
        return probe.wrap(real(name))

    for m in targets:
        if hasattr(m, "get_provider"):
            m.get_provider = patched
    return originals


def _restore(originals):
    for module, fn in originals:
        if fn is not None:
            module.get_provider = fn


# ---- The scenarios ----------------------------------------------------

SHORT_QUESTION = "What is the capital of Portugal?"
ANALYTIC_QUESTION = (
    "Compare optimistic and pessimistic concurrency control for a multi-user "
    "document editor, and say which you would pick and why."
)


async def _fresh_conversation(turns: int = 0):
    from storage.conversations import append_message, create_conversation
    cid = await create_conversation(mode="normal")
    for i in range(turns):
        await append_message(cid, "user", f"Question {i} about scheduling and calendars.")
        await append_message(cid, "navi", f"Answer {i}. " + ("Some substantive detail. " * 25))
    return cid


@scenario("chat_idle", "One short question, idle tier — the most common turn there is")
async def _chat_idle():
    from dispatcher.chat import run_stored_mode_chat
    cid = await _fresh_conversation()
    await run_stored_mode_chat("normal", cid, SHORT_QUESTION)


@scenario("chat_idle_with_history", "Same question, but on a conversation with 20 prior messages")
async def _chat_history():
    from dispatcher.chat import run_stored_mode_chat
    cid = await _fresh_conversation(turns=10)
    await run_stored_mode_chat("normal", cid, SHORT_QUESTION)


@scenario("chat_exploratory", "A question needing real analysis, exploratory tier")
async def _chat_exploratory():
    from dispatcher.chat import run_stored_mode_chat
    cid = await _fresh_conversation()
    await run_stored_mode_chat("normal", cid, ANALYTIC_QUESTION, tier="chat_exploratory")


@scenario("chat_serious", "The same question on the top tier — what escalation actually costs")
async def _chat_serious():
    from dispatcher.chat import run_stored_mode_chat
    cid = await _fresh_conversation()
    await run_stored_mode_chat("normal", cid, ANALYTIC_QUESTION, tier="chat_serious")


@scenario("context_compaction", "Consolidating a full context.md — the synthesis role")
async def _compaction():
    from dispatcher.compaction import compact_context
    from storage.context_store import append_entry
    cid = await _fresh_conversation(turns=3)
    for i in range(60):
        await append_entry(cid, f"The team ships widgets from warehouse {i} on Tuesdays.")
    await compact_context(cid)


@scenario("branch_spec", "Drafting a branch's scoped spec from a parent conversation")
async def _branch_spec():
    from dispatcher.branch import draft_branch_spec
    cid = await _fresh_conversation(turns=6)
    await draft_branch_spec(cid, "calendar scheduling rewrite")


@scenario("source_distil", "Reading one web page into a Source document")
async def _source_distil():
    from dispatcher.compaction import compact_conversation
    from tools.fetch import SOURCE_MAX_CHARS, fetch_document
    from tools.source_document import SOURCE_DOCUMENT_INSTRUCTION, build_structuring_messages
    page = await asyncio.to_thread(
        fetch_document, "https://en.wikipedia.org/wiki/Dijkstra%27s_algorithm", SOURCE_MAX_CHARS,
    )
    msgs = build_structuring_messages(page["title"], page["url"], page["markdown"])
    await compact_conversation(msgs, SOURCE_DOCUMENT_INSTRUCTION)


@scenario("source_grounding", "One claim adjudicated by safeguard-20b against its passage")
async def _grounding():
    from dispatcher.source_ingest import _ask_grounding
    from tools.source_document import load_grounding_policy
    await asyncio.to_thread(
        _ask_grounding, load_grounding_policy(),
        "A production-grade harness contains five layers.",
        ["A production-grade harness contains five layers: tool orchestration, "
         "verification loops, context and memory, guardrails, and observability."],
    )


@scenario("content_safety", "Screening one fetched page through Prompt Guard")
async def _safety():
    from tools.content_safety import screen_content
    await asyncio.to_thread(
        screen_content,
        "Dijkstra's algorithm finds the shortest paths between nodes in a weighted graph.",
    )


# ---- Runner -----------------------------------------------------------

def _preflight(force: bool = False) -> bool:
    """Checks every provider the chat tiers route to BEFORE spending
    anything, and refuses to start if one is unreachable.

    This exists because of a real run (2026-09-13): a full benchmark
    completed with every chat scenario answered by LLM7 — the SECOND
    fallback — because the shell had not sourced .env. Gemini's key lives
    only there, and Cloudflare reads CLOUDFLARE_ACCOUNT_ID straight from
    the environment, while LLM7's key is in the config database and so
    works anywhere. The run looked successful and measured the wrong
    models from start to finish.

    Noticing that afterwards, from the models column, costs a whole run.
    Noticing it here costs nothing. --force proceeds anyway, for
    deliberately measuring a degraded chain.
    """
    import os

    from config.store import config
    from providers.registry import CHAT_TIERS, get_dispatcher_role

    problems: list[str] = []
    seen: set[str] = set()
    for tier in CHAT_TIERS:
        try:
            role = get_dispatcher_role(context=tier)
        except Exception as e:  # noqa: BLE001
            problems.append(f"{tier}: no routing configured ({e})")
            continue
        chain = [{"provider": role["provider"], "model": role["model"]}] + role.get("fallback", [])
        for position, attempt in enumerate(chain):
            name = attempt["provider"]
            if name in seen:
                continue
            seen.add(name)
            label = "primary" if position == 0 else f"fallback {position}"
            if not config.get_provider_key(name):
                problems.append(f"{name} ({tier} {label}): no API key in the store or the environment")
            elif name == "cloudflare" and not os.environ.get("CLOUDFLARE_ACCOUNT_ID"):
                problems.append(f"cloudflare ({tier} {label}): CLOUDFLARE_ACCOUNT_ID not set")

    if not problems:
        return True
    print("Providers the chat tiers route to that this shell cannot reach:\n")
    for p in problems:
        print(f"  - {p}")
    print(
        "\nEvery call would silently fall through to whatever still works, and the\n"
        "run would measure the wrong models without saying so. Usually this means\n"
        ".env was not sourced:\n\n"
        "    set -a && source .env && set +a\n\n"
        "Pass --force to benchmark the degraded chain on purpose.\n"
    )
    return bool(force)


def _chain() -> list[tuple[str, str, str]]:
    """(tier, label, provider, model) for every attempt in every chat
    tier, in the order routing would actually try them."""
    from providers.registry import CHAT_TIERS, get_dispatcher_role
    out = []
    for tier in CHAT_TIERS:
        try:
            role = get_dispatcher_role(context=tier)
        except Exception:  # noqa: BLE001
            continue
        chain = [{"provider": role["provider"], "model": role["model"]}] + role.get("fallback", [])
        for i, a in enumerate(chain):
            out.append((tier, "primary" if i == 0 else f"fallback {i}", a["provider"], a["model"]))
    return out


def check_providers(live: bool = True) -> bool:
    """Answers "are the keys actually set?" — the question the preflight
    only half-answers.

    Presence is not the same as working. A key that is present but wrong,
    revoked or out of quota passes every static check and fails at the
    first real call, which is exactly when it is most expensive to find
    out. So by default this sends a genuinely tiny prompt to each distinct
    provider and reports what came back.

    Reports EVERY provider, not just the broken ones. "Nothing printed"
    is a terrible answer to "is this configured correctly?".
    """
    import os

    from config.store import config
    from providers.base import ChatMessage
    from providers.registry import get_provider

    rows = _chain()
    seen: dict[str, tuple[str, str]] = {}
    for tier, label, provider, model in rows:
        seen.setdefault(provider, (f"{tier} {label}", model))

    if _ENV_FILE:
        print(f"Loaded environment from {_ENV_FILE}\n")
    else:
        print("No .env file loaded - relying on whatever is already in this shell.\n")
    print(f"{'provider':14} {'first used as':22} {'key':10} {'live call':32}")
    print("-" * 82)
    all_ok = True
    for provider, (where, model) in seen.items():
        has_key = bool(config.get_provider_key(provider))
        if provider == "cloudflare" and has_key and not os.environ.get("CLOUDFLARE_ACCOUNT_ID"):
            has_key = False
            key_note = "NO ACCT"
        else:
            key_note = "found" if has_key else "MISSING"

        result = "skipped" if not live else "-"
        if live and has_key:
            try:
                p = get_provider(provider)
                r = p.chat(model=model, messages=[ChatMessage(role="user", content="Reply with: ok")])
                result = "ok" if (r.text or "").strip() else "empty reply"
            except Exception as e:  # noqa: BLE001
                result = f"FAILED: {str(e)[:24]}"
        elif live:
            result = "not attempted"

        ok = has_key and (not live or result == "ok")
        all_ok = all_ok and ok
        print(f"{provider:14} {where:22} {key_note:10} {result:32}")

    print()
    if all_ok:
        print("All providers in the chat chains are configured and answering.")
    else:
        print(
            "Something above is not usable. A benchmark would silently fall through "
            "to whatever still works and measure the wrong models.\n\n"
            "If a key reads MISSING, the shell most likely has not sourced .env:\n\n"
            "    set -a && source .env && set +a\n"
        )
    return all_ok


async def run_scenario(name: str, reps: int) -> dict:
    spec = SCENARIOS[name]
    runs = []
    for i in range(reps):
        probe = Probe()
        originals = _patched(probe)
        t0 = time.time()
        error = None
        try:
            await spec["fn"]()
        except Exception as e:  # noqa: BLE001 - a failed rep is data, not a crash
            error = f"{type(e).__name__}: {e}"[:200]
        finally:
            _restore(originals)
        run = {
            "rep": i + 1,
            "seconds": round(time.time() - t0, 2),
            "error": error,
            **probe.totals(),
            "calls_detail": probe.calls,
        }
        runs.append(run)
        mark = "!" if error else " "
        print(f"  {mark}rep {i+1}/{reps}: {run['total']:>7} tokens  "
              f"({run['prompt'] or 0} in / {run['completion'] or 0} out)  "
              f"{run['calls']} call(s)  {run['seconds']}s"
              + (f"  ERROR {error[:60]}" if error else ""))
    return {"scenario": name, "description": spec["description"], "runs": runs,
            "summary": _summarize(runs)}


def _summarize(runs: list[dict]) -> dict:
    # A rep counts as usable if it made a real call, even a cheap one.
    ok = [r for r in runs if not r["error"] and r["calls"] and not r["failed_calls"]]
    if not ok:
        return {"ok_runs": 0}
    totals = [r["total"] for r in ok]
    secs = [r["seconds"] for r in ok]
    return {
        "ok_runs": len(ok),
        "tokens_min": min(totals),
        "tokens_max": max(totals),
        "tokens_mean": round(statistics.mean(totals), 1),
        "tokens_stdev": round(statistics.stdev(totals), 1) if len(totals) > 1 else 0.0,
        # The headline number. 1.0 means every run cost the same; the
        # essay that prompted this harness found 2.1 in the wild, and
        # anything approaching that means the cost is not predictable.
        "variance_ratio": round(max(totals) / min(totals), 2) if min(totals) else None,
        "seconds_mean": round(statistics.mean(secs), 1),
        "seconds_max": max(secs),
        "prompt_mean": round(statistics.mean([r["prompt"] or 0 for r in ok]), 1),
        "completion_mean": round(statistics.mean([r["completion"] or 0 for r in ok]), 1),
        "cached_mean": (round(statistics.mean([r["cached"] for r in ok if r["cached"] is not None]), 1)
                        if any(r["cached"] is not None for r in ok) else None),
        "models": sorted({m for r in ok for m in r["models"]}),
    }


async def main() -> None:
    args = [a for a in sys.argv[1:]]
    if "--check" in args:
        check_providers(live="--no-live" not in args)
        return
    if "--list" in args:
        for name, spec in SCENARIOS.items():
            print(f"  {name:26} {spec['description']}")
        return
    reps = DEFAULT_REPS
    if "--reps" in args:
        reps = int(args[args.index("--reps") + 1])
        del args[args.index("--reps"):args.index("--reps") + 2]
    names = [a for a in args if not a.startswith("--")] or list(SCENARIOS)
    unknown = [n for n in names if n not in SCENARIOS]
    if unknown:
        print(f"Unknown scenario(s): {unknown}. Try --list.")
        return

    _isolate_storage()
    if not _preflight(force="--force" in sys.argv):
        return
    print(f"Benchmarking {len(names)} scenario(s), {reps} reps each — REAL API calls.\n")
    results = []
    for name in names:
        print(f"{name} — {SCENARIOS[name]['description']}")
        results.append(await run_scenario(name, reps))
        print()

    OUT_DIR.mkdir(exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = OUT_DIR / f"bench-{stamp}.json"
    path.write_text(json.dumps({
        "run_at": stamp, "reps": reps, "results": results,
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"{'scenario':26} {'mean':>8} {'min':>8} {'max':>8} {'var':>6} {'cached':>8} {'secs':>7}")
    print("-" * 76)
    for r in results:
        s = r["summary"]
        if not s.get("ok_runs"):
            print(f"{r['scenario']:26} {'ALL RUNS FAILED':>40}")
            continue
        cached = s["cached_mean"] if s["cached_mean"] is not None else "-"
        print(f"{r['scenario']:26} {s['tokens_mean']:>8} {s['tokens_min']:>8} "
              f"{s['tokens_max']:>8} {s['variance_ratio']:>6} {str(cached):>8} {s['seconds_mean']:>7}")
    print(f"\nWritten to {path}")


if __name__ == "__main__":
    asyncio.run(main())
