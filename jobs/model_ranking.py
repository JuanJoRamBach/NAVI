"""
jobs/model_ranking.py

Step 1 of the daily self-healing routing design (see the
navi-model-ranking-design memory, 2026-09-01 — fully specced there,
this is the first real implementation pass): fetch every provider's
LIVE model catalog, join it against Artificial Analysis' benchmark
data, and rank candidates per NAVI task (research/code/graph-data/
summarize/recap/note/remind/tailor/design-read).

Deliberately does NOT write into config/store.py's task_routing yet.
Same reasoning daily_model_digest.py already stated: auto-applying to
production routing from a job like this is a bigger, separate risk
than fetching and ranking is. This produces a JSON snapshot
(MODEL_RANKING_PATH) with the full catalog + computed picks; wiring
that into actual routing (auto-apply, or a Telegram digest for JuanJo
to approve via chat) is the next step, not this one.

Ollama Cloud is deliberately excluded from fetching entirely — no
public model-list API, and per the design its quota is unmeasurable,
so it never competes on merit here.
"""

import json
import os
import re
import time
from pathlib import Path

import requests

MODEL_RANKING_PATH = Path(__file__).parent.parent / "model_ranking_snapshot.json"
AA_CACHE_PATH = Path(__file__).parent.parent / "aa_benchmarks_cache.json"
AA_CACHE_MAX_AGE_S = 7 * 24 * 3600  # weekly — benchmark quality doesn't shift day to day

GROQ_MODELS_URL = "https://api.groq.com/openai/v1/models"
OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models"
LLM7_MODELS_URL = "https://api.llm7.io/v1/models"
MISTRAL_MODELS_URL = "https://api.mistral.ai/v1/models"
GMI_MODELS_URL = "https://api.gmi-serving.com/v1/models"
CLOUDFLARE_MODELS_URL = "https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/models/search"
AA_BULK_URL = "https://artificialanalysis.ai/api/v2/data/llms/models"

# Groq's /v1/models has no "category" field, so filtering is by name
# pattern + a couple of live-confirmed signals (2026-09-01, real pull):
# output_modalities != ["text"] catches Whisper (transcription) and the
# Orpheus TTS pair outright. "safeguard" is excluded on purpose — it's a
# moderation/classifier model NAVI already special-cases for topic
# classification (dispatcher/topic_classifier.py), not a general chat
# candidate. "guard" catches the prompt-guard classifiers (max_completion_
# tokens 512 — not a chat model at all). "allam" is SDAIA's Arabic-
# specialist model, excluded per the design's own call to keep this
# general-purpose (NAVI's real traffic is English/Spanish).
GROQ_EXCLUDE_PATTERNS = ("guard", "safeguard", "allam")

# Correction, 2026-09-01: a real pull of the models/search endpoint (with
# JuanJo's live key) showed the free/paid gate IS exposed programmatically
# after all — each model has a `require_workers_paid` property ("true" on
# exactly the models the pricing-page exclusion list named by hand: kimi-
# k2.6/k2.7-code, glm-5.2/5.3/5.3-flash, deepseek-v4-flash-0731/v4-pro-
# 0813). No more hand-maintained list needed — read it straight from the
# live response instead (see fetch_cloudflare_models below).

# Per-task selection requirements. "tier" is a preference, not a hard
# filter — small tasks prefer an SLM but will take an LLM if nothing
# smaller qualifies; this mirrors NAVI's existing manual choices
# (gpt-oss-20b for summarize/recap/note/tailor, bigger models for
# research/code) rather than inventing new judgment calls.
TASK_REQUIREMENTS = {
    "research": {"tools": True, "min_context": 8000, "tier": "large"},
    "code": {"tools": False, "min_context": 8000, "tier": "large"},
    "graph-data": {"tools": True, "min_context": 4000, "tier": "small"},
    "remind": {"tools": True, "min_context": 4000, "tier": "small"},
    "summarize": {"tools": False, "min_context": 4000, "tier": "small"},
    "recap": {"tools": False, "min_context": 4000, "tier": "small"},
    "note": {"tools": False, "min_context": 4000, "tier": "small"},
    "tailor": {"tools": False, "min_context": 4000, "tier": "small"},
    "design-read": {"tools": False, "min_context": 4000, "tier": "small", "vision": True},
}

SMALL_TIER_MAX_PARAMS_B = 30  # matches the 20b/27b models already used for "small" tasks

_SIZE_RE = re.compile(r"(\d+(?:\.\d+)?)b(?!it)\b", re.IGNORECASE)


def normalize_name(raw: str) -> str:
    """Lookup key ONLY — never stored/used as the real model id. Strips
    org/provider prefix, drops reasoning-effort suffixes like "(max)",
    collapses separators. Snapshot/version suffixes (e.g. -0731) are
    deliberately kept — different snapshots can score differently."""
    name = (raw or "").lower()
    if "/" in name:
        name = name.split("/", 1)[1]
    name = re.sub(r"\s*\((?:max|high|low|medium|thinking|reasoning)\)\s*", "", name)
    name = re.sub(r"[\s\-_.]+", "", name)
    return name


def extract_param_billions(slug: str) -> float | None:
    """Best-effort size signal, regex'd straight out of the slug (present
    in almost every slug across every provider, e.g. "120b", "27b"). For
    MoE slugs with both a total and active-param suffix (nemotron-3-
    ultra-550b-a55b), this returns the larger number (total params) — a
    known simplification, not a claim that active-param count doesn't
    matter for latency/cost."""
    matches = _SIZE_RE.findall(slug or "")
    if not matches:
        return None
    return max(float(m) for m in matches)


# ---- Per-provider fetch ----
# Each returns a list of {provider, id, context_length, tools, vision,
# free, param_b} — best-effort, never raises; a provider that fails to
# fetch just contributes an empty list so the others still populate.

def fetch_groq_models(api_key: str | None) -> list[dict]:
    if not api_key:
        return []
    try:
        resp = requests.get(GROQ_MODELS_URL, headers={"Authorization": f"Bearer {api_key}"}, timeout=20)
        resp.raise_for_status()
        data = (resp.json() or {}).get("data") or []
    except (requests.RequestException, ValueError):
        return []

    out = []
    for m in data:
        mid = m.get("id", "")
        if not m.get("active", True):
            continue
        if m.get("output_modalities") != ["text"]:
            continue  # drops Whisper (transcription) and Orpheus (speech)
        if any(p in mid.lower() for p in GROQ_EXCLUDE_PATTERNS):
            continue
        features = m.get("supported_features") or []
        out.append({
            "provider": "groq", "id": mid,
            "context_length": m.get("context_length") or m.get("context_window"),
            "tools": "tools" in features,
            "vision": "image" in (m.get("input_modalities") or []),
            "free": True,  # whole self-serve tier is rate-limited-free by nature
            "param_b": extract_param_billions(mid),
        })
    return out


def fetch_openrouter_models() -> list[dict]:
    try:
        resp = requests.get(OPENROUTER_MODELS_URL, timeout=30)
        resp.raise_for_status()
        data = (resp.json() or {}).get("data") or []
    except (requests.RequestException, ValueError):
        return []

    out = []
    for m in data:
        pricing = m.get("pricing") or {}
        if pricing.get("prompt") != "0" or pricing.get("completion") != "0":
            continue
        mid = m.get("id", "")
        out.append({
            "provider": "openrouter", "id": mid,
            "context_length": m.get("context_length"),
            "tools": "tools" in (m.get("supported_parameters") or []),
            "vision": "image" in (m.get("architecture", {}).get("input_modalities") or []),
            "free": True,
            "param_b": extract_param_billions(mid),
        })
    return out


def fetch_llm7_models() -> list[dict]:
    try:
        resp = requests.get(LLM7_MODELS_URL, timeout=20)
        resp.raise_for_status()
        data = (resp.json() or {}).get("data") or []
    except (requests.RequestException, ValueError):
        return []

    out = []
    for m in data:
        if m.get("tier") != "turbo":
            continue  # only turbo carries a free daily allotment
        mid = m.get("id", "")
        caps = m.get("capabilities") or {}
        out.append({
            "provider": "llm7", "id": mid,
            "context_length": (m.get("context_window") or {}).get("tokens"),
            "tools": bool(caps.get("tools")),
            "vision": bool(caps.get("vision")),
            "free": True,
            "param_b": extract_param_billions(mid),
            "availability_pct": m.get("availability_last_hour_percent"),
        })
    return out


def fetch_mistral_models(api_key: str | None) -> list[dict]:
    """Live-verified 2026-09-01 against a real key. Real gotchas found:
    the response lists every "base" model (chat, embeddings, OCR,
    moderation, audio/TTS all mixed together, no task filter available)
    AND every alias as its own separate entry with identical capabilities
    — e.g. "codestral-2508" and "codestral-latest" are the same model
    listed 4x under different names. Deduped by billing_model_name
    (confirmed shared across aliases), preferring the "-latest" id since
    that's what NAVI actually calls.

    No pricing/free-tier field exists anywhere in this response — the
    free "Experiment" tier is an account-level token budget, not a
    per-model flag, so free-vs-paid can't be read off this endpoint at
    all. Heuristic until a real usage/pricing signal exists (Mistral's
    own /v1/admin/usage is the authoritative source, not wired in here
    yet): Ministral/Mistral Small/Codestral are treated as the free-tier-
    eligible family (matches this session's own research into what NAVI
    actually targets); Mistral Medium/Large are excluded as clearly
    premium-tier models, not because of a confirmed price signal.
    """
    if not api_key:
        return []
    try:
        resp = requests.get(MISTRAL_MODELS_URL, headers={"Authorization": f"Bearer {api_key}"}, timeout=20)
        resp.raise_for_status()
        data = (resp.json() or {}).get("data") or []
    except (requests.RequestException, ValueError):
        return []

    by_billing: dict[str, dict] = {}
    for m in data:
        caps = m.get("capabilities") or {}
        if not caps.get("completion_chat"):
            continue  # drops embeddings/OCR/moderation/audio-only models
        mid = m.get("id", "")
        billing = m.get("billing_model_name", mid)
        entry = {
            "provider": "mistral", "id": mid,
            "context_length": m.get("max_context_length"),
            "tools": bool(caps.get("function_calling")),
            "vision": bool(caps.get("vision")),
            "free": mid.lower().startswith(("ministral-", "mistral-small", "codestral")),
            "param_b": extract_param_billions(mid),
        }
        if billing not in by_billing or mid.endswith("-latest"):
            by_billing[billing] = entry
    return list(by_billing.values())


def fetch_gmi_models(api_key: str | None) -> list[dict]:
    """Live-verified 2026-09-01 against a real key. Real gotcha JuanJo
    caught directly: promotional free models (currently MiniMax M3 AND
    M2.7, not just M3) appear TWICE in this list — once at normal paid
    pricing, once as a separate entry with the identical id but
    `is_free: true` and every pricing field zeroed. The is_free entry is
    the one that matters; dedupe by id and mark a model free if ANY of
    its entries carries is_free, rather than pattern-matching a model
    name (which would have missed M2.7's promo entirely).

    No capability/tool-support field exists anywhere in this response,
    unlike every other provider fetched here — "tools" defaults to False
    rather than assumed true, since there's no live signal to check it
    against.
    """
    if not api_key:
        return []
    try:
        resp = requests.get(GMI_MODELS_URL, headers={"Authorization": f"Bearer {api_key}"}, timeout=20)
        resp.raise_for_status()
        data = (resp.json() or {}).get("data") or []
    except (requests.RequestException, ValueError):
        return []

    by_id: dict[str, dict] = {}
    for m in data:
        mid = m.get("id", "")
        if not mid:
            continue
        is_free = bool(m.get("is_free"))
        if mid not in by_id:
            by_id[mid] = {
                "provider": "gmi", "id": mid,
                "context_length": m.get("context_length"),
                "tools": False,
                "vision": False,
                "free": is_free,
                "param_b": extract_param_billions(mid),
            }
        elif is_free:
            by_id[mid]["free"] = True
    return list(by_id.values())


def fetch_cloudflare_models(api_key: str | None) -> list[dict]:
    """Live-verified 2026-09-01 against a real key. Two real bugs a
    first-draft/undocumented-shape guess got wrong, fixed here:

    1. Each result's top-level "id" is Cloudflare's internal catalog
       UUID, NOT the model slug — the callable slug (what chat() needs,
       e.g. "@cf/openai/gpt-oss-120b") is the "name" field instead.
    2. The endpoint is genuinely paginated (result_info.count/total_count
       — a real pull returned 65 of 299 total on page 1) and covers every
       task type Cloudflare hosts (embeddings, TTS, ASR, image, dumb-pipe
       audio-turn-detection, not just text generation), so this both
       pages through the full catalog and filters to task.name ==
       "Text Generation" before returning anything.

    Free-vs-paid no longer needs a hand-maintained exclusion list (see
    the comment above this function's old constant) — each model's
    properties include require_workers_paid: "true" on exactly the
    models known to need a paid billing method regardless of Neuron
    budget, confirmed directly against a live response.
    """
    account_id = os.environ.get("CLOUDFLARE_ACCOUNT_ID")
    if not api_key or not account_id:
        return []

    all_results = []
    page = 1
    while True:
        try:
            resp = requests.get(
                CLOUDFLARE_MODELS_URL.format(account_id=account_id),
                headers={"Authorization": f"Bearer {api_key}"},
                params={"per_page": 100, "page": page},
                timeout=20,
            )
            resp.raise_for_status()
            payload = resp.json() or {}
        except (requests.RequestException, ValueError):
            break
        if not payload.get("success"):
            break
        batch = payload.get("result") or []
        all_results += batch
        info = payload.get("result_info") or {}
        if not batch or len(all_results) >= (info.get("total_count") or 0):
            break
        page += 1
        if page > 10:  # hard stop — 299 models / 100 per page is ~3 pages, this is generous headroom
            break

    out = []
    for m in all_results:
        if (m.get("task") or {}).get("name") != "Text Generation":
            continue
        mid = m.get("name", "")  # the callable slug — NOT m["id"], which is an internal UUID
        if not mid:
            continue
        short_id = mid.split("/")[-1]
        props = {p.get("property_id"): p.get("value") for p in (m.get("properties") or [])}
        out.append({
            "provider": "cloudflare", "id": mid,
            "context_length": int(props["context_window"]) if props.get("context_window") else None,
            "tools": props.get("function_calling") == "true",
            "vision": props.get("vision") == "true",
            "free": props.get("require_workers_paid") != "true",
            "param_b": extract_param_billions(short_id),
        })
    return out


# ---- Artificial Analysis benchmark data (weekly cache) ----

def fetch_aa_benchmarks(api_key: str | None, force: bool = False) -> dict:
    """Returns {normalized_name: {intelligence_index, coding_index, ...,
    output_tokens_per_second}}. Cached to disk — refetched only if the
    cache is missing/stale (see AA_CACHE_MAX_AGE_S), since AA's real
    confirmed rate limit is 100 requests/day and quality scores don't
    move day to day anyway."""
    if not force and AA_CACHE_PATH.exists():
        try:
            cached = json.loads(AA_CACHE_PATH.read_text())
            if time.time() - cached.get("_fetched_at", 0) < AA_CACHE_MAX_AGE_S:
                return cached.get("models", {})
        except (json.JSONDecodeError, OSError):
            pass

    if not api_key:
        # No key available — fall back to whatever's cached, even if
        # stale, rather than silently ranking with zero quality signal.
        if AA_CACHE_PATH.exists():
            try:
                return json.loads(AA_CACHE_PATH.read_text()).get("models", {})
            except (json.JSONDecodeError, OSError):
                pass
        return {}

    try:
        resp = requests.get(AA_BULK_URL, headers={"x-api-key": api_key}, timeout=60)
        resp.raise_for_status()
        raw_models = (resp.json() or {}).get("data") or []
    except (requests.RequestException, ValueError):
        return {}

    models = {}
    for m in raw_models:
        evals = m.get("evaluations") or {}
        entry = {
            "id": m.get("id"),
            "name": m.get("name"),
            "intelligence_index": evals.get("artificial_analysis_intelligence_index"),
            "coding_index": evals.get("artificial_analysis_coding_index"),
            "output_tokens_per_second": m.get("median_output_tokens_per_second"),
        }
        for key in (m.get("name"), m.get("slug")):
            if key:
                models[normalize_name(key)] = entry

    try:
        AA_CACHE_PATH.write_text(json.dumps({"_fetched_at": time.time(), "models": models}, indent=2))
    except OSError:
        pass
    return models


# ---- Ranking ----

def _score_candidate(m: dict, aa_index: dict) -> tuple:
    """Sort key, HIGHER is better. Order: quality (AA intelligence index,
    0 if unmatched — an unranked model doesn't get penalized to the
    bottom, just treated as a tie at the "no signal" level) then output
    speed as a cheap reliability/latency tiebreak, both from the same AA
    dataset so this is one lookup, not two."""
    bench = aa_index.get(normalize_name(m["id"])) or {}
    quality = bench.get("intelligence_index") or 0
    speed = bench.get("output_tokens_per_second") or 0
    return (quality, speed)


def rank_for_task(task: str, catalog: list[dict], aa_index: dict) -> dict:
    reqs = TASK_REQUIREMENTS.get(task, {})
    candidates = [
        m for m in catalog
        if m.get("free")
        and (not reqs.get("tools") or m.get("tools"))
        and (not reqs.get("vision") or m.get("vision"))
        and (m.get("context_length") or 0) >= reqs.get("min_context", 0)
    ]
    if not candidates:
        return {"primary": None, "fallback": [], "candidate_count": 0}

    tier = reqs.get("tier", "large")
    if tier == "small":
        tiered = [m for m in candidates if (m.get("param_b") or 0) and m["param_b"] <= SMALL_TIER_MAX_PARAMS_B]
    else:
        tiered = [m for m in candidates if (m.get("param_b") or 0) > SMALL_TIER_MAX_PARAMS_B]
    pool = tiered or candidates  # fall back to the full candidate set if the tier preference has no matches

    ranked = sorted(pool, key=lambda m: _score_candidate(m, aa_index), reverse=True)
    return {
        "primary": {"provider": ranked[0]["provider"], "model": ranked[0]["id"]},
        "fallback": [{"provider": m["provider"], "model": m["id"]} for m in ranked[1:3]],
        "candidate_count": len(candidates),
    }


def build_ranking_snapshot() -> dict:
    from config.store import config

    catalog = []
    catalog += fetch_groq_models(config.get_provider_key("groq"))
    catalog += fetch_openrouter_models()
    catalog += fetch_llm7_models()
    catalog += fetch_mistral_models(config.get_provider_key("mistral"))
    catalog += fetch_gmi_models(config.get_provider_key("gmi"))
    catalog += fetch_cloudflare_models(config.get_provider_key("cloudflare"))

    aa_index = fetch_aa_benchmarks(os.environ.get("AANALYSIS_API_KEY"))

    rankings = {task: rank_for_task(task, catalog, aa_index) for task in TASK_REQUIREMENTS}

    return {
        "fetched_at": time.time(),
        "catalog": catalog,
        "provider_counts": {
            p: len([m for m in catalog if m["provider"] == p])
            for p in ("groq", "openrouter", "llm7", "mistral", "gmi", "cloudflare")
        },
        "rankings": rankings,
    }


def main() -> None:
    snapshot = build_ranking_snapshot()
    MODEL_RANKING_PATH.write_text(json.dumps(snapshot, indent=2))

    print(f"Fetched {len(snapshot['catalog'])} free models across providers: {snapshot['provider_counts']}")
    for task, result in snapshot["rankings"].items():
        if result["primary"]:
            p = result["primary"]
            fb = ", ".join(f"{f['provider']}/{f['model']}" for f in result["fallback"])
            print(f"  {task}: {p['provider']}/{p['model']} (candidates: {result['candidate_count']}"
                  + (f", fallbacks: {fb}" if fb else "") + ")")
        else:
            print(f"  {task}: no qualifying free model found")


if __name__ == "__main__":
    main()
