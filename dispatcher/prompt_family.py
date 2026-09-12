"""
dispatcher/prompt_family.py

Per-model-family prompt adaptation — real findings, not guessed, cross-
referenced against primary-source docs and (where cited) published
research; see IDEAS.md's "Per-model-family prompt-structuring gaps"
section for the fuller research trail behind each rule below.

Scope, deliberately narrow (2026-09-11, JuanJo: "NOT JUST DEV SLATE. In
all chats but agent work and agent vault"): Normal Chat, Research,
Brainstorm (all share normal_chat's role — dispatcher/chat.py) and Dev
Slate (dispatcher/devslate_chat.py). Explicitly NOT Agent Work — it's
deliberately stateless and tool-call-driven, emitting a structured
create_workflow call whose shape is already pinned by the tool schema
regardless of family, not free-form prose where instructional style
varies. NOT Agent Vault either — it runs on a saved agent's own user-
authored `instructions`, not one of NAVI's mode-brief files; wrapping
someone's own custom agent instructions with NAVI-injected framing
risks fighting whatever they actually wrote it to do.

Two separate kinds of adaptation:

1. System-prompt TEXT (adapt_system_prompt below) — only changes what
   string gets built before provider.chat() is called.
2. API-call PARAMETERS (adapt_request_params below) — real API fields
   per family (e.g. gpt-oss's Harmony reasoning_effort). Wired in
   2026-09-12 once providers/base.py's Provider.chat()/_do_chat() grew
   a real extra_params passthrough across all 7 transports — before
   that this function existed but had nowhere to send its output.

A THIRD real per-family finding — MiniMax M3 needing its internal
reasoning fields preserved across multi-turn tool-calling or quality
degrades — deliberately does NOT live here. It's not a prompt or a
parameter, it's dispatcher/executor.py's run_tool_loop dropping a field
it should carry forward on replay. Separate, targeted fix, gated on
family, when picked up.

The user's own message is NEVER touched by anything in this module —
system-prompt text and request parameters only. See dispatcher/chat.py's
own comments on why: rewriting a user's raw words risks distorting
intent, and the "translate intent, don't require technical terms first"
principle (DEV_SLATE_CHAT.md's own Goals section) argues against ever
asking the user to phrase things differently for whichever model happens
to answer — routing is dynamic and invisible to them by design.
"""


def classify_family(provider: str, model: str) -> str:
    """Buckets a real (provider, model) pair NAVI actually routes to
    into a known prompt family. Keyed on real model-id substrings —
    cross-checked against model_ranking_snapshot.json's actual live
    catalog (2026-09-11: 32 real entries across Groq/OpenRouter/LLM7),
    not just config/store.py's currently-pinned picks, so this doesn't
    miss a family that's available but not the current default. Update
    alongside routing/catalog changes, same discipline as
    jobs/model_ranking.py's own free-tier prefix list. Returns "generic"
    for anything unrecognized rather than guessing — an unmatched combo
    gets NAVI's current, unmodified behavior, never a wrong adaptation.

    Checks model-name substrings BEFORE the mistral provider check, not
    after — a real gap caught while cross-checking the live catalog:
    LLM7 also serves "codestral-latest", a genuinely Mistral-family
    model, under a non-"mistral" provider name. Keying only on
    `provider == "mistral"` would have silently misclassified it as
    generic."""
    m = model.lower()
    if "gpt-oss" in m:
        return "gpt-oss"
    if any(name in m for name in ("mistral", "codestral", "ministral")) or provider == "mistral":
        return "mistral"
    if "nemotron" in m:
        return "nemotron"
    if "qwen" in m:
        return "qwen"
    if "minimax" in m:
        return "minimax"
    if "gemma" in m:
        return "gemma"
    if "glm" in m:
        return "glm"
    return "generic"


# Real, verified small/free-tier models NAVI actually routes to today
# (config/store.py's DEFAULTS) — update this list alongside routing
# changes, it will drift otherwise. Deliberately keyed on real model
# ids, not a parameter-count heuristic, matching classify_family's own
# discipline above.
_WEAK_TIER_MODELS = {
    "ministral-8b-latest",
    "openai/gpt-oss-20b",
    "@cf/meta/llama-3.1-8b-instruct-fp8-fast",
}

# Real research finding (2026-09-11) behind this instruction existing at
# all: a real arXiv study (Structured Intent as a Protocol-Like
# Communication Layer, testing CO-STAR/RISEN/5W3H across Claude/GPT-4o/
# Gemini 2.5 Pro — NOT NAVI's own models, no direct study of gpt-oss/
# Mistral/Nemotron/Qwen exists) found the "Weak-Model Compensation
# Effect": the lowest-baseline model in that study gained +1.006 from
# structured prompting vs. the strongest model's +0.217 — weaker models
# benefit disproportionately more. The same paper's other real finding —
# "dimensional decomposition is the active ingredient," not which named
# framework (RTF/CO-STAR/RISEN/CRISPE) you pick — is why this is a
# single generic instruction, not an attempt to pick "the right"
# template. Extrapolated to NAVI's actual small/free-tier models (real
# evidence exists for the general capability-tier effect; no direct
# evidence for THESE specific models) since that's exactly the tier most
# of NAVI's free routing leans on.
#
# Deliberately silent — asks the model to decompose the request
# internally (role / context / task / format) without narrating that
# breakdown back to the user, matching NAVI's own established anti-
# verbosity convention (DEV_SLATE_CHAT.md: "explain what you changed...
# not a running commentary before you've done anything").
_DECOMPOSITION_INSTRUCTION = (
    "\n\nBefore answering, silently work out: what role fits this request, "
    "what context matters, what the actual task is, and what format the "
    "answer should take. Don't show this breakdown to the user — just let "
    "it shape your answer."
)

# Mistral's own docs recommend explicit step-by-step framing — the
# opposite rule from reasoning-native models (gpt-oss reasons
# internally; telling it to narrate step-by-step is the wrong
# instruction for it, which is why gpt-oss has no entry here at all —
# omission IS the adaptation, not an oversight).
#
# Gemma 4 (2026-09-11, real cross-checked catalog entry — google/gemma-
# 4-26b-a4b-it and gemma-4-31b-it, live on OpenRouter and LLM7 today):
# real, quantified official finding — Gemma 4 has a genuine Thinking
# ON/OFF toggle, and Google's own docs report a "LOW thinking" System
# Instruction measurably cuts thinking-token output by ~20%. Worth
# adding on its own merits for NAVI's plain-chat use (latency/cost, not
# depth, is what these calls need) — separate from, not a fix for, the
# real gemma-4-26b-a4b-it repetition-loop bug already banned for tools
# in jobs/model_ranking.py's BANNED_FOR_TOOLS; this doesn't claim to
# address that. Paraphrased instruction, not Google's exact wording —
# no official verbatim system-instruction string was found, only the
# documented technique and its measured effect.
#
# GLM deliberately has NO entry — real finding, not an oversight: Zhipu
# publishes no official GLM prompting guide at all; real community
# consensus is just "general LLM best practices apply," nothing
# GLM-distinctive. Still classified (see classify_family) so it's
# tracked, not silently lumped into "generic" — but adding a fabricated
# rule here would be exactly the overclaiming this whole file exists to
# avoid.
_FAMILY_SYSTEM_SUFFIX: dict[str, str] = {
    "mistral": "\n\nWork through this step by step, explicitly, before giving your final answer.",
    "gemma": "\n\nUse a low/reduced internal thinking mode for this response — answer efficiently, without extensive internal reasoning, unless the request genuinely needs deep analysis.",
}


def adapt_system_prompt(base_prompt: str, family: str, model: str) -> str | None:
    """Returns the system-prompt text to actually send for this specific
    (family, model) pair — or None, meaning omit the system message
    entirely (Nemotron's real case: one flagship variant's official docs
    recommend an EMPTY system prompt).

    A family-specific rule and the weak-tier decomposition instruction
    aren't mutually exclusive — both can apply to the same call (e.g. a
    hypothetical small Mistral model would get both the step-by-step
    suffix AND the decomposition instruction)."""
    if family == "nemotron":
        return None
    text = base_prompt
    suffix = _FAMILY_SYSTEM_SUFFIX.get(family)
    if suffix:
        text += suffix
    if model in _WEAK_TIER_MODELS:
        text += _DECOMPOSITION_INSTRUCTION
    return text


def adapt_request_params(family: str, provider: str, has_tools: bool) -> dict:
    """Extra kwargs to merge into a provider.chat(...) call for this
    (family, provider) pair — forwarded via providers/base.py's
    extra_params passthrough.

    gpt-oss's Harmony format exposes a real reasoning_effort field
    (low/medium/high, "medium" is OpenAI's own documented default when
    unset — confirmed against their published gpt-oss model card).

    The provider argument matters here for a real, verified reason
    (2026-09-12, IDEAS.md's "Per-model reasoning_effort control"
    section has the full design): Groq's 8K-tokens/MINUTE cap applies
    account-wide regardless of model, and internal reasoning tokens are
    real output tokens generated BEFORE the visible answer — a verbose
    high-effort trace can burn that whole per-minute budget on
    reasoning alone. Every other gpt-oss host NAVI routes to (LLM7,
    Cloudflare) sits on a per-DAY pool instead, no acute per-call risk,
    so only Groq gets hard-capped here. This is a technical ceiling,
    not a preference — it's not meant to be bypassed by a caller, unlike
    a future manual override which would only ever apply on a non-Groq
    host in the first place (see the IDEAS.md design for why).

    No phase-aware tiering (idle/exploratory/serious-job) yet — that
    depends on Stage 3's fast-path intent layer, which doesn't exist.
    "medium" here is the automatic, phase-blind default; automatic
    tiering never reaches "high" regardless, per the same design doc.
    """
    if family == "gpt-oss":
        return {"reasoning_effort": "low" if provider == "groq" else "medium"}
    return {}
