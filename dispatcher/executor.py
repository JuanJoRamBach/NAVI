"""
dispatcher/executor.py

Runs a parsed list of Steps in order (never in parallel — later steps often
depend on earlier ones' actual output). For each step:

  1. Try the primary provider/model for that command.
  2. On failure, rotate through the configured fallback chain.
  3. If a fallback had to be used, mark the step "degraded" and record it,
     so every LATER step that receives this step's output as context also
     gets flagged as potentially contaminated — not just the one that failed.
  4. Build a final reply that's explicit about what happened, per the
     graceful-degradation-with-disclosure approach we settled on.
"""

import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

from config.store import config
from dispatcher.mode_briefs import get_mode_brief
from dispatcher.parser import Step
from dispatcher.reminders import add_reminder
from dispatcher.research_status import set_status
from dispatcher.slugify import assign_slugs
from providers.base import ChatMessage, ChatResponse, Provider, ProviderError
from providers.registry import ProviderNotConfigured, get_provider
from storage.filen import StorageError, save_bytes, save_result
from tools.charts import CHART_TOOL_CHOICE, CHART_TOOL_NAME, CHART_TOOL_SCHEMA, ChartError, render_chart
from tools.image_gen import ImageGenError, generate_image
from tools.registry import TOOL_SCHEMAS, schemas_for
from tools.registry import dispatch as dispatch_tool

# DeepSeek-V4-Flash-0731 via LLM7 — the synthesis-phase model for
# /research (see _run_research_step). Hardcoded rather than a
# task_routing entry since this role is very specific: one-shot,
# no-tools, huge-context synthesis over already-gathered material, not
# a general dispatcher role.
SYNTHESIS_PROVIDER = "llm7"
SYNTHESIS_MODEL = "DeepSeek-V4-Flash-0731"

# Retry a flaky synthesis call every 30s for 3 minutes before giving up
# and falling back to the gathering model's own synthesis instead. LLM7's
# free tier proved to have real shared-capacity hiccups ("model
# temporarily busy") under light load — worth a few retries before
# accepting the fallback's lower quality.
SYNTHESIS_RETRY_DELAY_S = 30
SYNTHESIS_MAX_ATTEMPTS = 6

# Caps the gathered-material document handed to the synthesis model at
# roughly gpt-oss-120b's ~130k-token context (assuming ~4 chars/token) —
# not DeepSeek's much larger ~400k. This is deliberate: if DeepSeek is
# unavailable, the fallback synthesis (see below) needs to read the same
# document, so the cap has to fit whichever model actually ends up
# reading it, not just the best case.
RESEARCH_DOC_CHAR_BUDGET = 130_000 * 4

# File extension per command — used when saving each step's output.
EXTENSION_FOR_COMMAND = {
    "research": "md",
    "code": "py",  # best-guess default; language-specific naming can improve this later
    "graph-data": "png",
    "create-image": "png",
    "summarize": "md",
    "recap": "md",
    "note": "md",
    "remind": "md",
    "tailor": "md",
    "design-read": "md",
}

# /summarize gets exactly one tool (fetch_page), not the full research
# belt — it's a single-phase digest, not a gather-then-synthesize
# pipeline like /research. The model decides whether to call it: if
# the input is already pasted text, there's nothing to fetch.
SUMMARIZE_SYSTEM_PROMPT = (
    "Produce a tight, faithful digest of the given content. If the message "
    "contains a URL, call fetch_page to read it first, then summarize what you "
    "fetched — don't just describe the link. If it's already pasted text, "
    "summarize that directly. Keep it dense: hit the key points, skip padding, "
    "no filler intro like 'Here is a summary'. Preserve concrete numbers, names, "
    "and dates from the source."
)

# /graph-data doesn't get the research tool belt — it gets exactly one
# forced tool (render_chart), so the model can't just answer in prose.
# The model supplies the numbers; matplotlib draws the pixels, so the
# chart can't hallucinate a wrong-looking trend.
GRAPH_DATA_SYSTEM_PROMPT = (
    "Extract the data needed to answer this into the render_chart tool call. "
    "Use real numbers only — if exact figures aren't available in the given "
    "context, use your best reasonable estimate rather than inventing precision "
    "you don't have. Don't reply in prose; the only valid response is the tool call."
)

# Ceiling on tool-call round-trips per step, so a model that keeps calling
# tools instead of answering can't spin forever and burn the day's quota.
MAX_TOOL_ITERATIONS = 5

# Tells the model to cite sources as Markdown links rather than pasting
# bare URLs — both messaging adapters render [text](url) as a clickable
# link (Telegram via an HTML conversion, Discord natively), so this is
# what actually makes citations clickable in the final reply.
CITATION_STYLE_PROMPT = (
    "When you cite a source you found via web_search or fetch_page, format it as a "
    "Markdown link: [short title](url). Don't paste bare URLs in your answer."
)


def run_tool_loop(
    provider: Provider, model: str, messages: list[ChatMessage], response: ChatResponse, context: dict
) -> tuple[ChatResponse, list[ChatMessage]]:
    """Executes any tool_calls in `response`, feeds results back to the
    model, and repeats until the model stops asking for tools or the
    iteration ceiling is hit. Returns (final ChatResponse, full message
    transcript) — the transcript lets a caller extract raw tool results
    (see _extract_tool_results) without needing the model's own final
    prose synthesis.

    Public (not `_`-prefixed) because dispatcher/chat.py reuses this for
    free-form mode-based chat, not just /research's command chain."""
    iterations = 0
    while response.tool_calls and iterations < MAX_TOOL_ITERATIONS:
        raw_choice = ((response.raw or {}).get("choices") or [{}])[0].get("message", {})
        messages = messages + [ChatMessage(
            role="assistant",
            content=response.text or "",
            tool_calls=raw_choice.get("tool_calls"),
        )]

        for tc in response.tool_calls:
            args = tc.arguments
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {}
            result_text = dispatch_tool(tc.name, args, context)
            messages = messages + [ChatMessage(
                role="tool", content=result_text, tool_call_id=tc.id, name=tc.name,
            )]

        response = provider.chat(model=model, messages=messages, tools=TOOL_SCHEMAS)
        iterations += 1

    return response, messages


def _extract_tool_results(messages: list[ChatMessage]) -> str:
    """Concatenates every tool-result message from a run_tool_loop
    transcript into one readable document — the raw gathered material
    (search snippets, fetched page text), not the model's own prose
    synthesis. Used by /research to hand DeepSeek the source material
    directly rather than a re-summarized version of it."""
    parts = []
    for m in messages:
        if m.role == "tool":
            parts.append(f"### Result from {m.name}\n{m.content}")
    return "\n\n".join(parts)


@dataclass
class StepResult:
    step: Step
    text: str
    degraded: bool = False
    fallback_used: dict | None = None
    error: str | None = None
    contaminated_by: list[str] = field(default_factory=list)  # commands whose degradation fed this step
    saved_path: str | None = None
    save_error: str | None = None
    # Set only for /graph-data — the rendered chart, sent as an attachment
    # rather than (or alongside) plain text.
    image_bytes: bytes | None = None
    image_filename: str | None = None
    # Provider-reported per-call cost, e.g. "2.7 Neurons" on Cloudflare.
    # None for providers with no comparable metric.
    usage_note: str | None = None
    # Set only for /research — `text` is the full report (used for the
    # Filen save and for chunked delivery where there's no real
    # attachment support, e.g. push/PWA); `snippet` is a short version
    # for channels that can send the full report as a real file
    # alongside it (Telegram's caption + attached .md).
    snippet: str | None = None


def _parse_tool_args(raw_args) -> dict:
    if isinstance(raw_args, str):
        try:
            return json.loads(raw_args)
        except json.JSONDecodeError:
            return {}
    return raw_args or {}


def _run_graph_data_step(
    provider: Provider, model: str, messages: list[ChatMessage]
) -> tuple[bytes, str, str]:
    """
    Forces a render_chart tool call, renders it, and returns
    (png_bytes, filename, caption_text). Raises ProviderError or
    ChartError on failure — callers handle those the same way a normal
    chat failure would (rotate to the next fallback).
    """
    response = provider.chat(
        model=model, messages=messages, tools=[CHART_TOOL_SCHEMA], tool_choice=CHART_TOOL_CHOICE,
    )
    if not response.tool_calls:
        raise ProviderError(f"{model} didn't call {CHART_TOOL_NAME} despite it being forced")

    args = _parse_tool_args(response.tool_calls[0].arguments)
    try:
        png_bytes = render_chart(
            chart_type=args["chart_type"],
            title=args["title"],
            labels=args["labels"],
            series=args["series"],
            x_label=args.get("x_label", ""),
            y_label=args.get("y_label", ""),
        )
    except (KeyError, ChartError) as e:
        raise ChartError(f"Model produced unusable chart data: {e}")

    title = args.get("title", "chart")
    filename = f"{title[:40].strip().replace(' ', '-') or 'chart'}.png"
    return png_bytes, filename, f"📊 {title}"


def _run_create_image_step(step: Step) -> StepResult:
    """
    /create-image has no LLM/provider involved at all — a fixed free
    endpoint (see tools/image_gen.py), not something task_routing's
    primary/fallback shape fits. Bypasses the routing/attempts loop below
    entirely rather than forcing a "provider" that doesn't exist onto it.
    """
    try:
        image_bytes = generate_image(step.text)
    except ImageGenError as e:
        return StepResult(step=step, text="", error=str(e))

    filename = f"{step.topic_slug or 'image'}.png"
    return StepResult(step=step, text=f"🎨 {step.text[:80]}", image_bytes=image_bytes, image_filename=filename)


def _make_snippet(text: str, max_chars: int = 600) -> str:
    """A short teaser for channels that can attach the full report as a
    real file alongside it (Telegram). Cuts at the last paragraph break
    before the limit when there is one, so it doesn't end mid-sentence."""
    if len(text) <= max_chars:
        return text
    cut = text[:max_chars]
    break_at = cut.rfind("\n\n")
    if break_at > max_chars // 2:  # only use it if it's not absurdly early
        cut = cut[:break_at]
    return cut.rstrip() + "…"


def _run_research_gather_phase(step: Step, prior_context: str | None) -> tuple[str, str, bool, dict | None, str | None]:
    """
    Phase 1 — runs the existing tool-calling loop (web_search/fetch_page/
    save_note) using /research's configured primary/fallback chain,
    unchanged from before. Returns (gathered_document, gathering_model's
    own draft answer, degraded, fallback_used, error). `gathered_document`
    is the raw tool-result transcript, not the model's prose — that's
    what phase 2 (_run_research_synthesis) actually works from.
    """
    routing = config.get_task_routing("research")
    if not routing:
        return "", "", False, None, "No routing configured for /research"

    attempts = [routing["primary"]] + routing.get("fallback", [])
    last_error = None

    for i, attempt in enumerate(attempts):
        model = attempt.get("model")
        if not model:
            continue
        try:
            provider = get_provider(attempt["provider"])
        except Exception as e:
            last_error = str(e)
            continue

        messages = [ChatMessage(role="system", content=CITATION_STYLE_PROMPT)]
        if prior_context:
            messages.append(ChatMessage(
                role="system",
                content=f"Context from a previous step in this chain:\n{prior_context}",
            ))
        messages.append(ChatMessage(role="user", content=step.text))

        try:
            set_status(f"Researching — gathering sources ({attempt['provider']}/{model})…")
            response = provider.chat(model=model, messages=messages, tools=TOOL_SCHEMAS)
            response, full_messages = run_tool_loop(
                provider, model, messages, response,
                context={"command": "research", "topic_slug": step.topic_slug},
            )
            doc = _extract_tool_results(full_messages)
            return doc, response.text or "", (i > 0), (attempt if i > 0 else None), None
        except ProviderError as e:
            last_error = str(e)
            continue

    return "", "", False, None, last_error or "All research providers failed during gathering"


def _run_research_synthesis(step: Step, gathered_doc: str, gather_fallback: dict | None, routing: dict) -> tuple[str, bool]:
    """
    Phase 2 — one-shot synthesis over the gathered material, via DeepSeek/
    LLM7 for its large context, retried on transient failure (30s x 6 =
    3min) before falling back to the same model that did the gathering.
    Returns (final_text, degraded).
    """
    doc = gathered_doc
    if len(doc) > RESEARCH_DOC_CHAR_BUDGET:
        doc = doc[:RESEARCH_DOC_CHAR_BUDGET] + "\n\n[...truncated — gathered material exceeded the safe context budget]"

    brief = get_mode_brief("research")
    synth_messages = [
        ChatMessage(role="system", content=brief.system_prompt),
        ChatMessage(role="user", content=f"Research question: {step.text}\n\nGathered material:\n{doc}"),
    ]

    try:
        synth_provider = get_provider(SYNTHESIS_PROVIDER)
    except ProviderNotConfigured:
        synth_provider = None

    if synth_provider:
        for attempt_num in range(1, SYNTHESIS_MAX_ATTEMPTS + 1):
            try:
                set_status(f"Researching — synthesizing with DeepSeek (attempt {attempt_num}/{SYNTHESIS_MAX_ATTEMPTS})…")
                resp = synth_provider.chat(model=SYNTHESIS_MODEL, messages=synth_messages)
                return resp.text or "", False
            except ProviderError:
                if attempt_num < SYNTHESIS_MAX_ATTEMPTS:
                    set_status(f"Researching — DeepSeek busy, retrying in {SYNTHESIS_RETRY_DELAY_S}s ({attempt_num}/{SYNTHESIS_MAX_ATTEMPTS})…")
                    time.sleep(SYNTHESIS_RETRY_DELAY_S)

    # DeepSeek never came through — fall back to the gathering model
    # itself doing the synthesis instead of losing the result outright.
    set_status("Researching — DeepSeek unavailable, finishing with the gathering model…")
    fallback_attempt = gather_fallback or routing["primary"]
    try:
        provider = get_provider(fallback_attempt["provider"])
        resp = provider.chat(model=fallback_attempt["model"], messages=synth_messages)
        return resp.text or "", True
    except ProviderError:
        return "", True


def _run_research_step(step: Step, prior_context: str | None) -> StepResult:
    routing = config.get_task_routing("research")
    if not routing:
        return StepResult(step=step, text="", error="No routing configured for /research")

    gathered_doc, draft_text, gather_degraded, gather_fallback, gather_error = _run_research_gather_phase(step, prior_context)
    if gather_error:
        set_status(None)
        return StepResult(step=step, text="", error=gather_error)

    if not gathered_doc.strip():
        # No tool calls happened — model answered from reasoning alone,
        # nothing gathered to synthesize from.
        set_status(None)
        return StepResult(step=step, text=draft_text, degraded=gather_degraded, fallback_used=gather_fallback)

    final_text, synth_degraded = _run_research_synthesis(step, gathered_doc, gather_fallback, routing)
    set_status(None)

    if not final_text:
        return StepResult(step=step, text="", error="Research gathering succeeded but synthesis failed on every attempt")

    return StepResult(
        step=step,
        text=final_text,
        snippet=_make_snippet(final_text),
        degraded=gather_degraded or synth_degraded,
        fallback_used=gather_fallback,
    )


# Recap distills; Note preserves. Both deliver to Telegram themselves via
# the send_to_telegram tool (named explicitly in prose — see tools/registry.py's
# note on DeepSeek needing it spelled out to reliably call it from casual phrasing).
RECAP_SYSTEM_PROMPT = (
    "Turn the given material into a durable recap worth keeping, structured "
    "like a memory note: a claim/fact stated plainly, a 'Why:' line giving the "
    "reasoning or motivation behind it, and an 'Open threads:' line for anything "
    "unresolved or worth following up on (omit that line if there's nothing open). "
    "Distill, don't just restate — cut anything not worth remembering later. "
    "Always call send_to_telegram with the finished recap so it's delivered, "
    "then also return it as your reply."
)
NOTE_SYSTEM_PROMPT = (
    "Lightly capture the given material as a casual note — something worth "
    "preserving but not worth structuring or distilling. Keep it close to how "
    "it was said; don't force it into a claim/reasoning format. A sentence or "
    "two of your own framing is fine if it helps future-you understand the "
    "context, but don't summarize away the specifics. "
    "Always call send_to_telegram with the finished note so it's delivered, "
    "then also return it as your reply."
)


def _run_text_transform_step(
    step: Step, prior_context: str | None, command: str, system_prompt: str, tool_names: list[str],
) -> StepResult:
    """Shared shape for single-phase, tool-optional commands (/summarize,
    /recap, /note): one provider call over the given text, with a small
    tool belt available. Differs from /research's two-phase gather-then-
    synthesize pipeline — these are simple enough for one call."""
    routing = config.get_task_routing(command)
    if not routing:
        return StepResult(step=step, text="", error=f"No routing configured for /{command}")

    attempts = [routing["primary"]] + routing.get("fallback", [])
    last_error = None

    for i, attempt in enumerate(attempts):
        model = attempt.get("model")
        if not model:
            continue
        try:
            provider = get_provider(attempt["provider"])
        except Exception as e:
            last_error = str(e)
            continue

        messages = [ChatMessage(role="system", content=system_prompt)]
        if prior_context:
            messages.append(ChatMessage(
                role="system",
                content=f"Context from a previous step in this chain:\n{prior_context}",
            ))
        messages.append(ChatMessage(role="user", content=step.text))

        try:
            response = provider.chat(model=model, messages=messages, tools=schemas_for(tool_names))
            response, _ = run_tool_loop(
                provider, model, messages, response,
                context={"command": command, "topic_slug": step.topic_slug},
            )
            return StepResult(
                step=step,
                text=response.text or "",
                degraded=(i > 0),
                fallback_used=attempt if i > 0 else None,
                usage_note=response.usage_note,
            )
        except ProviderError as e:
            last_error = str(e)
            continue

    return StepResult(step=step, text="", error=last_error or f"All {command} providers failed")


# /remind forces a tool call (same reasoning as /graph-data's render_chart)
# so the model can't just reply in prose "sure, I'll remind you" without
# actually recording anything. The model resolves whatever time phrasing
# the user gave into an absolute UTC timestamp itself — current time (both
# UTC and JuanJo's local Europe/Madrid) is given in the system prompt so
# relative ("in 20 minutes") and local ("tomorrow at 9am") phrasing both
# resolve correctly.
REMIND_TOOL_NAME = "set_reminder"
REMIND_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": REMIND_TOOL_NAME,
        "description": "Records a reminder with an absolute fire time.",
        "parameters": {
            "type": "object",
            "properties": {
                "fire_at_utc": {
                    "type": "string",
                    "description": "Absolute UTC timestamp, ISO 8601, e.g. 2026-08-27T14:30:00+00:00",
                },
                "message": {"type": "string", "description": "What to remind the user about."},
            },
            "required": ["fire_at_utc", "message"],
        },
    },
}
REMIND_TOOL_CHOICE = {"type": "function", "function": {"name": REMIND_TOOL_NAME}}

USER_TIMEZONE = "Europe/Madrid"


def _run_remind_step(step: Step, prior_context: str | None) -> StepResult:
    routing = config.get_task_routing("remind")
    if not routing:
        return StepResult(step=step, text="", error="No routing configured for /remind")

    now_utc = datetime.now(timezone.utc)
    now_local = now_utc.astimezone(ZoneInfo(USER_TIMEZONE))
    system_prompt = (
        f"Current time is {now_utc.isoformat()} (UTC), which is "
        f"{now_local.strftime('%Y-%m-%d %H:%M')} in {USER_TIMEZONE}. "
        "The user wants a reminder set. Resolve whatever time they gave — relative "
        f"('in 20 minutes') or local ('tomorrow at 9am', assume {USER_TIMEZONE} if no "
        "timezone is stated) — into an absolute UTC timestamp, and call set_reminder "
        "with that plus a short message describing what to remind them about."
    )

    attempts = [routing["primary"]] + routing.get("fallback", [])
    last_error = None

    for i, attempt in enumerate(attempts):
        model = attempt.get("model")
        if not model:
            continue
        try:
            provider = get_provider(attempt["provider"])
        except Exception as e:
            last_error = str(e)
            continue

        messages = [
            ChatMessage(role="system", content=system_prompt),
            ChatMessage(role="user", content=step.text),
        ]

        try:
            response = provider.chat(
                model=model, messages=messages, tools=[REMIND_TOOL_SCHEMA], tool_choice=REMIND_TOOL_CHOICE,
            )
            if not response.tool_calls:
                raise ProviderError(f"{model} didn't call {REMIND_TOOL_NAME} despite it being forced")

            args = _parse_tool_args(response.tool_calls[0].arguments)
            fire_at = datetime.fromisoformat(args["fire_at_utc"])
            message = args["message"]
            add_reminder(fire_at, message)

            local_str = fire_at.astimezone(ZoneInfo(USER_TIMEZONE)).strftime("%a %d %b, %H:%M")
            return StepResult(
                step=step,
                text=f"⏰ Reminder set for {local_str} ({USER_TIMEZONE}): {message}",
                degraded=(i > 0),
                fallback_used=attempt if i > 0 else None,
            )
        except (ProviderError, KeyError, ValueError) as e:
            last_error = str(e)
            continue

    return StepResult(step=step, text="", error=last_error or "Couldn't set the reminder — all providers failed")


def _run_summarize_step(step: Step, prior_context: str | None) -> StepResult:
    return _run_text_transform_step(step, prior_context, "summarize", SUMMARIZE_SYSTEM_PROMPT, ["fetch_page"])


# The only command whose input is an image, not text — routed to a
# vision-capable model (see config/store.py's task_routing entry) via a
# ChatMessage.content list (see providers/base.py) instead of a plain
# string. Currently only reachable via a Telegram photo (see
# messaging/telegram.py) — the PWA has no upload UI yet, deliberately
# deferred (see NAVI v2 handoff notes).
DESIGN_READ_SYSTEM_PROMPT = (
    "You're given a screenshot of a UI or design. Identify the design pattern(s) "
    "in use (e.g. 'card grid with sticky filter bar', 'stepped onboarding wizard', "
    "'inline validation form') — be specific, not generic. Then write a ready-to-"
    "paste prompt for Claude Code that would recreate this UI in a real codebase: "
    "specific about layout, spacing, states, and interaction, not vague. "
    "Structure your reply exactly as:\n\n"
    "Pattern: <name>\n\n"
    "Claude Code prompt:\n<the prompt, ready to paste as-is>"
)


def _run_design_read_step(step: Step, prior_context: str | None) -> StepResult:
    if not step.image_data_url:
        return StepResult(
            step=step, text="",
            error="No image attached — send a screenshot along with /design-read.",
        )

    routing = config.get_task_routing("design-read")
    if not routing:
        return StepResult(step=step, text="", error="No routing configured for /design-read")

    attempts = [routing["primary"]] + routing.get("fallback", [])
    last_error = None

    user_content = [
        {"type": "text", "text": step.text or "Read this design."},
        {"type": "image_url", "image_url": {"url": step.image_data_url}},
    ]

    for i, attempt in enumerate(attempts):
        model = attempt.get("model")
        if not model:
            continue
        try:
            provider = get_provider(attempt["provider"])
        except Exception as e:
            last_error = str(e)
            continue

        messages = [
            ChatMessage(role="system", content=DESIGN_READ_SYSTEM_PROMPT),
            ChatMessage(role="user", content=user_content),
        ]

        try:
            response = provider.chat(model=model, messages=messages)
            return StepResult(
                step=step,
                text=response.text or "",
                degraded=(i > 0),
                fallback_used=attempt if i > 0 else None,
                usage_note=response.usage_note,
            )
        except ProviderError as e:
            last_error = str(e)
            continue

    return StepResult(step=step, text="", error=last_error or "All design-read providers failed")


# The CV lives at a link JuanJo controls (portfolio site, hosted PDF, etc.)
# rather than pasted text or stored raw — set once via "/tailor cv: <link>",
# read back on every later /tailor call. A deterministic "cv:" prefix
# rather than guessing intent from a bare URL, since a job posting can
# also be just a link.
TAILOR_SYSTEM_PROMPT_TEMPLATE = (
    "You're helping tailor a job application. The user's CV is at this link — "
    "call fetch_page on it to read it: {cv_link}\n\n"
    "The job posting is given below in the user's message; if it's a URL, call "
    "fetch_page on that too, otherwise it's already pasted text.\n\n"
    "Produce two clearly separated sections:\n"
    "1. A tailored cover note — short, specific to this posting, grounded only "
    "in what's actually in the CV, no invented experience.\n"
    "2. An honest fit rundown: why they'd likely fit, and where they might not "
    "— a realistic read is more useful than a flattering one."
)


def _run_tailor_step(step: Step, prior_context: str | None) -> StepResult:
    text = step.text.strip()

    if text.lower().startswith("cv:"):
        link = text[3:].strip()
        if not link:
            return StepResult(step=step, text="", error="No link given after 'cv:'")
        config.set("cv_link", link)
        return StepResult(step=step, text=f"CV link saved: {link}")

    cv_link = config.get("cv_link")
    if not cv_link:
        return StepResult(
            step=step, text="",
            error="No CV link on file yet — set one first with /tailor cv: <link>",
        )

    tailor_step = Step(command=step.command, text=text, topic_slug=step.topic_slug)
    system_prompt = TAILOR_SYSTEM_PROMPT_TEMPLATE.format(cv_link=cv_link)
    return _run_text_transform_step(tailor_step, prior_context, "tailor", system_prompt, ["fetch_page"])


def _run_recap_step(step: Step, prior_context: str | None) -> StepResult:
    return _run_text_transform_step(step, prior_context, "recap", RECAP_SYSTEM_PROMPT, ["send_to_telegram"])


def _run_note_step(step: Step, prior_context: str | None) -> StepResult:
    return _run_text_transform_step(step, prior_context, "note", NOTE_SYSTEM_PROMPT, ["send_to_telegram"])


def _run_single_step(step: Step, prior_context: str | None) -> StepResult:
    if step.command == "create-image":
        return _run_create_image_step(step)
    if step.command == "research":
        return _run_research_step(step, prior_context)
    if step.command == "summarize":
        return _run_summarize_step(step, prior_context)
    if step.command == "recap":
        return _run_recap_step(step, prior_context)
    if step.command == "note":
        return _run_note_step(step, prior_context)
    if step.command == "remind":
        return _run_remind_step(step, prior_context)
    if step.command == "tailor":
        return _run_tailor_step(step, prior_context)
    if step.command == "design-read":
        return _run_design_read_step(step, prior_context)

    routing = config.get_task_routing(step.command)
    if not routing:
        return StepResult(step=step, text="", error=f"No routing configured for /{step.command}")

    attempts = [routing["primary"]] + routing.get("fallback", [])
    last_error = None

    for i, attempt in enumerate(attempts):
        model = attempt.get("model")
        if not model:
            continue  # e.g. create-image with nothing free today — skip to next fallback
        try:
            provider = get_provider(attempt["provider"])
        except Exception as e:
            last_error = str(e)
            continue

        is_graph_data = step.command == "graph-data"

        messages = []
        if is_graph_data:
            messages.append(ChatMessage(role="system", content=GRAPH_DATA_SYSTEM_PROMPT))
        if prior_context:
            messages.append(ChatMessage(
                role="system",
                content=f"Context from a previous step in this chain:\n{prior_context}",
            ))
        messages.append(ChatMessage(role="user", content=step.text))

        try:
            if is_graph_data:
                png_bytes, filename, caption = _run_graph_data_step(provider, model, messages)
                return StepResult(
                    step=step,
                    text=caption,
                    degraded=(i > 0),
                    fallback_used=attempt if i > 0 else None,
                    image_bytes=png_bytes,
                    image_filename=filename,
                )

            response = provider.chat(model=model, messages=messages)
            return StepResult(
                step=step,
                text=response.text or "",
                degraded=(i > 0),  # true if this wasn't the primary
                fallback_used=attempt if i > 0 else None,
                usage_note=response.usage_note,
            )
        except (ProviderError, ChartError) as e:
            last_error = str(e)
            continue

    return StepResult(step=step, text="", error=last_error or "All providers failed")


def run_chain(steps: list[Step]) -> list[StepResult]:
    assign_slugs(steps)

    results: list[StepResult] = []
    prior_context: str | None = None
    contamination_trail: list[str] = []  # commands that were degraded so far

    for step in steps:
        result = _run_single_step(step, prior_context)
        result.contaminated_by = list(contamination_trail)  # snapshot before this step's own result

        if result.error:
            # Stop the chain — an unrecoverable step (all fallbacks failed)
            # should not let later steps run on nothing.
            results.append(result)
            break

        if result.degraded:
            contamination_trail.append(step.command)

        # Persist to Filen — a result that only lives in the chat reply
        # isn't "saved" in any real sense (Render's disk is scratch-only).
        # A save failure doesn't abort the chain — the AI result still
        # exists and reaches the user — but it IS flagged, since silently
        # losing something the user thinks got archived is its own kind
        # of failure worth disclosing, same principle as the degradation
        # disclosure above.
        if result.image_bytes:
            try:
                result.saved_path = save_bytes(
                    command=step.command,
                    topic_slug=step.topic_slug,
                    filename=result.image_filename or f"{step.command}.png",
                    content=result.image_bytes,
                )
            except StorageError as e:
                result.save_error = str(e)
        elif result.text:
            ext = EXTENSION_FOR_COMMAND.get(step.command, "txt")
            filename = f"{step.command}.{ext}"
            try:
                result.saved_path = save_result(
                    command=step.command,
                    topic_slug=step.topic_slug,
                    filename=filename,
                    content=result.text,
                )
            except StorageError as e:
                result.save_error = str(e)

        prior_context = result.text
        results.append(result)

    return results


def format_summary(results: list[StepResult]) -> str:
    """Builds the final reply message, explicit about any degradation."""
    lines = []
    for r in results:
        if r.error:
            lines.append(f"\u274c /{r.step.command} failed completely: {r.error}")
            lines.append("Chain stopped here \u2014 later steps did not run.")
            break

        header = f"\u2705 /{r.step.command}"
        if r.degraded:
            fb = r.fallback_used
            header = (
                f"\u26a0\ufe0f /{r.step.command} \u2014 primary model failed, "
                f"fallback used ({fb['provider']}/{fb['model']}). "
                f"Result may not be exactly what you expected."
            )
        if r.contaminated_by:
            names = ", ".join(f"/{c}" for c in r.contaminated_by)
            header += f" Note: this step received context from an earlier degraded step ({names})."

        if r.saved_path:
            header += f"\n\U0001f4be Saved to {r.saved_path}"
        elif r.save_error:
            header += f"\n\u26a0\ufe0f Not saved to storage: {r.save_error}"

        lines.append(header)
        # A step with a snippet (currently only /research) is long-form —
        # the snippet goes inline, the full r.text is delivered as an
        # attached file instead (see server.py's attachment handling).
        lines.append(r.snippet or r.text)
        if r.usage_note:
            lines.append(f"⚡ {r.usage_note}")
        lines.append("")  # blank line between steps

    return "\n".join(lines).strip()
