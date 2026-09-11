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
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

from config.store import config
from dispatcher.mode_briefs import get_mode_brief
from dispatcher.parser import Step
from dispatcher.reminders import add_reminder
from dispatcher.slugify import assign_slugs
from providers.base import ChatMessage, ChatResponse, Provider, ProviderError
from providers.registry import ProviderNotConfigured, get_provider
from storage.filen import StorageError, save_bytes, save_result
from tools.charts import CHART_TOOL_CHOICE, CHART_TOOL_NAME, CHART_TOOL_SCHEMA, ChartError, render_chart
from tools.documents import DocumentRenderError, render as render_document
from tools.registry import TOOL_SCHEMAS, schemas_for
from tools.registry import dispatch as dispatch_tool

# Opt-in file request for commands that don't save one by default
# (/summarize, /recap, /note) — "--file" alone defaults to PDF (94% of
# organizations use PDF as their primary format for finished business
# documents, verified against real usage data rather than assumed), or
# an explicit format: "--file docx", "--file pptx". Deterministic
# regex, not an AI call, matching how dispatcher/parser.py itself
# handles command detection — no reason to spend a request just to
# notice a flag.
_FILE_REQUEST_RE = re.compile(r"--file(?:\s+(pdf|docx|pptx))?\b", re.IGNORECASE)


def _extract_file_request(text: str) -> tuple[str, str | None]:
    """Returns (text with the flag stripped out, requested format or None)."""
    match = _FILE_REQUEST_RE.search(text)
    if not match:
        return text, None
    fmt = (match.group(1) or "pdf").lower()
    cleaned = (text[:match.start()] + text[match.end():]).strip()
    return cleaned, fmt


def _attach_requested_file(result: "StepResult", file_format: str | None) -> "StepResult":
    """Renders result.text into the requested format and attaches it
    alongside the plain-text reply — additive, not a replacement, so the
    plain .md save always still happens regardless of what else gets
    attached."""
    if not file_format or not result.text:
        return result
    try:
        title = result.step.text[:80] or result.step.command
        result.rendered_file_bytes = render_document(file_format, title, result.text)
        result.rendered_file_name = f"{result.step.topic_slug or result.step.command}.{file_format}"
    except DocumentRenderError as e:
        result.save_error = f"Requested file format failed: {e}"
    return result


# File extension per command — used when saving each step's output.
EXTENSION_FOR_COMMAND = {
    "graph-data": "png",
    "summarize": "md",
    "recap": "md",
    "note": "md",
    "remind": "md",
}

# /summarize gets exactly one tool (fetch_page), not a full tool belt —
# it's a single-phase digest. The model decides whether to call it: if
# the input is already pasted text, there's nothing to fetch.
SUMMARIZE_SYSTEM_PROMPT = (
    "Produce a tight, faithful digest of the given content. If the message "
    "contains a URL, call fetch_page to read it first, then summarize what you "
    "fetched — don't just describe the link. If it's already pasted text, "
    "summarize that directly. Keep it dense: hit the key points, skip padding, "
    "no filler intro like 'Here is a summary'. Preserve concrete numbers, names, "
    "and dates from the source."
)

# /graph-data doesn't get a general tool belt — it gets exactly one
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

# Tools whose side effect is real and non-idempotent — calling one twice
# with identical arguments doesn't "check again," it repeats the action
# (a second Telegram message, a second scheduled workflow, a second run).
# Deliberately NOT every tool: get_run_status/list_workflow_runs/
# web_search/fetch_page can legitimately return a different answer on a
# repeat call (time passed, a run's status changed) — guarding those
# would serve stale/wrong data instead of a fresh check. 2026-09-03,
# JuanJo: one "send me a Telegram message" request produced 5, then 10,
# duplicate workflows — this guards the case that survives even after
# dispatcher/chat.py stopped retrying a NEW fallback provider: the SAME
# provider re-issuing the same tool call within its own turn.
_NON_IDEMPOTENT_TOOLS = {"create_workflow", "run_workflow", "send_to_telegram", "save_note", "save_source"}

# Some non-idempotent tools legitimately vary a field the identical-args
# dedup above would otherwise treat as "different." save_source's title/
# content wording can differ slightly between retries of the literal same
# page — the same shape of problem create_workflow's varying names caused
# (see _ONCE_PER_TURN_TOOLS below) — but unlike create_workflow, one real
# turn legitimately calls save_source more than once (one page can be
# relevant to several different search terms in the same batch), so a
# blanket once-per-turn rule would be wrong here. This narrows the dedup
# key to just the fields that actually identify "the same source for the
# same term," so wording differences in title/content don't defeat it —
# 2026-09-06, JuanJo: a single Sources batch saved 8 near-duplicate
# documents for one term, same root cause as the create_workflow incident.
_DEDUP_KEY_FIELDS = {"save_source": ("term", "url")}

# create_workflow specifically gets a HARDER rule than the identical-args
# dedup above: at most one real execution per run_tool_loop call, full
# stop, no matter what arguments the model uses on later attempts.
# 2026-09-03, JuanJo, real evidence from production: a single stateless
# Agent Work Chat turn (one "send Hola to my telegram" message, confirmed
# by identical creation_transcript timestamps) called create_workflow 4
# times, each with a DIFFERENT name ("Send Hola", "Send Hola Telegram",
# "Send Hola to Telegram", "Send Hola Telegram" again) — the identical-
# args guard above never caught it because the arguments genuinely
# differed each time; the model wasn't repeating itself, it was retrying
# with variations, seemingly not registering "Created workflow X." as a
# real success. Same root cause explained why no Telegram message ever
# arrived: none of the 4 was ever run — the model stayed stuck re-trying
# creation and never reached its own brief's "call run_workflow" step.
# One workflow per turn is also just the correct semantics here — Agent
# Work Chat's whole job for a given message is building ONE workflow (or
# running/checking one), never several.
_ONCE_PER_TURN_TOOLS = {"create_workflow"}

# Tells the model to cite sources as Markdown links rather than pasting
# bare URLs — both messaging adapters render [text](url) as a clickable
# link (Telegram via an HTML conversion, Discord natively), so this is
# what actually makes citations clickable in the final reply.
CITATION_STYLE_PROMPT = (
    "When you cite a source you found via web_search or fetch_page, format it as a "
    "Markdown link: [short title](url). Don't paste bare URLs in your answer."
)


def run_tool_loop(
    provider: Provider, model: str, messages: list[ChatMessage], response: ChatResponse, context: dict,
    tools: list[dict] | None = None,
) -> tuple[ChatResponse, list[ChatMessage], int]:
    """Executes any tool_calls in `response`, feeds results back to the
    model, and repeats until the model stops asking for tools or the
    iteration ceiling is hit. Returns (final ChatResponse, full message
    transcript, iteration count) — the transcript lets a caller extract
    raw tool results (see _extract_tool_results) without needing the
    model's own final prose synthesis. The iteration count feeds
    StepResult.attempt_count (2026-09-01) — "took too long" for a plan
    step means too many LLM round-trips, not wall-clock seconds, so this
    can't stay a discarded local variable anymore.

    `tools` should be whatever scoped list the caller's *first* call to
    the model already used — every follow-up call inside this loop reuses
    it, so a caller that scoped down to e.g. just fetch_page doesn't
    silently regain the full tool belt on iteration 2 (defaults to
    TOOL_SCHEMAS only for callers that genuinely want the whole belt).

    Public (not `_`-prefixed) because dispatcher/chat.py reuses this for
    free-form mode-based chat, not just the command chain here."""
    tools = tools if tools is not None else TOOL_SCHEMAS
    iterations = 0
    # Scoped to this one run_tool_loop call only — a fresh call (a new
    # provider attempt in dispatcher/chat.py's fallback chain) starts
    # empty, since that path is already guarded separately (see
    # dispatcher/chat.py's own DECISION-to-stop logic).
    already_executed: dict[tuple[str, str], str] = {}
    once_per_turn_executed: dict[str, str] = {}
    print(f"[run_tool_loop] start model={model} initial_tool_calls={[tc.name for tc in response.tool_calls]}")
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
            if tc.name in _NON_IDEMPOTENT_TOOLS:
                key_fields = _DEDUP_KEY_FIELDS.get(tc.name)
                key_args = {f: args.get(f) for f in key_fields} if key_fields else args
                dedup_key = (tc.name, json.dumps(key_args, sort_keys=True))
            else:
                dedup_key = None
            if tc.name in _ONCE_PER_TURN_TOOLS and tc.name in once_per_turn_executed:
                # Hard stop regardless of arguments — see _ONCE_PER_TURN_TOOLS'
                # own comment. Feeds the model an explicit correction (not
                # just a silent repeat of the same result, which is what the
                # softer identical-args guard below does) — the whole point
                # is nudging it toward its next real step (run_workflow, or
                # just reporting success) instead of retrying creation again.
                prior_result = once_per_turn_executed[tc.name]
                result_text = (
                    f"Already done this turn: {prior_result} Do not call {tc.name} again — "
                    "it already succeeded, calling it again would create a duplicate. If the user "
                    "wants it to run right now, call run_workflow with the workflow_id from that "
                    "result. Otherwise, just report it's been created."
                )
                print(f"[run_tool_loop] iteration={iterations} BLOCKED repeat call tool={tc.name} args={args} (already executed once this turn)")
            elif dedup_key is not None and dedup_key in already_executed:
                result_text = already_executed[dedup_key]
                print(f"[run_tool_loop] iteration={iterations} SKIPPED duplicate call tool={tc.name} args={args} (reusing prior result)")
            else:
                print(f"[run_tool_loop] iteration={iterations} CALLING tool={tc.name} args={args}")
                # chat_messages lets create_workflow (tools/registry.py) capture
                # the real conversation that led to it, for Agent Vault's
                # "Instructions" — every other tool call ignores the key.
                result_text = dispatch_tool(tc.name, args, {**context, "chat_messages": messages})
                print(f"[run_tool_loop] iteration={iterations} RESULT tool={tc.name} result={result_text[:300]!r}")
                if tc.name in _ONCE_PER_TURN_TOOLS:
                    once_per_turn_executed[tc.name] = result_text
                # save_source's content-quality guard (tools/registry.py)
                # sends the model a "NOT usable, try again" message rather
                # than a hard failure — deliberately meant to be retried
                # for the SAME (term, url) with better content. Caching
                # that rejection under the (term, url) dedup key would
                # permanently block the retry from ever re-executing,
                # replaying the same rejection forever instead — so a
                # rejected save_source call is excluded from the cache
                # (2026-09-06), while an actually-saved one still is.
                is_rejected_retry = tc.name == "save_source" and result_text.startswith("NOT usable")
                if dedup_key is not None and not is_rejected_retry:
                    already_executed[dedup_key] = result_text
            messages = messages + [ChatMessage(
                role="tool", content=result_text, tool_call_id=tc.id, name=tc.name,
            )]

        response = provider.chat(model=model, messages=messages, tools=tools)
        iterations += 1
        print(
            f"[run_tool_loop] iteration={iterations} model replied "
            f"text={(response.text or '')[:200]!r} next_tool_calls={[tc.name for tc in response.tool_calls]}"
        )

    print(f"[run_tool_loop] done after {iterations} iteration(s), final text={(response.text or '')[:200]!r}")
    return response, messages, iterations


def _extract_tool_results(messages: list[ChatMessage]) -> str:
    """Concatenates every tool-result message from a run_tool_loop
    transcript into one readable document — the raw gathered material
    (search snippets, fetched page text), not the model's own prose
    synthesis."""
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
    # Total LLM round-trips this step needed (fallback attempts + tool-
    # loop iterations combined) — "took too long" for a plan step means
    # too many tries, not wall-clock time (JuanJo's correction, 2026-09-01).
    # Defaults to 1 (a single clean call) for every construction site not
    # yet updated to pass a real count, deliberately not done
    # speculatively ahead of need.
    attempt_count: int = 1
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
    # Set when the user opted into a rendered file (e.g. "--file pdf" on
    # /summarize, /recap, /note) via _extract_file_request — an additional
    # file alongside the plain-text reply, not a replacement for it.
    rendered_file_bytes: bytes | None = None
    rendered_file_name: str | None = None
    rendered_file_saved_path: str | None = None  # "filen:..." — set once actually saved


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
    tool belt available — simple enough for one call, no gather-then-
    synthesize pipeline needed."""
    routing = config.get_task_routing(command)
    if not routing:
        return StepResult(step=step, text="", error=f"No routing configured for /{command}")

    attempts = config.get_attempts([routing["primary"]] + routing.get("fallback", []))
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
            scoped_tools = schemas_for(tool_names)
            response = provider.chat(model=model, messages=messages, tools=scoped_tools)
            response, _messages, iterations = run_tool_loop(
                provider, model, messages, response,
                context={"command": command, "topic_slug": step.topic_slug},
                tools=scoped_tools,
            )
            return StepResult(
                step=step,
                text=response.text or "",
                degraded=(i > 0),
                fallback_used=attempt if i > 0 else None,
                usage_note=response.usage_note,
                attempt_count=i + 1 + iterations,
            )
        except ProviderError as e:
            last_error = str(e)
            if e.is_rate_limit:
                config.mark_rate_limited(attempt["provider"], model)
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

    attempts = config.get_attempts([routing["primary"]] + routing.get("fallback", []))
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
            if isinstance(e, ProviderError) and e.is_rate_limit:
                config.mark_rate_limited(attempt["provider"], model)
            continue

    return StepResult(step=step, text="", error=last_error or "Couldn't set the reminder — all providers failed")


def _run_summarize_step(step: Step, prior_context: str | None) -> StepResult:
    text, file_format = _extract_file_request(step.text)
    working_step = Step(command=step.command, text=text, topic_slug=step.topic_slug)
    result = _run_text_transform_step(working_step, prior_context, "summarize", SUMMARIZE_SYSTEM_PROMPT, ["fetch_page"])
    return _attach_requested_file(result, file_format)


def _run_recap_step(step: Step, prior_context: str | None) -> StepResult:
    text, file_format = _extract_file_request(step.text)
    working_step = Step(command=step.command, text=text, topic_slug=step.topic_slug)
    result = _run_text_transform_step(working_step, prior_context, "recap", RECAP_SYSTEM_PROMPT, ["send_to_telegram"])
    return _attach_requested_file(result, file_format)


def _run_note_step(step: Step, prior_context: str | None) -> StepResult:
    text, file_format = _extract_file_request(step.text)
    working_step = Step(command=step.command, text=text, topic_slug=step.topic_slug)
    result = _run_text_transform_step(working_step, prior_context, "note", NOTE_SYSTEM_PROMPT, ["send_to_telegram"])
    return _attach_requested_file(result, file_format)


def _run_single_step(step: Step, prior_context: str | None) -> StepResult:
    if step.command == "summarize":
        return _run_summarize_step(step, prior_context)
    if step.command == "recap":
        return _run_recap_step(step, prior_context)
    if step.command == "note":
        return _run_note_step(step, prior_context)
    if step.command == "remind":
        return _run_remind_step(step, prior_context)

    routing = config.get_task_routing(step.command)
    if not routing:
        return StepResult(step=step, text="", error=f"No routing configured for /{step.command}")

    attempts = config.get_attempts([routing["primary"]] + routing.get("fallback", []))
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
            if isinstance(e, ProviderError) and e.is_rate_limit:
                config.mark_rate_limited(attempt["provider"], model)
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

        # A requested file render (e.g. "--file pdf") is additive — saved
        # alongside whatever the plain-text branch above already did, not
        # instead of it.
        if result.rendered_file_bytes and result.rendered_file_name:
            try:
                result.rendered_file_saved_path = save_bytes(
                    command=step.command,
                    topic_slug=step.topic_slug,
                    filename=result.rendered_file_name,
                    content=result.rendered_file_bytes,
                )
            except StorageError as e:
                result.save_error = (result.save_error or "") + f" Requested file not saved: {e}"

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
        lines.append(r.text)
        if r.usage_note:
            lines.append(f"⚡ {r.usage_note}")
        lines.append("")  # blank line between steps

    return "\n".join(lines).strip()
