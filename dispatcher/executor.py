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
from dispatcher.mode_briefs import get_mode_brief, get_phase_brief
from dispatcher.parser import Step
from dispatcher.reminders import add_reminder
from dispatcher.research_status import set_status
from dispatcher.slugify import assign_slugs
from providers.base import ChatMessage, ChatResponse, Provider, ProviderError
from providers.registry import ProviderNotConfigured, get_provider
from storage.filen import StorageError, save_bytes, save_result
from tools.charts import CHART_TOOL_CHOICE, CHART_TOOL_NAME, CHART_TOOL_SCHEMA, ChartError, render_chart
from tools.documents import DocumentRenderError, render as render_document
from tools.image_gen import ImageGenError, generate_image
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


# /code saved with a hardcoded ".py" regardless of what language the
# model actually generated — detected instead from the language tag on
# the first fenced code block in the reply (```python, ```javascript,
# etc.), which every provider reliably includes for generated code.
# Falls back to EXTENSION_FOR_COMMAND's "py" default when nothing
# matches, rather than guessing.
_CODE_LANG_EXTENSIONS = {
    "python": "py", "py": "py",
    "javascript": "js", "js": "js",
    "typescript": "ts", "ts": "ts",
    "jsx": "jsx", "tsx": "tsx",
    "go": "go", "golang": "go",
    "rust": "rs", "rs": "rs",
    "java": "java",
    "c": "c",
    "cpp": "cpp", "c++": "cpp",
    "csharp": "cs", "c#": "cs", "cs": "cs",
    "ruby": "rb", "rb": "rb",
    "php": "php",
    "swift": "swift",
    "kotlin": "kt",
    "html": "html",
    "css": "css",
    "sql": "sql",
    "bash": "sh", "sh": "sh", "shell": "sh",
    "json": "json",
    "yaml": "yaml", "yml": "yaml",
    "xml": "xml",
}
_CODE_BLOCK_RE = re.compile(r"```(\w+)\n(.*?)```", re.DOTALL)
# Catches a fence the model opened but never closed — a real, observed
# formatting slip (e.g. a CSS block cut off with no trailing ```). Only
# searched in the text *after* the last complete match, so a properly
# closed block never gets double-counted as also being this trailing one.
_UNCLOSED_CODE_BLOCK_RE = re.compile(r"```(\w+)\n(.*)\Z", re.DOTALL)


def _extract_code_blocks(text: str) -> list[tuple[str, str]]:
    """Returns [(language, code), ...] for every fenced block in the
    reply, in order — not just the first, so an html+css+js reply can be
    bundled into one viewable page instead of only ever looking at the
    first block. A trailing block with no closing fence is still
    included (as "everything to the end") rather than silently dropped."""
    blocks = [(lang.lower(), code.strip("\n")) for lang, code in _CODE_BLOCK_RE.findall(text)]
    consumed_end = 0
    for m in _CODE_BLOCK_RE.finditer(text):
        consumed_end = m.end()
    trailing = _UNCLOSED_CODE_BLOCK_RE.search(text[consumed_end:])
    if trailing:
        blocks.append((trailing.group(1).lower(), trailing.group(2).strip("\n")))
    return blocks


# Filenames for /code's separate-files mode — real project-shaped names
# for the languages that pair up in a typical web reply, rather than
# generic "code.css"/"code.js".
_SEPARATE_FILE_NAME_FOR_LANG = {"html": "index", "css": "styles", "javascript": "script", "js": "script"}


@dataclass
class CodeFile:
    filename: str
    content: str
    ext: str
    viewable: bool = False


def _build_code_artifact(text: str, base_name: str) -> list[CodeFile]:
    """Turns a /code reply into real downloadable file(s) — deliberately
    NOT the raw chat reply (prose explanation + markdown fences), just
    the actual generated code, so the download button hands over real
    source, not a markdown transcript.

    Bundles into ONE self-contained HTML document (CSS/JS inlined) only
    when there's an HTML block AND 3+ total code blocks — JuanJo's call:
    a 2-block html+css reply is exactly the "two real files" case (an
    index.html that legitimately wants to sit next to its own styles.css)
    and shouldn't get silently merged; 3+ is where bundling into one
    viewable page actually saves real friction. Below that threshold, or
    with no HTML at all, every block becomes its own separate file.
    Falls back to the raw reply text as one .txt file if no fenced block
    was found at all, so nothing is silently dropped.
    """
    blocks = _extract_code_blocks(text)
    if not blocks:
        return [CodeFile(filename=f"{base_name}.txt", content=text, ext="txt")]

    html_blocks = [code for lang, code in blocks if lang == "html"]
    if html_blocks and len(blocks) >= 3:
        html = html_blocks[0]
        css_blocks = [code for lang, code in blocks if lang == "css"]
        js_blocks = [code for lang, code in blocks if lang in ("javascript", "js")]
        if css_blocks and "<style" not in html:
            style_tag = "<style>\n" + "\n".join(css_blocks) + "\n</style>"
            html = html.replace("</head>", f"{style_tag}\n</head>") if "</head>" in html else f"{style_tag}\n{html}"
        if js_blocks and "<script" not in html:
            script_tag = "<script>\n" + "\n".join(js_blocks) + "\n</script>"
            html = html.replace("</body>", f"{script_tag}\n</body>") if "</body>" in html else f"{html}\n{script_tag}"
        return [CodeFile(filename=f"{base_name}.html", content=html, ext="html", viewable=True)]

    files = []
    used_names: set[str] = set()
    for lang, code in blocks:
        ext = _CODE_LANG_EXTENSIONS.get(lang, "txt")
        stem = _SEPARATE_FILE_NAME_FOR_LANG.get(lang, base_name)
        name = f"{stem}.{ext}"
        suffix = 2
        while name in used_names:
            name = f"{stem}_{suffix}.{ext}"
            suffix += 1
        used_names.add(name)
        files.append(CodeFile(filename=name, content=code, ext=ext, viewable=(lang == "html")))
    return files


def _attach_requested_file(result: "StepResult", file_format: str | None) -> "StepResult":
    """Renders result.text into the requested format and attaches it
    alongside the plain-text reply — additive, not a replacement, same
    reasoning as /research always keeping its plain .md save regardless
    of what else gets attached."""
    if not file_format or not result.text:
        return result
    try:
        title = result.step.text[:80] or result.step.command
        result.rendered_file_bytes = render_document(file_format, title, result.text)
        result.rendered_file_name = f"{result.step.topic_slug or result.step.command}.{file_format}"
    except DocumentRenderError as e:
        result.save_error = f"Requested file format failed: {e}"
    return result

# LLM7's turbo (free) tier — the synthesis-phase model for /research
# (see _run_research_step). Hardcoded rather than a task_routing entry
# since this role is very specific: one-shot, huge-context synthesis
# over already-gathered material, not a general dispatcher role.
#
# Was DeepSeek-V4-Flash-0731 — moved to LLM7's paid "pro" tier
# (usage_based_only) at some point after this was first wired in, with
# no billing/payment method on this account. Every retry against it was
# hitting a permanent wall, not a transient "busy" state — the full 3min
# retry window burned on every single /research run for nothing before
# ever reaching the fallback. Verified live against LLM7's /v1/models
# endpoint (2026-08-27): gpt-oss is confirmed still on the free turbo
# tier (usage_based_only: false), has tool calling + reasoning, and
# 92.9% recent availability — a real number, not DeepSeek's permanent 0%.
SYNTHESIS_PROVIDER = "llm7"
SYNTHESIS_MODEL = "gpt-oss"

# Retry a flaky synthesis call every 30s for 3 minutes before giving up
# and falling back to the gathering model's own synthesis instead. LLM7's
# free tier proved to have real shared-capacity hiccups ("model
# temporarily busy") under light load — worth a few retries before
# accepting the fallback's lower quality.
SYNTHESIS_RETRY_DELAY_S = 30
SYNTHESIS_MAX_ATTEMPTS = 6

# Caps the gathered-material document handed to the synthesis model.
# gpt-oss's real context is 131,072 tokens (verified live) — noticeably
# smaller than DeepSeek's ~400k-1M this was originally sized against, so
# the document alone can't be allowed to eat the whole window: it still
# has to leave real room for the system prompt, the question, tool
# round-trips (ANALYSIS.md can call fetch_page), and the model's own
# output. Capped well under the ceiling on purpose, not right up against
# it — if the fallback synthesis model ends up being someone else
# entirely, this budget has to fit that one too, not just the best case.
RESEARCH_DOC_CHAR_BUDGET = 100_000 * 4

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
_NON_IDEMPOTENT_TOOLS = {"create_workflow", "run_workflow", "send_to_telegram", "save_note"}

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
    free-form mode-based chat, not just /research's command chain."""
    tools = tools if tools is not None else TOOL_SCHEMAS
    iterations = 0
    # Scoped to this one run_tool_loop call only — a fresh call (a new
    # provider attempt in dispatcher/chat.py's fallback chain) starts
    # empty, since that path is already guarded separately (see
    # dispatcher/chat.py's own DECISION-to-stop logic).
    already_executed: dict[tuple[str, str], str] = {}
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
            dedup_key = (tc.name, json.dumps(args, sort_keys=True)) if tc.name in _NON_IDEMPOTENT_TOOLS else None
            if dedup_key is not None and dedup_key in already_executed:
                result_text = already_executed[dedup_key]
                print(f"[run_tool_loop] iteration={iterations} SKIPPED duplicate call tool={tc.name} args={args} (reusing prior result)")
            else:
                print(f"[run_tool_loop] iteration={iterations} CALLING tool={tc.name} args={args}")
                # chat_messages lets create_workflow (tools/registry.py) capture
                # the real conversation that led to it, for Agent Vault's
                # "Instructions" — every other tool call ignores the key.
                result_text = dispatch_tool(tc.name, args, {**context, "chat_messages": messages})
                print(f"[run_tool_loop] iteration={iterations} RESULT tool={tc.name} result={result_text[:300]!r}")
                if dedup_key is not None:
                    already_executed[dedup_key] = result_text
            messages = messages + [ChatMessage(
                role="tool", content=result_text, tool_call_id=tc.id, name=tc.name,
            )]

        response = provider.chat(model=model, messages=messages, tools=tools)
        iterations += 1
        print(
            f"[run_tool_loop] iteration={iterations} model replied "
            f"text={response.text[:200]!r} next_tool_calls={[tc.name for tc in response.tool_calls]}"
        )

    print(f"[run_tool_loop] done after {iterations} iteration(s), final text={response.text[:200]!r}")
    return response, messages, iterations


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
    # Total LLM round-trips this step needed (fallback attempts + tool-
    # loop iterations combined) — "took too long" for a plan step means
    # too many tries, not wall-clock time (JuanJo's correction, 2026-09-01).
    # Defaults to 1 (a single clean call) for every construction site not
    # yet updated to pass a real count — only the /research gather+
    # synthesis path is wired so far; the other command types still
    # default here until they get the same treatment, deliberately not
    # done speculatively ahead of need.
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
    # Set only for /research — `text` is the full report (used for the
    # Filen save and for chunked delivery where there's no real
    # attachment support, e.g. push/PWA); `snippet` is a short version
    # for channels that can send the full report as a real file
    # alongside it (Telegram's caption + attached .md).
    snippet: str | None = None
    # Set when the user opted into a rendered file (e.g. "--file pdf" on
    # /summarize, /recap, /note) via _extract_file_request. Delivered
    # the same way /research's attachment is — an additional file
    # alongside the plain-text reply, not a replacement for it.
    rendered_file_bytes: bytes | None = None
    rendered_file_name: str | None = None
    rendered_file_saved_path: str | None = None  # "filen:..." — set once actually saved
    # Set only for /code — the extracted code file(s) (see
    # _build_code_artifact): one entry for a single-language reply or a
    # bundled HTML+CSS+JS page, multiple entries for a reply that
    # produced separate files (e.g. index.html + styles.css). Each gets
    # its own Filen save and its own download chip — real source, not
    # the raw chat reply's prose + markdown fences.
    code_files: list[CodeFile] = field(default_factory=list)
    code_saved: list[tuple[str, str]] = field(default_factory=list)  # [(filename, "filen:...saved_path"), ...]


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


def _run_research_gather_phase(step: Step, prior_context: str | None) -> tuple[str, str, bool, dict | None, str | None, int]:
    """
    Phase 1 — runs the tool-calling loop (web_search/fetch_page/save_note,
    scoped via GATHERING.md — not the full unscoped TOOL_SCHEMAS, which
    included send_to_telegram for no reason relevant to gathering) using
    /research's configured primary/fallback chain. Returns
    (gathered_document, gathering_model's own draft answer, degraded,
    fallback_used, error, attempt_count). `gathered_document` is the raw
    tool-result transcript, not the model's prose — that's what phase 2
    (_run_research_synthesis) actually works from. attempt_count is
    failed-provider-attempts-before-success + 1 + tool-loop iterations
    within the successful attempt (or every attempt tried, if all failed).
    """
    routing = config.get_task_routing("research")
    if not routing:
        return "", "", False, None, "No routing configured for /research", 0

    gathering_brief = get_phase_brief("GATHERING.md")
    gathering_tools = schemas_for(gathering_brief.tools)

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

        messages = [ChatMessage(role="system", content=gathering_brief.system_prompt)]
        if prior_context:
            messages.append(ChatMessage(
                role="system",
                content=f"Context from a previous step in this chain:\n{prior_context}",
            ))
        messages.append(ChatMessage(role="user", content=step.text))

        try:
            set_status(f"Researching — gathering sources ({attempt['provider']}/{model})…")
            response = provider.chat(model=model, messages=messages, tools=gathering_tools)
            response, full_messages, iterations = run_tool_loop(
                provider, model, messages, response,
                context={"command": "research", "topic_slug": step.topic_slug},
                tools=gathering_tools,
            )
            doc = _extract_tool_results(full_messages)
            if doc.strip():
                # Saved as its own artifact — separate from the final
                # synthesized report — so the raw gathered material is
                # inspectable afterward (what was actually scraped, whether
                # it contains characters that might trip an upstream
                # parser bug like ollama/ollama#17836) instead of only
                # existing as an in-memory variable that vanishes the
                # moment this function returns or the process crashes.
                # Best-effort: a save failure here doesn't abort the
                # command, same reasoning as the final result's save.
                try:
                    save_result(command="research", topic_slug=step.topic_slug, filename="gathered.md", content=doc)
                except StorageError:
                    pass
            return doc, response.text or "", (i > 0), (attempt if i > 0 else None), None, i + 1 + iterations
        except ProviderError as e:
            last_error = str(e)
            continue

    return "", "", False, None, last_error or "All research providers failed during gathering", len(attempts)


def _run_research_synthesis(step: Step, gathered_doc: str, gather_fallback: dict | None, routing: dict) -> tuple[str, bool, int]:
    """
    Phase 2 — synthesis over the gathered material, via DeepSeek/LLM7 for
    its large context, retried on transient failure (30s x 6 = 3min)
    before falling back to the same model that did the gathering.
    Returns (final_text, degraded, attempt_count).

    Uses ANALYSIS.md (its own dedicated brief — NOT RESEARCHER.md's, see
    ollama/ollama#17836 in the project notes for why that mattered: a
    prior version reused RESEARCHER.md's system prompt here, which
    describes a full tool belt this call was never actually given, and
    the model narrating/attempting a tool call the request didn't
    support is a plausible trigger for a real crash hit in production).
    ANALYSIS.md scopes exactly one real tool (fetch_page) so a model that
    identifies a genuine gap in the gathered material — "I need one more
    specific page on X" — can actually act on it via a real tool loop,
    instead of either hallucinating a tool call that doesn't exist or
    silently ignoring a gap it correctly spotted.
    """
    doc = gathered_doc
    if len(doc) > RESEARCH_DOC_CHAR_BUDGET:
        doc = doc[:RESEARCH_DOC_CHAR_BUDGET] + "\n\n[...truncated — gathered material exceeded the safe context budget]"

    analysis_brief = get_phase_brief("ANALYSIS.md")
    analysis_tools = schemas_for(analysis_brief.tools)
    synth_messages = [
        ChatMessage(role="system", content=analysis_brief.system_prompt),
        ChatMessage(role="user", content=f"Research question: {step.text}\n\nGathered material:\n{doc}"),
    ]

    def _run_synthesis_call(provider: Provider, model: str) -> tuple[str, int]:
        response = provider.chat(model=model, messages=synth_messages, tools=analysis_tools)
        response, _messages, iterations = run_tool_loop(
            provider, model, synth_messages, response,
            context={"command": "research", "topic_slug": step.topic_slug},
            tools=analysis_tools,
        )
        return response.text or "", iterations

    try:
        synth_provider = get_provider(SYNTHESIS_PROVIDER)
    except ProviderNotConfigured:
        synth_provider = None

    if synth_provider:
        for attempt_num in range(1, SYNTHESIS_MAX_ATTEMPTS + 1):
            try:
                set_status(f"Researching — synthesizing with {SYNTHESIS_MODEL} (attempt {attempt_num}/{SYNTHESIS_MAX_ATTEMPTS})…")
                text, iterations = _run_synthesis_call(synth_provider, SYNTHESIS_MODEL)
                return text, False, attempt_num + iterations
            except ProviderError:
                if attempt_num < SYNTHESIS_MAX_ATTEMPTS:
                    set_status(f"Researching — {SYNTHESIS_MODEL} busy, retrying in {SYNTHESIS_RETRY_DELAY_S}s ({attempt_num}/{SYNTHESIS_MAX_ATTEMPTS})…")
                    time.sleep(SYNTHESIS_RETRY_DELAY_S)

    # Synthesis model never came through — fall back to the gathering model
    # itself doing the synthesis instead of losing the result outright.
    set_status(f"Researching — {SYNTHESIS_MODEL} unavailable, finishing with the gathering model…")
    fallback_attempt = gather_fallback or routing["primary"]
    try:
        provider = get_provider(fallback_attempt["provider"])
        text, iterations = _run_synthesis_call(provider, fallback_attempt["model"])
        return text, True, SYNTHESIS_MAX_ATTEMPTS + 1 + iterations
    except ProviderError:
        return "", True, SYNTHESIS_MAX_ATTEMPTS + 1


def _run_research_step(step: Step, prior_context: str | None) -> StepResult:
    routing = config.get_task_routing("research")
    if not routing:
        return StepResult(step=step, text="", error="No routing configured for /research")

    gathered_doc, draft_text, gather_degraded, gather_fallback, gather_error, gather_attempts = _run_research_gather_phase(step, prior_context)
    if gather_error:
        set_status(None)
        return StepResult(step=step, text="", error=gather_error, attempt_count=gather_attempts)

    if not gathered_doc.strip():
        # No tool calls happened — model answered from reasoning alone,
        # nothing gathered to synthesize from.
        set_status(None)
        return StepResult(step=step, text=draft_text, degraded=gather_degraded, fallback_used=gather_fallback, attempt_count=gather_attempts)

    final_text, synth_degraded, synth_attempts = _run_research_synthesis(step, gathered_doc, gather_fallback, routing)
    set_status(None)
    total_attempts = gather_attempts + synth_attempts

    if not final_text:
        return StepResult(step=step, text="", error="Research gathering succeeded but synthesis failed on every attempt", attempt_count=total_attempts)

    return StepResult(
        step=step,
        text=final_text,
        snippet=_make_snippet(final_text),
        degraded=gather_degraded or synth_degraded,
        fallback_used=gather_fallback,
        attempt_count=total_attempts,
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
    text, file_format = _extract_file_request(step.text)
    working_step = Step(command=step.command, text=text, topic_slug=step.topic_slug)
    result = _run_text_transform_step(working_step, prior_context, "summarize", SUMMARIZE_SYSTEM_PROMPT, ["fetch_page"])
    return _attach_requested_file(result, file_format)


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
            code_files = []
            if step.command == "code":
                code_files = _build_code_artifact(response.text or "", step.topic_slug or "code")
            return StepResult(
                step=step,
                text=response.text or "",
                degraded=(i > 0),  # true if this wasn't the primary
                fallback_used=attempt if i > 0 else None,
                usage_note=response.usage_note,
                code_files=code_files,
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
        elif result.code_files:
            # /code — one save per extracted file (see _build_code_artifact:
            # a bundled HTML page is still exactly one entry here, a
            # separate-files reply is several). Each failure is disclosed
            # individually rather than aborting the rest.
            for cf in result.code_files:
                try:
                    saved = save_result(
                        command=step.command, topic_slug=step.topic_slug,
                        filename=cf.filename, content=cf.content,
                    )
                    result.code_saved.append((cf.filename, saved))
                except StorageError as e:
                    result.save_error = (result.save_error or "") + f" {cf.filename} not saved: {e}"
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
        # instead of it, same reasoning as /research always keeping its
        # plain .md regardless of what else gets attached.
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
        # A step with a snippet (currently only /research) is long-form —
        # the snippet goes inline, the full r.text is delivered as an
        # attached file instead (see server.py's attachment handling).
        lines.append(r.snippet or r.text)
        if r.usage_note:
            lines.append(f"⚡ {r.usage_note}")
        lines.append("")  # blank line between steps

    return "\n".join(lines).strip()
