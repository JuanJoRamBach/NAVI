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
from dataclasses import dataclass, field

from config.store import config
from dispatcher.parser import Step
from dispatcher.slugify import assign_slugs
from providers.base import ChatMessage, ChatResponse, Provider, ProviderError
from providers.registry import get_provider
from storage.filen import StorageError, save_result
from tools.registry import TOOL_SCHEMAS
from tools.registry import dispatch as dispatch_tool

# File extension per command — used when saving each step's output.
EXTENSION_FOR_COMMAND = {
    "research": "md",
    "code": "py",  # best-guess default; language-specific naming can improve this later
    "graph-data": "md",
    "create-image": "png",
    "brainstorm": "md",
}

# Which commands get the tool belt (web_search, fetch_page, save_note)
# available to their provider calls. Research is the only one that needs
# to look things up live; the others work from the prompt text alone.
TOOL_ENABLED_COMMANDS = {"research"}

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


def _run_tool_loop(
    provider: Provider, model: str, messages: list[ChatMessage], response: ChatResponse, context: dict
) -> ChatResponse:
    """Executes any tool_calls in `response`, feeds results back to the
    model, and repeats until the model stops asking for tools or the
    iteration ceiling is hit. Returns the final ChatResponse."""
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

    return response


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


def _run_single_step(step: Step, prior_context: str | None) -> StepResult:
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

        use_tools = step.command in TOOL_ENABLED_COMMANDS
        tools = TOOL_SCHEMAS if use_tools else None

        messages = []
        if use_tools:
            messages.append(ChatMessage(role="system", content=CITATION_STYLE_PROMPT))
        if prior_context:
            messages.append(ChatMessage(
                role="system",
                content=f"Context from a previous step in this chain:\n{prior_context}",
            ))
        messages.append(ChatMessage(role="user", content=step.text))

        try:
            response = provider.chat(model=model, messages=messages, tools=tools)
            if use_tools:
                response = _run_tool_loop(
                    provider, model, messages, response,
                    context={"command": step.command, "topic_slug": step.topic_slug},
                )
            return StepResult(
                step=step,
                text=response.text or "",
                degraded=(i > 0),  # true if this wasn't the primary
                fallback_used=attempt if i > 0 else None,
            )
        except ProviderError as e:
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
        if result.text:
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
        lines.append(r.text)
        lines.append("")  # blank line between steps

    return "\n".join(lines).strip()
