"""
dispatcher/research.py

Real dispatcher wiring for Research mode's plan-then-execute design
(RESEARCHER.md + RESEARCH_EXECUTE_PLAN.md, redesigned 2026-09-11 but never
actually connected to anything — MODE_FILES only ever mapped "research" to
the planning-stage brief, so a user who accepted a drafted plan had no way
to actually trigger the execution stage; RESEARCHER.md also, until now,
drafted AND presented the plan itself in the same continuous chat, which
how_to_handle_context.md's 2026-09-12 design explicitly moved to a
dispatcher-mediated checkpoint instead — see that file's "Research and
Brainstorm's context design" section for the full rationale).

Real state machine, tracked via storage/conversations.py's existing
per-conversation task_state blob (already generic, just unused by Research
until now):

    planning              -> ordinary clarifying turns (RESEARCHER.md),
                              until the model calls propose_plan_ready.
    awaiting_readiness    -> dispatcher already asked "ready to draft?" —
                              a "yes" triggers the compact_conversation
                              draft; anything else falls back to more
                              clarifying (still "planning" in spirit).
    awaiting_plan_confirm -> dispatcher already presented the drafted
                              plan — "yes" triggers real execution,
                              "cancel" resets, anything else goes back to
                              clarifying.
    done                  -> a finished research turn; the NEXT message
                              starts a fresh planning cycle in the same
                              conversation rather than getting stuck.

Deliberately NOT wired for Brainstorm yet — that mode still goes through
dispatcher/chat.py's run_stored_mode_chat unchanged; BRAINSTORM.md needs
the equivalent propose_plan_ready-style trigger added first (flagged in
how_to_handle_context.md, not done).

Known, deliberate scope limit: RESEARCH_EXECUTE_PLAN.md permits the model
one clarifying ask_user_choice call mid-gathering if it hits a genuine
gap the plan didn't anticipate. This first wiring pass does NOT support
resuming a paused execution from that point — the tool-loop transcript
isn't persisted the way Dev Slate's file-relay state is. If it happens,
the broad except around _run_execution's tool loop below turns it into
"try the next fallback" rather than a crash, and if every attempt hits
it, the user sees a real failure message and can just re-confirm the
plan to retry from scratch. Real resume support is future work, not
silently pretended to exist here.
"""

import asyncio
import json
from datetime import datetime, timezone

from config.store import config
from dispatcher.compaction import compact_conversation, strip_code_fence
from dispatcher.executor import _parse_tool_args, run_tool_loop
from dispatcher.mode_briefs import get_mode_brief
from dispatcher.prompt_family import adapt_request_params, adapt_system_prompt, classify_family
from dispatcher.slugify import slugify
from providers.base import ChatMessage, ProviderError
from providers.registry import ProviderNotConfigured, get_dispatcher_role, get_provider
from storage.conversations import append_message, get_messages, get_task_state, set_task_state
from storage.filen import StorageError, file_download_url, save_bytes
from tools.registry import schemas_for
from tools.report_render import ReportRenderError, convert_docx_to_pdf, render_research_report_docx

RECENT_MESSAGE_WINDOW = 20

STAGE_PLANNING = "planning"
STAGE_AWAITING_READINESS = "awaiting_readiness"
STAGE_AWAITING_PLAN_CONFIRM = "awaiting_plan_confirm"
STAGE_DONE = "done"

READY_CHECK_QUESTION = "I think I have enough to draft a research plan. Ready for me to draft it?"
READY_CHECK_OPTIONS = ["Yes, draft the plan", "Not yet — let me add more"]
PLAN_CONFIRM_OPTIONS = ["Looks good, start researching", "Let me adjust something", "Cancel"]

# Mirrors RESEARCHER.md's old (now-removed) "plan's structure" section —
# moved here since the drafting itself moved from the model's own
# continuous chat turn to this dedicated compact_conversation call, and a
# schema belongs next to the code that enforces it, not duplicated in two
# places that could drift apart.
PLAN_DRAFT_INSTRUCTION = """You are drafting a research plan from a conversation between a user and an assistant that was clarifying what to research. Read the full conversation and produce a plan the user can review before real research work begins.

Reply with ONLY a single JSON object, no prose, no markdown fence, matching exactly this shape:
{
  "goal": "string — the actual question this research needs to answer, one sentence",
  "method": ["string", "..."] ,
  "deliverable": "string",
  "verification": "string"
}

- "method": 2 to 4 concrete, non-overlapping sub-questions the research should pursue — not a single vague search, each should give a distinct thing to look for. Fold in any specific source the user already mentioned as part of the relevant sub-question's own wording rather than inventing a separate field for it.
- "deliverable": what "professional" means for this specific request — who it's for, whether it should end in recommendations or is purely descriptive, and any depth/format expectation the user stated or implied. Don't invent formality the user never asked for.
- "verification": what "enough" looks like — how the user will know the research actually answered the goal, not just produced material.

Re-read your own draft once before answering: check for a sub-question redundant with another, scope too vague to actually search against, or a goal the sub-questions don't fully cover. Fix silently — output only the corrected version, don't narrate what you fixed.

Base this ENTIRELY on what the conversation actually established — never invent a detail the user didn't state or clearly imply."""


def _format_plan_markdown(plan: dict) -> str:
    method_lines = "\n".join(f"- {q}" for q in plan.get("method") or [])
    return (
        f"**Goal:** {plan.get('goal', '')}\n\n"
        f"**Method:**\n{method_lines}\n\n"
        f"**Deliverable:** {plan.get('deliverable', '')}\n\n"
        f"**Verification:** {plan.get('verification', '')}"
    )


def _is_affirmative(text: str, affirmative_label: str) -> bool:
    """The exact button label is the reliable signal — ChoiceButtons.onPick
    sends the clicked option's text back verbatim (navi-pwa/App.tsx). The
    prefix check underneath is a soft fallback for a user who types their
    own words instead of clicking."""
    t = text.strip()
    if t == affirmative_label:
        return True
    return t.lower().startswith(("yes", "yeah", "yep", "looks good", "start", "sure", "go ahead", "sounds good"))


def _history_to_messages(history: list[dict]) -> list[ChatMessage]:
    messages = []
    for m in history:
        if m["role"] == "navi" and m["content"].startswith("⚠️"):
            continue
        messages.append(ChatMessage(role="assistant" if m["role"] == "navi" else m["role"], content=m["content"]))
    return messages


async def run_research_chat(conversation_id: str, text: str) -> dict:
    """Entry point — server.py calls this for mode == "research" instead
    of dispatcher/chat.py's generic run_stored_mode_chat, since Research
    now needs real stage-tracking that mode has no equivalent of."""
    await append_message(conversation_id, "user", text)
    task_state = await get_task_state(conversation_id) or {}
    stage = task_state.get("stage", STAGE_PLANNING)

    if stage == STAGE_AWAITING_READINESS:
        if _is_affirmative(text, READY_CHECK_OPTIONS[0]):
            return await _draft_and_present_plan(conversation_id)
        # "not yet" or free-form — back to ordinary clarifying. Persisted
        # immediately, not just reassigned locally — get_task_state reads
        # fresh from storage on the NEXT call, so a local-only reassignment
        # here would leave the DB still saying "awaiting_readiness" and
        # misroute the turn after this one.
        await set_task_state(conversation_id, {"stage": STAGE_PLANNING, "plan": None})
        stage = STAGE_PLANNING

    if stage == STAGE_AWAITING_PLAN_CONFIRM:
        if _is_affirmative(text, PLAN_CONFIRM_OPTIONS[0]):
            plan = task_state.get("plan")
            if plan:
                return await _run_execution(conversation_id, plan)
            # Corrupted/missing plan in task_state — don't crash, just
            # restart planning rather than trust a state that isn't there.
            await set_task_state(conversation_id, {"stage": STAGE_PLANNING, "plan": None})
            stage = STAGE_PLANNING
        elif t := text.strip():
            if t == PLAN_CONFIRM_OPTIONS[2] or t.lower().startswith("cancel"):
                await set_task_state(conversation_id, {"stage": STAGE_PLANNING, "plan": None})
                reply = "Okay, cancelled. Let me know what you'd like to research and we'll start fresh."
                await append_message(conversation_id, "navi", reply)
                return {"text": reply, "provider": None, "model": None}
            # "Let me adjust something" or any other written feedback —
            # back to clarifying, with this message as the next input.
            await set_task_state(conversation_id, {"stage": STAGE_PLANNING, "plan": task_state.get("plan")})
            stage = STAGE_PLANNING

    if stage == STAGE_DONE:
        # A finished research turn just starts a fresh planning cycle on
        # the next message instead of getting stuck — same conversation,
        # a new topic. Persisted (see the comment above on why a local-
        # only reassignment isn't enough).
        await set_task_state(conversation_id, {"stage": STAGE_PLANNING, "plan": None})
        stage = STAGE_PLANNING

    return await _run_planning_turn(conversation_id)


async def _run_planning_turn(conversation_id: str) -> dict:
    brief = get_mode_brief("research")
    history = await get_messages(conversation_id, limit=RECENT_MESSAGE_WINDOW)
    tools = schemas_for(brief.tools)
    history_messages = _history_to_messages(history)
    if not history_messages:
        history_messages = [ChatMessage(role="user", content="")]
    history_messages[-1].content = f"{history_messages[-1].content}\n\n[Current UTC time: {datetime.now(timezone.utc).isoformat()}]"

    try:
        role = get_dispatcher_role(context="chat")
    except ProviderNotConfigured as e:
        error_text = f"⚠️ Can't reply right now — research isn't configured: {e}"
        await append_message(conversation_id, "navi", error_text)
        return {"text": error_text, "provider": None, "model": None}

    attempts = config.get_attempts([{"provider": role["provider"], "model": role["model"]}] + role.get("fallback", []))
    last_error = None
    for i, attempt in enumerate(attempts):
        try:
            provider = get_provider(attempt["provider"])
        except Exception as e:
            last_error = str(e)
            continue
        family = classify_family(attempt["provider"], attempt["model"])
        system_content = adapt_system_prompt(brief.system_prompt, family, attempt["model"])
        extra_params = adapt_request_params(family, attempt["provider"], has_tools=True) or None
        messages = list(history_messages)
        if system_content is not None:
            messages.insert(0, ChatMessage(role="system", content=system_content))
        try:
            response = await asyncio.to_thread(
                provider.chat, model=attempt["model"], messages=messages, tools=tools, extra_params=extra_params,
            )
            ready_call = next((tc for tc in response.tool_calls if tc.name == "propose_plan_ready"), None)
            if ready_call:
                await set_task_state(conversation_id, {"stage": STAGE_AWAITING_READINESS, "plan": None})
                await append_message(conversation_id, "navi", READY_CHECK_QUESTION, provider=attempt["provider"], model=attempt["model"])
                return {
                    "text": READY_CHECK_QUESTION, "provider": attempt["provider"], "model": attempt["model"],
                    "choices": list(READY_CHECK_OPTIONS),
                }
            choice_call = next((tc for tc in response.tool_calls if tc.name == "ask_user_choice"), None)
            if choice_call:
                args = _parse_tool_args(choice_call.arguments)
                question = args.get("question", "")
                options = args.get("options") or []
                await append_message(conversation_id, "navi", question, provider=attempt["provider"], model=attempt["model"])
                return {"text": question, "provider": attempt["provider"], "model": attempt["model"], "choices": options}
            if not response.text and not response.tool_calls:
                last_error = f"{attempt['provider']}/{attempt['model']} returned neither text nor a tool call"
                continue
            reply = response.text or "(empty reply)"
            if i > 0:
                reply += f"\n\n⚡ (primary was unavailable, answered via {attempt['provider']}/{attempt['model']} instead)"
            await append_message(conversation_id, "navi", reply, provider=attempt["provider"], model=attempt["model"])
            return {"text": reply, "provider": attempt["provider"], "model": attempt["model"], "usage_note": response.usage_note}
        except ProviderError as e:
            last_error = str(e)
            if e.is_rate_limit:
                config.mark_rate_limited(attempt["provider"], attempt["model"])
            continue

    error_text = f"⚠️ research failed on every configured provider: {last_error}"
    await append_message(conversation_id, "navi", error_text)
    return {"text": error_text, "provider": None, "model": None}


async def _draft_and_present_plan(conversation_id: str) -> dict:
    """The real synthesis moment (how_to_handle_context.md) — reads the
    FULL stored history, not the RECENT_MESSAGE_WINDOW slice the ordinary
    back-and-forth uses, since this is exactly the point a flat recency
    window risks silently dropping something established early on."""
    full_history = await get_messages(conversation_id)
    messages = _history_to_messages(full_history)
    plan = await compact_conversation(messages, PLAN_DRAFT_INSTRUCTION)
    if not plan or not plan.get("goal"):
        await set_task_state(conversation_id, {"stage": STAGE_PLANNING, "plan": None})
        error_text = "⚠️ Couldn't draft a plan from this conversation — let's keep clarifying a bit more, or say more about what you're after."
        await append_message(conversation_id, "navi", error_text)
        return {"text": error_text, "provider": None, "model": None}

    await set_task_state(conversation_id, {"stage": STAGE_AWAITING_PLAN_CONFIRM, "plan": plan})
    reply = _format_plan_markdown(plan)
    await append_message(conversation_id, "navi", reply)
    return {"text": reply, "provider": None, "model": None, "choices": list(PLAN_CONFIRM_OPTIONS)}


def _parse_report_json(text: str) -> dict | None:
    stripped = strip_code_fence(text or "")
    if not stripped:
        return None
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) and parsed.get("title") else None


async def _run_execution(conversation_id: str, plan: dict) -> dict:
    brief = get_mode_brief("research_execute")
    tools = schemas_for(brief.tools)
    task_message = (
        f"The user has reviewed and accepted this research plan:\n\n{_format_plan_markdown(plan)}\n\n"
        "Carry it out now, per your instructions, and produce the final JSON deliverable."
    )

    try:
        role = get_dispatcher_role(context="chat")
    except ProviderNotConfigured as e:
        error_text = f"⚠️ Can't research right now — research isn't configured: {e}"
        await append_message(conversation_id, "navi", error_text)
        return {"text": error_text, "provider": None, "model": None}

    attempts = config.get_attempts([{"provider": role["provider"], "model": role["model"]}] + role.get("fallback", []))
    last_error = None
    for attempt in attempts:
        try:
            provider = get_provider(attempt["provider"])
        except Exception as e:
            last_error = str(e)
            continue
        family = classify_family(attempt["provider"], attempt["model"])
        system_content = adapt_system_prompt(brief.system_prompt, family, attempt["model"])
        extra_params = adapt_request_params(family, attempt["provider"], has_tools=True) or None
        messages = [ChatMessage(role="user", content=task_message)]
        if system_content is not None:
            messages.insert(0, ChatMessage(role="system", content=system_content))
        try:
            response = await asyncio.to_thread(
                provider.chat, model=attempt["model"], messages=messages, tools=tools, extra_params=extra_params,
            )
            if tools and response.tool_calls:
                response, _sent_messages, _iterations = await asyncio.to_thread(
                    run_tool_loop, provider, attempt["model"], messages, response,
                    context={"command": "research-execute", "topic_slug": slugify(plan.get("goal") or "research")},
                    tools=tools, extra_params=extra_params,
                )
            report = _parse_report_json(response.text or "")
            if not report:
                last_error = f"{attempt['provider']}/{attempt['model']} didn't return a valid report"
                continue
            return await _finish_research(conversation_id, attempt, report)
        except ProviderError as e:
            last_error = str(e)
            if e.is_rate_limit:
                config.mark_rate_limited(attempt["provider"], attempt["model"])
            continue
        except Exception as e:
            # Broad on purpose, scoped to just this call: a mid-gathering
            # ask_user_choice (RESEARCH_EXECUTE_PLAN.md permits exactly
            # one) would hit tools/registry.py's dispatch() — which
            # doesn't handle it, by design — and raise ToolExecutionError
            # from inside run_tool_loop. Real resume support for that case
            # doesn't exist yet (see this module's own docstring) — until
            # it does, this turns an unmodeled failure into "try the next
            # fallback" instead of an unhandled 500.
            last_error = f"{attempt['provider']}/{attempt['model']} execution error: {e}"
            continue

    # Plan stays intact (task_state untouched) — the user can just click
    # "Looks good, start researching" again to retry, no need to re-draft.
    error_text = f"⚠️ Research execution failed on every configured provider: {last_error}"
    await append_message(conversation_id, "navi", error_text)
    return {"text": error_text, "provider": None, "model": None}


async def _finish_research(conversation_id: str, attempt: dict, report: dict) -> dict:
    topic_slug = slugify(report.get("title") or "research-report")
    links = []

    try:
        docx_bytes = await asyncio.to_thread(render_research_report_docx, report)
    except ReportRenderError as e:
        await set_task_state(conversation_id, {"stage": STAGE_DONE, "plan": None})
        error_text = f"⚠️ Research finished, but the report couldn't be rendered into a document: {e}"
        await append_message(conversation_id, "navi", error_text, provider=attempt["provider"], model=attempt["model"])
        return {"text": error_text, "provider": attempt["provider"], "model": attempt["model"]}

    try:
        docx_path = await asyncio.to_thread(save_bytes, "research", topic_slug, f"{topic_slug}.docx", docx_bytes)
        docx_url = file_download_url(docx_path)
        if docx_url:
            links.append(f"📎 {topic_slug}.docx: {docx_url}")
    except StorageError:
        pass

    # PDF conversion is best-effort — Gotenberg (tools/report_render.py's
    # GOTENBERG_URL) may not be deployed on this server yet (it wasn't as
    # of the pipeline's own initial commit); the DOCX alone is still a
    # real, complete deliverable either way.
    try:
        pdf_bytes = await asyncio.to_thread(convert_docx_to_pdf, docx_bytes)
        pdf_path = await asyncio.to_thread(save_bytes, "research", topic_slug, f"{topic_slug}.pdf", pdf_bytes)
        pdf_url = file_download_url(pdf_path)
        if pdf_url:
            links.append(f"📎 {topic_slug}.pdf: {pdf_url}")
    except (ReportRenderError, StorageError):
        pass

    summary = (report.get("executive_summary") or "").strip()
    reply = f"**{report.get('title', 'Research report')}**\n\n{summary}"
    if links:
        reply += "\n\n" + "\n".join(links)
    else:
        reply += "\n\n⚠️ The report was generated but couldn't be saved anywhere downloadable — check storage configuration."

    await set_task_state(conversation_id, {"stage": STAGE_DONE, "plan": None})
    await append_message(conversation_id, "navi", reply, provider=attempt["provider"], model=attempt["model"])
    return {"text": reply, "provider": attempt["provider"], "model": attempt["model"]}
