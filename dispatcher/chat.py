"""
dispatcher/chat.py

Free-form chat — a message that isn't a typed /command. Replaces the old
fixed reply from the role now named normal_chat (renamed 2026-09-01 from
dispatcher_chat — see config/store.py's migration): loads the active
mode's brief (system prompt
+ allowed tools from dispatcher/modes/), then lets the model decide
whether to use any of those tools, resolving calls via the same loop
/research's command chain uses (run_tool_loop, capped at
MAX_TOOL_ITERATIONS).

run_mode_chat (below) stays fully stateless — kept as-is for any caller
that still wants single-message behavior. run_stored_mode_chat is the
new persisted sibling (2026-09-01): first real multi-turn memory for
Normal/Research/Brainstorm/Agent Work, previously only Dev Slate had
this. Deliberately the dumbest version that could work — a flat recency
window (RECENT_MESSAGE_WINDOW, same first-cut approach as Dev Slate's
own), no topic classifier, no compaction — see how_to_handle_context.md:
the goal right now is to find out empirically where plain context
actually breaks, not to pre-solve failures nobody's hit yet.
"""

import asyncio
import re
from datetime import datetime, timezone

from config.store import config
from dispatcher.compaction import CONTEXT_TRIGGER_TOKENS, compact_context
from dispatcher.executor import (
    CITATION_STYLE_PROMPT,
    _extract_tool_results,
    _parse_tool_args,
    run_tool_loop,
    strip_reasoning_tags,
)
from dispatcher.branch import (
    COMPLETE_OPTIONS, completion_as_entry, draft_completion, format_completion_markdown,
)
from dispatcher import events
from dispatcher.events import Emitter, emit_status
from dispatcher.mode_briefs import get_mode_brief
from dispatcher.prompt_family import adapt_request_params, adapt_system_prompt, classify_family
from dispatcher.provider_debug import save_failed_exchange
from providers.base import ChatMessage, ProviderError
from providers.registry import (
    ProviderNotConfigured, get_dispatcher_role, get_provider, is_byok_provider, next_chat_tier,
    role_name_for_context,
)
from storage.context_store import (
    SOURCE_ASSISTANT, SOURCE_BRANCH_RESULT, append_entry, build_context_block,
    estimate_tokens, get_friction_since, record_friction, record_friction_sync,
)
from storage.usage import current_call_context, set_call_context
from storage.conversations import (
    append_message, get_conversation, get_messages, get_task_state, set_task_state,
)
from tools.registry import schemas_for

RECENT_MESSAGE_WINDOW = 20

_CREATE_WORKFLOW_ID_RE = re.compile(r"Created workflow ([0-9a-fA-F-]{36})\.")

# A real, observed failure mode (2026-09-06, JuanJo: a normal_chat reply
# repeated the same paragraph 4 times in one message) — not fixable by
# picking a "better" model alone, since any model can degenerate into a
# repetition loop under the wrong conditions; this is a deterministic,
# model-agnostic safety net applied to every chat reply regardless of
# which provider produced it. Threshold at 40 chars deliberately excludes
# short recurring lines (a bullet marker, "---", "Thanks!") that can
# legitimately repeat in real content — only substantial prose repeating
# verbatim is the actual degenerate-loop signature.
_REPEAT_MIN_PARAGRAPH_CHARS = 40


def _collapse_repeated_paragraphs(text: str) -> str:
    """If the same substantial paragraph appears again later in the
    reply, cuts the reply at the point the repeat starts — everything
    from there on is the model looping, not real additional content."""
    paragraphs = text.split("\n\n")
    seen: set[str] = set()
    kept: list[str] = []
    for p in paragraphs:
        normalized = p.strip()
        if len(normalized) > _REPEAT_MIN_PARAGRAPH_CHARS:
            if normalized in seen:
                break
            seen.add(normalized)
        kept.append(p)
    collapsed = "\n\n".join(kept)
    if collapsed != text:
        # The safety net actually tripped, which is a real quality signal
        # about the model that produced this — logged here, inside the
        # function, rather than at one of the three call sites that used
        # to be responsible for noticing. Only one of the three ever was,
        # which is the same hand-placement problem the whole friction
        # move is about: put the record where the event is.
        ctx = current_call_context()
        record_friction_sync(
            "repetition_loop", severity=2,
            conversation_id=ctx.get("conversation_id"),
            detail=f"cut {len(text) - len(collapsed)} chars of repeat",
            # Attributed to whichever model actually answered this turn —
            # the attempt loop puts it on the ambient context, so this
            # stays correct on a fallback without being passed down.
            provider=ctx.get("provider"), model=ctx.get("model"),
        )
    return collapsed


# A ceiling on how much replayed conversation rides along each turn.
#
# RECENT_MESSAGE_WINDOW alone counts MESSAGES, which is the wrong unit and
# has been a known gap since the window was built: what providers meter,
# and what actually degrades a model's attention, is tokens. Twenty short
# exchanges and twenty long structured answers cost wildly different
# amounts for the same cap. A real turn was observed at 7,393 prompt
# tokens for a one-line question, most of it replayed history.
#
# So both limits apply: at most RECENT_MESSAGE_WINDOW messages, and at
# most this many tokens of them. A chat of quick back-and-forth keeps all
# twenty; a chat of long answers keeps fewer, which is the correct
# behaviour in both cases.
HISTORY_TOKEN_BUDGET = 2_500

# Never trim below this many of the most recent messages, whatever they
# cost. Continuity is the window's actual job — the last couple of
# exchanges are what keep a reply feeling like it belongs to the
# conversation rather than arriving out of nowhere. A single enormous
# message is allowed to blow the budget rather than leave the model
# answering with no idea what was just said.
HISTORY_MIN_MESSAGES = 4


def _trim_history_to_budget(messages: list[ChatMessage]) -> list[ChatMessage]:
    """Keeps the newest messages that fit HISTORY_TOKEN_BUDGET.

    Walks from the newest backwards, because the newest are the ones that
    must survive: the final entry is the user's actual question. Dropping
    happens at the OLD end, which is the same end RECENT_MESSAGE_WINDOW
    already drops from, so this only ever tightens an existing boundary —
    it never removes something the previous rule would have kept for a
    reason of its own.

    Anything dropped here is not lost from the conversation: it stays in
    storage, and whatever mattered about it should already be in
    context.md, which is exactly the job context.md was built to do.
    """
    kept: list[ChatMessage] = []
    total = 0
    for m in reversed(messages):
        cost = estimate_tokens(m.content)
        if kept and len(kept) >= HISTORY_MIN_MESSAGES and total + cost > HISTORY_TOKEN_BUDGET:
            break
        kept.append(m)
        total += cost
    kept.reverse()
    if len(kept) < len(messages):
        print(f"[chat] history trimmed {len(messages)} -> {len(kept)} messages ({total} tokens, budget {HISTORY_TOKEN_BUDGET})")
    return kept


# How long a NORMAL CHAT turn may take before NAVI says something, and
# before it stops waiting and moves to the fallback.
#
# These are per-ROLE numbers, not per-provider ones. The transports each
# carry their own timeout (Groq 30s, Cloudflare and Gemini 60s, Ollama
# 190s) and those describe the TRANSPORT's patience — how long a socket
# should stay open. This describes the USER's patience, which is a
# different quantity and the one that actually matters for an interactive
# reply. A background compaction pass taking 45s is fine; a person
# watching a chat box for 45s is not.
#
# The numbers come from research rather than from NAVI's own latency
# statistics, deliberately. A threshold derived from p95 would be circular
# — it moves when behaviour moves, and killing slow calls removes them
# from the distribution, which lowers p95, which kills more. Human
# tolerance does not drift like that. A 2026 CHI study (240 participants,
# TTFT at 2/9/20s) found 20s is where waiting stops reading as deliberation
# and starts reading as inefficiency, and web-app satisfaction research
# puts abandonment around 12s. 15 sits between the two: past the point
# where a wait is comfortable, before the point where it reads as broken.
#
# JuanJo, 2026-09-15: "15 seconds sounds good for now, with testing we
# will see if we change it." Treat both as starting points to tune against
# real use, not as derived constants.
CHAT_WARN_AFTER_S = 12.0
CHAT_GIVE_UP_AFTER_S = 15.0

# SCOPE, confirmed 2026-09-16 after JuanJo compared this to waiting on a
# real coding-agent session ("I wait 2 minutes... some tasks take 8 to 10
# minutes"): these two numbers govern ONE quick chat reply, nothing else.
# _call_with_watchdog has exactly one call site, inside
# run_stored_mode_chat, which is Normal Chat and Brainstorm only.
# Research's execution stage (dispatcher/research.py) and Agent Work
# (dispatcher/agent_work.py) each run their OWN attempt loop with a plain
# provider.chat() call, bounded only by that transport's own connection
# timeout (60-190s) times up to MAX_TOOL_ITERATIONS rounds — an
# 8-10-minute job was never going to hit this, because it never calls
# this function. Do not widen these numbers to accommodate a long job,
# and do not narrow Research/Agent Work to match these — they are
# answering a different question ("is this one reply taking too long for
# someone watching a chat box") than a multi-step session is.


async def _call_with_watchdog(
    provider, model: str, messages: list[ChatMessage], tools, extra_params,
    emit, next_label: str | None,
):
    """Runs one provider call, narrating it, and stops waiting past the budget.

    Returns the ChatResponse, or None if the budget ran out — in which case
    the caller should move to its next attempt.

    IMPORTANT, AND STATED PLAINLY: this stops WAITING, it does not stop the
    CALL. `asyncio.to_thread` cannot be cancelled, so the request keeps
    running in its worker thread, finishes in its own time, and its tokens
    are still spent and still recorded by providers/base.py. What the user
    is spared is the wait, not the cost.

    That is a real limitation of a non-streaming transport, not an
    oversight: aborting for real needs a response we are reading
    incrementally, which is what `stream=True` gives us. When streaming
    lands, this function is where the genuine abort goes.

    The abandonment is recorded rather than swallowed, so the gap between
    "a call we paid for" and "an answer anyone saw" stays visible in the
    data instead of looking like an ordinary successful call.
    """
    # Bridge tokens from the provider's worker thread back to this loop.
    #
    # provider.chat is synchronous and runs in a thread; `emit` is a
    # coroutine belonging to the request's event loop. run_coroutine_
    # threadsafe is the sanctioned crossing between the two. Order is
    # preserved because the coroutines are scheduled in call order onto a
    # FIFO queue — tokens cannot arrive shuffled.
    #
    # No on_token when nobody is listening: a provider that would stream
    # answers blocking instead, which is exactly right for /chat/send.
    on_token = None
    if emit is not None:
        loop = asyncio.get_running_loop()

        def on_token(chunk: str) -> None:  # noqa: F811
            asyncio.run_coroutine_threadsafe(emit(events.TOKEN, {"text": chunk}), loop)

    task = asyncio.create_task(
        asyncio.to_thread(
            provider.chat, model=model, messages=messages, tools=tools,
            extra_params=extra_params, on_token=on_token,
        )
    )
    done, _pending = await asyncio.wait({task}, timeout=CHAT_WARN_AFTER_S)
    if not done:
        await emit_status(emit, events.taking_long(model), kind="slow")
        remaining = CHAT_GIVE_UP_AFTER_S - CHAT_WARN_AFTER_S
        done, _pending = await asyncio.wait({task}, timeout=remaining)

    if done:
        return task.result()  # re-raises ProviderError for the caller to handle

    # Budget spent. Let the orphaned task finish quietly rather than leave
    # its exception unretrieved (which asyncio would otherwise log as an
    # unhandled error on a perfectly understood situation).
    def _swallow(finished: asyncio.Task) -> None:
        # Retrieving the exception is what stops asyncio logging "Task
        # exception was never retrieved" for a situation we understand
        # completely. But .exception() RAISES on a cancelled task rather
        # than returning, and this task can absolutely be cancelled — the
        # event loop shutting down mid-call is the ordinary case. Guarding
        # is the whole point; without it the cleanup throws its own error.
        if not finished.cancelled():
            finished.exception()

    task.add_done_callback(_swallow)
    await emit_status(
        emit,
        events.switching(model, next_label) if next_label
        else f"{model} didn't answer in time.",
        kind="switch",
    )
    record_friction_sync(
        "answer_abandoned", severity=3,
        conversation_id=current_call_context().get("conversation_id"),
        detail=f"stopped waiting after {CHAT_GIVE_UP_AFTER_S:.0f}s; the call was still paid for",
        provider=getattr(provider, "name", None), model=model,
    )
    return None


async def _maybe_compact_context(conversation_id: str) -> bool:
    """Fires a compaction pass if context.md has crossed its token ceiling.

    Runs INLINE, at the end of the turn that crossed the line, after the
    reply is already persisted. That placement is deliberate and is what
    satisfies "block the user's input until compaction finishes"
    (how_to_handle_context.md) with zero new infrastructure: the frontend
    already disables the composer for the duration of an in-flight
    /chat/send request, so extending that request IS the lock. No lock
    column, no status endpoint, no polling loop, no way for a second
    message to race a half-written snapshot.

    The real cost is one visibly slower turn when it fires. Accepted for
    v1 — moving this to a background job with a "Tidying up context…"
    status is a genuine later refinement (the dead async-polling path in
    navi-pwa's App.tsx is still sitting there for it), not a prerequisite.

    Never raises: a failed compaction leaves the existing context in place
    and the conversation keeps working, slightly over budget, rather than
    the turn blowing up over housekeeping.

    Returns True on the one turn a conversation is judged to have hit the
    floor — see _hit_the_floor below. False every other time, including
    every turn where nothing compacted at all.
    """
    try:
        _block, tokens = await build_context_block(conversation_id)
        if tokens < CONTEXT_TRIGGER_TOKENS:
            return False
        print(f"[chat] context.md at {tokens} tokens (ceiling {CONTEXT_TRIGGER_TOKENS}) — compacting")
        report = await compact_context(conversation_id)
        if not report.get("ok"):
            await record_friction(
                "compaction_failed", severity=3, conversation_id=conversation_id,
                detail=str(report.get("reason")),
            )
        elif report.get("still_above_target"):
            await record_friction(
                "compaction_above_target", severity=2, conversation_id=conversation_id,
                detail=f"{report.get('after_tokens')} tokens after compaction",
            )
            return await _hit_the_floor(conversation_id)
    except Exception as e:
        print(f"[chat] context compaction check failed (non-fatal): {e}")
    return False


# How many times a conversation has to fail to compact under target before
# it counts as having hit the floor.
#
# Two, not one. A single pass can miss target for a mundane reason — a
# weak model call, an unlucky run. Twice means the material is genuinely
# irreducible: the first pass already took out what was takeable and the
# second still couldn't get under. One is noise, two is a pattern.
#
# The cost asymmetry points the same way. Suggesting a fresh chat too
# eagerly trains people to dismiss the suggestion, which destroys it as a
# signal; suggesting it one pass late costs almost nothing.
FLOOR_FAILURES = 2

# What the user actually reads when the floor is hit. Says what is
# happening and what to do about it, and deliberately does not say
# anything is wrong — nothing is. A conversation that has accumulated
# more than it can compress is one that has been used a lot.
FLOOR_NOTICE = (
    "This chat is carrying about as much as it usefully can — tidying it up isn't "
    "buying much room any more. If the next thing you want is a distinct piece of "
    "work, it'll go better in its own chat, started with a brief from this one."
)


async def _hit_the_floor(conversation_id: str) -> bool:
    """Has this conversation stopped being compactable?

    The escape valve from the original compaction design: past some point
    everything left is genuinely load-bearing, and further passes can only
    shrink it by destroying something real. That is not a compaction bug
    to fix by compacting harder — it means the conversation is done
    growing, and the next piece of work belongs in its own chat.

    Counted cumulatively rather than consecutively, deliberately. A
    conversation that has missed target twice at all is one where
    compaction is not keeping up, and whether a luckier pass happened to
    land between them does not change that. It is also the version that
    can be computed correctly from what is actually recorded — only
    failures are, so "consecutive" would be inferred rather than known.

    Offered once per conversation, never repeated: a suggestion that
    reappears every turn is nagging, and gets dismissed reflexively.
    """
    events = await get_friction_since(0, conversation_id)
    if any(e["kind"] == "context_floor_reached" for e in events):
        return False
    if sum(1 for e in events if e["kind"] == "compaction_above_target") < FLOOR_FAILURES:
        return False
    await record_friction(
        "context_floor_reached", severity=3, conversation_id=conversation_id,
        detail=f"{FLOOR_FAILURES} compaction passes could not reach target",
    )
    return True


def _extract_created_workflow_id(sent_messages: list[ChatMessage]) -> str | None:
    """Agent Work Chat's whole point is that the model, not the user,
    builds the graph — but the frontend still needs to know WHICH
    workflow just got created so it can load it onto the canvas as real
    nodes (2026-09-03, JuanJo: a workflow the chat created should show up
    as nodes, not just an entry in the Workflows list). tools/registry.py's
    create_workflow branch returns a fixed "Created workflow <id>." string
    as its tool-result content; that's the only place the id exists once
    run_tool_loop has finished, so pulling it out of the transcript here
    is simpler than widening dispatch()'s return contract for every tool.
    Returns the LAST match if create_workflow was somehow called more
    than once in a single turn."""
    found = None
    for m in sent_messages:
        if m.role == "tool" and m.name == "create_workflow" and isinstance(m.content, str):
            match = _CREATE_WORKFLOW_ID_RE.search(m.content)
            if match:
                found = match.group(1)
    return found


def run_mode_chat(mode: str, text: str) -> str:
    brief = get_mode_brief(mode)

    # Every existing mode (normal/research/brainstorm) shares normal_chat's
    # role — only Agent Work gets its own (agent_work role, same reasoning
    # as Dev Slate having dev_slate_chat: it needs real tool-calling
    # reliability, not whatever's cheapest for everyday chat). Context
    # string intentionally differs from `mode` itself for every other mode
    # (stays "chat") so this doesn't silently change normal_chat/research/
    # brainstorm's existing behavior.
    role_context = "agent_work" if mode == "agent_work" else "chat"
    set_call_context(role=role_name_for_context(role_context), mode=mode, tier=role_context)
    try:
        role = get_dispatcher_role(context=role_context)
    except ProviderNotConfigured as e:
        return f"⚠️ Can't reply right now — {role_context} isn't configured: {e}"

    # Combined into one system message, not two — see run_stored_mode_chat
    # below for why (Cloudflare rejects more than one system-role entry).
    tools = schemas_for(brief.tools) if brief.tools else None
    base_system_content = f"{brief.system_prompt}\n\n{CITATION_STYLE_PROMPT}" if tools else brief.system_prompt

    # Same-model-family fallback chain (added 2026-08-29 after a real
    # "Groq rate limited" hard failure) — try the primary, then each
    # configured fallback in order, same pattern as /research's gathering
    # phase (executor.py). Any one succeeding returns immediately.
    attempts = config.get_attempts([{"provider": role["provider"], "model": role["model"]}] + role.get("fallback", []))
    last_error = None
    for i, attempt in enumerate(attempts):
        set_call_context(attempt=i, provider=attempt["provider"], model=attempt["model"])
        try:
            provider = get_provider(attempt["provider"])
        except Exception as e:
            last_error = str(e)
            continue
        # Per-family system-prompt + request-param adaptation
        # (2026-09-11/12, dispatcher/prompt_family.py) — built fresh per
        # attempt, not once before the loop, since a fallback chain can
        # hand this request to a genuinely different model family.
        # Deliberately skipped for agent_work — see prompt_family.py's
        # own scope docstring for why (stateless, tool-call-driven, no
        # free-form prose to adapt; its own reasoning_effort treatment
        # needs a separate, not-yet-built complexity classifier — see
        # IDEAS.md).
        if mode == "agent_work":
            system_content = base_system_content
            extra_params = None
        else:
            family = classify_family(attempt["provider"], attempt["model"])
            system_content = adapt_system_prompt(base_system_content, family, attempt["model"])
            extra_params = adapt_request_params(family, attempt["provider"], has_tools=bool(tools)) or None
        messages = [ChatMessage(role="user", content=text)]
        if system_content is not None:
            messages.insert(0, ChatMessage(role="system", content=system_content))
        try:
            response = provider.chat(model=attempt["model"], messages=messages, tools=tools, extra_params=extra_params)
            if tools and response.tool_calls:
                # Free-form chat has no StepResult to attach an attempt
                # count to (that's a /research-command-chain concept) —
                # discard it here, not silently drop it by accident.
                response, _messages, _iterations = run_tool_loop(
                    provider, attempt["model"], messages, response,
                    context={"command": f"chat-{mode}", "topic_slug": "chat"},
                    tools=tools, extra_params=extra_params,
                )
            reply = _collapse_repeated_paragraphs(strip_reasoning_tags(response.text) or "(empty reply)")
            if i > 0:
                reply += f"\n\n⚡ (Groq was busy, answered via {attempt['provider']}/{attempt['model']} instead)"
            elif response.usage_note:
                reply += f"\n\n⚡ {response.usage_note}"
            return reply
        except ProviderError as e:
            last_error = str(e)
            continue

    return f"⚠️ normal_chat failed on every configured provider: {last_error}"


# Agent Work's old "review changes" mode (2026-09-01 - 2026-09-03) lived
# here as a prompt instruction telling the model to describe its plan and
# wait for an explicit "yes" on a LATER turn before actually calling
# create_workflow/run_workflow — the only option available without a live
# connection to pause on, mirroring Dev Slate's EditModeSelector concept.
# Removed 2026-09-03 once agent_work went stateless (JuanJo: "no context.md
# for the chat... it should just create"): a confirm-on-a-later-turn design
# cannot work once the model no longer sees its own earlier turns.
# auto_accept is kept as a parameter (harmless no-op for agent_work now,
# still meaningful for nothing else) purely so no caller needs updating.


async def _propose_branch_completion(conversation_id: str, provider: str, model: str) -> dict | None:
    """Turns a model's completion claim into a handover the user can
    actually judge, and parks the branch until they answer.

    Returns None if the handover can't be drafted — the caller then just
    lets the turn answer normally, which is strictly better than telling
    the user their work is finished and then failing to say what it was.
    """
    doc = await draft_completion(conversation_id)
    if not doc:
        return None
    await set_task_state(conversation_id, {"branch_stage": "awaiting_complete", "completion": doc})
    reply = (
        "Here's what I'd hand back to the main chat:\n\n"
        + format_completion_markdown(doc)
        + "\n\nAccepting closes this chat and sends that summary back."
    )
    await append_message(conversation_id, "navi", reply, provider=provider, model=model)
    return {
        "text": reply, "provider": provider, "model": model,
        "choices": list(COMPLETE_OPTIONS),
    }


async def _settle_pending_branch_completion(conversation_id: str, text: str) -> dict | None:
    """Reads the user's answer to a pending handover. Returns None when
    there's nothing pending, so the turn proceeds normally.

    Three outcomes, and the third one matters most: accept closes the
    branch, an explicit decline resumes work, and ANYTHING ELSE also
    resumes work rather than being swallowed as an answer. Someone who
    ignores the question and keeps typing has plainly not accepted, and
    treating an unrelated message as consent would close their work on
    their behalf.
    """
    state = await get_task_state(conversation_id) or {}
    if state.get("branch_stage") != "awaiting_complete":
        return None
    doc = state.get("completion") or {}
    await set_task_state(conversation_id, None)

    if not _is_branch_accept(text):
        # The branch claimed it was finished and the user did not agree.
        # Same free ground-truth label as research's plan checkpoint: the
        # model made a judgement, a human overruled it, and the disagreement
        # is recorded rather than thrown away. Deliberately NOT severity 3
        # — "not yet" on a completion claim is ordinary, where a cancelled
        # research plan means the whole draft missed.
        await record_friction(
            "completion_rejected", severity=2, conversation_id=conversation_id,
            detail=f"branch proposed done, user continued: {text.strip()[:200]}",
        )
        return None  # declined, or moved on — either way, keep working

    conversation = await get_conversation(conversation_id)
    parent_id = (conversation or {}).get("parent_id")
    scope = str(doc.get("scope") or "").strip()
    if parent_id:
        # ONE entry, not one per line: the main chat asked for a feature,
        # not for this chat's working notes. Handing everything back would
        # rebuild there exactly the bloat splitting the work off avoided.
        await append_entry(
            parent_id, completion_as_entry(scope or "a separate chat", doc),
            source=SOURCE_BRANCH_RESULT,
        )
    await set_task_state(conversation_id, {"branch_stage": "closed", "completion": doc})
    reply = (
        "Done — sent back to the main chat. This chat is closed; "
        "anything further belongs there, or in a new one."
    )
    await append_message(conversation_id, "navi", reply)
    return {"text": reply, "provider": None, "model": None, "branch_closed": True}


def _is_branch_accept(text: str) -> bool:
    """The exact button label is the reliable signal — ChoiceButtons sends
    the clicked option back verbatim. The prefix check under it is a soft
    fallback for someone who types instead of clicking, and is deliberately
    narrow: a false accept closes work that isn't finished, while a false
    miss just means asking again."""
    t = text.strip()
    if t == COMPLETE_OPTIONS[0]:
        return True
    return t.lower().startswith(("accept", "yes, accept", "looks good, close", "close it"))


async def run_stored_mode_chat(
    mode: str, conversation_id: str, text: str, auto_accept: bool = True,
    reasoning_effort: str | None = None, tier: str = "chat", _escalated_from: str | None = None,
    emit: Emitter | None = None,
) -> dict:
    """Persisted sibling of run_mode_chat above — appends the user's
    message, replays a windowed slice of REAL history (not just this one
    message, except for agent_work — see below) alongside the mode's
    brief, calls the model, persists the reply. Returns {text, provider,
    model} (provider/model reflect whichever fallback actually answered,
    mirroring dispatcher/devslate_chat.py's run_devslate_turn, which this
    is deliberately modeled on — same role-selection/fallback/tool-loop
    shape as run_mode_chat above, just with storage/conversations.py
    wrapped around it instead of nothing.

    reasoning_effort (2026-09-12): the manual override from the PWA's
    slider next to the model picker — passed straight through to
    adapt_request_params, which only actually applies it for a non-Groq
    gpt-oss attempt (see that function's own docstring). Harmless no-op
    for every other family/provider, so callers that never send it
    (typed /commands, older clients) are unaffected.

    tier (2026-09-13): which capability tier answers this turn — "chat"
    (idle, the default and where every turn starts), "chat_exploratory",
    or "chat_serious". Escalation is NOT decided by a classifier: the
    idle model calls request_stronger_model when it judges a message
    beyond it, and this function re-runs itself one tier up. Recognising
    "this is beyond me" is a far easier task than answering it, which is
    what makes it safe to hand a small model. `_escalated_from` is set
    only on those internal re-runs — it exists to prevent the user's
    message being appended to history twice, not as a public parameter."""
    if not _escalated_from:
        await append_message(conversation_id, "user", text)
        # A branch waiting on the user to accept its handover gets first
        # look at this message. Only ever set on a branch that proposed
        # completion, so every ordinary conversation skips it on a null
        # task_state and this costs a dict lookup.
        settled = await _settle_pending_branch_completion(conversation_id, text)
        if settled:
            return settled

    brief = get_mode_brief(mode)
    history = await get_messages(conversation_id, limit=RECENT_MESSAGE_WINDOW)

    role_context = "agent_work" if mode == "agent_work" else tier
    # Tag every provider call this turn makes — including the tool loop's
    # continuation calls, which inherit it through asyncio.to_thread
    # without being tagged again. Set before the role lookup so a
    # ProviderNotConfigured path is still attributable.
    set_call_context(
        role=role_name_for_context(role_context), mode=mode, tier=tier,
        conversation_id=conversation_id,
    )
    try:
        role = get_dispatcher_role(context=role_context)
    except ProviderNotConfigured as e:
        error_text = f"⚠️ Can't reply right now — {role_context} isn't configured: {e}"
        await append_message(conversation_id, "navi", error_text)
        return {"text": error_text, "provider": None, "model": None}

    # Ordering here is a real invariant, not incidental (JuanJo,
    # 2026-09-01): static content first (brief/citation prompt — byte-
    # identical every call), then growing context (history), with the
    # new message naturally landing last since it's already the final
    # row `history` returns. This is what lets a provider's prefix-based
    # prompt caching (Groq/OpenRouter automatic, Cloudflare automatic
    # baseline, Mistral manual via prompt_cache_key though not wired up
    # yet — see providers/*.py's own docstrings) actually hit: the
    # unchanging prefix (static + already-seen history) stays identical
    # turn to turn, only the newest message is genuinely new. Don't
    # insert anything that changes between calls (a live timestamp, a
    # per-turn-computed block) ahead of the growing-but-stable part, or
    # it breaks the prefix match for every provider's cache at once.
    # 2026-09-02: these three were previously separate ChatMessage entries
    # (all consecutively first, before any history) — correct per the
    # usual "system messages must lead" convention, but the Cloudflare 400
    # ("System message must be at the beginning") persisted even after the
    # trailing UTC-time message was removed. agent_work is the one mode
    # that reliably stacks all three at once (brief + citation format,
    # since it has tools + the review instruction, since auto_accept
    # defaults off) — strong circumstantial evidence Cloudflare's real
    # rule is "at most one system message, and it must be message[0]," not
    # just "system messages must be consecutively first." Combining into
    # one message satisfies either reading and can't regress anything.
    tool_names = list(brief.tools or [])
    # Finishing is only a real gesture in a branch: an ordinary chat has no
    # parent to hand work back to and no acceptance criteria to have met.
    # Withheld here rather than dropped after the fact, so the model never
    # sees an option it cannot meaningfully use — the same discipline that
    # scopes every other mode's tools to what that mode can actually do.
    if "propose_branch_complete" in tool_names:
        conversation = await get_conversation(conversation_id)
        if not (conversation and conversation.get("parent_id")):
            tool_names.remove("propose_branch_complete")
    # Same rule, same reason: the top tier has nowhere to escalate TO
    # (next_chat_tier returns None), so shipping this schema there buys a
    # guaranteed-useless option and pays ~190 prompt tokens per turn for
    # the privilege. It also removes the only way to reach the
    # escalation-at-ceiling branch by accident.
    if "request_stronger_model" in tool_names and not next_chat_tier(tier):
        tool_names.remove("request_stronger_model")
    # A model picked through someone's own key (DeepSeek, Claude) is an
    # explicit choice, usually made to see how THAT model does. Escalating
    # to the next free tier would quietly answer with a different model and
    # spoil the comparison, so there is nothing to escalate to.
    if "request_stronger_model" in tool_names and is_byok_provider(role.get("provider", "")):
        tool_names.remove("request_stronger_model")
    tools = schemas_for(tool_names) if tool_names else None
    base_system_parts = [brief.system_prompt]
    if tools:
        base_system_parts.append(CITATION_STYLE_PROMPT)
    # context.md (2026-09-13) — the conversation's distilled durable memory,
    # riding along on every turn so a fact established 50 messages ago still
    # reaches the model after RECENT_MESSAGE_WINDOW scrolled past it. Sits
    # inside the SAME single system message as the brief, not a second one
    # (Cloudflare rejects any system message that isn't message[0] — see the
    # long comment above), and BEFORE the history block, which keeps the
    # documented prefix-caching invariant intact: it only changes when a
    # compaction pass runs or a new insight is flagged, not per-turn.
    #
    # agent_work is excluded by construction — it's deliberately stateless
    # (see below), so durable memory would contradict its whole design.
    # Fetched here, but attached to the LAST message far below — NOT to
    # this leading system block. See that call site for why that matters.
    context_block = ""
    if mode != "agent_work":
        context_block, _context_tokens = await build_context_block(conversation_id)
    # AGENT_WORK_REVIEW_INSTRUCTION's "confirm on a LATER message" design
    # requires the model to remember its own proposal on a future turn —
    # incompatible with agent_work now being stateless (2026-09-03,
    # JuanJo: "no context.md for the chat... just needs the LLM to create
    # the json schema with the steps"). auto_accept stays a harmless no-op
    # parameter for every other mode.
    #
    # Not a system ChatMessage yet — base_system_content gets adapted per
    # family (dispatcher/prompt_family.py, 2026-09-11) fresh for each
    # fallback attempt below, since a fallback chain can hand this same
    # request to a genuinely different model family. Everything below
    # this point (history_messages) is the part that stays IDENTICAL
    # across attempts.
    base_system_content = "\n\n".join(base_system_parts)
    history_messages: list[ChatMessage] = []
    # The just-appended user message is already the last row `history`
    # returns (get_messages reads it back from storage) — not double
    # counted. "navi" -> "assistant" matches storage's own role
    # convention (see storage/conversations.py / devslate_chat.py).
    #
    # A past failure's own error text (always prefixed "⚠️" — see every
    # error_text/return above) is skipped here, not replayed as if it were
    # a real prior reply (2026-09-02, JuanJo: "are we giving the errors as
    # context in the chat? that might be [messing] us too"). A retry in
    # the same conversation would otherwise feed the model its own past
    # failure's raw error text as supposed prior context on every
    # subsequent attempt — noise at best, actively confusing at worst.
    # Still shown to the user in the UI (this only filters what's SENT to
    # the model, not what's persisted/displayed) — see get_messages calls
    # elsewhere, unaffected by this.
    #
    # agent_work is deliberately stateless (2026-09-03, JuanJo: "we
    # actually make Agent Work Chat have no context. it should just
    # create, it doesn't need any context") — each message is a
    # standalone "build/run this" instruction, not a turn in an ongoing
    # conversation. Replaying old turns was also part of what let a
    # flaky model's confusion compound across turns (a stale tool result
    # or an earlier vague reply sitting in context, nudging a later
    # attempt toward repeating work). Still fully persisted via
    # append_message above/below for the frontend's own display
    # history — this only changes what's SENT to the model.
    if mode == "agent_work":
        history_messages.append(ChatMessage(role="user", content=text))
    else:
        for m in history:
            if m["role"] == "navi" and m["content"].startswith("⚠️"):
                continue
            history_messages.append(ChatMessage(role="assistant" if m["role"] == "navi" else m["role"], content=m["content"]))
        history_messages = _trim_history_to_budget(history_messages)
    # UTC time grounding (JuanJo, 2026-09-01: "if it asks for something
    # close to 'do it in X time', must send the messages with a UTC
    # signal") — the model has no inherent sense of "now," so a request
    # like "in 5 minutes" or "every hour starting now" is unresolvable
    # without this.
    #
    # 2026-09-02: originally a separate trailing system message, which
    # broke outright on Cloudflare (400: "System message must be at the
    # beginning") — Cloudflare's OpenAI-compatible endpoint rejects any
    # system-role message that isn't the very first one, and every other
    # system message here (brief/citation/review instruction) already IS
    # first, consecutively. Rather than move the time signal to the front
    # (which would poison the whole history block's cache-prefix stability
    # with a value that changes every call), it's appended directly onto
    # the final user message's own content instead — that message was
    # already guaranteed unique this turn, so this costs nothing
    # additional for prefix-based caching while keeping every system
    # message genuinely first. The stored copy (already persisted above,
    # via append_message) is untouched — only this outgoing copy changes.
    # context.md — the conversation's distilled durable memory, so a fact
    # established 50 messages ago still reaches the model after
    # RECENT_MESSAGE_WINDOW scrolled past it.
    #
    # Attached to the FINAL message rather than the leading system block,
    # and that placement is the entire point. It first shipped inside the
    # system message, whose own comment claimed this was cache-safe
    # because the block "only changes when a compaction pass runs or a new
    # insight is flagged, not per-turn." That reasoning was wrong in its
    # conclusion: anything at the FRONT that ever changes invalidates the
    # prefix for everything behind it, so one flagged insight threw away
    # the cache for the whole conversation — brief, tool schemas and every
    # replayed message alike. Observed live 2026-09-13: 7,393 prompt
    # tokens with ZERO cached, on a turn that should have reused most of
    # them.
    #
    # The UTC line directly below had already established this exact rule
    # and this exact fix ("which would poison the whole history block's
    # cache-prefix stability with a value that changes every call"). Same
    # treatment for the same reason: the final message is already unique
    # this turn, so riding along on it costs nothing a cache could have
    # saved.
    #
    # Placed before the user's own words rather than after, so the question
    # itself stays last — the strongest position for the thing actually
    # being answered.
    if context_block:
        history_messages[-1].content = (
            "## What you already know about this conversation\n"
            "Durable memory from earlier in this conversation, distilled. Treat it as "
            "established background, not as something the user just said — don't "
            "re-confirm it back to them unprompted.\n\n"
            + context_block
            + "\n\n---\n\n"
            + history_messages[-1].content
        )
    history_messages[-1].content = f"{history_messages[-1].content}\n\n[Current UTC time: {datetime.now(timezone.utc).isoformat()}]"

    attempts = config.get_attempts([{"provider": role["provider"], "model": role["model"]}] + role.get("fallback", []))
    attempt_labels = [f"{a['provider']}/{a['model']}" for a in attempts]
    print(f"[run_stored_mode_chat] mode={mode} conversation={conversation_id} attempts={attempt_labels}")
    last_error = None
    for i, attempt in enumerate(attempts):
        print(f"[run_stored_mode_chat] attempt {i}: {attempt['provider']}/{attempt['model']}")
        # attempt > 0 IS the fallback signal, recorded as data rather than
        # narrated. The existing fallback_used friction event is written by
        # hand in one branch further down; this is the same fact, available
        # for every call on every role without anyone remembering to log it.
        set_call_context(attempt=i, provider=attempt["provider"], model=attempt["model"])
        try:
            provider = get_provider(attempt["provider"])
        except Exception as e:
            last_error = str(e)
            print(f"[run_stored_mode_chat] attempt {i} get_provider failed: {e}")
            continue
        # Per-family system-prompt + request-param adaptation
        # (2026-09-11/12) — built fresh per attempt; skipped for
        # agent_work (see prompt_family.py's own scope docstring on why;
        # its reasoning_effort treatment needs a separate, not-yet-built
        # complexity classifier — see IDEAS.md).
        if mode == "agent_work":
            system_content = base_system_content
            extra_params = None
        else:
            family = classify_family(attempt["provider"], attempt["model"])
            system_content = adapt_system_prompt(base_system_content, family, attempt["model"])
            extra_params = adapt_request_params(
                family, attempt["provider"], has_tools=bool(tools), reasoning_effort=reasoning_effort,
            ) or None
        messages = list(history_messages)
        if system_content is not None:
            messages.insert(0, ChatMessage(role="system", content=system_content))
        try:
            sent_messages = messages
            # The label for whatever we would move to if this one runs out
            # of time — named in the message rather than left vague, so the
            # user reads a decision being made and not just a complaint.
            _next = attempts[i + 1] if i + 1 < len(attempts) else None
            next_label = f"{_next['provider']}/{_next['model']}" if _next else None
            await emit_status(
                emit,
                events.asking(attempt["model"]) if i == 0
                else events.unavailable(attempts[i - 1]["model"], attempt["model"]),
                kind="asking" if i == 0 else "switch",
            )
            response = await _call_with_watchdog(
                provider, attempt["model"], messages, tools, extra_params, emit, next_label,
            )
            if response is None:
                # Budget spent. _call_with_watchdog has already told the
                # user and recorded it; just move on to the next attempt.
                last_error = f"{attempt['provider']}/{attempt['model']} exceeded the {CHAT_GIVE_UP_AFTER_S:.0f}s chat budget"
                continue
            print(
                f"[run_stored_mode_chat] attempt {i} FIRST reply: "
                f"text={(response.text or '')[:200]!r} tool_calls={[tc.name for tc in response.tool_calls]}"
            )
            # flag_key_insight is intercepted FIRST and is NON-TERMINAL —
            # it's recorded and then execution carries straight on, so a
            # memory write never costs the user their actual reply (see
            # tools/registry.py's own note on why this one differs from the
            # other three intercepted tools). Stripped from response.tool_calls
            # afterward so the tool loop below doesn't try to dispatch it —
            # dispatch() has no handler and would raise ToolExecutionError.
            insight_calls = [tc for tc in response.tool_calls if tc.name == "flag_key_insight"]
            if insight_calls:
                response.tool_calls = [tc for tc in response.tool_calls if tc.name != "flag_key_insight"]
                for tc in insight_calls:
                    insight = (_parse_tool_args(tc.arguments).get("insight") or "").strip()
                    if insight:
                        # Provenance matters at consolidation time, not now:
                        # SOURCE_ASSISTANT records that a model wrote this
                        # down, so compaction can refuse to promote it into
                        # a user-stated fact. See storage/context_store.py.
                        await append_entry(conversation_id, insight, source=SOURCE_ASSISTANT)
                        print(f"[run_stored_mode_chat] flagged key insight: {insight[:120]!r}")

            # Capability escalation (2026-09-13). Checked before every
            # other interception: if the model says it can't handle this,
            # nothing else it produced this turn is worth acting on.
            escalate_call = next((tc for tc in response.tool_calls if tc.name == "request_stronger_model"), None)
            if escalate_call:
                higher = next_chat_tier(tier)
                reason = (_parse_tool_args(escalate_call.arguments).get("reason") or "").strip()
                if higher:
                    print(f"[run_stored_mode_chat] escalating {tier} -> {higher}: {reason[:120]!r}")
                    await record_friction(
                        "tier_escalation", severity=1, conversation_id=conversation_id,
                        detail=f"{tier} -> {higher} ({attempt['provider']}/{attempt['model']}): {reason[:200]}",
                        provider=attempt["provider"], model=attempt["model"],
                    )
                    # Told, not hidden. The turn is about to be re-run on a
                    # different model, which the user would otherwise
                    # experience as an unexplained extra wait.
                    await emit_status(emit, events.escalating(higher), kind="escalate")
                    # Anything already streamed came from the weaker model
                    # and is about to be replaced by a different answer, so
                    # tell the client to drop it. Without this the user
                    # would watch one reply be silently overwritten by
                    # another, which reads as a glitch rather than as the
                    # deliberate hand-off it is.
                    if emit is not None:
                        await emit(events.RESET, {"reason": "escalated to a stronger model"})
                    # Re-run the same turn one tier up. _escalated_from
                    # stops the user's message being appended twice —
                    # it's already in history from the first pass.
                    return await run_stored_mode_chat(
                        mode, conversation_id, text, auto_accept=auto_accept,
                        reasoning_effort=reasoning_effort, tier=higher, _escalated_from=tier,
                        emit=emit,  # the re-run narrates itself too
                    )
                # Already at the ceiling. Don't loop, don't fail — let the
                # top-tier model answer as best it can, which is strictly
                # better than telling the user nothing. Logged at higher
                # severity because it means the strongest tier declared
                # itself insufficient, which is worth knowing about.
                print(f"[run_stored_mode_chat] escalation requested at ceiling tier {tier} — answering anyway")
                await record_friction(
                    "escalation_at_ceiling", severity=3, conversation_id=conversation_id,
                    detail=f"{tier}: {reason[:200]}",
                    provider=attempt["provider"], model=attempt["model"],
                )
                response.tool_calls = [tc for tc in response.tool_calls if tc.name != "request_stronger_model"]

            research_mode_call = next((tc for tc in response.tool_calls if tc.name == "propose_research_mode"), None)
            if research_mode_call:
                # Stage 3's "fast-path intent layer" (IDEAS.md,
                # 2026-09-12) — only ever offered when NORMAL_CHAT.md's
                # tools list includes propose_research_mode, so this only
                # fires for mode == "normal" in practice. Intercepted the
                # same way ask_user_choice is: no server-side action,
                # calling it IS the model flagging a scope shift, and the
                # DISPATCHER (not the model) phrases the actual offer —
                # same "LLM proposes, dispatcher decides" principle
                # Research mode's own propose_plan_ready checkpoint uses.
                # `suggested_mode` rides on the return value (not
                # persisted anywhere) so navi-pwa's App.tsx can flip
                # chatMode client-side the moment "yes" is clicked —
                # Normal/Research/Brainstorm already share one
                # conversation_id (mode is a per-request parameter, never
                # stored), so switching is just sending the next message
                # under a different mode, no seed/handoff needed.
                args = _parse_tool_args(research_mode_call.arguments)
                reason = args.get("reason") or "This looks like it could use real research rather than a quick answer."
                question = f"{reason} Want me to switch this to Research mode?"
                options = ["Yes, switch to Research mode", "No, keep chatting here"]
                await append_message(conversation_id, "navi", question, provider=attempt["provider"], model=attempt["model"])
                return {
                    "text": question, "provider": attempt["provider"], "model": attempt["model"],
                    "choices": options, "suggested_mode": "research",
                }
            branch_complete_call = next((tc for tc in response.tool_calls if tc.name == "propose_branch_complete"), None)
            if branch_complete_call:
                # Same "propose, don't declare" shape as the two checkpoints
                # above. The model calling this is a CLAIM that the work is
                # done; the dispatcher writes the handover and the user
                # accepts it. A model that certifies its own work is just
                # marking its own homework.
                #
                # Guarded on actually being a branch: a chat with no parent
                # has nothing to hand back to and no acceptance criteria to
                # have met, so the call is meaningless there and is dropped
                # rather than acted on.
                conversation = await get_conversation(conversation_id)
                if conversation and conversation.get("parent_id"):
                    result = await _propose_branch_completion(
                        conversation_id, attempt["provider"], attempt["model"],
                    )
                    if result:
                        return result
                response.tool_calls = [tc for tc in response.tool_calls if tc.name != "propose_branch_complete"]

            choice_call = next((tc for tc in response.tool_calls if tc.name == "ask_user_choice"), None)
            if choice_call:
                # Intercepted BEFORE run_tool_loop, not executed through it
                # — this tool has no server-side action; calling it IS the
                # model handing a question back to the user. question is
                # persisted/returned as the reply text (so it reads
                # naturally without the tool call), options ride alongside
                # for the frontend to render as clickable buttons. Doesn't
                # survive a page refresh (only the question text is
                # persisted) — same known limit as usage_note.
                args = _parse_tool_args(choice_call.arguments)
                question = args.get("question", "")
                options = args.get("options") or []
                await append_message(conversation_id, "navi", question, provider=attempt["provider"], model=attempt["model"])
                return {
                    "text": question, "provider": attempt["provider"], "model": attempt["model"],
                    "usage_note": response.usage_note, "choices": options,
                }
            if tools and response.tool_calls:
                print(f"[run_stored_mode_chat] attempt {i}: entering run_tool_loop")
                # Name the tools BEFORE running them: this is the part of a
                # slow turn that is genuinely doing visible work, and it is
                # the strongest material the labor-illusion research says
                # to show. "Searching the web…" is a far better account of
                # a 20-second wait than "Thinking…".
                for tc in response.tool_calls:
                    await emit_status(emit, events.running_tool(tc.name), kind="tool")
                response, sent_messages, iterations = await asyncio.to_thread(
                    run_tool_loop, provider, attempt["model"], messages, response,
                    context={"command": f"chat-{mode}", "topic_slug": "chat"}, tools=tools, extra_params=extra_params,
                )
                created_workflow_id = _extract_created_workflow_id(sent_messages)
                print(
                    f"[run_stored_mode_chat] attempt {i}: run_tool_loop returned iterations={iterations} "
                    f"text={(response.text or '')[:200]!r} tool_calls={[tc.name for tc in response.tool_calls]} "
                    f"created_workflow_id={created_workflow_id}"
                )
                # tool_loop_exhausted used to be recorded here, because
                # run_tool_loop is sync and the friction writer was async
                # only. It is recorded inside run_tool_loop itself now
                # (2026-09-14) — which is what makes it fire for that
                # function's other four callers too, not just this one.
                if iterations > 0 and not response.text and not response.tool_calls:
                    # run_tool_loop actually executed a real tool call here
                    # (e.g. create_workflow persisted a row, send_to_telegram
                    # sent a message) — falling through to `continue` below
                    # would retry the NEXT fallback provider from scratch,
                    # replaying the same request and re-running that same
                    # side effect again. 2026-09-03, JuanJo: one "send me a
                    # Telegram message" request produced 5 duplicate
                    # workflows this way, one per fallback provider that
                    # also came back with an empty wrap-up. Once execution
                    # already happened, an empty wrap-up is a done-but-
                    # unsummarized outcome, not a failure to retry.
                    reply = "Done — the action completed, but I didn't get a summary back. Check Workflows / Run History for the result."
                    print(f"[run_stored_mode_chat] attempt {i}: DECISION = stop here (real tool call already ran, empty wrap-up) — NOT retrying fallback")
                    await append_message(conversation_id, "navi", reply, provider=attempt["provider"], model=attempt["model"])
                    return {
                        "text": reply, "provider": attempt["provider"], "model": attempt["model"],
                        "usage_note": response.usage_note,
                        **({"created_workflow_id": created_workflow_id} if created_workflow_id else {}),
                    }
            else:
                created_workflow_id = None
            if not response.text and not response.tool_calls:
                # A real failure mode, not a valid (if terse) answer — a
                # model that returns neither text nor a tool call did
                # nothing at all (2026-09-02: gpt-oss-20b via Cloudflare,
                # asked to create a workflow, returned a completely blank
                # response — no create_workflow call, nothing). Treat it
                # the same as a ProviderError so the next attempt in the
                # fallback chain actually gets tried, instead of silently
                # "succeeding" with an unhelpful "(empty reply)" placeholder
                # and nothing having happened.
                last_error = f"{attempt['provider']}/{attempt['model']} returned neither text nor a tool call"
                print(f"[run_stored_mode_chat] attempt {i}: DECISION = retry next fallback ({last_error})")
                await asyncio.to_thread(
                    save_failed_exchange, role_context, attempt["provider"], attempt["model"],
                    sent_messages, last_error, response.raw,
                )
                continue
            if not response.text and sent_messages is not messages:
                # Reached with tool_calls still non-empty (run_tool_loop hit
                # MAX_TOOL_ITERATIONS while the model kept asking for more
                # tool calls, e.g. repeat web_search attempts, instead of
                # ever writing a summary) — neither guard above catches this
                # specific shape, since both require tool_calls to be empty.
                # Real gathered material (search snippets, page content)
                # very likely exists in sent_messages regardless; throwing
                # it away for a bare "(empty reply)" is the exact bug a real
                # user hit in Normal Chat (2026-09-03): asked for 3 AI news
                # stories summarized, got "(empty reply)" twice in a row —
                # the raw results were sitting right there in the transcript.
                reply = _extract_tool_results(sent_messages) or "(empty reply)"
            else:
                reply = strip_reasoning_tags(response.text) or "(empty reply)"
            # _collapse_repeated_paragraphs records the repetition_loop
            # signal itself now — see its own note.
            reply = _collapse_repeated_paragraphs(reply)
            if i > 0:
                reply += f"\n\n⚡ (primary was unavailable, answered via {attempt['provider']}/{attempt['model']} instead)"
                await record_friction(
                    "fallback_used", severity=1, conversation_id=conversation_id,
                    detail=f"attempt {i}: {attempt['provider']}/{attempt['model']}",
                )
            print(f"[run_stored_mode_chat] attempt {i}: DECISION = success, returning (created_workflow_id={created_workflow_id})")
            await append_message(conversation_id, "navi", reply, provider=attempt["provider"], model=attempt["model"])
            # Ceiling check, AFTER the reply is persisted — see
            # _maybe_compact_context's own docstring for why compaction
            # runs inline here rather than as a background job.
            hit_floor = False
            if mode != "agent_work":
                hit_floor = await _maybe_compact_context(conversation_id)
            if hit_floor:
                # The escape valve, surfaced. Its own persisted message,
                # not text appended to the reply: the user asked something
                # and still gets their answer intact, and a note about the
                # conversation reads as exactly that. Persisting it also
                # means it survives a reload, unlike usage_note/choices.
                await append_message(conversation_id, "navi", FLOOR_NOTICE)
            return {
                "text": reply, "provider": attempt["provider"], "model": attempt["model"],
                "usage_note": response.usage_note,
                **({"created_workflow_id": created_workflow_id} if created_workflow_id else {}),
                **({"branch_suggestion": FLOOR_NOTICE} if hit_floor else {}),
            }
        except ProviderError as e:
            last_error = str(e)
            print(f"[run_stored_mode_chat] attempt {i}: DECISION = retry next fallback (ProviderError: {last_error})")
            await asyncio.to_thread(
                save_failed_exchange, role_context, attempt["provider"], attempt["model"], messages, last_error,
            )
            continue

    error_text = f"⚠️ {role_context} failed on every configured provider: {last_error}"
    await append_message(conversation_id, "navi", error_text)
    return {"text": error_text, "provider": None, "model": None}


async def run_agent_vault_chat(agent: dict, conversation_id: str, text: str) -> dict:
    """Persisted chat with one specific saved agent (Agent Vault) — sibling
    of run_stored_mode_chat above, but the system prompt/tools come from
    the agent's OWN row (storage/agents.py) instead of a fixed mode brief,
    closing the gap AgentVaultChat.tsx's own header comment flagged
    (2026-09-10: it was posting through agent_work's generic /chat/send,
    same brief/tools/model as every other Agent Work chat, regardless of
    which saved agent's window it was). Keeps full history (unlike
    agent_work's deliberately stateless design above) — a saved-agent
    chat is an ongoing conversation with that one agent, not a one-shot
    "build this" instruction.
    """
    await append_message(conversation_id, "user", text)
    history = await get_messages(conversation_id, limit=RECENT_MESSAGE_WINDOW)

    tools = schemas_for(agent["tools"]) if agent["tools"] else None
    system_parts = [agent["instructions"]]
    if tools:
        system_parts.append(CITATION_STYLE_PROMPT)
    messages = [ChatMessage(role="system", content="\n\n".join(system_parts))]
    for m in history:
        if m["role"] == "navi" and m["content"].startswith("⚠️"):
            continue
        messages.append(ChatMessage(role="assistant" if m["role"] == "navi" else m["role"], content=m["content"]))
    messages[-1].content = f"{messages[-1].content}\n\n[Current UTC time: {datetime.now(timezone.utc).isoformat()}]"

    set_call_context(
        role=role_name_for_context("agent_work"), mode="agent_vault",
        conversation_id=conversation_id,
    )
    try:
        role = get_dispatcher_role(context="agent_work")
    except ProviderNotConfigured as e:
        error_text = f"⚠️ Can't reply right now — agent_work isn't configured: {e}"
        await append_message(conversation_id, "navi", error_text)
        return {"text": error_text, "provider": None, "model": None}

    # A pinned model on the agent's own row (storage/agents.py's `model`
    # field — always null today, no UI sets one yet, see AgentVault.tsx)
    # takes priority over the shared agent_work role's own primary, but
    # still falls back through that role's configured chain if it fails —
    # same "extend, don't replace, the fallback chain" pattern the model
    # picker below (ModelBadge, still wired to the shared role) relies on.
    primary = {"provider": role["provider"], "model": role["model"]}
    if agent.get("model") and "/" in agent["model"]:
        pinned_provider, pinned_model = agent["model"].split("/", 1)
        primary = {"provider": pinned_provider, "model": pinned_model}
    attempts = config.get_attempts([primary] + role.get("fallback", []))
    last_error = None
    for i, attempt in enumerate(attempts):
        set_call_context(attempt=i, provider=attempt["provider"], model=attempt["model"])
        try:
            provider = get_provider(attempt["provider"])
        except Exception as e:
            last_error = str(e)
            continue
        try:
            sent_messages = messages
            response = await asyncio.to_thread(provider.chat, model=attempt["model"], messages=messages, tools=tools)
            choice_call = next((tc for tc in response.tool_calls if tc.name == "ask_user_choice"), None)
            if choice_call:
                args = _parse_tool_args(choice_call.arguments)
                question = args.get("question", "")
                options = args.get("options") or []
                await append_message(conversation_id, "navi", question, provider=attempt["provider"], model=attempt["model"])
                return {
                    "text": question, "provider": attempt["provider"], "model": attempt["model"],
                    "usage_note": response.usage_note, "choices": options,
                }
            if tools and response.tool_calls:
                response, sent_messages, iterations = await asyncio.to_thread(
                    run_tool_loop, provider, attempt["model"], messages, response,
                    context={"command": "chat-agent_vault", "topic_slug": "chat"}, tools=tools,
                )
                if iterations > 0 and not response.text and not response.tool_calls:
                    reply = "Done — the action completed, but I didn't get a summary back."
                    await append_message(conversation_id, "navi", reply, provider=attempt["provider"], model=attempt["model"])
                    return {"text": reply, "provider": attempt["provider"], "model": attempt["model"], "usage_note": response.usage_note}
            if not response.text and not response.tool_calls:
                last_error = f"{attempt['provider']}/{attempt['model']} returned neither text nor a tool call"
                await asyncio.to_thread(
                    save_failed_exchange, "agent_vault", attempt["provider"], attempt["model"],
                    sent_messages, last_error, response.raw,
                )
                continue
            if not response.text and sent_messages is not messages:
                reply = _extract_tool_results(sent_messages) or "(empty reply)"
            else:
                reply = strip_reasoning_tags(response.text) or "(empty reply)"
            reply = _collapse_repeated_paragraphs(reply)
            if i > 0:
                reply += f"\n\n⚡ (primary was unavailable, answered via {attempt['provider']}/{attempt['model']} instead)"
            await append_message(conversation_id, "navi", reply, provider=attempt["provider"], model=attempt["model"])
            return {"text": reply, "provider": attempt["provider"], "model": attempt["model"], "usage_note": response.usage_note}
        except ProviderError as e:
            last_error = str(e)
            await asyncio.to_thread(
                save_failed_exchange, "agent_vault", attempt["provider"], attempt["model"], messages, last_error,
            )
            continue

    error_text = f"⚠️ agent chat failed on every configured provider: {last_error}"
    await append_message(conversation_id, "navi", error_text)
    return {"text": error_text, "provider": None, "model": None}
