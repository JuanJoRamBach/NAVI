"""
dispatcher/agent_work.py

Executes an "Agent Work" workflow: a graph of nodes (see storage/agent_work.py
for the schema) walked in topological order, one node at a time, each run
through the same run_tool_loop primitive /research and mode chat already
share (dispatcher/executor.py) rather than a new orchestration engine.

A real topological sort (Kahn's algorithm), not a hardcoded linear walk —
v1 workflows only ever produce linear graphs (one edge in, one out, per
node), but the walker itself already handles branching/merging, so the
eventual node-graph visual builder needs no executor rewrite, only a UI
that can author non-linear graphs.

Execution happens in a background thread (mirroring server.py's existing
threading.Thread pattern for /research and Telegram webhooks) since
provider.chat() is a blocking HTTP call and must not stall FastAPI's event
loop; the thread gets its own asyncio.run() to talk to the (async)
storage layer, since aiosqlite connections aren't loop-agnostic.
"""

import asyncio
import threading
import time
from collections import deque
from datetime import datetime, timezone
from typing import Callable

from dispatcher.executor import _extract_tool_results, _parse_tool_args, run_tool_loop
from dispatcher.provider_debug import save_failed_exchange
from providers.base import ChatMessage, ChatResponse, ProviderError
from providers.registry import ProviderNotConfigured, get_dispatcher_role, get_provider
from storage.agent_work import (
    complete_step, create_step, create_run, due_workflows, get_workflow,
    set_step_input, update_run_status, update_workflow_trigger,
)
from tools.notes import NoteError, save_note
from tools.registry import schemas_for
from tools.telegram_send import TelegramSendError, send_to_telegram

# --- Node functions (2026-09-02) ---
# Each workflow step is executed by a plain function with FIXED logic —
# not a generic "give the LLM a prompt and hope it calls the right tool"
# handler, and not a class hierarchy either. Real prior art checked
# before choosing this shape: LangGraph (the leading code-first graph
# framework — the category NAVI's Agent Work actually belongs to, not
# n8n's visual/config-driven category) states it plainly — "nodes and
# edges are nothing more than functions, they can contain an LLM or just
# good ol' code." One function per node KIND (inferred from which single
# tool the node declares, matching the established "one step, one tool"
# convention every workflow already uses).
#
# Two real kinds of node, not one shape forced onto both (2026-09-02,
# JuanJo: "send to telegram is a deterministic function. the content for
# it is not... generating content or use a set text MUST be
# differentiated... we only use an LLM when it's actually needed"):
#   - DETERMINISTIC ACTION nodes (send_to_telegram, save_note) — the
#     real action is plain Python, called directly, never gated on an
#     LLM tool call succeeding. Their "content" is whatever's already
#     available: prior_context if a preceding step produced it, else the
#     node's own prompt text taken literally (someone — the chat model
#     while building the workflow, or a person via the manual/visual
#     builder — already wrote the actual text once, at CREATION time;
#     there's nothing left to compose at RUN time). Zero LLM calls for a
#     workflow like "send exactly this message" — the whole run is one
#     deterministic API call.
#   - LLM-DIRECTED nodes (web_search, fetch_page) — the model genuinely
#     has a judgment call to make (what query, interpreting what came
#     back), so tool-calling through run_tool_loop stays appropriate.
# A "generate content" step, when a workflow genuinely needs run-time
# composition from live data, is just the existing no-tools text node
# (_run_text_node) feeding a deterministic action node via prior_context
# — no new node kind needed, the pieces already compose.

TEXT_NODE_SYSTEM_PROMPT = (
    "You are executing one step of an automated NAVI workflow, running "
    "unattended (no user available to answer follow-up questions). This "
    "is a pure text-generation step — there is nothing to call or send. "
    "Write the requested text directly and completely."
)
WEB_SEARCH_NODE_SYSTEM_PROMPT = (
    "You are executing one step of an automated NAVI workflow: "
    "researching something via web search, running unattended. Call "
    "web_search with a query that covers the prompt below, then "
    "summarize what you actually found in your reply so a later step "
    "can use it — never answer from your own general knowledge instead "
    "of actually searching."
)
FETCH_PAGE_NODE_SYSTEM_PROMPT = (
    "You are executing one step of an automated NAVI workflow: reading a "
    "specific URL, running unattended. Call fetch_page with the URL "
    "described in the prompt below, then summarize what you actually "
    "found."
)
GENERIC_MULTI_TOOL_NODE_SYSTEM_PROMPT = (
    "You are executing one step of an automated NAVI workflow, running "
    "unattended (no user available to answer follow-up questions). Do "
    "the step's task directly using whichever of your tools it actually "
    "needs, and report the concrete result — don't just describe what "
    "you would do instead of doing it."
)


class WorkflowError(Exception):
    pass


def _node_system_prompt(prompt: str, prior_context: str | None) -> str:
    if not prior_context:
        return prompt
    return f"{prompt}\n\nOutput from the prior step(s) this one depends on:\n\n{prior_context}\n\nUse this as needed to complete your own task below."


def _call_for_node(
    debug_context: str, messages: list[ChatMessage],
    tools: list[dict] | None = None, tool_choice: str | dict | None = None,
) -> tuple[ChatResponse, object, str]:
    """The one provider/fallback attempt loop every node function shares
    — try the 'agent_work' role's primary, then each configured
    fallback. A response with neither text nor a tool call is treated as
    a failure worth retrying (2026-09-02 incident: three different
    Cloudflare models "completed" by returning nothing at all), not a
    success with nothing to show for it. Every failed attempt is saved
    to Filen via save_failed_exchange. Returns (response, provider
    instance, model name) on success — the provider instance is handed
    back so a caller that needs run_tool_loop doesn't have to re-resolve
    it. Raises WorkflowError once every attempt is exhausted."""
    try:
        role = get_dispatcher_role(context="agent_work")
    except ProviderNotConfigured as e:
        raise WorkflowError(f"role 'agent_work' isn't configured: {e}")

    attempts = [{"provider": role["provider"], "model": role["model"]}] + role.get("fallback", [])
    last_error = None
    for attempt in attempts:
        try:
            provider = get_provider(attempt["provider"])
        except Exception as e:
            last_error = str(e)
            continue
        try:
            response = provider.chat(model=attempt["model"], messages=messages, tools=tools, tool_choice=tool_choice)
        except ProviderError as e:
            last_error = str(e)
            save_failed_exchange(debug_context, attempt["provider"], attempt["model"], messages, last_error)
            continue
        if not response.text and not response.tool_calls:
            last_error = f"{attempt['provider']}/{attempt['model']} returned neither text nor a tool call"
            save_failed_exchange(debug_context, attempt["provider"], attempt["model"], messages, last_error, response.raw)
            continue
        return response, provider, attempt["model"]

    raise WorkflowError(f"every configured provider failed: {last_error}")


def _run_tool_forced_node(system_prompt: str, tool_name: str, prompt: str, prior_context: str | None, debug_context: str) -> str:
    """Shared body for the LLM-directed node kinds (web_search,
    fetch_page) — the model genuinely has a judgment call to make here
    (what to search, how to interpret a fetched page), unlike
    send_to_telegram/save_note which are pure deterministic dispatch
    (see _run_send_telegram_node's docstring). The only thing that
    differs between web_search and fetch_page is which tool and which
    fixed prompt, so this is the one place that logic lives."""
    tools = schemas_for([tool_name])
    messages = [
        ChatMessage(role="system", content=_node_system_prompt(system_prompt, prior_context)),
        ChatMessage(role="user", content=prompt),
    ]
    tool_choice = {"type": "function", "function": {"name": tool_name}}
    response, provider, model = _call_for_node(debug_context, messages, tools=tools, tool_choice=tool_choice)
    if response.tool_calls:
        response, sent_messages, _iterations = run_tool_loop(
            provider, model, messages, response,
            context={"command": "agent_work", "topic_slug": tool_name}, tools=tools,
        )
        if not response.text:
            # The tool call itself genuinely succeeded (real search
            # results/page content exist in sent_messages) — the model
            # just failed to write a summary of them, the same flaky-
            # model "empty reply after a successful tool call" pattern
            # seen throughout 2026-09-03. Falling back to "(empty reply)"
            # here silently threw away real material and then got sent
            # to Telegram verbatim by the next node. The raw tool output
            # (same helper /research already uses to hand a synthesizer
            # source material directly) is strictly better than nothing.
            raw = _extract_tool_results(sent_messages)
            if raw:
                return raw
    return response.text or "(empty reply)"


def _run_text_node(prompt: str, prior_context: str | None) -> str:
    messages = [
        ChatMessage(role="system", content=_node_system_prompt(TEXT_NODE_SYSTEM_PROMPT, prior_context)),
        ChatMessage(role="user", content=prompt),
    ]
    response, _provider, _model = _call_for_node("agent_work_step:text", messages)
    return response.text or "(empty reply)"


def _run_send_telegram_node(prompt: str, prior_context: str | None) -> str:
    """No LLM call, ever — send_to_telegram is a deterministic action
    with one input (the message text), and that text already exists by
    the time this runs: prior_context if a preceding step produced it
    live, otherwise the node's own prompt taken as the literal message
    (already-composed at workflow-creation time, not something to
    re-generate now). Raises WorkflowError on a real send failure
    (missing credentials, Telegram API error) — same disclosure
    principle as every other node."""
    text = prior_context or prompt
    if not text:
        raise WorkflowError("send_to_telegram step has no text to send (empty prompt, no prior step output)")
    try:
        return send_to_telegram(text)
    except TelegramSendError as e:
        raise WorkflowError(str(e))


def _run_web_search_node(prompt: str, prior_context: str | None) -> str:
    return _run_tool_forced_node(WEB_SEARCH_NODE_SYSTEM_PROMPT, "web_search", prompt, prior_context, "agent_work_step:web_search")


def _run_fetch_page_node(prompt: str, prior_context: str | None) -> str:
    return _run_tool_forced_node(FETCH_PAGE_NODE_SYSTEM_PROMPT, "fetch_page", prompt, prior_context, "agent_work_step:fetch_page")


def _run_save_note_node(prompt: str, prior_context: str | None) -> str:
    """No LLM call, ever — same reasoning as _run_send_telegram_node.
    save_note additionally needs a filename, which nothing has ever
    asked a human or a chat model to specify (create_workflow's steps
    schema has no field for it) — derived deterministically instead of
    inventing an LLM call just to name a file."""
    content = prior_context or prompt
    if not content:
        raise WorkflowError("save_note step has no content to save (empty prompt, no prior step output)")
    filename = f"step-{int(time.time())}.md"
    try:
        return save_note(command="agent_work", topic_slug="workflow", filename=filename, content=content)
    except NoteError as e:
        raise WorkflowError(str(e))


def _run_input_node(prompt: str, prior_context: str | None) -> str:
    """No LLM call, ever (2026-09-03, JuanJo: "whatever instruction has
    the Input and Output nodes, are deterministic, unless they want an
    LLM input node" — that variant isn't built, this is the plain
    default). An Input node IS its own literal configured value — its
    whole job is marking "this is what comes in from outside" (the same
    role a fan-out group's {{item}} already plays informally, just
    generalized to the whole workflow), not generating or interpreting
    anything. prior_context is accepted for signature symmetry with
    every other node function but deliberately ignored — an Input node
    has no meaningful predecessor by construction; if the graph gives it
    one anyway, its own configured value still wins."""
    if not prompt:
        raise WorkflowError("Input step has no value set.")
    return prompt


def _run_output_node(prompt: str, prior_context: str | None) -> str:
    """No LLM call, ever — same reasoning as _run_input_node. An Output
    node's whole job is returning whatever fed into it (a sub-agent
    handing a computed value back to whatever embeds it, per the Agent
    Vault design — see storage/agents.py), not taking a real-world
    action itself the way send_to_telegram/save_note do. Falls back to
    its own literal prompt only if genuinely nothing upstream produced
    anything, so an Output node never silently returns empty."""
    result = prior_context or prompt
    if not result:
        raise WorkflowError("Output step has nothing to return (no prior step output, no literal value set).")
    return result


def _run_generic_multi_tool_node(tool_names: list[str], prompt: str, prior_context: str | None) -> str:
    """Safety net for a node with more than one tool — not a named kind
    of its own (nothing in the chat-facing tool catalog produces this
    today, since AGENT_WORK_CHAT.md's steps are one-tool-each by
    convention), but a node built some other way (the future manual/
    visual graph editor, most likely) isn't restricted to that
    convention, so this keeps a multi-tool node working rather than
    crashing on an unrecognized shape."""
    tools = schemas_for(tool_names)
    messages = [
        ChatMessage(role="system", content=_node_system_prompt(GENERIC_MULTI_TOOL_NODE_SYSTEM_PROMPT, prior_context)),
        ChatMessage(role="user", content=prompt),
    ]
    response, provider, model = _call_for_node("agent_work_step:multi_tool", messages, tools=tools, tool_choice="required")
    if response.tool_calls:
        response, _messages, _iterations = run_tool_loop(
            provider, model, messages, response,
            context={"command": "agent_work", "topic_slug": "multi_tool"}, tools=tools,
        )
    return response.text or "(empty reply)"


# Dispatch table — a node's single declared tool selects its handler.
# Keyed by tool name, not by some separate "kind" field on the node
# itself, since the tool IS what determines the fixed logic/prompt a
# node needs; no reason to duplicate that as a second piece of data that
# could drift out of sync with the tools list.
SINGLE_TOOL_NODE_HANDLERS: dict[str, Callable[[str, str | None], str]] = {
    "send_to_telegram": _run_send_telegram_node,
    "web_search": _run_web_search_node,
    "fetch_page": _run_fetch_page_node,
    "save_note": _run_save_note_node,
    "input": _run_input_node,
    "output": _run_output_node,
}


def _topological_order(graph: dict) -> list[dict]:
    """Kahn's algorithm. Raises WorkflowError on a cycle or a dangling edge
    reference — fails the run cleanly rather than executing a partial/wrong
    order."""
    nodes = {n["id"]: n for n in graph.get("nodes", [])}
    incoming = {nid: 0 for nid in nodes}
    adjacency: dict[str, list[str]] = {nid: [] for nid in nodes}
    for edge in graph.get("edges", []):
        src, dst = edge["from"], edge["to"]
        if src not in nodes or dst not in nodes:
            raise WorkflowError(f"edge references unknown node: {edge}")
        adjacency[src].append(dst)
        incoming[dst] += 1

    queue = deque(nid for nid, degree in incoming.items() if degree == 0)
    order = []
    while queue:
        nid = queue.popleft()
        order.append(nodes[nid])
        for nxt in adjacency[nid]:
            incoming[nxt] -= 1
            if incoming[nxt] == 0:
                queue.append(nxt)

    if len(order) != len(nodes):
        raise WorkflowError("workflow graph has a cycle")
    return order


def _run_node(node: dict, prior_context: str | None = None) -> str:
    """Router, not an executor — picks the node function whose fixed
    logic matches this node's declared tool(s), and calls it. See the
    "Node functions" section above for what each one actually does.

    prior_context (2026-09-02): the completed output of this node's direct
    predecessors in the graph, if any — same "prior_context: str | None"
    shape dispatcher/executor.py's _run_remind_step already takes. Without
    this, a step genuinely depending on a prior step's result (e.g.
    "research news" -> "send what you found") had no way to see it; each
    node ran in total isolation from every other node.

    A node's optional "role" field (a per-node model-role override) is no
    longer read here — every node function now always uses the
    'agent_work' role. Not a real regression: no path that actually
    creates a node (the chat tool, the manual form) has ever set this
    field, so nothing exercised it before either. Worth restoring, keyed
    per node function, if a future caller (the visual graph builder)
    actually wants it."""
    prompt = node.get("prompt", "")
    tool_names = node.get("tools") or []

    if not tool_names:
        return _run_text_node(prompt, prior_context)
    if len(tool_names) == 1 and tool_names[0] in SINGLE_TOOL_NODE_HANDLERS:
        return SINGLE_TOOL_NODE_HANDLERS[tool_names[0]](prompt, prior_context)
    return _run_generic_multi_tool_node(tool_names, prompt, prior_context)


def _substitute_item(text: str | None, item: str) -> str | None:
    """The only templating this graph supports — a fan-out group's nodes
    reference the current loop item as the literal string "{{item}}" in
    their prompt. Deliberately not real Jinja2 (Conductor's own choice)
    or anything more general — one substitution, one variable name, no
    new dependency for a v1 slice."""
    return text if text is None else text.replace("{{item}}", item)


async def _execute_run(run_id: str, graph: dict) -> None:
    try:
        order = _topological_order(graph)
    except WorkflowError as e:
        await update_run_status(run_id, "failed", error=str(e))
        return

    # Direct predecessors per node, straight from the edge list — the same
    # graph.get("edges", []) _topological_order already walks, just indexed
    # the other direction (by destination instead of source).
    predecessors: dict[str, list[str]] = {n["id"]: [] for n in order}
    for edge in graph.get("edges", []):
        if edge["to"] in predecessors:
            predecessors[edge["to"]].append(edge["from"])

    # Fan-out groups (2026-09-03) — Agent Work's "sub-flows". A group
    # only affects execution if it was given an "items" list (built
    # purely for visual organization otherwise, see
    # navi-pwa/src/AgentWorkGraphEditor.tsx's Group node — those groups
    # never appear in graph["groups"] at all). node_group maps each
    # member node id to its group, so the main loop below can tell in
    # O(1) whether a given node needs to run once or once per item.
    node_group: dict[str, dict] = {
        nid: group
        for group in graph.get("groups", []) if group.get("items")
        for nid in group.get("node_ids", [])
    }

    # Composite-keyed by design: a node OUTSIDE any fan-out group (or in
    # a different one) keys its single output under its plain node id —
    # unchanged from before this feature existed. A node INSIDE a fan-out
    # group keys each iteration's output under "<node_id>#<item_index>",
    # since it genuinely produces one output per item, not one overall.
    outputs: dict[str, str] = {}
    seq = 0

    await update_run_status(run_id, "running")
    for node in order:
        group = node_group.get(node["id"])
        items = group["items"] if group else [None]  # [None] = run exactly once, no substitution

        for item_index, item in enumerate(items):
            prior_context_parts = []
            for pid in predecessors.get(node["id"], []):
                # A predecessor in the SAME fan-out group ran once per
                # item too — use THIS iteration's output from it. A
                # predecessor outside the group (or in a different one)
                # ran once total; that single output feeds every
                # iteration equally — e.g. a node before the group that
                # supplies shared context to each pass.
                same_group_predecessor = pid in node_group and node_group[pid] is group
                key = f"{pid}#{item_index}" if same_group_predecessor else pid
                if key in outputs:
                    prior_context_parts.append(f"[{pid}]: {outputs[key]}")
            prior_context = "\n\n".join(prior_context_parts) or None

            run_node = dict(node) if item is None else {**node, "prompt": _substitute_item(node.get("prompt"), item)}
            step_label = node["id"] if item is None else f"{node['id']} (item {item_index + 1}/{len(items)})"

            step_id = await create_step(run_id, node["id"], seq)
            seq += 1
            step_input = {"prompt": run_node.get("prompt"), "role": run_node.get("role"), "tools": run_node.get("tools")}
            if item is not None:
                step_input["item"] = item
            await set_step_input(step_id, step_input)
            try:
                output = await asyncio.to_thread(_run_node, run_node, prior_context)
                await complete_step(step_id, "completed", output=output)
                if item is not None:
                    outputs[f"{node['id']}#{item_index}"] = output
                # Plain key always gets written too, even inside a
                # fan-out — last iteration wins. A downstream node OUTSIDE
                # the group has no per-item concept of its own, so "the
                # most recent thing this node produced" is the only
                # sensible single value to hand it (same convention as a
                # variable reassigned each pass of a loop in any
                # language) — without this, a node placed right after a
                # fan-out group would see no context from it at all.
                outputs[node["id"]] = output
            except WorkflowError as e:
                await complete_step(step_id, "failed", error=str(e))
                await update_run_status(run_id, "failed", error=f"node '{step_label}' failed: {e}")
                return

    await update_run_status(run_id, "completed")


def _execute_run_thread(run_id: str, graph: dict) -> None:
    asyncio.run(_execute_run(run_id, graph))


async def start_run(graph: dict, workflow_id: str | None = None, trigger_source: str = "manual") -> str:
    """Creates the run row synchronously (so the caller — a FastAPI route —
    can return run_id immediately) then executes the graph in a background
    thread. Matches server.py's existing async-kickoff pattern for
    /research (threading.Thread + a pollable status), just with real
    per-run persistence instead of one global status string."""
    run_id = await create_run(workflow_id, trigger_source)
    threading.Thread(target=_execute_run_thread, args=(run_id, graph), daemon=True).start()
    return run_id


async def start_workflow_run(workflow_id: str, trigger_source: str = "manual") -> str:
    workflow = await get_workflow(workflow_id)
    if not workflow:
        raise WorkflowError(f"no workflow with id {workflow_id}")
    return await start_run(workflow["graph"], workflow_id=workflow_id, trigger_source=trigger_source)


RESOLVE_SCHEDULE_TOOL_NAME = "set_schedule"
RESOLVE_SCHEDULE_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": RESOLVE_SCHEDULE_TOOL_NAME,
        "description": "Report the resolved schedule for a workflow trigger, computed from the description and the current UTC time you were given.",
        "parameters": {
            "type": "object",
            "properties": {
                "first_run_at_utc": {
                    "type": "string",
                    "description": "ISO 8601 UTC timestamp of the first run, e.g. 2026-09-02T14:30:00+00:00.",
                },
                "interval_seconds": {
                    "type": "integer",
                    "description": "Seconds between runs. Omit (or 0) for a one-off, non-repeating run.",
                },
                "remaining_runs": {
                    "type": ["integer", "null"],
                    "description": "How many times it should fire, counting the first run. null (or omit) means no expiration set — fires indefinitely. Only set a number if the request gave a real count.",
                },
            },
            "required": ["first_run_at_utc"],
        },
    },
}
RESOLVE_SCHEDULE_TOOL_CHOICE = {"type": "function", "function": {"name": RESOLVE_SCHEDULE_TOOL_NAME}}

# Real prior incident (2026-09-02): "send a message in 5 mins" got resolved
# as interval_seconds=300, remaining_runs=null — the model read the delay
# before the single run as a recurrence cadence and fired every 5 minutes
# indefinitely until manually killed. Prompting alone already failed once
# here, so this is a deterministic backstop, not just better wording: if
# the ORIGINAL description (not the model's own paraphrase of it) doesn't
# contain a real recurrence cue, a repeating trigger is impossible to
# construct no matter what the model returns.
_RECURRENCE_CUES = (
    "every", "each ", "daily", "weekly", "hourly", "monthly", "repeat",
    "recurring", "recur", "again and again", "keep doing", "indefinitely",
    "until i say", "until you're told", "repeating:",
)


def _looks_recurring(description: str) -> bool:
    d = description.lower()
    return any(cue in d for cue in _RECURRENCE_CUES)


def resolve_schedule(description: str) -> dict:
    """Mirrors dispatcher/executor.py's _run_remind_step: gives the model
    the current UTC time in a system prompt, then FORCES a tool call so it
    can't skip resolution or answer in prose — the model's only job here
    is turning a plain-language schedule description into concrete
    numbers, isolated from the broader "build the workflow" task.

    Raises WorkflowError if every configured provider fails, or if the
    model's tool call can't be parsed — the caller (tools/workflows.py's
    create_workflow) should surface that as a tool error back to the
    chat rather than silently falling back to "manual"."""
    try:
        role = get_dispatcher_role(context="agent_work")
    except ProviderNotConfigured as e:
        raise WorkflowError(f"role 'agent_work' isn't configured: {e}")

    now = datetime.now(timezone.utc)
    messages = [
        ChatMessage(
            role="system",
            content=(
                f"Current UTC time: {now.isoformat()}\n"
                "Resolve the schedule description into a concrete first run time "
                "and (if it repeats) an interval, using the current UTC time above "
                "as your only source of 'now' — never guess. Call set_schedule with "
                "the result."
            ),
        ),
        ChatMessage(role="user", content=description),
    ]

    attempts = [{"provider": role["provider"], "model": role["model"]}] + role.get("fallback", [])
    last_error = None
    for attempt in attempts:
        try:
            provider = get_provider(attempt["provider"])
        except Exception as e:
            last_error = str(e)
            continue
        try:
            response = provider.chat(
                model=attempt["model"], messages=messages,
                tools=[RESOLVE_SCHEDULE_TOOL_SCHEMA], tool_choice=RESOLVE_SCHEDULE_TOOL_CHOICE,
            )
        except ProviderError as e:
            last_error = str(e)
            save_failed_exchange("resolve_schedule", attempt["provider"], attempt["model"], messages, last_error)
            continue

        if not response.tool_calls:
            last_error = "model didn't call set_schedule"
            save_failed_exchange("resolve_schedule", attempt["provider"], attempt["model"], messages, last_error, response.raw)
            continue
        try:
            args = _parse_tool_args(response.tool_calls[0].arguments)
            first_run = datetime.fromisoformat(args["first_run_at_utc"])
            interval = int(args.get("interval_seconds") or 0)
            remaining = args.get("remaining_runs")
            if interval and not _looks_recurring(description):
                # The model returned a repeat interval, but nothing in the
                # actual request said this should recur — force it back to
                # a one-off rather than trust a number that shouldn't exist.
                interval = 0
                remaining = None
            trigger = {
                "type": "scheduled",
                "interval_seconds": interval,
                "next_run_at": first_run.timestamp(),
            }
            if remaining is not None:
                trigger["remaining_runs"] = int(remaining)
            return trigger
        except (KeyError, ValueError, TypeError) as e:
            last_error = f"couldn't parse set_schedule call: {e}"
            save_failed_exchange("resolve_schedule", attempt["provider"], attempt["model"], messages, last_error, response.raw)
            continue

    raise WorkflowError(f"couldn't resolve schedule: {last_error}")


async def check_due_workflows() -> int:
    """Starts a run for every scheduled workflow whose trigger.next_run_at
    has passed, then rolls next_run_at forward by interval_seconds so it
    doesn't refire on the next check — UNLESS trigger.remaining_runs has
    just been exhausted (see below), in which case next_run_at is cleared
    instead, so due_workflows()'s own existing "next_run_at is set and
    past" check naturally stops picking this workflow up again. No new
    "is this exhausted" branch needed anywhere else — reusing the check
    that already exists for every other reason a trigger might have no
    next_run_at. Returns how many runs it started.

    trigger.remaining_runs (2026-09-01, JuanJo: "a counter that tells how
    many times it must be repeated... if that counter is null, it means
    it's scheduled until removed") — real prior art for this exact shape:
    Quartz Scheduler's SimpleTrigger.repeatCount (a positive integer to
    fire N more times, a sentinel for unlimited). null here (not a magic
    number — real, sourced REST convention for "no expiration set", e.g.
    GitLab's own token-expiration API) means unlimited, matching what
    every trigger already did before this field existed. Absent key
    reads identically to explicit null via dict.get(), but explicit null
    is preferred when WRITING one (create_workflow, the manual form) —
    self-documents "deliberately unlimited" rather than leaving it
    ambiguous whether the field was just never considered.

    Extracted out of server.py's GET /agent/workflows/due (2026-09-01) so
    both that route (kept as a manual poke/health-check) and the
    in-process scheduler (dispatcher/scheduler.py) call the exact same
    logic — one code path regardless of what triggers the check."""
    started = 0
    for workflow in await due_workflows():
        await start_run(workflow["graph"], workflow_id=workflow["id"], trigger_source="scheduled")
        trigger = workflow["trigger"]
        interval = trigger.get("interval_seconds")
        remaining = trigger.get("remaining_runs")
        if remaining is not None:
            remaining -= 1
            trigger["remaining_runs"] = remaining
        if interval and (remaining is None or remaining > 0):
            trigger["next_run_at"] = time.time() + interval
        else:
            trigger["next_run_at"] = None
        await update_workflow_trigger(workflow["id"], trigger)
        started += 1
    return started
