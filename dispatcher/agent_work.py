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
import json
import random
import re
import secrets
import tempfile
import threading
import time
import uuid
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from dispatcher.executor import _extract_tool_results, _parse_tool_args, run_tool_loop
from dispatcher.provider_debug import save_failed_exchange
from providers.base import ChatMessage, ChatResponse, ProviderError
from providers.registry import ProviderNotConfigured, get_dispatcher_role, get_provider
from storage.agent_work import (
    complete_step, create_step, create_run, due_workflows, get_workflow,
    get_workflow_by_webhook_token, set_step_input, update_run_status, update_workflow_trigger,
)
from tools.documents import DocumentRenderError, render_pdf
from tools.gmail_send import GmailSendError, send_gmail_message
from tools.notes import NoteError, save_note
from tools.registry import schemas_for
from tools.telegram_send import TelegramSendError, send_file_to_telegram, send_to_telegram

# Real prior art (2026-09-06): LangGraph's whole graph operates on ONE
# typed state object — every node returns a partial update, any later
# node can reference any earlier one's output by name, not just its
# immediate predecessor's raw string. _execute_run's own `outputs` dict
# (keyed by node id, or "node_id#item_index" inside a fan-out — see
# below) already IS that shared state; this just exposes it to a node's
# OWN config fields too; instead of collapsing everything into one
# joined prior_context string, a field can name exactly which earlier
# node it wants: "to": "{{state.n2}}" pulls n2's real output directly,
# independent of whichever nodes happen to be direct predecessors.
# Whole-value only, not embedded string templating — a field's value
# either IS a state reference or is a literal, no templating engine.
#
# Optional dot-path suffix (2026-09-07): "{{state.n2.customer.email}}"
# pulls one field out of n2's output instead of the whole thing — real
# need, not speculative: a webhook trigger's output is the raw JSON
# payload it was called with, and "get one field out of an API payload"
# is n8n/Zapier's single most common real use case for this exact
# mechanism (checked directly against both — this isn't a guess). Only
# meaningful when the referenced node's output actually IS JSON; see
# _resolve_json_path's own docstring for what happens when it isn't.
_STATE_REF_RE = re.compile(r"^\{\{state\.([A-Za-z0-9_-]+)((?:\.[A-Za-z0-9_-]+)*)\}\}$")


def _resolve_json_path(raw_value: str, path_parts: list[str]) -> str:
    """Walks a dot-path into a node's output, parsed as JSON. Raises
    WorkflowError on any failure (not JSON, missing key, indexing into
    something that isn't a dict/list) rather than silently falling back
    to an empty string — a wrong path is a real, catchable mistake;
    "found the field" and "resolved to nothing" must never look the
    same here, unlike the plain whole-value reference above (which keeps
    its existing empty-string-on-miss behavior, unchanged, for every
    workflow already relying on that)."""
    path = ".".join(path_parts)
    try:
        current = json.loads(raw_value)
    except (TypeError, ValueError):
        raise WorkflowError(f"Can't look up '{path}' — the referenced step's output isn't JSON.")
    for part in path_parts:
        if isinstance(current, dict):
            if part not in current:
                raise WorkflowError(f"'{part}' isn't a field in the referenced step's output (looking up '{path}').")
            current = current[part]
        elif isinstance(current, list):
            try:
                index = int(part)
            except ValueError:
                raise WorkflowError(f"'{part}' isn't a valid list index (looking up '{path}').")
            if not (0 <= index < len(current)):
                raise WorkflowError(f"Index {part} is out of range (looking up '{path}').")
            current = current[index]
        else:
            raise WorkflowError(f"Can't look up '{part}' — that part of the referenced output isn't an object or list (looking up '{path}').")
    return current if isinstance(current, str) else json.dumps(current)


def _resolve_state_refs(node: dict, outputs: dict[str, str]) -> dict:
    resolved = dict(node)
    for key, value in node.items():
        if isinstance(value, str):
            m = _STATE_REF_RE.match(value)
            if m:
                node_id, path_suffix = m.group(1), m.group(2)
                base = outputs.get(node_id, "")
                path_parts = [p for p in path_suffix.split(".") if p]
                resolved[key] = _resolve_json_path(base, path_parts) if path_parts else base
    return resolved

# Marks a node's string output as a reference to a real file on disk
# rather than literal text to send/save as-is — currently only produced
# by an Output node with output_type="pdf" (see _run_output_node) and
# consumed by _run_send_telegram_node. A plain temp-file convention, not
# a new storage layer: the file only needs to survive from one step to
# the next within the SAME workflow run, which already executes start to
# finish in one process (see this module's own docstring).
FILE_OUTPUT_PREFIX = "navi-file://"

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
CHOOSE_PATH_NODE_SYSTEM_PROMPT_TEMPLATE = (
    "You are executing one step of an automated NAVI workflow: deciding "
    "which branch applies, running unattended. Given the condition below "
    "and anything the prior step(s) produced, respond with EXACTLY one "
    "of these branch labels and nothing else, no punctuation or "
    "explanation: {labels}"
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
    principle as every other node.

    FILE_OUTPUT_PREFIX (2026-09-03) is the one exception to "text" —
    when the immediately preceding step was an Output node with
    output_type="pdf", prior_context is a marked file path rather than
    real message text; sent as a real Telegram document attachment
    (send_file_to_telegram, tools/telegram_send.py) instead of stuffing
    a local temp path into a chat message."""
    text = prior_context or prompt
    if not text:
        raise WorkflowError("send_to_telegram step has no text to send (empty prompt, no prior step output)")
    try:
        if text.startswith(FILE_OUTPUT_PREFIX):
            path = text[len(FILE_OUTPUT_PREFIX):]
            return send_file_to_telegram(path, Path(path).name)
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


def _run_send_email_node(node: dict, prior_context: str | None) -> str:
    """No LLM call, ever — same reasoning as _run_send_telegram_node/
    _run_save_note_node (2026-09-06: "the dispatcher is the one that
    reads it, never an LLM" — this is that same principle, extended to
    sending). `to` and `body` are the node's own configured fields
    (each independently resolved against the shared run state via
    {{state.<node_id>}} in _execute_run before this ever runs), not
    something invented at run time. A bulk/dataset recipient list is
    real, designed scope, NOT built here yet — this handles a single
    address or a literal comma-separated list only; the dispatcher-only,
    never-an-LLM constraint for reading a real client database applies
    regardless of which real data source that ends up being. Bypasses
    the connected Gmail MCP server (confirmed no send capability at all)
    via tools/gmail_send.py's direct REST call."""
    to = node.get("to")
    body = node.get("body") or prior_context
    if not to:
        raise WorkflowError("Send Email step has no 'to' recipient configured.")
    if not body:
        raise WorkflowError("Send Email step has no body (no literal value, no prior step output).")
    subject = node.get("subject") or "Message from NAVI"
    try:
        message_id = send_gmail_message(to, subject, body, html=True)
    except GmailSendError as e:
        raise WorkflowError(str(e))
    return f"Sent to {to} (message {message_id})"


# A workflow node blocking for longer than this ties up its own
# background thread indefinitely with no way to persist and resume later
# — real for arbitrarily long delays would need a "resume at this
# timestamp" mechanism this codebase doesn't have (same family of gap as
# scheduled workflows before dispatcher/scheduler.py existed). 1 hour
# covers every real "space these out" / "wait for X to catch up" use
# case without risking a run silently pinned in memory for a full day.
_MAX_DELAY_SECONDS = 3600.0


def _run_delay_node(node: dict, prior_context: str | None) -> str:
    """Pauses this run for a fixed duration, then passes whatever fed
    into it straight through unchanged — the plain "wait N seconds"
    primitive every automation tool has (n8n's Wait, Zapier's Delay,
    Make's Sleep). time.sleep, not asyncio.sleep, is correct here: this
    always runs inside asyncio.to_thread's own worker thread (see
    _run_node_with_resilience's timeout wrapper), never on the event loop
    itself, so blocking it doesn't stall anything else in the process."""
    try:
        seconds = float(node.get("seconds"))
    except (TypeError, ValueError):
        raise WorkflowError("Delay step has no valid number of seconds set.")
    if seconds < 0:
        raise WorkflowError("Delay step's seconds can't be negative.")
    time.sleep(min(seconds, _MAX_DELAY_SECONDS))
    return prior_context or ""


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


def _run_output_node(prompt: str, prior_context: str | None, output_type: str | None = None) -> str:
    """No LLM call, ever — same reasoning as _run_input_node. An Output
    node's default job is returning whatever fed into it (a sub-agent
    handing a computed value back to whatever embeds it, per the Agent
    Vault design — see storage/agents.py), not taking a real-world
    action itself the way send_to_telegram/save_note do.

    output_type (2026-09-03, JuanJo: "create an output node before
    sending to telegram with pdf as output, we already have that" — the
    Agent Vault output_type vocabulary, chat/pdf/markdown) is the one
    exception: "pdf" renders the text into a real file (via
    tools/documents.py's render_pdf, already Unicode-safe — same
    renderer /research's own file-export path uses) and returns a
    FILE_OUTPUT_PREFIX-marked path instead of the raw text, which
    _run_send_telegram_node below recognizes and sends as an attachment
    rather than stuffing a file path into a chat message. Anything else
    (None, "chat", "markdown") is the original plain pass-through —
    markdown text renders fine as-is in a Telegram message, no separate
    handling needed."""
    result = prior_context or prompt
    if not result:
        raise WorkflowError("Output step has nothing to return (no prior step output, no literal value set).")
    if output_type == "pdf":
        try:
            pdf_bytes = render_pdf("NAVI Workflow Output", result)
        except DocumentRenderError as e:
            raise WorkflowError(f"PDF rendering failed: {e}")
        path = Path(tempfile.gettempdir()) / f"navi-output-{uuid.uuid4().hex}.pdf"
        path.write_bytes(pdf_bytes)
        return f"{FILE_OUTPUT_PREFIX}{path}"
    return result


def _run_choose_path_node(prompt: str, prior_context: str | None, labels: list[str]) -> str:
    """A real model call (2026-09-04) — unlike Input/Output's plain pass-
    through, deciding which branch applies genuinely needs judgment
    (JuanJo, 2026-09-03: "a conditional branch node is what we need").
    `labels` are the edge labels the canvas collected for every edge
    LEAVING this node (see AgentWorkGraphEditor.tsx's edge-label UI) —
    the model must pick exactly one, and its choice becomes the node's
    own string output, the same "the node's output IS what happened"
    shape every other node already has. _execute_run below matches this
    return value against the edge labels to decide which branch actually
    runs and which get skipped.

    Falls back to the first label — or one literally labeled "else" /
    "default" / "otherwise" if any exists, matching Zapier Paths' and
    Make's own default-route convention — when the model's reply doesn't
    cleanly match any label, rather than crashing the whole run over a
    flaky/free model returning noise (the exact failure mode this
    codebase hit repeatedly on 2026-09-03)."""
    if not labels:
        raise WorkflowError("Choose a Path step has no labeled branches — label at least one outgoing edge.")
    if len(labels) == 1:
        return labels[0]
    if not prompt:
        raise WorkflowError("Choose a Path step has no condition set.")
    messages = [
        ChatMessage(
            role="system",
            content=_node_system_prompt(CHOOSE_PATH_NODE_SYSTEM_PROMPT_TEMPLATE.format(labels=", ".join(labels)), prior_context),
        ),
        ChatMessage(role="user", content=prompt),
    ]
    response, _provider, _model = _call_for_node("agent_work_step:choose_path", messages)
    reply = (response.text or "").strip().strip(".\"'")
    for label in labels:
        if reply.lower() == label.lower():
            return label
    for label in labels:
        if label.lower() in reply.lower():
            return label
    fallback = next((label for label in labels if label.lower() in ("else", "default", "otherwise")), labels[0])
    return fallback


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


def _run_node(node: dict, prior_context: str | None = None, outgoing_labels: list[str] | None = None) -> str:
    """Router, not an executor — picks the node function whose fixed
    logic matches this node's declared tool(s), and calls it. See the
    "Node functions" section above for what each one actually does.

    prior_context (2026-09-02): the completed output of this node's direct
    predecessors in the graph, if any — same "prior_context: str | None"
    shape dispatcher/executor.py's _run_remind_step already takes. Without
    this, a step genuinely depending on a prior step's result (e.g.
    "research news" -> "send what you found") had no way to see it; each
    node ran in total isolation from every other node.

    outgoing_labels (2026-09-04): only meaningful for a choose_path node
    — the labels collected off this node's own outgoing edges, computed
    by _execute_run (which has the graph) since a bare node dict has no
    concept of its own edges.

    A node's optional "role" field (a per-node model-role override) is no
    longer read here — every node function now always uses the
    'agent_work' role. Not a real regression: no path that actually
    creates a node (the chat tool, the manual form) has ever set this
    field, so nothing exercised it before either. Worth restoring, keyed
    per node function, if a future caller (the visual graph builder)
    actually wants it."""
    # Real `kind` discriminator (2026-09-06) — checked first. A node kind
    # with fields beyond plain prompt/prior_context (send_email's to/body,
    # output's output_type) is looked up directly rather than overloading
    # the `tools` list as the selector; a plain kind with no extra fields
    # still just wraps the existing SINGLE_TOOL_NODE_HANDLERS entry, so
    # nothing about their own logic changes, only how they get selected.
    kind = node.get("kind")
    if kind == "send_email":
        return _run_send_email_node(node, prior_context)
    if kind == "delay":
        return _run_delay_node(node, prior_context)
    if kind == "output":
        return _run_output_node(node.get("prompt", ""), prior_context, node.get("output_type"))
    if kind == "choose_path":
        return _run_choose_path_node(node.get("prompt", ""), prior_context, outgoing_labels or [])
    if kind == "text":
        return _run_text_node(node.get("prompt", ""), prior_context)
    if kind in SINGLE_TOOL_NODE_HANDLERS:
        return SINGLE_TOOL_NODE_HANDLERS[kind](node.get("prompt", ""), prior_context)

    # Legacy path — no `kind` field at all (a workflow saved before this
    # existed, or built through a caller that hasn't been updated to set
    # it yet): dispatch off `tools` exactly as this router always has.
    prompt = node.get("prompt", "")
    tool_names = node.get("tools") or []

    if not tool_names:
        return _run_text_node(prompt, prior_context)
    if len(tool_names) == 1 and tool_names[0] in ("output", "choose_path"):
        # The two node kinds whose behavior varies by something beyond
        # prompt/prior_context — kept as narrow special cases rather than
        # widening every other handler's shared 2-arg signature for them.
        if tool_names[0] == "output":
            return _run_output_node(prompt, prior_context, node.get("output_type"))
        return _run_choose_path_node(prompt, prior_context, outgoing_labels or [])
    if len(tool_names) == 1 and tool_names[0] in SINGLE_TOOL_NODE_HANDLERS:
        return SINGLE_TOOL_NODE_HANDLERS[tool_names[0]](prompt, prior_context)
    return _run_generic_multi_tool_node(tool_names, prompt, prior_context)


# Deterministic, network/IO-bound node kinds — the ones with no retry of
# their own. LLM-backed kinds (writeText/generateAi/choose_path/text,
# the legacy no-`kind` text path, and the generic multi-tool fallback)
# already get a provider-fallback retry from _call_for_node; wrapping
# them in a SECOND retry here would just multiply latency for no benefit,
# so they get exactly one attempt in _run_node_with_resilience below
# (still timeout-guarded — every node kind gets that part).
_RETRYABLE_KINDS = {"send_to_telegram", "web_search", "fetch_page", "save_note", "send_email"}

# Reliability numbers (2026-09-07) — researched, not guessed, against the
# actual tools this design is modeled on: LangGraph's own published
# RetryPolicy default is EXACTLY max_attempts=3, initial_interval=0.5s,
# backoff_factor=2.0, max_interval=128s, jitter=True (confirmed against
# LangChain's own reference docs) — used verbatim here rather than
# inventing different numbers, since NAVI's whole reliability model for
# Agent Work is already explicitly built off LangGraph's fault-tolerance
# primitives (RetryPolicy/TimeoutPolicy/error_handler). n8n's own "sane
# baseline" for its opt-in per-node retry independently lands on the same
# 3-attempt count, for what that's worth as a second data point. The 60s
# per-node timeout isn't as precisely sourced — n8n and Langflow both
# reference something in this range (n8n caps its own per-retry wait at
# 5s, Langflow's community discussion treats 60s as the ordinary
# component-timeout ceiling worth raising past) rather than publishing
# one canonical default, so 60s here is a reasonable middle point, not a
# copied constant the way the retry numbers are.
_NODE_TIMEOUT_SECONDS = 60
_MAX_NODE_ATTEMPTS = 3
_RETRY_INITIAL_INTERVAL = 0.5
_RETRY_BACKOFF_FACTOR = 2.0
_RETRY_MAX_INTERVAL = 128.0


def _is_retryable_node(run_node: dict) -> bool:
    """Mirrors _run_node's own kind-vs-legacy-tools dispatch logic (see its
    docstring) so retry eligibility matches whatever _run_node will
    actually treat this node as, including a workflow saved before the
    `kind` field existed."""
    kind = run_node.get("kind")
    if kind in _RETRYABLE_KINDS:
        return True
    if kind is not None:
        return False
    tools = run_node.get("tools") or []
    return len(tools) == 1 and tools[0] in _RETRYABLE_KINDS


def _timeout_for_node(run_node: dict) -> float:
    """A delay node's own configured wait is an intentional, expected
    duration, not a hang to guard against — the fixed _NODE_TIMEOUT_SECONDS
    would otherwise kill any delay longer than 60s and misreport it as a
    timeout. +5s buffer over the delay itself (capped the same way
    _run_delay_node caps its own time.sleep) covers normal scheduling
    jitter without weakening the timeout's real purpose for every other
    node kind, which still gets the plain fixed default."""
    if run_node.get("kind") == "delay":
        try:
            seconds = float(run_node.get("seconds"))
        except (TypeError, ValueError):
            seconds = 0.0
        return min(max(seconds, 0.0), _MAX_DELAY_SECONDS) + 5.0
    return _NODE_TIMEOUT_SECONDS


async def _run_node_with_resilience(
    run_node: dict, prior_context: str | None, outgoing_labels: list[str] | None,
) -> str:
    """Wraps a single node's execution in a wall-clock timeout (every node
    kind) plus a bounded exponential-backoff retry (deterministic action
    kinds only — see _RETRYABLE_KINDS above). Raises WorkflowError on
    final failure either way, same as a plain _run_node call would — this
    is purely "survive a transient blip before giving up," not a change
    to _execute_run's own all-or-nothing run-failure semantics (that's a
    separate, later step: continue-on-error/error edges)."""
    attempts = _MAX_NODE_ATTEMPTS if _is_retryable_node(run_node) else 1
    timeout = _timeout_for_node(run_node)
    last_error: BaseException = WorkflowError("node never ran")
    for attempt in range(attempts):
        try:
            return await asyncio.wait_for(
                asyncio.to_thread(_run_node, run_node, prior_context, outgoing_labels),
                timeout=timeout,
            )
        except asyncio.TimeoutError:
            last_error = WorkflowError(f"timed out after {timeout:g}s")
        except WorkflowError as e:
            last_error = e
        if attempt + 1 >= attempts:
            break
        interval = min(_RETRY_INITIAL_INTERVAL * (_RETRY_BACKOFF_FACTOR ** attempt), _RETRY_MAX_INTERVAL)
        interval *= 1 + random.uniform(-0.1, 0.1)  # jitter, +/-10%, matching LangGraph's own default
        await asyncio.sleep(interval)
    raise last_error


def _substitute_item(text: str | None, item: str) -> str | None:
    """The only templating this graph supports — a fan-out group's nodes
    reference the current loop item as the literal string "{{item}}" in
    their prompt. Deliberately not real Jinja2 (Conductor's own choice)
    or anything more general — one substitution, one variable name, no
    new dependency for a v1 slice."""
    return text if text is None else text.replace("{{item}}", item)


async def _execute_run(run_id: str, graph: dict, initial_outputs: dict[str, str] | None = None) -> None:
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

    # Labeled edges leaving a choose_path node — its own only source of
    # "what are the possible branches" (a bare node dict has no concept
    # of its own edges). Unlabeled edges out of a choose_path node are
    # simply never eligible to be chosen or pruned; label at least one.
    outgoing_labels_by_node: dict[str, list[str]] = {}
    for edge in graph.get("edges", []):
        label = edge.get("label")
        if label:
            outgoing_labels_by_node.setdefault(edge["from"], []).append(label)

    # choose_path pruning (2026-09-04): a node is skipped iff EVERY edge
    # feeding it is dead — either its source was itself skipped, or the
    # specific edge was pruned by a choose_path decision. Evaluated once
    # per node, in the SAME topological order already being walked below,
    # so every predecessor's live/dead status is already final by the
    # time a node is checked — no separate forward pass needed, and (the
    # part a naive forward-BFS-from-the-pruned-branch gets wrong) a node
    # reachable via BOTH a taken and an untaken branch correctly stays
    # live, since at least one of its incoming edges is live.
    skipped: set[str] = set()
    pruned_edges: set[tuple[str, str]] = set()

    def _is_skipped(node_id: str) -> bool:
        preds = predecessors.get(node_id, [])
        if not preds:
            return False  # a root node always runs
        return all(pid in skipped or (pid, node_id) in pruned_edges for pid in preds)

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
    # Pre-seeded from outside the graph entirely — currently only a
    # webhook trigger's incoming payload (2026-09-07), set before this
    # function is even called (see start_webhook_run). Copied, not
    # aliased, so mutating `outputs` below never reaches back into the
    # caller's dict.
    outputs: dict[str, str] = dict(initial_outputs or {})
    seq = 0

    await update_run_status(run_id, "running")
    for node in order:
        if _is_skipped(node["id"]):
            # A branch choose_path didn't take, or something only
            # reachable through one — recorded as a real step (not
            # silently absent) so the run history shows what was skipped
            # and why, matching every other node's own transparency.
            skipped.add(node["id"])
            step_id = await create_step(run_id, node["id"], seq)
            seq += 1
            await set_step_input(step_id, {"prompt": node.get("prompt"), "role": node.get("role"), "tools": node.get("tools")})
            await complete_step(step_id, "skipped", output="Skipped — branch not taken.")
            continue

        if node["id"] in outputs:
            # Pre-seeded (a webhook trigger node, whose "output" IS the
            # payload the call arrived with, not something to compute) —
            # recorded as a real, already-completed step for the same run-
            # history transparency every other node gets, but _run_node is
            # never called on it: there's nothing to run, and no handler
            # for a trigger-only kind exists (deliberately — see
            # find_webhook_trigger_node_id's own docstring).
            step_id = await create_step(run_id, node["id"], seq)
            seq += 1
            await set_step_input(step_id, {"prompt": node.get("prompt"), "role": node.get("role"), "tools": node.get("tools")})
            await complete_step(step_id, "completed", output=outputs[node["id"]])
            continue

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
            try:
                run_node = _resolve_state_refs(run_node, outputs)
            except WorkflowError as e:
                # A bad {{state...}} path (typo'd field, wrong node,
                # non-JSON output) is a real, reportable failure — same
                # "fail the run cleanly with a real error" treatment as
                # every other node failure below, not an uncaught
                # exception that would silently kill this whole
                # background thread with the run stuck at "running"
                # forever and nothing to show for why.
                step_id = await create_step(run_id, node["id"], seq)
                seq += 1
                await set_step_input(step_id, {"prompt": node.get("prompt"), "role": node.get("role"), "tools": node.get("tools")})
                await complete_step(step_id, "failed", error=str(e))
                await update_run_status(run_id, "failed", error=f"node '{step_label}' failed: {e}")
                return

            step_id = await create_step(run_id, node["id"], seq)
            seq += 1
            step_input = {
                "prompt": run_node.get("prompt"), "role": run_node.get("role"), "tools": run_node.get("tools"),
                # Real values used THIS run, already resolved through
                # _resolve_state_refs — a run's history should show what
                # actually got sent, not the unresolved "{{state.n1}}"
                # placeholder, same transparency principle as everything
                # else this file records.
                **{k: run_node[k] for k in ("kind", "to", "body", "subject", "output_type") if run_node.get(k) is not None},
            }
            if item is not None:
                step_input["item"] = item
            await set_step_input(step_id, step_input)
            try:
                is_choose_path = run_node.get("kind") == "choose_path" or run_node.get("tools") == ["choose_path"]
                output = await _run_node_with_resilience(
                    run_node, prior_context,
                    outgoing_labels_by_node.get(node["id"]) if is_choose_path else None,
                )
                await complete_step(step_id, "completed", output=output)
                if is_choose_path:
                    # The chosen label IS this node's output — prune every
                    # OTHER labeled edge leaving it; _is_skipped above
                    # propagates that forward to whatever's only reachable
                    # through a pruned edge, on each later node's own turn.
                    for e in graph.get("edges", []):
                        if e["from"] == node["id"] and e.get("label") and e["label"] != output:
                            pruned_edges.add((e["from"], e["to"]))
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


def _execute_run_thread(run_id: str, graph: dict, initial_outputs: dict[str, str] | None = None) -> None:
    asyncio.run(_execute_run(run_id, graph, initial_outputs))


async def start_run(
    graph: dict, workflow_id: str | None = None, trigger_source: str = "manual",
    initial_outputs: dict[str, str] | None = None,
) -> str:
    """Creates the run row synchronously (so the caller — a FastAPI route —
    can return run_id immediately) then executes the graph in a background
    thread. Matches server.py's existing async-kickoff pattern for
    /research (threading.Thread + a pollable status), just with real
    per-run persistence instead of one global status string."""
    run_id = await create_run(workflow_id, trigger_source)
    threading.Thread(target=_execute_run_thread, args=(run_id, graph, initial_outputs), daemon=True).start()
    return run_id


async def start_workflow_run(
    workflow_id: str, trigger_source: str = "manual", initial_outputs: dict[str, str] | None = None,
) -> str:
    workflow = await get_workflow(workflow_id)
    if not workflow:
        raise WorkflowError(f"no workflow with id {workflow_id}")
    return await start_run(
        workflow["graph"], workflow_id=workflow_id, trigger_source=trigger_source, initial_outputs=initial_outputs,
    )


def find_webhook_trigger_node_id(graph: dict) -> str | None:
    """The graph-entry node a webhook payload gets seeded into (see
    _execute_run's pre-seeded-output branch) — first node whose backend
    `kind` is "webhookTrigger" (navi-pwa's agentWorkGraphConvert.ts sets
    this via BACKEND_KIND_FOR_NODE_KIND, same mechanism send_email
    already uses for its own kind discriminator). No dedicated
    _run_node handler exists for this kind, deliberately: this node is
    never executed in the normal sense, its "output" already IS the
    payload the call arrived with by the time the topological walk
    reaches it. A graph with none returns None — the webhook still
    legitimately means "run this now," the payload is just unused."""
    for node in graph.get("nodes", []):
        if node.get("kind") == "webhookTrigger":
            return node["id"]
    return None


async def start_webhook_run(workflow: dict, payload: dict | list | str) -> str:
    """Fires a workflow whose trigger is `{"type": "webhook", ...}` —
    called from server.py's public /agent/webhooks/{token} route, already
    past the token check by the time this runs."""
    graph = workflow["graph"]
    trigger_node_id = find_webhook_trigger_node_id(graph)
    initial_outputs = {}
    if trigger_node_id is not None:
        initial_outputs[trigger_node_id] = payload if isinstance(payload, str) else json.dumps(payload)
    return await start_run(
        graph, workflow_id=workflow["id"], trigger_source="webhook", initial_outputs=initial_outputs,
    )


def generate_webhook_token() -> str:
    return secrets.token_urlsafe(24)


async def set_webhook_trigger(workflow_id: str) -> str:
    """Idempotent by design: returns the EXISTING token if this workflow's
    trigger is already a webhook, rather than rotating it — a fresh token
    every time this is called would silently invalidate whatever URL the
    user already pasted into Stripe/GitHub/wherever. Raises WorkflowError
    for an unknown workflow_id, same convention as start_workflow_run."""
    workflow = await get_workflow(workflow_id)
    if not workflow:
        raise WorkflowError(f"no workflow with id {workflow_id}")
    trigger = workflow["trigger"]
    if trigger.get("type") == "webhook" and trigger.get("token"):
        return trigger["token"]
    token = generate_webhook_token()
    await update_workflow_trigger(workflow_id, {"type": "webhook", "token": token})
    return token


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
