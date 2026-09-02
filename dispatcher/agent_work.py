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

from dispatcher.executor import CITATION_STYLE_PROMPT, run_tool_loop
from providers.base import ChatMessage, ProviderError
from providers.registry import ProviderNotConfigured, get_dispatcher_role, get_provider
from storage.agent_work import (
    complete_step, create_step, create_run, due_workflows, get_workflow,
    set_step_input, update_run_status, update_workflow_trigger,
)
from tools.registry import schemas_for

AGENT_WORK_SYSTEM_PROMPT = (
    "You are executing one step of an automated NAVI workflow, running "
    "unattended (no user available to answer follow-up questions). Do the "
    "step's task directly and report the concrete result — don't ask "
    "clarifying questions."
)


class WorkflowError(Exception):
    pass


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


def _run_node(node: dict) -> str:
    """Synchronous — same primary+fallback attempt loop as
    dispatcher/chat.py's run_mode_chat, reusing the 'agent_work' role
    rather than picking a model itself. Raises WorkflowError with a
    human-readable message on total failure (every provider exhausted)."""
    role_context = node.get("role") or "agent_work"
    try:
        role = get_dispatcher_role(context=role_context)
    except ProviderNotConfigured as e:
        raise WorkflowError(f"role '{role_context}' isn't configured: {e}")

    node_tools = schemas_for(node["tools"]) if node.get("tools") else None
    messages = [ChatMessage(role="system", content=AGENT_WORK_SYSTEM_PROMPT)]
    if node_tools:
        messages.append(ChatMessage(role="system", content=CITATION_STYLE_PROMPT))
    messages.append(ChatMessage(role="user", content=node.get("prompt", "")))

    attempts = [{"provider": role["provider"], "model": role["model"]}] + role.get("fallback", [])
    last_error = None
    for attempt in attempts:
        try:
            provider = get_provider(attempt["provider"])
        except Exception as e:
            last_error = str(e)
            continue
        try:
            response = provider.chat(model=attempt["model"], messages=messages, tools=node_tools)
            if node_tools and response.tool_calls:
                response, _messages, _iterations = run_tool_loop(
                    provider, attempt["model"], messages, response,
                    context={"command": "agent_work", "topic_slug": node.get("id", "step")},
                    tools=node_tools,
                )
            return response.text or "(empty reply)"
        except ProviderError as e:
            last_error = str(e)
            continue

    raise WorkflowError(f"every configured provider failed: {last_error}")


async def _execute_run(run_id: str, graph: dict) -> None:
    try:
        order = _topological_order(graph)
    except WorkflowError as e:
        await update_run_status(run_id, "failed", error=str(e))
        return

    await update_run_status(run_id, "running")
    for seq, node in enumerate(order):
        step_id = await create_step(run_id, node["id"], seq)
        await set_step_input(step_id, {"prompt": node.get("prompt"), "role": node.get("role"), "tools": node.get("tools")})
        try:
            output = await asyncio.to_thread(_run_node, node)
            await complete_step(step_id, "completed", output=output)
        except WorkflowError as e:
            await complete_step(step_id, "failed", error=str(e))
            await update_run_status(run_id, "failed", error=f"node '{node['id']}' failed: {e}")
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
