"""
tools/workflows.py

Lets a model (via /agent tool-calling, once a mode brief opts in — none do
yet) define and kick off Agent Work workflows directly from chat, same
"thin wrapper over storage/dispatcher" shape as tools/notes.py.

dispatch()'s call site (dispatcher/executor.py's run_tool_loop) is a plain
sync function, but storage/agent_work.py and dispatcher/agent_work.py are
async (aiosqlite). A workflow node itself can call these tools too (a
step's own model deciding to kick off a sub-workflow) — that call happens
inside dispatcher/agent_work.py's own asyncio.run() on a background
thread, so a naive asyncio.run() here would raise "cannot be called from a
running event loop" in that case. Running each call on its own throwaway
thread (never the caller's thread) sidesteps that entirely, whether or not
the caller already has a loop running.
"""

import asyncio
from concurrent.futures import ThreadPoolExecutor

from storage.agent_work import create_workflow as _create_workflow
from storage.agent_work import get_run, get_run_steps, list_runs as _list_runs

_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="agent-work-tool")


def _run_async(coro):
    return _executor.submit(lambda: asyncio.run(coro)).result()


class WorkflowToolError(Exception):
    pass


def create_workflow(name: str, description: str | None, graph: dict, trigger: dict | None = None) -> str:
    """Returns the new workflow's id. `graph` must be {"nodes": [...],
    "edges": [...]} — see storage/agent_work.py's module docstring for the
    node/edge shape. Defaults to a manual trigger if none given."""
    return _run_async(_create_workflow(name, description, graph, trigger or {"type": "manual"}))


def run_workflow(workflow_id: str) -> str:
    """Kicks off a manual run of a saved workflow, returns the new run id
    immediately — execution continues in the background."""
    from dispatcher.agent_work import WorkflowError, start_workflow_run
    try:
        return _run_async(start_workflow_run(workflow_id, trigger_source="manual"))
    except WorkflowError as e:
        raise WorkflowToolError(str(e))


def get_run_status(run_id: str) -> dict:
    run = _run_async(get_run(run_id))
    if not run:
        raise WorkflowToolError(f"no run with id {run_id}")
    steps = _run_async(get_run_steps(run_id))
    return {**run, "steps": steps}


def list_workflow_runs(workflow_id: str | None = None, status: str | None = None) -> list[dict]:
    return _run_async(_list_runs(workflow_id=workflow_id, status=status))
