"""
tools/devslate_tools.py

Dev Slate's tool belt — deliberately NOT wired through tools/registry.py's
dispatch(), because three of these four tools can't execute on this
server at all: the user's project files live on their own machine (see
HANDOFF context, 2026-09-01 — "files on the user's PC, not on Lightsail"),
so read_file/write_file/grep have to be relayed to the browser over the
open Dev Slate WebSocket and awaited, not run as a local Python call the
way every other tool in this codebase is. update_task_state is the one
exception — that's real server-side state (storage/conversations.py), so
it executes directly here, no relay needed.

TOOL_SCHEMAS follows the same OpenAI function-calling shape tools/registry.py
uses, so provider.chat(tools=...) needs no special-casing to accept these.
"""

import uuid

from storage.conversations import set_task_state

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a file from the user's local project folder. Returns its "
                            "text content. Use this before editing a file you haven't seen "
                            "the current contents of yet.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path relative to the project root, e.g. 'src/auth.ts'."},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "grep",
            "description": "Search the user's local project files for a text pattern. Returns "
                            "matching file paths and line numbers with a short surrounding "
                            "snippet. Use this to find where something is defined or used "
                            "before reading whole files blind.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "Plain text or regex to search for."},
                    "path_glob": {"type": "string", "description": "Optional glob to scope the search, e.g. 'src/**/*.ts'."},
                },
                "required": ["pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Propose a change to a file in the user's local project folder — "
                            "creating it if it doesn't exist yet, overwriting it if it does. "
                            "By default the user reviews a diff before anything actually lands "
                            "on disk; if they've turned on auto-accept, it applies immediately. "
                            "Either way, don't also describe the change in prose — the diff is "
                            "the description.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path relative to the project root."},
                    "content": {"type": "string", "description": "The full new content of the file."},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "update_task_state",
            "description": "Rewrite this Slate's running task-state summary — the goal, key "
                            "decisions made, and what's been built so far. This is what a "
                            "sub-Slate or a future session starts from, so keep it factual and "
                            "current, not a transcript. Call it when something worth "
                            "remembering happens, not after every message.",
            "parameters": {
                "type": "object",
                "properties": {
                    "goal": {"type": "string", "description": "One or two sentences: what this Slate is building."},
                    "decisions": {
                        "type": "array", "items": {"type": "string"},
                        "description": "Key decisions made so far, each one sentence.",
                    },
                    "built": {
                        "type": "array", "items": {"type": "string"},
                        "description": "What's actually been built/completed so far, each one sentence.",
                    },
                },
                "required": ["goal"],
            },
        },
    },
]

# Tools that execute on THIS server, directly — everything else in
# TOOL_SCHEMAS is relayed to the browser instead (see dispatch_devslate_tool
# in dispatcher/devslate_chat.py, which handles the relay + await itself
# since it needs the live WebSocket connection this module deliberately
# doesn't import).
LOCAL_TOOL_NAMES = {"update_task_state"}


def schemas_for(names: list[str]) -> list[dict]:
    return [s for s in TOOL_SCHEMAS if s["function"]["name"] in names]


class ToolExecutionError(Exception):
    pass


async def dispatch_local(name: str, arguments: dict, conversation_id: str) -> str:
    """Handles the one tool that never needs the browser — everything
    relay-bound is handled by the caller (dispatcher/devslate_chat.py),
    which owns the WebSocket connection this function has no access to."""
    if name != "update_task_state":
        raise ToolExecutionError(f"{name} is not a local tool — route it through the WebSocket relay instead.")
    state = {
        "goal": arguments.get("goal", ""),
        "decisions": arguments.get("decisions", []),
        "built": arguments.get("built", []),
    }
    await set_task_state(conversation_id, state)
    return "Task state updated."


def new_tool_call_id() -> str:
    return str(uuid.uuid4())


def format_task_state_for_prompt(state: dict | None) -> str | None:
    """Layer 3 as it appears in the model's context — a compact block,
    not the raw JSON. Returns None (caller skips the message entirely)
    when there's nothing to show yet, e.g. a brand-new Slate."""
    if not state or not state.get("goal"):
        return None
    lines = [f"Goal: {state['goal']}"]
    if state.get("decisions"):
        lines.append("Decisions so far:")
        lines += [f"- {d}" for d in state["decisions"]]
    if state.get("built"):
        lines.append("Built so far:")
        lines += [f"- {b}" for b in state["built"]]
    return "\n".join(lines)
