"""
tools/registry.py

The callable tool belt exposed to provider chat calls (OpenAI-compatible
"tools" schema — both Groq and OpenRouter speak this format). Wired in for
/research at minimum, per the brief; other commands can opt in later by
passing the same TOOL_SCHEMAS list.

dispatch() is the single entry point the executor calls when a model
response comes back with tool_calls — it maps a tool name + arguments to
the actual Python call and returns a plain string to feed back to the
model as the tool result.
"""

import json

from tools.fetch import FetchError, fetch_page
from tools.notes import NoteError, save_note
from tools.search import SearchError, web_search
from tools.telegram_send import TelegramSendError, send_to_telegram
from tools.workflows import (
    WorkflowToolError,
    create_workflow,
    get_run_status,
    list_workflow_runs,
    run_workflow,
)

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web via DuckDuckGo and get back a list of "
                            "titles, URLs, and snippets. Use this to find sources "
                            "before answering a research question.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The search query."},
                    "max_results": {
                        "type": "integer",
                        "description": "How many results to return (default 5).",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fetch_page",
            "description": "Fetch a URL and return its extracted plain-text content. "
                            "Use this to read a source found via web_search before "
                            "citing or summarizing it.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "The URL to fetch."},
                },
                "required": ["url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "send_to_telegram",
            "description": "Sends a text message to the user's Telegram, regardless of "
                            "which chat channel this conversation is happening in. Use when "
                            "the user asks to send/save something to Telegram, or wants a "
                            "document/finding delivered there.",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "The message or document content to send."},
                },
                "required": ["text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "save_note",
            "description": "Save an intermediate note or source excerpt to persistent "
                            "storage, separate from the final result. Use sparingly — "
                            "only for something worth keeping beyond the final answer.",
            "parameters": {
                "type": "object",
                "properties": {
                    "filename": {"type": "string", "description": "e.g. 'source-1.md'"},
                    "content": {"type": "string", "description": "The note content."},
                },
                "required": ["filename", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_workflow",
            "description": "Define a new Agent Work workflow — a graph of steps, each its "
                            "own prompt to a model, that can be run manually or on a schedule. "
                            "graph is {\"nodes\": [{\"id\", \"prompt\", \"tools\"?}], \"edges\": "
                            "[{\"from\", \"to\"}]} — a single-node graph with no edges is a "
                            "one-step workflow.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Short workflow name."},
                    "description": {"type": "string", "description": "What this workflow does."},
                    "graph": {
                        "type": "object",
                        "description": "{\"nodes\": [...], \"edges\": [...]} — see tool description.",
                    },
                    "trigger": {
                        "type": "object",
                        "description": "{\"type\": \"manual\"} (default) or {\"type\": \"scheduled\", "
                                        "\"interval_seconds\", \"next_run_at\"} (epoch seconds).",
                    },
                },
                "required": ["name", "graph"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_workflow",
            "description": "Manually starts a run of a saved Agent Work workflow. Returns "
                            "immediately with the new run's id — execution continues in the "
                            "background; use get_run_status to check on it later.",
            "parameters": {
                "type": "object",
                "properties": {
                    "workflow_id": {"type": "string", "description": "The workflow's id, from create_workflow."},
                },
                "required": ["workflow_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_run_status",
            "description": "Checks the status and step-by-step log of an Agent Work run.",
            "parameters": {
                "type": "object",
                "properties": {
                    "run_id": {"type": "string", "description": "The run's id, from run_workflow."},
                },
                "required": ["run_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_workflow_runs",
            "description": "Lists recent Agent Work runs, optionally filtered by workflow or status.",
            "parameters": {
                "type": "object",
                "properties": {
                    "workflow_id": {"type": "string", "description": "Only runs of this workflow."},
                    "status": {"type": "string", "description": "queued | running | completed | failed"},
                },
            },
        },
    },
]


def schemas_for(names: list[str]) -> list[dict]:
    """Filters TOOL_SCHEMAS down to the subset a mode's brief allows —
    used so a mode's frontmatter `tools: [...]` list controls what the
    model can actually call, not just what it's told in prose."""
    return [s for s in TOOL_SCHEMAS if s["function"]["name"] in names]


class ToolExecutionError(Exception):
    pass


def dispatch(name: str, arguments: dict, context: dict) -> str:
    """
    Runs a tool call and returns its result as a plain string (what gets
    fed back to the model as the tool message content).

    `context` carries executor-owned state a tool needs but the model
    shouldn't have to supply itself — currently just the command and
    topic_slug for save_note, so notes land in the same Filen folder as
    the step's final output.
    """
    try:
        if name == "web_search":
            query = arguments["query"]
            max_results = int(arguments.get("max_results", 5))
            results = web_search(query, max_results=max_results)
            if not results:
                return "No results found."
            lines = [f"- {r['title']} ({r['url']}): {r['snippet']}" for r in results]
            return "\n".join(lines)

        if name == "fetch_page":
            return fetch_page(arguments["url"])

        if name == "save_note":
            path = save_note(
                command=context.get("command", "research"),
                topic_slug=context.get("topic_slug", "untitled"),
                filename=arguments["filename"],
                content=arguments["content"],
            )
            return f"Saved to {path}"

        if name == "send_to_telegram":
            return send_to_telegram(arguments["text"])

        if name == "create_workflow":
            workflow_id = create_workflow(
                arguments["name"], arguments.get("description"), arguments["graph"], arguments.get("trigger"),
            )
            return f"Created workflow {workflow_id}."

        if name == "run_workflow":
            run_id = run_workflow(arguments["workflow_id"])
            return f"Started run {run_id}."

        if name == "get_run_status":
            return json.dumps(get_run_status(arguments["run_id"]))

        if name == "list_workflow_runs":
            return json.dumps(list_workflow_runs(arguments.get("workflow_id"), arguments.get("status")))

        raise ToolExecutionError(f"Unknown tool: {name}")

    except (SearchError, FetchError, NoteError, TelegramSendError, WorkflowToolError) as e:
        # A failed tool call isn't fatal to the step — it's reported back
        # to the model as a tool result, same as a successful one, so the
        # model can decide how to proceed (retry, try another source, note
        # the gap in its answer) instead of the whole step blowing up.
        return f"Tool error: {e}"
    except KeyError as e:
        return f"Tool error: missing required argument {e}"
