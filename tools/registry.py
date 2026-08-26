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

from tools.fetch import FetchError, fetch_page
from tools.notes import NoteError, save_note
from tools.search import SearchError, web_search

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

        raise ToolExecutionError(f"Unknown tool: {name}")

    except (SearchError, FetchError, NoteError) as e:
        # A failed tool call isn't fatal to the step — it's reported back
        # to the model as a tool result, same as a successful one, so the
        # model can decide how to proceed (retry, try another source, note
        # the gap in its answer) instead of the whole step blowing up.
        return f"Tool error: {e}"
    except KeyError as e:
        return f"Tool error: missing required argument {e}"
