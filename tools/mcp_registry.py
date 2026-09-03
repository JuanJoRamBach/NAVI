"""
tools/mcp_registry.py

Builds the LLM-facing tool schema list for connected MCP servers and
routes a proposed call back into dispatcher/mcp_client.py. Deliberately
separate from tools/registry.py (NAVI's own hand-rolled tools) per
JuanJo's own instruction — "I don't want to mix them." A third-party
server's tools never sit in the same schema list or dispatch function as
tools/registry.py's own code; they only merge at the very last step
(schemas_for_connected_servers below), and only for servers a given
workspace actually has connected — same per-connection gating decided
earlier in this design (don't even show the model a tool for a
disconnected service).

Naming: every tool is namespaced mcp__<server>__<tool>, so a collision
with an internal tool name — or another server's tool — is structurally
impossible, not just a convention someone has to remember to follow.
"""

from config.store import config
from dispatcher.mcp_client import MCPError, call_tool, is_write_tool

MCP_TOOL_PREFIX = "mcp__"


def _namespaced(server: str, tool: str) -> str:
    return f"{MCP_TOOL_PREFIX}{server}__{tool}"


def _split_namespaced(name: str) -> tuple[str, str] | None:
    if not name.startswith(MCP_TOOL_PREFIX):
        return None
    rest = name[len(MCP_TOOL_PREFIX):]
    if "__" not in rest:
        return None
    server, tool = rest.split("__", 1)
    return server, tool


def is_mcp_tool(name: str) -> bool:
    return name.startswith(MCP_TOOL_PREFIX)


def schemas_for_connected_servers() -> list[dict]:
    """One OpenAI-compatible function schema per APPROVED tool on every
    connected server. Reads the pinned baseline in config/store.py, not a
    live server call — a "new" or "changed" tool surfaced by
    dispatcher/mcp_client.py's discover_tools() is deliberately absent
    here until a human approves it; the LLM never sees an unapproved
    tool at all, not even to refuse calling it."""
    schemas = []
    for server_name, conn in config.list_mcp_connections().items():
        if not conn.get("connected"):
            continue
        for tool_name, baseline in conn.get("tools", {}).items():
            kind = "read-only" if baseline["read_only"] else "WRITE — requires confirmation"
            schemas.append({
                "type": "function",
                "function": {
                    "name": _namespaced(server_name, tool_name),
                    "description": f"[{server_name}, {kind}] {baseline['description']}",
                    "parameters": baseline["input_schema"] or {"type": "object", "properties": {}},
                },
            })
    return schemas


class MCPToolExecutionError(Exception):
    pass


def dispatch(name: str, arguments: dict, context: dict) -> str:
    """Mirrors tools/registry.py's dispatch() shape exactly, so a caller
    (tools/registry.py's own dispatch, once the two are merged at the
    executor level) can route mcp__* names here without changing its own
    signature. `context` carries the same executor-owned state as the
    internal dispatch — here it's also where an already-granted write
    confirmation is threaded through, via context["mcp_confirmed_writes"]
    (a set of namespaced tool names the user has explicitly approved for
    this run). No confirmation UI is wired up yet on the frontend side —
    until it is, that set stays empty, so every write tool safely refuses
    by default rather than silently executing."""
    split = _split_namespaced(name)
    if split is None:
        raise MCPToolExecutionError(f"Not an MCP tool: {name}")
    server_name, tool_name = split

    try:
        if is_write_tool(server_name, tool_name):
            confirmed = name in (context.get("mcp_confirmed_writes") or set())
            if not confirmed:
                return (
                    f"'{tool_name}' on '{server_name}' makes a real change and needs explicit "
                    f"user confirmation before it can run — it was not executed."
                )
        return call_tool(server_name, tool_name, arguments)
    except MCPError as e:
        # Same "a failed tool call isn't fatal, report it back to the
        # model" shape as tools/registry.py's own dispatch() — a
        # dropped connection or a rug-pull halt shouldn't crash the
        # whole turn, just this one call.
        return f"Tool error: {e}"
