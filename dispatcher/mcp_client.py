"""
dispatcher/mcp_client.py

NAVI's MCP client — the only code in this codebase with a live connection
to a third-party MCP server (GitHub, Slack, Notion, etc). The LLM never
gets a handle to any of this; it only ever produces a structured
{tool, arguments} proposal (see tools/mcp_registry.py), which lands here
to actually be validated and executed.

Security model (2026-09-03 design):
  - **Rug-pull defense**: every tool's definition (name + description +
    input schema) is hashed at approval time and pinned in config/store.py.
    Before trusting a tool again, its live definition is re-fetched and
    compared — see MCP-38/policylayer.com research on this exact attack.
    A mismatch halts that tool rather than silently adopting the change;
    it has to go through approve_tools() again, same as a brand-new tool.
  - **Tool poisoning defense**: descriptions are sanitized before they're
    ever hashed or shown to the LLM — hidden-instruction markers stripped,
    unusually-formatted descriptions rejected outright (OWASP's MCP Tool
    Poisoning guidance).
  - **Read/write classification**: seeded from the tool's own
    readOnlyHint/destructiveHint annotations (a real MCP spec field,
    2025-03-26 revision) but stored as NAVI's own effective judgment — a
    hint is self-declared by the server, not verified, so it's a starting
    point for human review, not a trust boundary on its own.
  - **Output sanitization**: a tool's *result* is exactly as capable of
    hiding instructions as its description was — sanitize_content() is
    reused for both.

Bridging async MCP <-> NAVI's sync dispatcher (see dispatcher/devslate_chat.py's
own docstring for the fuller reasoning already established in this repo):
every provider/tool call in tools/registry.py's dispatch() is synchronous
by design, called from run_tool_loop, itself either called directly or via
asyncio.to_thread() from an async route — never from inside a running
event loop. So each public function here does asyncio.run() internally
rather than exposing an async API outward; that matches how the rest of
the dispatcher already bridges blocking work, instead of introducing a
second async/sync convention.
"""

import asyncio
import hashlib
import json
import re

from config.store import config

try:
    import httpx2
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
    from mcp.client.streamable_http import streamable_http_client
except ImportError:  # pragma: no cover - dependency not installed yet in some envs
    ClientSession = None


class MCPError(Exception):
    pass


class MCPConnectionError(MCPError):
    pass


class MCPToolChangedError(MCPError):
    """Raised when a tool's live definition no longer matches its pinned
    baseline — the rug-pull check. The caller (tools/mcp_registry.py)
    surfaces this as a tool-error string, same shape as any other failed
    call, rather than silently proceeding with the new definition."""
    pass


# Markers real MCP tool-poisoning writeups (OWASP, Microsoft's own MCP
# security post) call out as the actual vehicle for hidden instructions
# embedded in tool metadata or results. Stripped, not just flagged — a
# stripped marker can't smuggle anything even if the surrounding text
# still gets shown to the model.
_HIDDEN_INSTRUCTION_PATTERNS = [
    re.compile(r"<\s*(system|important|instructions?)\s*>.*?<\s*/\s*\1\s*>", re.IGNORECASE | re.DOTALL),
    re.compile(r"<!--.*?-->", re.DOTALL),  # HTML comments
]
_MAX_DESCRIPTION_LENGTH = 2000  # generous for a real tool description; unusual length is itself a signal

# A hard ceiling on every MCP round trip, connect through disconnect.
# Confirmed by hand (2026-09-03): a session whose cleanup path never gets
# to run cleanly — closing a stdio subprocess transport after an
# exception was raised mid-session, in this case — can hang indefinitely
# with no exception and no timeout of its own. Whether that's an SDK
# quirk or a genuinely unreliable/hostile server on the other end doesn't
# matter: NAVI's dispatcher must never block forever on code it doesn't
# control, which is the whole reason this client exists as a boundary in
# the first place. asyncio.wait_for wraps every public entry point below.
_MCP_CALL_TIMEOUT_SECONDS = 30


def sanitize_content(text: str | None) -> str | None:
    """Shared defense for both tool descriptions (at connect time) and
    tool results (at call time) — same threat class either way: untrusted
    third-party content that could carry hidden instructions. Returns
    None if the content is rejected outright (not just stripped)."""
    if text is None:
        return None
    cleaned = text
    for pattern in _HIDDEN_INSTRUCTION_PATTERNS:
        cleaned = pattern.sub("", cleaned)
    if len(cleaned) > _MAX_DESCRIPTION_LENGTH:
        return None
    return cleaned.strip()


def _hash_tool_def(name: str, description: str, input_schema: dict) -> str:
    """Deterministic hash of what actually defines a tool's behavior to the
    LLM — name, (sanitized) description, and its parameter schema. Order-
    independent on the schema via sort_keys, so equivalent JSON that
    happens to serialize differently doesn't false-positive as a change."""
    payload = json.dumps(
        {"name": name, "description": description, "input_schema": input_schema},
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()

def _connection_params(conn: dict):
    if conn["transport"] == "stdio":
        # PYTHONUNBUFFERED forced on, always — a stdio server whose
        # language runtime block-buffers stdout when it isn't attached
        # to a real terminal (Python's default) sits on its own
        # already-computed JSON-RPC replies until the buffer fills,
        # while the client waits on a response that's technically ready
        # but never flushed. Confirmed by hand: without this, a real
        # Python test server hung session.initialize() indefinitely —
        # not a timeout, a genuine deadlock. Harmless to set for
        # non-Python servers (an unused env var), so it's unconditional
        # rather than conditioned on the command being "python".
        env = {"PYTHONUNBUFFERED": "1", **(conn.get("env") or {})}
        return StdioServerParameters(command=conn["command"], args=conn.get("args") or [], env=env)
    if conn["transport"] == "http":
        return conn["url"]
    raise MCPConnectionError(f"Unknown transport: {conn['transport']}")


class _Session:
    """Thin async context manager wrapping the transport-specific
    connection so callers don't need to branch on stdio vs. http
    themselves — the one place that distinction actually matters."""

    def __init__(self, conn: dict):
        self._conn = conn
        self._transport_cm = None
        self._session_cm = None

    async def __aenter__(self) -> "ClientSession":
        if ClientSession is None:
            raise MCPConnectionError("The 'mcp' package isn't installed — add it to requirements.txt.")
        if self._conn["transport"] == "stdio":
            params = _connection_params(self._conn)
            self._transport_cm = stdio_client(params)
            read, write = await self._transport_cm.__aenter__()
        else:
            # streamable_http_client itself has no headers param — auth
            # goes through an httpx client instance instead (checked
            # against the installed SDK's real signature, not assumed).
            http_client = (
                httpx2.AsyncClient(headers={"Authorization": self._conn["auth_header"]})
                if self._conn.get("auth_header") else None
            )
            self._transport_cm = streamable_http_client(self._conn["url"], http_client=http_client)
            read, write, _ = await self._transport_cm.__aenter__()
        self._session_cm = ClientSession(read, write)
        session = await self._session_cm.__aenter__()
        await session.initialize()
        return session

    async def __aexit__(self, *exc):
        if self._session_cm is not None:
            await self._session_cm.__aexit__(*exc)
        if self._transport_cm is not None:
            await self._transport_cm.__aexit__(*exc)


async def _async_discover_tools(server_name: str) -> list[dict]:
    """Connects, lists tools, sanitizes descriptions, and diffs each one
    against its pinned baseline. Returns one descriptor per tool:
    {name, description, input_schema, read_only, status}, where status is
    "approved" (unchanged, safe to expose to the LLM), "new" (never seen
    before), or "changed" (rug-pull candidate) — "new" and "changed" both
    need approve_tools() before tools/mcp_registry.py will build a schema
    for them."""
    conn = config.get_mcp_connection(server_name)
    if conn is None:
        raise MCPConnectionError(f"No connection configured for '{server_name}'.")

    results = []
    async with _Session(conn) as session:
        listing = await session.list_tools()
        for tool in listing.tools:
            description = sanitize_content(tool.description or "")
            if description is None:
                # Rejected outright — not exposed to the LLM at all, not
                # even as "new". A real tool description doesn't need to
                # be this long or carry this formatting; treat it as
                # actively suspicious rather than silently truncating it.
                continue

            input_schema = tool.input_schema or {}
            tool_hash = _hash_tool_def(tool.name, description, input_schema)
            baseline = config.get_mcp_tool_baseline(server_name, tool.name)

            read_only_hint = getattr(tool.annotations, "read_only_hint", None) if tool.annotations else None

            if baseline is None:
                status = "new"
                read_only = bool(read_only_hint) if read_only_hint is not None else False
            elif baseline["hash"] != tool_hash:
                status = "changed"
                read_only = baseline["read_only"]  # keep prior classification until re-approved
            else:
                status = "approved"
                read_only = baseline["read_only"]

            results.append({
                "name": tool.name, "description": description, "input_schema": input_schema,
                "hash": tool_hash, "read_only": read_only, "status": status,
            })
    return results


def discover_tools(server_name: str) -> list[dict]:
    """Sync entry point — see module docstring on the async/sync bridge."""
    try:
        return asyncio.run(asyncio.wait_for(_async_discover_tools(server_name), timeout=_MCP_CALL_TIMEOUT_SECONDS))
    except asyncio.TimeoutError:
        raise MCPConnectionError(f"'{server_name}' didn't respond within {_MCP_CALL_TIMEOUT_SECONDS}s.")


def approve_tools(server_name: str, tools: list[dict]):
    """Pins the given tools' current definitions as the new trusted
    baseline — called only after a human has reviewed a "new" or
    "changed" tool (surfaced by discover_tools) and explicitly approved
    it. `tools` is the subset of discover_tools()'s output being approved,
    read_only overridable per-tool at approval time if the human disagrees
    with the server's own hint."""
    for t in tools:
        config.set_mcp_tool_baseline(
            server_name, t["name"], t["hash"], t["description"], t["input_schema"], t["read_only"],
        )


async def _async_call_tool(server_name: str, tool_name: str, arguments: dict) -> str:
    conn = config.get_mcp_connection(server_name)
    if conn is None:
        raise MCPConnectionError(f"No connection configured for '{server_name}'.")

    baseline = config.get_mcp_tool_baseline(server_name, tool_name)
    if baseline is None:
        raise MCPError(f"'{tool_name}' on '{server_name}' has never been approved — nothing to call.")

    async with _Session(conn) as session:
        # Re-verify against the pinned baseline on every single call, not
        # just at connect time — this is the actual rug-pull check, not
        # just discovery-time bookkeeping. A server that behaved at
        # connect time and changed since gets caught here before its new
        # behavior ever runs.
        listing = await session.list_tools()
        live_tool = next((t for t in listing.tools if t.name == tool_name), None)
        if live_tool is None:
            raise MCPToolChangedError(f"'{tool_name}' no longer exists on '{server_name}'.")
        description = sanitize_content(live_tool.description or "") or ""
        live_hash = _hash_tool_def(live_tool.name, description, live_tool.input_schema or {})
        if live_hash != baseline["hash"]:
            raise MCPToolChangedError(
                f"'{tool_name}' on '{server_name}' changed since it was approved — re-approval required before it can run again."
            )

        result = await session.call_tool(tool_name, arguments=arguments)

    text_parts = [c.text for c in result.content if hasattr(c, "text")]
    raw = "\n".join(text_parts) if text_parts else str(result.structured_content or "")
    sanitized = sanitize_content(raw)
    if sanitized is None:
        raise MCPError(f"'{tool_name}' on '{server_name}' returned content that failed sanitization — not shown to the model.")
    if result.is_error:
        raise MCPError(sanitized)
    return sanitized


def call_tool(server_name: str, tool_name: str, arguments: dict) -> str:
    """Sync entry point. Does NOT decide whether this call needs human
    confirmation first — that's tools/mcp_registry.py's job, using
    is_write_tool() below, before this is ever reached. This function's
    only gate is the rug-pull check, which is mandatory and unconditional
    regardless of read/write classification."""
    try:
        return asyncio.run(asyncio.wait_for(
            _async_call_tool(server_name, tool_name, arguments), timeout=_MCP_CALL_TIMEOUT_SECONDS,
        ))
    except asyncio.TimeoutError:
        raise MCPConnectionError(f"'{tool_name}' on '{server_name}' didn't respond within {_MCP_CALL_TIMEOUT_SECONDS}s.")


def is_write_tool(server_name: str, tool_name: str) -> bool:
    baseline = config.get_mcp_tool_baseline(server_name, tool_name)
    if baseline is None:
        return True  # unknown tool defaults to the more restrictive assumption
    return not baseline["read_only"]
