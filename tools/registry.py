"""
tools/registry.py

The callable tool belt exposed to provider chat calls (OpenAI-compatible
"tools" schema — both Groq and OpenRouter speak this format). Any
command or chat mode can opt in by passing the same TOOL_SCHEMAS list
(or a scoped subset via schemas_for).

dispatch() is the single entry point the executor calls when a model
response comes back with tool_calls — it maps a tool name + arguments to
the actual Python call and returns a plain string to feed back to the
model as the tool result.
"""

import json
from urllib.parse import urlparse as _urlparse

from tools.content_safety import screened
from tools.document_save import DocumentError, create_document
from tools.fetch import FetchError, fetch_page
from tools.notes import NoteError, save_note
from tools.search import SearchError, web_search
from tools.sources import SourceSaveError, save_source_document
from tools.telegram_send import TelegramSendError, send_to_telegram
from tools.workflows import (
    WorkflowToolError,
    create_workflow,
    get_run_status,
    list_workflow_runs,
    run_workflow,
)
from tools.mcp_registry import is_mcp_tool, schemas_for_connected_servers
from tools.mcp_registry import dispatch as dispatch_mcp_tool

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
            "name": "create_document",
            "description": "Save a finished document as a real downloadable file and get back "
                            "a link to it — use when the user asks for something written up as a "
                            "file/document to keep or share (a report, an itinerary, a list), not "
                            "for a normal chat reply. Give it the complete, real content — never a "
                            "placeholder or a description of what the document would contain.",
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {"type": "string", "description": "Short title — used to group this with related saves."},
                    "filename": {"type": "string", "description": "e.g. 'trip-itinerary.md' or 'summary.html'. Extension controls how it's served."},
                    "content": {"type": "string", "description": "The complete document content."},
                },
                "required": ["title", "filename", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "save_source",
            "description": "Save a web page as a real Source document for the user to review "
                            "later — use this ONLY after you've fetched the page (fetch_page) "
                            "and judged it genuinely relevant to the search term you were given. "
                            "Do NOT call this for every search result — only the ones actually "
                            "worth keeping. Never invent a url or content you haven't fetched.",
            "parameters": {
                "type": "object",
                "properties": {
                    "term": {"type": "string", "description": "The search term this result answers."},
                    "title": {"type": "string", "description": "The page's real title."},
                    "url": {"type": "string", "description": "The exact URL you fetched."},
                    "content": {"type": "string", "description": "The relevant extracted content — trim to what actually matters, not the whole raw page."},
                },
                "required": ["term", "title", "url", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_workflow",
            "description": "Define a new Agent Work workflow — an ORDERED list of steps, each "
                            "its own model call, run manually or on a schedule. You decide how "
                            "many steps the task needs and what each one's prompt says; the "
                            "dispatcher wires them into a chain itself — don't invent node ids "
                            "or edges. A single-item list is a one-step workflow. Use `kind` to "
                            "pick a step's real behavior — a step of a DETERMINISTIC kind (e.g. "
                            "send_email) never calls you again at run time to compose anything; "
                            "its fields must already be the complete, real content.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Short workflow name."},
                    "description": {"type": "string", "description": "What this workflow does."},
                    "steps": {
                        "type": "array",
                        "description": "Ordered — step 1 runs first, then step 2, etc. Each "
                                        "step's prompt must stand alone (no 'as discussed above' "
                                        "or references to this conversation) since the step's "
                                        "model call never sees this chat. To use an EARLIER step's "
                                        "real output as a later step's field value (not just the "
                                        "immediately-preceding one), set that field to exactly "
                                        "\"{{state.n1}}\" (n1/n2/... matching that earlier step's "
                                        "position, 1-indexed) instead of writing prose.",
                        "items": {
                            "type": "object",
                            "properties": {
                                "prompt": {"type": "string", "description": "Complete, self-contained instruction for this step. For a deterministic kind (send_email, save_note, send_to_telegram) with no prior step feeding it, this is the literal content itself, not an instruction."},
                                "kind": {
                                    "type": "string",
                                    "enum": ["text", "send_email", "output", "choose_path", "web_search", "fetch_page", "save_note", "send_to_telegram"],
                                    "description": "The step's real behavior. 'text' (default if omitted) generates text with a model call. 'send_email' SENDS A REAL EMAIL — no LLM call, requires 'to' and 'body' below. 'output' returns a value (optionally 'output_type': 'pdf'). The rest match the tool of the same name.",
                                },
                                "to": {"type": "string", "description": "send_email only, REQUIRED for it: recipient address, or comma-separated for multiple. Can be \"{{state.n1}}\" to reuse an earlier step's output."},
                                "body": {"type": "string", "description": "send_email only, REQUIRED for it: the complete real HTML (or plain text) email body — links and buttons are real HTML anchor/button markup, not described in prose. Can be \"{{state.n1}}\" to reuse an earlier step's output."},
                                "subject": {"type": "string", "description": "send_email only, optional: the email subject line."},
                                "output_type": {"type": "string", "enum": ["pdf", "chat", "markdown"], "description": "output kind only, optional: 'pdf' renders the value into a real file; omit for plain text/markdown pass-through."},
                                "tools": {"type": "array", "items": {"type": "string"}, "description": "Legacy — prefer 'kind' above. Tool names this step may call, if any."},
                            },
                            "required": ["prompt"],
                        },
                    },
                    "trigger_description": {
                        "type": "string",
                        "description": "Plain language, e.g. 'once, in 20 minutes', 'every day "
                                        "at 9am UTC', 'every hour, 5 times'. Omit entirely for a "
                                        "manual-only workflow. Resolved into a concrete schedule "
                                        "separately using the real current time — describe it in "
                                        "the user's own terms, don't compute times yourself.",
                    },
                },
                "required": ["name", "steps"],
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
    {
        "type": "function",
        "function": {
            "name": "ask_user_choice",
            "description": "Presents the user with a small set of concrete options to "
                            "pick from with one click, instead of asking a question in "
                            "plain text and waiting for them to type a full reply. Use "
                            "this for a genuine multi-way decision — narrowing scope, "
                            "confirming before an action, choosing between real "
                            "alternatives — not for every question. The user can still "
                            "type something else instead of picking an option.",
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "The complete message to show the user — this IS "
                                        "what's displayed, write it whole and self-"
                                        "contained (not just a short prompt).",
                    },
                    "options": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "2 to 5 short, concrete option labels, each "
                                        "readable at a glance.",
                    },
                },
                "required": ["question", "options"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "request_stronger_model",
            "description": "Call this INSTEAD OF answering when this request is genuinely "
                            "beyond you — it needs deeper reasoning, more careful analysis, "
                            "or more capability than you can give a good answer with. A "
                            "stronger model will then take over this same message and "
                            "answer it properly. Recognising your own limit is the whole "
                            "job here: it is far better to hand off than to produce a "
                            "shallow or wrong answer confidently. But don't reach for it "
                            "on anything you can genuinely handle — most messages are "
                            "ordinary and you should just answer them.",
            "parameters": {
                "type": "object",
                "properties": {
                    "reason": {
                        "type": "string",
                        "description": "One short sentence on what makes this beyond you — "
                                        "used for routing diagnostics, never shown to the "
                                        "user.",
                    },
                },
                "required": ["reason"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "flag_key_insight",
            "description": "Records ONE durable fact from this exchange into the "
                            "conversation's long-term memory, so it survives after the "
                            "recent-message window scrolls past it. Call this ALONGSIDE "
                            "your normal reply (not instead of it) when the user has "
                            "established something worth remembering later: a fact about "
                            "them or their work, a stated preference, a constraint, or a "
                            "decision they've made and why. Do NOT call it for ordinary "
                            "conversational turns, for anything you inferred rather than "
                            "were told, or for something already obvious from earlier in "
                            "the conversation — a memory full of noise is worse than a "
                            "short one. Most turns should not call this at all.",
            "parameters": {
                "type": "object",
                "properties": {
                    "insight": {
                        "type": "string",
                        "description": "The fact, in one self-contained sentence that will "
                                        "still make sense months from now with no "
                                        "surrounding conversation — write \"JuanJo prefers "
                                        "terse replies without preamble\", not \"he prefers "
                                        "that\".",
                    },
                },
                "required": ["insight"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "propose_research_mode",
            "description": "Call this INSTEAD OF replying when the user's message has "
                            "genuinely shifted from a quick question into something that "
                            "deserves real research — a request for an in-depth "
                            "investigation, a competitive/literature review, or a "
                            "data-driven question where getting the scope right matters "
                            "more than getting an answer fast. Not for every substantive "
                            "question — most things, even meaty ones, are still a normal "
                            "chat answer. Only call this when a proper plan (breaking the "
                            "question into sub-questions, gathering real sources) would "
                            "genuinely serve them better than what you'd say right now.",
            "parameters": {
                "type": "object",
                "properties": {
                    "reason": {
                        "type": "string",
                        "description": "One short sentence, shown to the user, on why this "
                                        "looks like it needs real research rather than a "
                                        "quick answer.",
                    },
                },
                "required": ["reason"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "propose_branch_complete",
            "description": "Call this INSTEAD OF replying when the work this chat "
                            "was opened to do is finished — when the 'Done when' "
                            "criteria in your context have actually been met. Do not "
                            "call it because a lot has been done, or because a "
                            "milestone feels reached: check each criterion and only "
                            "call this if every one of them is genuinely satisfied. "
                            "If any is not, keep working and say what is left. The "
                            "user reviews and accepts — you are proposing that the "
                            "work is done, not declaring it.",
            "parameters": {
                "type": "object",
                "properties": {
                    "summary": {
                        "type": "string",
                        "description": "One or two sentences on what was achieved. "
                                        "State plainly anything that was NOT achieved.",
                    },
                },
                "required": ["summary"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "propose_plan_ready",
            "description": "Call this once clarification is genuinely done and there's "
                            "enough to draft a research plan — instead of drafting the "
                            "plan text yourself. The dispatcher takes it from here: it "
                            "confirms with the user, then drafts the actual plan from the "
                            "full conversation. Don't describe or summarize the plan in "
                            "your reply when calling this — just call it.",
            "parameters": {
                "type": "object",
                "properties": {},
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_knowledge",
            "description": "Searches what this company has written down about itself and, "
                            "when this chat is in a project, about that project: policies, "
                            "client facts, decisions, how things are done here. Use it "
                            "BEFORE assuming anything company- or client-specific that the "
                            "briefs above don't already answer (who a client is, a price "
                            "rule, a house style, a past decision). Don't use it for general "
                            "knowledge or for things the user just told you.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "A few keywords, not a sentence: \"refund policy\", "
                                        "\"Acme invoicing\".",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "suggest_knowledge",
            "description": "Suggests adding ONE lasting fact to the company's knowledge, or "
                            "the current project's, for an Owner, Admin or project editor to "
                            "approve. Nothing is added until a person approves it. Use it "
                            "when the user has stated something that will matter to other "
                            "people's work too (a client preference, a rule, a decision and "
                            "its reason), not for things that only matter in this chat — "
                            "those belong in flag_key_insight. Tell the user in your reply "
                            "that you've suggested it.",
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {"type": "string", "description": "A short label, under 120 characters."},
                    "text": {
                        "type": "string",
                        "description": "The fact, self-contained, readable months from now by "
                                        "someone who never saw this chat.",
                    },
                    "scope": {
                        "type": "string",
                        "enum": ["project", "company"],
                        "description": "\"project\" when it's about this chat's project or "
                                        "client (the usual case), \"company\" when it holds for "
                                        "the whole company.",
                    },
                },
                "required": ["title", "text", "scope"],
            },
        },
    },
]

# ask_user_choice, propose_research_mode, propose_plan_ready,
# propose_branch_complete, flag_key_insight and request_stronger_model are
# deliberately NOT handled in dispatch() below — all six are intercepted
# earlier, in
# run_stored_mode_chat/run_devslate_turn/dispatcher/research.py, before a
# call ever reaches the normal execute-and-continue tool loop.
#
# request_stronger_model (2026-09-13) is how NAVI does capability tiering
# without paying for a separate classifier call on every turn: each turn
# starts on the cheap idle model, and escalation only happens when the
# model itself says it can't handle the message. Recognising "this is
# beyond me" is a much easier task than answering, which is why a small
# model can be trusted with it. Cost is one cheap call on the common
# path, one cheap + one real call when it escalates.
#
# flag_key_insight (2026-09-13) differs from the other three in one
# important way: it is NON-TERMINAL. The other three ARE the turn's
# outcome — calling one is how the model hands a question, a readiness
# decision, or a scope-shift back to the dispatcher, and nothing else
# happens that turn. flag_key_insight rides ALONGSIDE a normal reply: the
# dispatcher records it into context.md's store and then carries on using
# response.text as the actual answer, so a memory write never costs the
# user their reply. Calling ask_user_choice is how the
# model hands a question back to the user; calling propose_research_mode
# (2026-09-12, Stage 3's "fast-path intent layer," IDEAS.md) is how it
# flags a genuine scope shift toward real research, leaving the actual
# mode-switch decision to the dispatcher/user rather than narrating it
# itself; calling propose_plan_ready is how it hands the readiness
# DECISION back to the dispatcher (2026-09-12, how_to_handle_context.md's
# "dispatcher-mediated, not model-narrated" design) — none of the three
# has a server-side action to run.


def schemas_for(names: list[str]) -> list[dict]:
    """Filters TOOL_SCHEMAS down to the subset a mode's brief allows —
    used so a mode's frontmatter `tools: [...]` list controls what the
    model can actually call, not just what it's told in prose.

    A mode opts into MCP tools with the sentinel "mcp" in its tools list,
    not by naming individual mcp__<server>__<tool> entries — those are
    dynamic per company/connection, so a mode's static frontmatter can't
    enumerate them ahead of time. When present, every APPROVED tool on
    every CONNECTED server is attached (see tools/mcp_registry.py's own
    per-connection gating — a disconnected service's tools are never
    attached regardless of this sentinel)."""
    schemas = [s for s in TOOL_SCHEMAS if s["function"]["name"] in names]
    if "mcp" in names:
        schemas += schemas_for_connected_servers()
    return schemas


def _build_creation_transcript(chat_messages: list | None) -> str | None:
    """Renders the Agent Work Chat exchange that led to this create_workflow
    call into plain "User: ..." / "NAVI: ..." lines — this becomes Agent
    Vault's "Instructions" if the workflow is later starred (2026-09-03,
    JuanJo: instructions should be "when the Agent is created using that
    tab... fed to the Agent Work's chat LLM to create the workflow").
    Only user/assistant text turns are kept — tool-result payloads
    (search snippets, run-status JSON) are noise for a human reading this
    back as "what did I ask for," not part of the actual conversation.
    `chat_messages` is executor.run_tool_loop's own transcript, threaded
    in via context; absent (e.g. a manually-built workflow never went
    through this loop) or empty, this returns None."""
    if not chat_messages:
        return None
    lines = []
    for m in chat_messages:
        role = getattr(m, "role", None)
        content = getattr(m, "content", None)
        if role not in ("user", "assistant") or not content:
            continue
        if isinstance(content, list):
            content = " ".join(part.get("text", "") for part in content if isinstance(part, dict) and part.get("type") == "text")
        content = content.strip()
        if not content:
            continue
        lines.append(f"{'User' if role == 'user' else 'NAVI'}: {content}")
    return "\n\n".join(lines) if lines else None


class ToolExecutionError(Exception):
    pass


def _dispatch_knowledge(name: str, arguments: dict, context: dict) -> str:
    """Company and project knowledge (storage/knowledge.py, 2026-09-24).

    Scope comes from `context`, set by the dispatcher from the conversation
    itself, never from the model: a chat can only search the company's
    knowledge and its OWN project's, whatever a model asks for. Which
    project a conversation is in was already checked against the person's
    access when it was set (server.py's _apply_project)."""
    from storage import knowledge

    project_id = context.get("project_id")
    if name == "search_knowledge":
        scopes = [knowledge.COMPANY] + ([knowledge.project_scope(project_id)] if project_id else [])
        hits = knowledge.search(scopes, arguments.get("query") or "")
        if not hits:
            return "Nothing in the company's knowledge matches that. Don't guess company-specific facts; ask the user."
        lines = []
        for h in hits:
            where = "company" if h["scope"] == knowledge.COMPANY else "this project"
            lines.append(f"[{where}] {h['title']}: {h['body']}")
        return "\n\n".join(lines)

    scope_arg = (arguments.get("scope") or "project").lower()
    if scope_arg == "project" and not project_id:
        scope_arg = "company"  # a chat outside any project can only suggest for the company
    scope = knowledge.COMPANY if scope_arg == "company" else knowledge.project_scope(project_id)
    try:
        knowledge.suggest_entry_from_chat(
            scope, arguments.get("title") or "", arguments.get("text") or "", context.get("conversation_id"),
        )
    except knowledge.KnowledgeError as e:
        return f"Not suggested: {e}"
    who = "an Owner or Admin" if scope == knowledge.COMPANY else "one of the project's editors"
    return f"Suggested. It waits for {who} to approve it before NAVI uses it. Tell the user you've suggested it."


def dispatch(name: str, arguments: dict, context: dict) -> str:
    """
    Runs a tool call and returns its result as a plain string (what gets
    fed back to the model as the tool message content).

    `context` carries executor-owned state a tool needs but the model
    shouldn't have to supply itself — currently just the command and
    topic_slug for save_note, so notes land in the same Filen folder as
    the step's final output.
    """
    # Real per-tool dispatch frequency (2026-09-11) — isolated in its own
    # try/except, same pattern providers/base.py's Provider.chat() already
    # uses for record_usage, so a tracking failure can never break the
    # actual tool call below it.
    try:
        from storage.usage import record_tool_call
        record_tool_call(name)
    except Exception:
        pass

    try:
        if is_mcp_tool(name):
            # Routed, never executed inline here — dispatcher/mcp_client.py
            # is the only code with a live connection to the real server,
            # and tools/mcp_registry.py's own dispatch() is what applies
            # the write-confirmation gate before it ever gets there. This
            # function never touches an MCP server directly, on purpose —
            # JuanJo: "I don't want to mix them."
            return dispatch_mcp_tool(name, arguments, context)

        if name in ("search_knowledge", "suggest_knowledge"):
            return _dispatch_knowledge(name, arguments, context)

        if name == "web_search":
            query = arguments["query"]
            max_results = int(arguments.get("max_results", 5))
            # Trusted-site enforcement lives HERE, not in a prompt asking
            # the model to "only search trusted-sites.com" — the model
            # never even sees a result outside the registry it was
            # given. Only the source-fetch batch loop (dispatcher/
            # source_fetch.py) threads "trusted_sites" into context;
            # every other caller (plain chat, /research) leaves it unset
            # and web_search behaves exactly as it always has.
            trusted_sites = context.get("trusted_sites")
            if trusted_sites:
                # Rewritten 2026-09-06 after a real batch searched
                # generically then filtered the top max_results down to
                # trusted domains AFTER the fact — a trusted domain
                # ranked outside the top 5 of a generic query (verified:
                # anthropic.com was position 7 for "AI agent harness
                # design") was never even fetched, so the filter had
                # nothing to keep. Fixed per-entry instead of post-hoc
                # (JuanJo: "it should use the url directly to do web
                # search there, that's the point of the trusted sites"):
                from storage.sources import domain_of

                def _as_url(site: str) -> str:
                    s = site.strip()
                    return s if "://" in s else f"https://{s}"

                results = []
                for site in trusted_sites:
                    if not site.strip():
                        continue
                    full_url = _as_url(site)
                    path = _urlparse(full_url).path
                    if path and path != "/":
                        # An exact page, not just a domain — the caller
                        # already knows the page it wants, so there's
                        # nothing to search for. Handed straight back as
                        # a pre-known candidate; the model still fetches
                        # and judges it like any other result, just
                        # without a wasted search step.
                        results.append({
                            "title": full_url, "url": full_url,
                            "snippet": "(trusted page — already known, fetch it directly to judge relevance)",
                        })
                    else:
                        # A bare domain has no single page yet — scope
                        # the search itself to this domain (DuckDuckGo's
                        # own site: operator) instead of searching the
                        # whole web and hoping this domain happens to
                        # rank in the top max_results generically.
                        domain = domain_of(full_url)
                        results.extend(web_search(f"site:{domain} {query}", max_results=max_results))
                if not results:
                    return "No results found on any trusted site for this term."
                lines = [f"- {r['title']} ({r['url']}): {r['snippet']}" for r in results]
                return screened("\n".join(lines))

            results = web_search(query, max_results=max_results)
            if not results:
                return "No results found."
            lines = [f"- {r['title']} ({r['url']}): {r['snippet']}" for r in results]
            return screened("\n".join(lines))

        if name == "save_source":
            # Enforced in code, not just asked for in the tool description
            # — same "dispatcher enforces, model doesn't get to just
            # comply or not" philosophy already used for trusted-site
            # scoping and duplicate-call blocking. Real failure this
            # catches (2026-09-06, JuanJo, live batch): a smaller model
            # called save_source with `content` that was just the page
            # title again ("Harnessing Design for Long-Running Apps"),
            # producing a real document row backed by a real Filen file
            # that was useless to actually review. A minimum length AND
            # "meaningfully longer than the title" check catches both an
            # exact echo and a lightly-reworded one — a real relevant
            # excerpt is always substantially longer than a title.
            title, content = arguments["title"], arguments["content"]
            min_len = max(200, len(title.strip()) + 50)
            if len(content.strip()) < min_len:
                # Still saved, not silently dropped — 2026-09-06, JuanJo:
                # "That kind of stuff must be informed to the user. why
                # some were rejected." Pre-marked 'rejected' with a real
                # reason, so the human sees the term DID get a candidate
                # and can inspect the actual (too-thin) extracted text
                # that caused the rejection, instead of the term just
                # having zero documents with no explanation anywhere.
                reason = "Auto-rejected: content was too short / just repeated the title, not real extracted page content."
                doc_id = save_source_document(
                    batch_id=context["batch_id"],
                    term=arguments["term"], title=title,
                    url=arguments["url"], content=content,
                    status="rejected", reason=reason,
                )
                return (
                    f"NOT usable (saved as rejected, doc {doc_id}) — the content you gave for '{title}' "
                    "was too short to be real extracted page content (it looked like just the title "
                    "repeated). Look at what fetch_page actually returned and extract several real "
                    "sentences or paragraphs relevant to the search term, or skip this page entirely "
                    "if there's nothing substantial worth keeping."
                )
            doc_id = save_source_document(
                batch_id=context["batch_id"],
                term=arguments["term"], title=title,
                url=arguments["url"], content=content,
            )
            return f"Saved source document {doc_id} for review."

        if name == "fetch_page":
            return screened(fetch_page(arguments["url"]))

        if name == "save_note":
            path = save_note(
                command=context.get("command", "research"),
                topic_slug=context.get("topic_slug", "untitled"),
                filename=arguments["filename"],
                content=arguments["content"],
            )
            return f"Saved to {path}"

        if name == "create_document":
            try:
                url = create_document(
                    command=context.get("command", "chat"),
                    title=arguments["title"],
                    filename=arguments["filename"],
                    content=arguments["content"],
                )
            except DocumentError as e:
                return f"Failed to save document: {e}"
            return f"Document saved — download link: {url}"

        if name == "send_to_telegram":
            return send_to_telegram(arguments["text"])

        if name == "create_workflow":
            workflow_id = create_workflow(
                arguments["name"], arguments.get("description"), arguments["steps"], arguments.get("trigger_description"),
                creation_transcript=_build_creation_transcript(context.get("chat_messages")),
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

    except (SearchError, FetchError, NoteError, SourceSaveError, TelegramSendError, WorkflowToolError) as e:
        # A failed tool call isn't fatal to the step — it's reported back
        # to the model as a tool result, same as a successful one, so the
        # model can decide how to proceed (retry, try another source, note
        # the gap in its answer) instead of the whole step blowing up.
        return f"Tool error: {e}"
    except KeyError as e:
        return f"Tool error: missing required argument {e}"
