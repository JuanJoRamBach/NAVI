"""
dispatcher/compaction.py

compact_conversation — the shared schema-constrained "read a conversation,
produce a structured result" primitive designed in how_to_handle_context.md
(2026-08-30/2026-09-01 design). First real caller: Research mode's
plan-drafting synthesis step (dispatcher/research.py, 2026-09-12) — the
Agent_Work_Context.md handoff and weekly chat-canvas compaction are the
other two designed-but-not-yet-built callers for this same function, same
shape ("N messages in, one schema-constrained JSON object out"), different
instruction/schema per caller.

Runs on its own `context_synthesis` role (2026-09-13), NOT the shared
"chat" role it originally borrowed. Two real reasons, both in
config/store.py's own comment on that role: compaction is a long-context
high-reasoning job (Nemotron 3 Super beats gpt-oss-120b 91.75 vs 22.30 on
RULER long-context retrieval — the exact skill this needs), and routing it
to Ollama Cloud's session-metered free tier keeps it from competing with
live chat for the same token-metered daily allowance.
"""

import asyncio
import json
import re

from config.store import config
from providers.base import ChatMessage, ProviderError
from providers.registry import ProviderNotConfigured, get_dispatcher_role, get_provider
from storage.context_store import (
    build_context_block,
    estimate_tokens,
    get_entries,
    replace_live_snapshot,
)
from storage.conversations import get_messages

# The ceiling that triggers a compaction pass, and what a pass aims to
# leave behind. Two numbers, not one, on purpose: if the trigger and the
# target were equal, the very next flagged insight would push context.md
# straight back over the line and compaction would fire on nearly every
# turn. The gap is the headroom that makes compaction an occasional event.
# Both are starting points to tune against real usage, not derived
# constants — see how_to_handle_context.md's own reasoning for why a flat
# ceiling sized to the smallest permitted model was rejected.
CONTEXT_TRIGGER_TOKENS = 12_000
CONTEXT_TARGET_TOKENS = 6_000

# Below this, an entry's survival check is skipped — a very short flagged
# note ("fiscal year starts in April") has too few distinctive words for
# overlap scoring to say anything meaningful, and short notes are cheap to
# carry forward verbatim anyway.
_MIN_WORDS_FOR_COVERAGE_CHECK = 4

# An entry counts as "survived" if at least this fraction of its
# distinctive words appear anywhere in the new compacted text. Deliberately
# lenient: the check exists to catch an entry that was DROPPED, not to
# force the compactor to reuse the original wording — rephrasing is the
# whole point of compacting.
_COVERAGE_THRESHOLD = 0.5

_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "is", "are", "was", "were", "be", "been", "being",
    "to", "of", "in", "on", "at", "for", "with", "by", "from", "as", "that", "this", "these",
    "those", "it", "its", "they", "them", "their", "we", "our", "you", "your", "i", "he", "she",
    "has", "have", "had", "do", "does", "did", "will", "would", "should", "could", "can", "may",
    "not", "no", "if", "then", "than", "so", "up", "out", "about", "into", "over", "after",
}

CONTEXT_COMPACTION_INSTRUCTION = """You are compacting the durable memory of an ongoing conversation. You are given the FULL raw conversation history. Produce the distilled memory that should be carried forward into every future turn.

Reply with ONLY a single JSON object, no prose, no markdown fence, matching exactly this shape:
{
  "core_constraints": ["..."],
  "active_state": ["..."],
  "key_decisions": ["..."]
}

- "core_constraints": stable facts, rules, preferences and requirements that are unlikely to change — the things that would still be true next week. Who the user is, what they're building, constraints they've stated, how they want things done.
- "active_state": what is currently in flight — the task underway, open blockers, things explicitly still undecided. This is the section most likely to be wrong next week, and that's expected.
- "key_decisions": decisions actually made, each with the reason it was made. "Chose X over Y because Z." A decision without its reason is much less useful later — include the why.

Rules that matter more than completeness:
- NEVER invent a detail the conversation didn't actually establish. An empty array is correct when a section genuinely has nothing in it.
- Anything that came from a tool result, a fetched web page, or search output is NOT a statement by the user. Do not phrase it as one. If it matters, attribute it ("the fetched docs said X"), don't promote it to a user-stated fact.
- Be specific over comprehensive. "Prefers concise replies" is worth keeping; "discussed several topics" is not.
- Merge duplicates. If the same fact was established three times, it appears once."""


def _content_words(text: str) -> set[str]:
    words = re.findall(r"[a-z0-9]+", (text or "").lower())
    return {w for w in words if w not in _STOPWORDS and len(w) > 2}


def _entry_survived(entry_content: str, compacted_text: str, compacted_words: set[str]) -> bool:
    """Deterministic survival check — no second LLM call, matching the
    'dispatcher checks, not a model' rule in how_to_handle_context.md."""
    words = _content_words(entry_content)
    if len(words) < _MIN_WORDS_FOR_COVERAGE_CHECK:
        # Too short to score; treat as not-survived so it gets carried
        # forward verbatim. Cheap, and errs toward keeping things.
        return entry_content.strip().lower() in compacted_text.lower()
    covered = len(words & compacted_words) / len(words)
    return covered >= _COVERAGE_THRESHOLD


def _render_sections(parsed: dict) -> str:
    """Renders the compaction JSON into the markdown block that actually
    gets injected. Sectioned rather than one blob specifically so a future
    pass can evict from a section — you can't evict from a paragraph."""
    labels = [
        ("core_constraints", "## Core constraints"),
        ("active_state", "## Active state"),
        ("key_decisions", "## Key decisions"),
    ]
    parts = []
    for key, heading in labels:
        items = [str(i).strip() for i in (parsed.get(key) or []) if str(i).strip()]
        if items:
            parts.append(heading + "\n" + "\n".join(f"- {i}" for i in items))
    return "\n\n".join(parts)


async def compact_context(conversation_id: str) -> dict:
    """Runs one compaction pass for a conversation and installs the result
    as the new live snapshot. Returns a small report dict (never raises) —
    callers treat a failed compaction as "leave the existing context
    alone and carry on," never as a reason to fail the chat turn.

    Reads the FULL raw chat log, never the previous snapshot and never the
    flagged entries themselves — compacting a compaction is the documented
    cause of summarization drift (see how_to_handle_context.md). The
    flagged entries are used ONLY afterward, as the integrity check.

    The integrity property is guaranteed by CONSTRUCTION, not by trusting
    the model to comply: any flagged entry the new text doesn't
    demonstrably cover is appended verbatim under "Preserved". That makes
    "every previously flagged item survived" true by definition, with no
    retry loop and no way for a sloppy compaction to silently lose
    something.
    """
    messages = await get_messages(conversation_id)
    if not messages:
        return {"ok": False, "reason": "no history to compact"}

    entries = await get_entries(conversation_id)
    _before_text, before_tokens = await build_context_block(conversation_id)

    transcript = [
        ChatMessage(role="assistant" if m["role"] == "navi" else "user", content=m["content"])
        for m in messages
        if not (m["role"] == "navi" and m["content"].startswith("⚠️"))
    ]
    if not transcript:
        return {"ok": False, "reason": "no usable history to compact"}

    parsed = await compact_conversation(transcript, CONTEXT_COMPACTION_INSTRUCTION)
    if not parsed:
        return {"ok": False, "reason": "compaction call failed or returned unparseable output"}

    compacted = _render_sections(parsed)
    if not compacted.strip():
        return {"ok": False, "reason": "compaction produced an empty result"}

    compacted_words = _content_words(compacted)
    rescued = [
        e["content"] for e in entries
        if not _entry_survived(e["content"], compacted, compacted_words)
    ]
    if rescued:
        compacted += "\n\n## Preserved\n" + "\n".join(f"- {c}" for c in rescued)

    await replace_live_snapshot(conversation_id, compacted)
    after_tokens = estimate_tokens(compacted)

    report = {
        "ok": True,
        "before_tokens": before_tokens,
        "after_tokens": after_tokens,
        "entries_checked": len(entries),
        "entries_rescued": len(rescued),
        # Real signal for the escape valve (how_to_handle_context.md's
        # "hit the floor" question, still open): a pass that barely shrank
        # anything, or left the result still above target, is what that
        # design is meant to eventually detect. Reported, not acted on yet.
        "still_above_target": after_tokens > CONTEXT_TARGET_TOKENS,
    }
    print(
        f"[compact_context] conversation={conversation_id} {before_tokens} -> {after_tokens} tokens, "
        f"{len(rescued)}/{len(entries)} entries rescued verbatim, "
        f"still_above_target={report['still_above_target']}"
    )
    return report


def strip_code_fence(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = text[3:]
        if text.lower().startswith("json"):
            text = text[4:]
        if text.endswith("```"):
            text = text[:-3]
    return text.strip()


async def compact_conversation(messages: list[ChatMessage], instruction: str) -> dict | None:
    """Runs one schema-constrained extraction call over `messages` — the
    caller's job to pass the FULL relevant history, not a windowed slice;
    that's the entire point of this primitive over just replaying the
    normal recency window (see how_to_handle_context.md's "Is 20 messages
    actually enough?" section). `instruction` is the system prompt
    describing the desired JSON shape and any drafting guidance.

    Returns the parsed JSON object, or None if every attempt in the
    fallback chain failed, returned empty, or produced something that
    doesn't parse as JSON — callers MUST handle the None case explicitly
    (e.g. telling the user to try again), this never raises to signal
    failure, matching every other provider-call site in this codebase."""
    try:
        role = get_dispatcher_role(context="context_synthesis")
    except ProviderNotConfigured:
        return None

    call_messages = [ChatMessage(role="system", content=instruction)] + messages
    attempts = config.get_attempts([{"provider": role["provider"], "model": role["model"]}] + role.get("fallback", []))
    for attempt in attempts:
        try:
            provider = get_provider(attempt["provider"])
        except Exception:
            continue
        try:
            response = await asyncio.to_thread(provider.chat, model=attempt["model"], messages=call_messages)
        except ProviderError as e:
            if e.is_rate_limit:
                config.mark_rate_limited(attempt["provider"], attempt["model"])
            continue
        text = strip_code_fence(response.text or "")
        if not text:
            continue
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None
