"""Branch chats — a clean job space for one feature, seeded with a spec.

The problem this solves, concretely: branching used to copy the parent's
messages into the client's scrollback and then open a BRAND NEW, EMPTY
server conversation. `/chat/send` only ever carries text + mode +
conversation_id, never history, so the model saw none of it. The user
read a conversation the model had no access to.

The fix is not "send more history" — that recreates the bloat a branch
exists to escape. It is spec-driven development, which is now an
established pattern for exactly this (GitHub's Spec Kit, AWS's Kiro):
write the intent down, hand THAT to the worker, and validate the work
against it. Spec Kit's framing is the useful one — intent, not code, is
the source of truth, and when something doesn't make sense you go back
to the spec.

So a branch receives a spec compacted from the parent and scoped by the
name the user typed. That name was already being collected and thrown
away as a title; it is the scoping signal this whole mechanism needs.

Two properties worth stating outright, because they are load-bearing:

1. The spec is DISSECTED into individual context entries, never stored as
   one blob. Each line is then uniform with every other memory from turn
   one: individually retirable, covered by consolidation's integrity
   check for free, and visible in the conversation's own token budget.

2. The spec is simultaneously the branch's CONTEXT and its COMPLETION
   CONTRACT. That follows from how the industry actually separates these
   ideas: acceptance criteria are per-item, written in advance, agreed
   between the person who wants the work and the people doing it. Which
   is exactly what the `acceptance` field is. Those entries are PINNED —
   consolidation may never retire them, because losing them would not
   merely forget a fact, it would destroy the branch's ability to know
   whether it is finished.

Deliberately NOT here: a Definition of Done (a project-wide checklist
applying to all work, a separate feature with its own home question), and
role-gated sign-off. On the latter, DORA's research is unambiguous that
approval by someone outside the team — a board or a senior manager — is
negatively correlated with lead time, deployment frequency and restore
time, with no correlation at all to change failure rate. The person who
owns the branch accepts it. An optional role gate for companies that
genuinely need one is a later choice, not a default.
"""

from __future__ import annotations

from dispatcher.compaction import compact_conversation
from providers.base import ChatMessage
from storage.context_store import SOURCE_BRANCH_BRIEF, append_entry, build_context_block
from storage.conversations import get_messages

# Below this many real messages in the parent, drafting a spec is paying
# a model call to summarise almost nothing. The branch still opens — it
# just opens with the name the user typed as its goal and nothing else,
# which is honestly all the parent had to give.
MIN_PARENT_MESSAGES_FOR_SPEC = 4

SPEC_INSTRUCTION = """You are writing a SPEC that hands one piece of work from an ongoing conversation to a fresh, separate working session. That session will see ONLY this spec — none of the conversation you are reading. Anything you leave out is genuinely lost to it.

The work being handed over is: "{scope}"

Read the whole conversation and extract only what bears on THAT work. The conversation covers other things; those are not your problem, and including them defeats the purpose.

Reply with ONLY a single JSON object, no prose, no markdown fence, matching exactly this shape:
{{
  "goal": "string — one sentence: what this piece of work is",
  "established": ["string", "..."],
  "constraints": ["string", "..."],
  "acceptance": ["string", "..."],
  "out_of_scope": ["string", "..."],
  "unspecified": ["string", "..."]
}}

- "established": facts already settled in the conversation that the work depends on. One self-contained fact per string — the reader has no other context to resolve a pronoun or a "that" against.
- "constraints": decisions already made that this work must respect and must NOT reopen. Include the reason where the conversation gave one; a constraint without its reason gets argued with.
- "acceptance": how anyone will know this work is actually finished. Each one must be checkable — something you could look at the finished work and answer yes or no about. "Works well" is not checkable. If the conversation never established a real bar, return fewer items rather than inventing plausible ones.
- "out_of_scope": things explicitly deferred or ruled out. This is what keeps the fresh session from wandering back into the rest of the conversation.
- "unspecified": real open questions this work will hit that the conversation never settled. Listing a gap is far more useful than filling it with a guess.

Every list may be empty. Never pad a list to look complete.

Base this ENTIRELY on what the conversation actually established. Do not invent a requirement, a constraint or a standard the conversation never stated. Where the conversation shows a fact came from a tool result or a fetched page rather than from the user, do not restate it as something the user decided."""

# Prefix each entry gets when the spec is dissected. The label survives
# into every future prompt, so it has to read as a fact about the work
# rather than as a field name from a form.
_SECTION_LABELS = {
    "established": "Established",
    "constraints": "Constraint",
    "acceptance": "Done when",
    "out_of_scope": "Out of scope",
    "unspecified": "Open question",
}


def _clean_list(spec: dict, key: str) -> list[str]:
    raw = spec.get(key) or []
    if not isinstance(raw, list):
        return []
    return [str(item).strip() for item in raw if str(item or "").strip()]


def format_spec_markdown(spec: dict) -> str:
    """The spec as the user reviews it at the checkpoint. Sections with
    nothing in them are omitted rather than shown empty — an empty
    "Constraints" heading reads as a missing answer instead of as a real
    "there aren't any". Acceptance leads, because that is the part the
    user is actually agreeing to."""
    parts = [f"**{str(spec.get('goal') or '').strip()}**"]
    for key, heading in (
        ("acceptance", "Done when"),
        ("established", "Established"),
        ("constraints", "Constraints"),
        ("out_of_scope", "Out of scope"),
        ("unspecified", "Still open"),
    ):
        items = _clean_list(spec, key)
        if items:
            parts.append(f"**{heading}:**\n" + "\n".join(f"- {i}" for i in items))
    return "\n\n".join(parts)


async def draft_branch_spec(parent_conversation_id: str, scope: str) -> dict | None:
    """Compacts the parent into a spec for `scope`. Returns None when
    there is genuinely nothing to compact or the model call failed — the
    caller decides what to do about it, since "open the branch anyway
    with just the name" is a perfectly good outcome and not an error.

    Reads the parent's FULL history AND its live context block. The
    history alone is not enough: anything the parent has already
    consolidated lives in its context block as a distilled judgment, and
    that is frequently the most relevant material of all.
    """
    history = await get_messages(parent_conversation_id)
    real = [m for m in history if not (m["role"] == "navi" and m["content"].startswith("⚠️"))]
    if len(real) < MIN_PARENT_MESSAGES_FOR_SPEC:
        return None

    messages = [
        ChatMessage(role="assistant" if m["role"] == "navi" else m["role"], content=m["content"])
        for m in real
    ]
    parent_context, _tokens = await build_context_block(parent_conversation_id)
    if parent_context.strip():
        messages.insert(0, ChatMessage(
            role="user",
            content=(
                "Before the conversation itself, here is what has already been "
                "established and consolidated in it:\n\n" + parent_context
            ),
        ))

    spec = await compact_conversation(messages, SPEC_INSTRUCTION.format(scope=scope))
    if not isinstance(spec, dict) or not str(spec.get("goal") or "").strip():
        return None
    return spec


async def seed_branch_context(branch_conversation_id: str, spec: dict) -> int:
    """Dissects an accepted spec into individual context entries. Returns
    how many were written.

    The goal line is written first so it reads as the branch's subject on
    every future turn, and is pinned for the same reason the acceptance
    criteria are — a branch that forgets what it is for cannot finish.
    """
    written = 0
    goal = str(spec.get("goal") or "").strip()
    if goal:
        await append_entry(
            branch_conversation_id, f"This branch exists to: {goal}",
            source=SOURCE_BRANCH_BRIEF, pinned=True,
        )
        written += 1

    for key, label in _SECTION_LABELS.items():
        for item in _clean_list(spec, key):
            await append_entry(
                branch_conversation_id, f"{label}: {item}",
                source=SOURCE_BRANCH_BRIEF,
                pinned=(key == "acceptance"),
            )
            written += 1
    return written
