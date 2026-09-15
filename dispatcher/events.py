"""
dispatcher/events.py

The OUTWARD half of the audit.

`storage/context_store.py`'s friction log records what went wrong, for us,
after the fact. This records what is happening, for the user, while it
happens. Same moments in the code, same underlying signals, two different
audiences — which is the whole reason they are designed together rather
than as two parallel systems: a state worth showing the user is almost
always a state worth recording, and vice versa.

WHY THIS EXISTS AT ALL. Until now a chat turn was a single blocking
request that returned the finished reply, and the PWA showed the static
string "Thinking…" for however long that took. A p50 of 6.5s is fine that
way. A p95 of 45s is not: the user has no way to tell a slow answer from a
dead one, and the interface says exactly the same thing in both cases.

Real research, rather than instinct, says visible effort matters more than
raw speed here. Buell & Norton's operational-transparency work (Management
Science, 2011) found people rated a service that showed its work as more
valuable, and preferred a SLOWER one that showed its work to a faster one
that hid it. A 2026 CHI study of LLM latency specifically (240
participants, time-to-first-token at 2/9/20s) found responses at 9s were
rated MORE useful and MORE thoughtful than responses at 2s — but that 20s
"risks backfiring, reframing latency as inefficiency rather than
deliberation". Waiting reads as thinking, up to a point. Past that point
it reads as broken, and the fix is not only to be faster but to say what
is going on.

DESIGN. An emitter is an optional async callback threaded through the
dispatcher. `None` means "nobody is listening" and every emit becomes a
no-op, so the existing non-streaming `/chat/send` path behaves exactly as
it did before — this cannot slow down or break a caller that does not opt
in.

The dispatcher writes this copy, never the model. Same principle as every
other checkpoint in NAVI: the model proposes, the dispatcher decides, and
what the user reads about NAVI's own behaviour is authored by NAVI rather
than narrated by whatever model happens to be answering.
"""

from __future__ import annotations

from typing import Any, Awaitable, Callable

# An emitter takes (event_type, payload) and delivers it to whoever is
# listening. Returns nothing; must never raise into the dispatcher.
Emitter = Callable[[str, dict[str, Any]], Awaitable[None]]

# ---- Event types ----
#
# Deliberately few. Every one of these has to be rendered by a client, and
# a vocabulary that grows per feature ends up with events nothing displays.

# Human-readable progress. {"text": str, "kind": str}
STATUS = "status"
# A chunk of the answer as it is generated (Phase B). {"text": str}
TOKEN = "token"
# Discard anything streamed so far and start over — a tier escalation
# re-runs the turn on a stronger model, so text already shown is no longer
# the answer. {"reason": str}
RESET = "reset"
# The turn is finished; carries exactly what POST /chat/send returns, so a
# streaming client and a plain one end up with the same final object.
DONE = "done"
# The turn failed outright. {"text": str}
ERROR = "error"


async def noop_emit(event_type: str, payload: dict[str, Any]) -> None:
    """The default. Does nothing, cheaply."""
    return None


async def emit_status(emit: Emitter | None, text: str, kind: str = "info") -> None:
    """Sends one progress line, if anyone is listening.

    Swallows everything. A status update is decoration around real work —
    a client that disconnected mid-turn, or a slow consumer, must never
    take down the chat turn it was describing. Same rule the friction
    writer and the usage recorder already follow.
    """
    if emit is None:
        return
    try:
        await emit(STATUS, {"text": text, "kind": kind})
    except Exception as e:  # noqa: BLE001
        print(f"[events] status emit failed (non-fatal): {e}")


# ---- The copy ----
#
# Written here, once, rather than inline at each call site, for two
# reasons. It is user-facing product copy and belongs somewhere someone
# can read it all at once; and the same state is reachable from several
# places, so inlining guarantees they drift apart in wording.
#
# Voice: say what is happening and what NAVI is doing about it. Never
# apologise, never blame the provider by implication, never claim
# certainty about someone else's system ("is down", "is broken") when all
# we actually observed is that it did not answer us in time.

def asking(model: str) -> str:
    return f"Asking {model}…"


def taking_long(model: str) -> str:
    """Shown while still waiting, before giving up. Says NAVI is actively
    checking rather than passively stuck — which is the honest reading:
    the watchdog really is monitoring the call."""
    return f"This answer is taking longer than usual — checking on {model}…"


def switching(slow_model: str, next_model: str) -> str:
    return f"{slow_model} didn't answer in time. Switching to {next_model}…"


def unavailable(failed_model: str, next_model: str) -> str:
    """Distinct from `switching`: this one actually failed rather than
    just being slow, and conflating the two would misdescribe what
    happened in a log the user can read."""
    return f"{failed_model} was unavailable. Switching to {next_model}…"


def escalating(next_model: str) -> str:
    return f"This needs a stronger model — switching to {next_model}…"


def running_tool(tool_name: str) -> str:
    return TOOL_STATUS.get(tool_name, f"Running {tool_name}…")


# Plain language per tool, from the reader's side rather than the code's.
# A tool absent from here falls back to its own name, which is ugly but
# honest — better than inventing a friendly phrase for something new that
# might describe it wrongly.
TOOL_STATUS: dict[str, str] = {
    "web_search": "Searching the web…",
    "fetch_page": "Reading a page…",
    "save_source": "Saving a source…",
    "create_document": "Writing the document…",
    "save_note": "Saving a note…",
    "send_to_telegram": "Sending the message…",
    "create_workflow": "Building the workflow…",
    "run_workflow": "Running the workflow…",
    "get_run_status": "Checking the run…",
    "list_workflow_runs": "Looking up past runs…",
    "flag_key_insight": "Noting that for later…",
}
