"""
dispatcher/parser.py

The free, deterministic first pass on every incoming message. No AI call
happens here — this is pure string matching, same input always produces
the same output. Three possible results:

1. One or more literal commands found -> a list of Steps, ready to execute.
2. Something that looks like a near-miss typo of a command (e.g. "/reserch",
   or "reserch X" with the slash missing) -> flagged for user confirmation,
   since silently either running it as a command OR silently treating it as
   plain chat could both be wrong.
3. No command, no near-miss -> plain chat. dispatcher_chat handles it
   directly as a normal AI reply, not a routing decision.
"""

import re
from dataclasses import dataclass

COMMANDS = ["research", "code", "graph-data", "create-image", "summarize"]

# Max edit distance to flag as a "near miss" worth confirming. 2 catches
# single-letter typos and transpositions without being so loose it flags
# unrelated words.
NEAR_MISS_THRESHOLD = 2


@dataclass
class Step:
    command: str
    text: str
    topic_slug: str = ""  # filled in by slugify.py after parsing


@dataclass
class ParseResult:
    kind: str  # "commands" | "near_miss" | "plain_chat"
    steps: list[Step] | None = None
    near_miss_word: str | None = None
    near_miss_suggestion: str | None = None
    raw_text: str = ""


def _levenshtein(a: str, b: str) -> int:
    """Standard edit distance, no dependencies."""
    if len(a) < len(b):
        a, b = b, a
    if len(b) == 0:
        return len(a)
    prev_row = list(range(len(b) + 1))
    for i, ca in enumerate(a):
        curr_row = [i + 1]
        for j, cb in enumerate(b):
            curr_row.append(min(
                prev_row[j + 1] + 1,      # deletion
                curr_row[j] + 1,          # insertion
                prev_row[j] + (ca != cb),  # substitution
            ))
        prev_row = curr_row
    return prev_row[-1]


def _find_near_miss(word: str) -> str | None:
    """Returns the closest command name if within threshold, else None."""
    word = word.lower().lstrip("/")
    best, best_dist = None, NEAR_MISS_THRESHOLD + 1
    for cmd in COMMANDS:
        dist = _levenshtein(word, cmd)
        if dist < best_dist:
            best, best_dist = cmd, dist
    return best if best_dist <= NEAR_MISS_THRESHOLD else None


def parse_message(text: str) -> ParseResult:
    text = text.strip()

    # Build a regex matching only the literal whitelist, as real commands
    # (word boundary before the slash, command name, then boundary/space).
    command_pattern = "|".join(re.escape(c) for c in COMMANDS)
    marker_re = re.compile(rf"(?<!\S)/({command_pattern})(?=\s|$)", re.IGNORECASE)

    matches = list(marker_re.finditer(text))

    if matches:
        steps = []
        for i, m in enumerate(matches):
            cmd = m.group(1).lower()
            start = m.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            step_text = text[start:end].strip(" ,.\n")
            # Strip a trailing connector word ("then") left over from chaining
            # syntax like "...X, then /graph-data..." — it's punctuation
            # between commands, not part of the instruction itself.
            step_text = re.sub(r"[\s,]*\bthen\b\s*$", "", step_text, flags=re.IGNORECASE).strip(" ,.\n")
            steps.append(Step(command=cmd, text=step_text))
        return ParseResult(kind="commands", steps=steps, raw_text=text)

    # No literal command found. Check every "word" for a near-miss, whether
    # it had a stray slash (e.g. "/reserch") or was typed bare (e.g. "reserch").
    for word in re.findall(r"/?\w[\w-]*", text):
        if len(word.lstrip("/")) < 4:
            continue  # skip short words, too noisy for edit-distance matching
        suggestion = _find_near_miss(word)
        if suggestion:
            return ParseResult(
                kind="near_miss",
                near_miss_word=word,
                near_miss_suggestion=suggestion,
                raw_text=text,
            )

    return ParseResult(kind="plain_chat", raw_text=text)
