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
3. No command, no near-miss -> plain chat. normal_chat handles it
   directly as a normal AI reply, not a routing decision.
"""

import re
from dataclasses import dataclass

COMMANDS = ["research", "graph-data", "summarize", "recap", "note", "remind"]

# Max edit distance to flag as a "near miss" worth confirming.
# Slash-prefixed typos (e.g. "/cade") are a strong command-intent signal,
# so 2 is safe there — catches single-letter typos and transpositions.
# Bare words with no slash need a tighter bar: a short, generic command
# like "code" is within distance 2 of plenty of ordinary English words
# ("some" vs "code" is exactly 2), so at threshold 2 a normal sentence
# containing one of them gets wrongly flagged as a typo mid-conversation.
# Distance 1 is tight enough to stop matching incidental vocabulary while
# still catching a genuinely dropped-slash typo.
NEAR_MISS_THRESHOLD_SLASH = 2
NEAR_MISS_THRESHOLD_BARE = 1


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


def _find_near_miss(word: str, threshold: int) -> str | None:
    """Returns the closest command name if within threshold, else None.
    Requires a real edit distance (>0) — a bare word that's an EXACT match
    to a command name (e.g. plain "note" or "code" appearing in an
    ordinary sentence, no slash) isn't a typo of anything, it's just that
    word being used normally. Slash-prefixed exact matches never reach
    this function at all (parse_message's regex catches those first), so
    this only ever excludes the bare-word case."""
    word = word.lower().lstrip("/")
    best, best_dist = None, threshold + 1
    for cmd in COMMANDS:
        dist = _levenshtein(word, cmd)
        if dist < best_dist:
            best, best_dist = cmd, dist
    return best if 0 < best_dist <= threshold else None


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
        threshold = NEAR_MISS_THRESHOLD_SLASH if word.startswith("/") else NEAR_MISS_THRESHOLD_BARE
        suggestion = _find_near_miss(word, threshold)
        if suggestion:
            return ParseResult(
                kind="near_miss",
                near_miss_word=word,
                near_miss_suggestion=suggestion,
                raw_text=text,
            )

    return ParseResult(kind="plain_chat", raw_text=text)
