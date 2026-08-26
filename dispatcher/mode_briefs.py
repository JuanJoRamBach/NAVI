"""
dispatcher/mode_briefs.py

Loads the per-chat-mode system prompt + allowed-tools list from the .md
briefs in dispatcher/modes/. Each file starts with a small frontmatter
block:

    ---
    tools: [web_search, fetch_page]
    ---
    # Research Chat Mode
    ...

Parsed by hand rather than pulling in a YAML dependency — the frontmatter
here is always exactly one `tools: [...]` line, nothing more.
"""

import re
from dataclasses import dataclass
from pathlib import Path

MODES_DIR = Path(__file__).parent / "modes"

MODE_FILES = {
    "normal": "NORMAL_CHAT.md",
    "research": "RESEARCHER.md",
    "brainstorm": "BRAINSTORM.md",
}

_TOOLS_LINE = re.compile(r"^tools:\s*\[(.*?)\]\s*$", re.MULTILINE)


@dataclass
class ModeBrief:
    system_prompt: str
    tools: list[str]


def _parse_frontmatter(raw: str) -> tuple[list[str], str]:
    if not raw.startswith("---"):
        return [], raw
    end = raw.find("---", 3)
    if end == -1:
        return [], raw
    frontmatter, body = raw[3:end], raw[end + 3:].lstrip("\n")
    match = _TOOLS_LINE.search(frontmatter)
    tools = [t.strip() for t in match.group(1).split(",") if t.strip()] if match else []
    return tools, body


# Briefs rarely change at runtime — read once per process, not per message.
_cache: dict[str, ModeBrief] = {}


def get_mode_brief(mode: str) -> ModeBrief:
    """Unknown modes fall back to Normal rather than raising — a stale or
    unexpected mode string from the client shouldn't break the chat."""
    key = mode if mode in MODE_FILES else "normal"
    if key in _cache:
        return _cache[key]

    raw = (MODES_DIR / MODE_FILES[key]).read_text(encoding="utf-8")
    tools, body = _parse_frontmatter(raw)
    brief = ModeBrief(system_prompt=body.strip(), tools=tools)
    _cache[key] = brief
    return brief
