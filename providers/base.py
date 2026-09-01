"""
providers/base.py

The shared contract every provider (OpenRouter, Groq, NVIDIA NIM, whatever
gets added later) implements. Modeled on the transport-interface pattern
from LocalCodeCli: one shape in, one shape out, so swapping or adding a
provider never touches calling code.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

# Request counts, keyed by (provider_name, model) — always tracked at this
# granularity regardless of whether a provider's real rate limit is
# per-model (Groq) or shared across all models (OpenRouter, Cloudflare,
# LLM7, Mistral); rolling per-model counts up into a provider-wide total
# is a read-time concern, not a storage-time one. Module-level, not on
# Provider instances — get_provider() in registry.py hands back a fresh
# instance on every call (not a singleton), so instance state would reset
# to zero every time and never accumulate anything. Counts the ATTEMPT
# (incremented before the transport call, regardless of outcome) since
# rate limits are generally enforced server-side the moment a request
# arrives, before the response is known — undercounting a failed-but-
# received request would be the wrong direction to err in here.
# Process-lifetime only for now (resets on restart) — persisting this
# across restarts and reconciling it against each provider's own admin/
# usage-style endpoint is the not-yet-built piece from the model-ranking
# design conversation, not something to half-build speculatively here.
_REQUEST_COUNTS: dict[tuple[str, str], int] = {}


def get_request_counts() -> dict[tuple[str, str], int]:
    """Read-only snapshot of (provider, model) -> request count so far
    this process. Whatever eventually persists/reports/reconciles this
    reads it from here rather than reaching into the private dict."""
    return dict(_REQUEST_COUNTS)


@dataclass
class ChatMessage:
    role: str  # "system" | "user" | "assistant" | "tool"
    # Plain text for every existing use. A list of OpenAI-format content
    # parts (e.g. [{"type": "text", ...}, {"type": "image_url", ...}]) is
    # also accepted — every provider's _serialize_message forwards
    # `content` through untouched, so vision content (see /design-read)
    # needs no per-provider changes, just this wider type.
    content: str | list[dict]
    tool_call_id: str | None = None
    name: str | None = None
    # Raw OpenAI-format tool_calls list, set on an assistant message that
    # requested tool calls — needed to replay that turn back to the API
    # when continuing a tool-call loop (see dispatcher/executor.py).
    tool_calls: list[dict] | None = None


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict


@dataclass
class ChatResponse:
    text: str | None
    tool_calls: list[ToolCall] = field(default_factory=list)
    model_used: str = ""
    raw: dict | None = None
    # Optional human-readable cost/usage note (e.g. "2.7 Neurons" on
    # Cloudflare). Providers with no comparable per-call cost metric
    # (Groq, OpenRouter, Ollama Cloud) just leave this None.
    usage_note: str | None = None


class ProviderError(Exception):
    """Raised for any provider failure. Callers decide whether to rotate."""
    def __init__(self, message: str, is_rate_limit: bool = False):
        super().__init__(message)
        self.is_rate_limit = is_rate_limit


class Provider(ABC):
    """Base class every concrete provider transport implements."""

    name: str = "base"

    def __init__(self, api_key: str):
        self.api_key = api_key

    def chat(
        self,
        model: str,
        messages: list[ChatMessage],
        tools: list[dict] | None = None,
        tool_choice: str | dict | None = None,
    ) -> ChatResponse:
        """
        Send a chat completion request. Raises ProviderError on failure.

        tool_choice lets a caller force a specific tool (e.g. /graph-data
        forcing render_chart) instead of leaving it to "auto", which is
        the default when tools are provided but tool_choice isn't set.

        Concrete (not abstract) on purpose — this is the one place every
        provider's call gets counted (see _REQUEST_COUNTS above), so it
        can't be reimplemented per-transport without either duplicating
        the counting line five times or someone eventually forgetting to.
        Every existing caller keeps calling .chat() exactly as before;
        this signature and behavior are unchanged, only the transport-
        specific work moved to _do_chat().
        """
        key = (self.name, model)
        _REQUEST_COUNTS[key] = _REQUEST_COUNTS.get(key, 0) + 1
        return self._do_chat(model, messages, tools=tools, tool_choice=tool_choice)

    @abstractmethod
    def _do_chat(
        self,
        model: str,
        messages: list[ChatMessage],
        tools: list[dict] | None = None,
        tool_choice: str | dict | None = None,
    ) -> ChatResponse:
        """Provider-specific transport — build the request, call the API,
        parse the response. Same contract chat() used to document
        directly; implement this exactly as chat() was implemented before
        this split. Never call this directly — call .chat() so the
        request gets counted."""
        raise NotImplementedError
