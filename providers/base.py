"""
providers/base.py

The shared contract every provider (OpenRouter, Groq, NVIDIA NIM, whatever
gets added later) implements. Modeled on the transport-interface pattern
from LocalCodeCli: one shape in, one shape out, so swapping or adding a
provider never touches calling code.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


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

    @abstractmethod
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
        """
        raise NotImplementedError
