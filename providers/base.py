"""
providers/base.py

The shared contract every provider (OpenRouter, Groq, NVIDIA NIM, whatever
gets added later) implements. Modeled on the transport-interface pattern
from LocalCodeCli: one shape in, one shape out, so swapping or adding a
provider never touches calling code.
"""

import time
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
    """Raised for any provider failure. Callers decide whether to rotate.

    Two flags, deliberately separate, because they mean different things
    and must be treated differently (2026-09-14, after a real Gemini 503):

    is_rate_limit — YOU have hit a limit. Routine on free tiers, expected,
        and absorbed by the fallback chain by design.
    is_overloaded — THEY are having a bad minute: any 5xx. Not about your
        quota at all, and unlike a 429 it says something about the
        provider's reliability that is worth recording and eventually
        ranking on.

    Both should cool the endpoint down; only one should be treated as
    routine. Collapsing them into a single flag would either lose the
    backoff for 5xx (what happened before this) or hide provider outages
    inside the "routine, don't record" bucket.
    """
    def __init__(self, message: str, is_rate_limit: bool = False, is_overloaded: bool = False):
        super().__init__(message)
        self.is_rate_limit = is_rate_limit
        self.is_overloaded = is_overloaded


# Which provider-call outcomes are worth a friction row, and how heavily.
#
# NOT every failure. `rate_limit` and `cancelled` are deliberately absent:
# a 429 is routine on free tiers, it is exactly what the fallback chain
# exists to absorb, and it is ALREADY recorded per call in usage_calls
# with its own error_kind — writing it here too would both duplicate the
# fact and bury the rarer signals under the commonest one. The same cost
# asymmetry that set the "hit the floor" rule at two failures applies: a
# signal that fires constantly stops being read.
#
# A timeout is different in kind, not just degree. It pays the full input
# token cost and returns nothing, so it is the single most expensive
# outcome a call can have — and unlike a 429 it usually means a real
# mismatch between the work and the model, not a busy minute.
_FRICTION_FOR_ERROR: dict[str, int] = {
    "timeout": 3,
    # A provider 5xx. Recorded, unlike a 429, precisely because it is NOT
    # routine: it says the provider could not serve anyone, which is a
    # reliability fact about them worth accumulating — and it is what the
    # trust index needs in order to eventually demote a model that is
    # unavailable more often than its benchmark score suggests.
    "overloaded": 2,
    "error": 2,
    "empty_response": 2,
}


def _cool_down(kind: str, provider_name: str, model: str) -> None:
    """Demotes an endpoint that just told us it cannot serve us.

    Done HERE rather than at each call site, for the same reason the
    friction record is: this is the one point every provider call in NAVI
    passes through, so a new caller inherits the behaviour instead of
    having to remember it. Before this, the cooldown was hand-written at
    ten call sites and checked only is_rate_limit — so a 503 never cooled
    anything down, and Gemini stayed first in the chain while Google was
    overloaded, paying a wasted call on every single turn.

    Silent on every other kind. A timeout is ambiguous (it can be one slow
    request rather than a sick endpoint) and a malformed response is a
    bug, not a capacity signal — neither justifies sidelining a role's
    primary, and over-eager demotion would quietly route good traffic to
    weaker models.
    """
    from config.store import config

    windows = {
        "rate_limit": None,  # None = the default rate-limit window
        "overloaded": config.OVERLOADED_COOLDOWN_SECONDS,
    }
    if kind not in windows:
        return
    try:
        config.mark_rate_limited(provider_name, model, seconds=windows[kind])
    except Exception as e:  # noqa: BLE001
        print(f"[Provider.chat] cooldown marking failed (non-fatal): {e}")


def _outgoing_tokens(messages: list["ChatMessage"]) -> int:
    """Rough size of what was SENT, for a call that came back with nothing.

    A failed call has no `usage` object to read — the provider never got
    far enough to report one — so the only way to say what a timeout
    actually cost is to measure the payload that went out. Reuses
    context_store's own estimator rather than inventing a second one, and
    inherits its honesty: it is an estimate, and rows written from here
    say so.
    """
    try:
        from storage.context_store import estimate_tokens

        total = 0
        for m in messages:
            if isinstance(m.content, str):
                total += estimate_tokens(m.content)
            elif isinstance(m.content, list):
                for part in m.content:
                    if isinstance(part, dict) and isinstance(part.get("text"), str):
                        total += estimate_tokens(part["text"])
        return total
    except Exception:
        return 0


def _record_friction(
    kind: str, provider_name: str, model: str, call_ctx: dict, detail: str | None,
    wasted_tokens: int | None = None, estimated: bool = False,
) -> None:
    """Logs a degraded provider outcome, if this kind warrants one.

    Written HERE, at the one point every provider call in NAVI funnels
    through, rather than at each call site. Before this, friction was
    hand-recorded in dispatcher/chat.py alone — so Agent Work, Dev Slate,
    Research, Sources and every job could fail all day without producing a
    single row, which is exactly why the log was empty while real
    incidents (Ollama's 182-second cap, a Mistral 400, a source that
    wouldn't distil) were happening weekly.

    conversation_id rides in on the same ambient call context that tags
    usage rows, so a row written down here is still attributable to the
    conversation that caused it without any caller passing it down.
    """
    severity = _FRICTION_FOR_ERROR.get(kind)
    if severity is None:
        return
    try:
        from storage.context_store import record_friction_sync

        where = f"{provider_name}/{model}"
        note = f"{where}: {detail[:200]}" if detail else where
        if wasted_tokens and estimated:
            note += f" (~{wasted_tokens} tokens sent, estimated)"
        elif wasted_tokens:
            note += f" ({wasted_tokens} tokens spent)"
        record_friction_sync(
            f"provider_{kind}" if kind != "empty_response" else kind,
            severity=severity,
            conversation_id=call_ctx.get("conversation_id"),
            detail=note,
            wasted_tokens=wasted_tokens,
            provider=provider_name,
            model=model,
        )
    except Exception:
        pass  # an auxiliary signal must never break a real chat request


def _classify_error(exc: BaseException) -> str:
    """A coarse bucket for a failed call: rate_limit / timeout /
    cancelled / error.

    Deliberately coarse. The point of this field is to make "why do calls
    on this role fail" answerable at a glance without reading logs —
    four buckets does that, and a finer taxonomy invented now, before
    anything reads the column, would be a guess dressed as data. Every
    transport already funnels its real failures into ProviderError, so
    `is_rate_limit` is authoritative where it's set; the timeout check is
    a string match because that is genuinely how the distinction arrives
    (requests' own timeout exceptions are caught and re-raised as
    ProviderError inside each transport, losing the type).
    """
    import asyncio

    if isinstance(exc, asyncio.CancelledError):
        return "cancelled"
    if isinstance(exc, ProviderError) and exc.is_rate_limit:
        return "rate_limit"
    # Checked after rate_limit: a provider that sets both means the 429
    # reading, which is the more specific claim.
    if isinstance(exc, ProviderError) and exc.is_overloaded:
        return "overloaded"
    text = str(exc).lower()
    if "timeout" in text or "timed out" in text:
        return "timeout"
    return "error"


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
        extra_params: dict | None = None,
    ) -> ChatResponse:
        """
        Send a chat completion request. Raises ProviderError on failure.

        tool_choice lets a caller force a specific tool (e.g. /graph-data
        forcing render_chart) instead of leaving it to "auto", which is
        the default when tools are provided but tool_choice isn't set.

        extra_params (2026-09-12) — real per-family API fields NAVI
        never had a way to send before this (e.g. gpt-oss's Harmony
        reasoning_effort, see dispatcher/prompt_family.py's
        adapt_request_params, the actual first caller). Merged directly
        into the request payload by each provider's own _do_chat — kept
        as a bare dict rather than named parameters here so a new
        per-family field never needs touching this shared base class or
        all 7 transports' signatures again, just the one transport that
        cares. None/empty is always a safe no-op for every provider.

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
        # Counts the ATTEMPT, before _do_chat runs — rate limits are
        # generally enforced server-side the moment a request arrives,
        # before the response (or failure) is known, so a failed call
        # still needs to count here, same reasoning _REQUEST_COUNTS'
        # own docstring already documents. Real prompt/completion TOKEN
        # counts can only be known after a real response exists, so
        # those are recorded separately, below, only on success —
        # this call intentionally only ever contributes `requests`.
        from storage.usage import record_usage
        try:
            record_usage(self.name, model, requests=1)
        except Exception:
            pass  # usage tracking must never break a real chat request

        # Defense in depth (2026-09-06) — jobs/model_ranking.py's own
        # BANNED_FOR_TOOLS already keeps a known-unreliable-with-tools
        # model from ever being RANKED/SELECTED for a tools-requiring
        # task, but this is the one place every call funnels through
        # regardless of how a model got chosen (a stale manual pin, a
        # role config edited by hand, a future caller that bypasses
        # ranking entirely) — so this is the real backstop, not the
        # primary fix. Strips tools rather than raising: a banned-for-
        # tools model can still answer in plain text, which is strictly
        # better than a hard failure for whatever turn this is.
        if tools:
            from jobs.model_ranking import BANNED_FOR_TOOLS
            if model in BANNED_FOR_TOOLS:
                print(f"[Provider.chat] refusing to send tools to banned-for-tools model '{model}' — stripping tools/tool_choice for this call")
                tools = None
                tool_choice = None

        # Timed, and recorded per call whether it succeeds or fails
        # (2026-09-14). usage_daily above is an aggregate keyed by
        # (provider, model, day) — it cannot say which ROLE asked, which
        # tier answered, how long it took, or how often a call failed,
        # and every one of those is needed before a friction count can
        # become a friction RATE. See storage/usage.py's record_call.
        from storage.usage import current_call_context, record_call
        call_ctx = current_call_context()
        started = time.monotonic()
        try:
            response = self._do_chat(model, messages, tools=tools, tool_choice=tool_choice, extra_params=extra_params)
        except BaseException as e:
            # A failed call is still a call. Recording only successes
            # would make every rate computed from this table wrong in the
            # flattering direction — see record_call's own docstring.
            # BaseException rather than Exception so a cancelled request
            # (asyncio.CancelledError, a real and routine outcome when a
            # client disconnects mid-turn) is counted rather than
            # silently missing from the denominator.
            kind = _classify_error(e)
            record_call(
                self.name, model, ok=False,
                latency_ms=int((time.monotonic() - started) * 1000),
                error_kind=kind, context=call_ctx,
            )
            _record_friction(
                kind, self.name, model, call_ctx, str(e),
                wasted_tokens=_outgoing_tokens(messages), estimated=True,
            )
            _cool_down(kind, self.name, model)
            raise

        # Generic, provider-agnostic real TOKEN persistence (2026-09-10) —
        # for the Usage counters panel AND the real savings-summary
        # baseline (storage/usage.py's get_savings_summary). Extracted
        # from `usage`, which every OpenAI-compatible response includes
        # unconditionally (confirmed across all 7 transports) — previously
        # only 2 of 7 providers (cloudflare.py, llm7.py) recorded a token
        # count anywhere, each from inside its own _do_chat. Centralizing
        # here closes that gap for all 7 at once, in the one place every
        # call already funnels through, instead of duplicating this per
        # transport. requests=0 here on purpose — the attempt was already
        # counted above, before _do_chat; this call only ever contributes
        # tokens. A provider with an extra, richer signal beyond token
        # counts (Cloudflare's Neurons, Groq's rate-limit headers) still
        # records THAT separately from inside its own _do_chat.
        usage = (response.raw or {}).get("usage") or {}
        try:
            record_usage(
                self.name, model,
                tokens=usage.get("total_tokens") or 0,
                prompt_tokens=usage.get("prompt_tokens") or 0,
                completion_tokens=usage.get("completion_tokens") or 0,
            )
        except Exception:
            pass  # usage tracking must never break a real chat request

        # Same `usage` object, one row per call instead of a daily total.
        # cached_tokens is read here for the first time anywhere that
        # persists it: 5 of 8 transports already pull it out of
        # prompt_tokens_details for their human-readable usage_note and
        # then throw the number away (flagged as a real gap in IDEAS.md's
        # cached-token audit). Reading it at this one point closes that
        # for every provider that reports it, and costs a dict lookup for
        # the ones that don't.
        record_call(
            self.name, model, ok=True,
            prompt_tokens=usage.get("prompt_tokens") or 0,
            completion_tokens=usage.get("completion_tokens") or 0,
            cached_tokens=(usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0,
            total_tokens=usage.get("total_tokens") or 0,
            latency_ms=int((time.monotonic() - started) * 1000),
            context=call_ctx,
        )

        # A call that succeeded and returned nothing. Previously invisible
        # everywhere: it isn't an error, so no transport raises on it, and
        # ok=1 in usage_calls doesn't distinguish it from a real answer.
        # dispatcher/chat.py already treats this as a failure worth
        # rotating to the next provider for — it just never recorded that
        # it happened.
        if not response.text and not response.tool_calls:
            # Here the cost is MEASURED, not estimated: the call succeeded,
            # so the provider reported real prompt tokens — every one of
            # which bought nothing.
            _record_friction(
                "empty_response", self.name, model, call_ctx, None,
                wasted_tokens=usage.get("prompt_tokens") or None,
            )

        return response

    @abstractmethod
    def _do_chat(
        self,
        model: str,
        messages: list[ChatMessage],
        tools: list[dict] | None = None,
        tool_choice: str | dict | None = None,
        extra_params: dict | None = None,
    ) -> ChatResponse:
        """Provider-specific transport — build the request, call the API,
        parse the response. Same contract chat() used to document
        directly; implement this exactly as chat() was implemented before
        this split. Never call this directly — call .chat() so the
        request gets counted.

        extra_params: merge directly into the JSON payload before
        sending (`payload.update(extra_params)` or equivalent) — see
        chat()'s own docstring above."""
        raise NotImplementedError
