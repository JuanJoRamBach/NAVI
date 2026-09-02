"""
dispatcher/provider_debug.py

Saves the full outgoing/incoming exchange with a model whenever a chat
turn fails — a real "returned neither text nor a tool call" incident
(2026-09-02, three different Cloudflare models) was impossible to
diagnose further without seeing the actual raw response, and there was
nowhere durable to look. Reuses storage/filen.py's save_result (same
rclone-backed Filen remote every other command already saves to) rather
than inventing new storage — synchronous/blocking, so async callers
should run it via asyncio.to_thread, matching how provider.chat() calls
are already dispatched elsewhere.

Best-effort by design: a debug-save failure must never break the actual
chat flow it's trying to explain — caught, not raised. But NOT silently —
a bare `except: pass` here would defeat the entire point of this module
(2026-09-02: exactly that happened on the first version, and the actual
rclone/Filen failure reason was lost). Printed instead, so it lands in
whatever captures stdout (journalctl on the Lightsail service) — visible
without being fatal.
"""

import json
import sys
import traceback
from datetime import datetime, timezone

from providers.base import ChatMessage
from storage.filen import save_result

FOLDER = "provider-debug"


def save_failed_exchange(
    context: str, provider: str, model: str, messages: list[ChatMessage],
    reason: str, raw: dict | None = None,
) -> None:
    try:
        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "context": context,
            "provider": provider,
            "model": model,
            "reason": reason,
            "outgoing_messages": [
                {
                    "role": m.role, "content": m.content, "name": m.name,
                    "tool_call_id": m.tool_call_id, "tool_calls": m.tool_calls,
                }
                for m in messages
            ],
            "raw_response": raw,
        }
        content = json.dumps(payload, indent=2, default=str)
        filename = f"{provider}_{model.replace('/', '_').replace('@', '')}.json"
        remote_path = save_result(FOLDER, context, filename, content)
        print(f"[provider_debug] saved failed exchange to {remote_path}", file=sys.stderr)
    except Exception:
        print("[provider_debug] FAILED to save debug exchange:", file=sys.stderr)
        traceback.print_exc()
