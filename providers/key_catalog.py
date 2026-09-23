"""
providers/key_catalog.py

What Settings → Your API keys offers, and how each key is checked before
it is saved (2026-09-23).

Two kinds of provider:
- FREE: the eight NAVI already routes to on their free tiers. Bringing a
  key for one of these swaps in the person's own account (their own
  quota, or a paid tier) for NAVI's key. No model list is stored for them:
  their models already reach the picker through the daily ranking.
- PAID: providers NAVI never uses by default (providers/byok.py). Their
  model list is fetched when the key is saved, since that list is the only
  way their models reach the picker.

Each check makes one cheap, authenticated read — never a chat call, so
checking a key costs nothing on the account it belongs to.
"""

import requests

from providers.base import ProviderError
from providers.byok import BYOK_TRANSPORTS

# In the order Settings lists them.
FREE_PROVIDERS: dict[str, str] = {
    "groq": "Groq",
    "llm7": "LLM7",
    "cloudflare": "Cloudflare",
    "gmi": "GMI Cloud",
    "openrouter": "OpenRouter",
    "mistral": "Mistral",
    "ollama_cloud": "Ollama Cloud",
    "gemini": "Google AI Studio",
}

# LLM7's model list works with no key at all, and it has no key-info
# endpoint, so there is no read that tells a good key from a bad one. Its
# key is saved unchecked and the UI says so, rather than pretending.
UNCHECKABLE = {"llm7"}


def _probe(label: str, url: str, *, headers: dict | None = None, params: dict | None = None) -> None:
    try:
        resp = requests.get(url, headers=headers or {}, params=params, timeout=15)
    except requests.RequestException as e:
        raise ProviderError(f"Couldn't reach {label}: {e}")
    # Google answers a bad key with 400 API_KEY_INVALID rather than 401.
    if resp.status_code in (400, 401, 403):
        raise ProviderError(f"{label} rejected this key ({resp.status_code}).")
    if resp.status_code >= 400:
        raise ProviderError(f"{label} error {resp.status_code}: {resp.text[:200]}")


def check_free_key(provider: str, api_key: str, account_id: str | None = None) -> None:
    """Raises ProviderError if the provider rejects the key."""
    bearer = {"Authorization": f"Bearer {api_key}"}
    label = FREE_PROVIDERS[provider]
    if provider == "groq":
        _probe(label, "https://api.groq.com/openai/v1/models", headers=bearer)
    elif provider == "openrouter":
        # /models is public; /key is the one read that needs a real key.
        _probe(label, "https://openrouter.ai/api/v1/key", headers=bearer)
    elif provider == "mistral":
        _probe(label, "https://api.mistral.ai/v1/models", headers=bearer)
    elif provider == "gmi":
        _probe(label, "https://api.gmi-serving.com/v1/models", headers=bearer)
    elif provider == "ollama_cloud":
        _probe(label, "https://ollama.com/v1/models", headers=bearer)
    elif provider == "gemini":
        _probe(label, "https://generativelanguage.googleapis.com/v1beta/models", params={"key": api_key})
    elif provider == "cloudflare":
        if not account_id:
            raise ProviderError("Cloudflare also needs your Account ID.")
        # Checks the token AND that it can reach Workers AI on that account,
        # in one read — a valid token for the wrong account fails here too.
        _probe(
            label,
            f"https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/models/search",
            headers=bearer, params={"per_page": 1},
        )
    elif provider in UNCHECKABLE:
        return


def catalog() -> list[dict]:
    """The "Bring your key" dropdown: paid companies first, then the free
    providers, then "Other" (added by the UI, since it has its own form)."""
    paid = [{"id": p, "label": cls.label, "kind": "paid"} for p, cls in BYOK_TRANSPORTS.items()]
    free = [
        {"id": p, "label": label, "kind": "free",
         "needs_account_id": p == "cloudflare", "checkable": p not in UNCHECKABLE}
        for p, label in FREE_PROVIDERS.items()
    ]
    return paid + free
