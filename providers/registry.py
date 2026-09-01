"""
providers/registry.py

Maps a provider name to its transport class, and hands back a ready-to-use
instance with the current API key pulled from the config store. Adding a
new provider later (NVIDIA NIM, Cerebras, whatever) means: write the
transport class, add one line here. Nothing else in the codebase changes.
"""

from config.store import config
from providers.base import Provider
from providers.cloudflare import CloudflareProvider
from providers.groq import GroqProvider
from providers.llm7 import LLM7Provider
from providers.mistral import MistralProvider
from providers.ollama_cloud import OllamaCloudProvider
from providers.openrouter import OpenRouterProvider

_TRANSPORTS: dict[str, type[Provider]] = {
    "openrouter": OpenRouterProvider,
    "groq": GroqProvider,
    "ollama_cloud": OllamaCloudProvider,
    "cloudflare": CloudflareProvider,
    "llm7": LLM7Provider,
    "mistral": MistralProvider,
    # "nvidia_nim": NvidiaNimProvider,  # add when built
}


class ProviderNotConfigured(Exception):
    pass


def get_provider(name: str) -> Provider:
    """Returns a ready-to-use provider instance, or raises if no key is set."""
    if name not in _TRANSPORTS:
        raise ValueError(f"Unknown provider: {name}")

    api_key = config.get_provider_key(name)
    if not api_key:
        raise ProviderNotConfigured(
            f"No API key set for '{name}' yet. Send it to the agent to configure it."
        )

    return _TRANSPORTS[name](api_key=api_key)


def get_dispatcher(context: str = "chat") -> tuple[Provider, str]:
    """
    Returns (provider_instance, model_name) for a dispatcher role's
    primary — same-model fallback (if configured) is via get_dispatcher_role.

    context="chat" (default): live, interactive messages.
    context="autonomous": the two GitHub Actions jobs (daily digest, daily
        opportunity scan) — uses a separate quota bucket on purpose, see
        config/store.py DEFAULTS for the reasoning.
    """
    role = get_dispatcher_role(context)
    return get_provider(role["provider"]), role["model"]


_ROLE_NAME_FOR_CONTEXT = {
    # "chat" maps to "normal_chat" (renamed 2026-09-01 from
    # "dispatcher_chat" — see config/store.py's migration), not the
    # f"dispatcher_{context}" formula every other context still follows.
    "chat": "normal_chat",
    "autonomous": "dispatcher_autonomous",
}


def get_dispatcher_role(context: str = "chat") -> dict:
    """Returns the raw role dict ({provider, model, fallback}) for a
    dispatcher role — callers that need to retry across the fallback
    chain (see dispatcher/chat.py) use this instead of get_dispatcher."""
    role_name = _ROLE_NAME_FOR_CONTEXT.get(context, f"dispatcher_{context}")
    role = config.get_role(role_name)
    if not role:
        raise ProviderNotConfigured(f"Role '{role_name}' not configured.")
    return role
