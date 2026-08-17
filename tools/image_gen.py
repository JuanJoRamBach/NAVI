"""
tools/image_gen.py

/create-image's backend: OVHcloud AI Endpoints' anonymous Stable
Diffusion XL tier. No API key, no account, no signup — verified directly
against the real endpoint before wiring in. Rate-limited to 2 requests/
minute per IP (confirmed by triggering it during verification), no
published daily cap since it needs no account to track one against.

Unlike /graph-data, no model/provider routing is involved at all — this
is a single fixed free endpoint, not something task_routing or the
Provider abstraction fits (there's no chat/tool-calling shape here, just
a prompt in, an image out).
"""

import requests

BASE_URL = "https://stable-diffusion-xl.endpoints.kepler.ai.cloud.ovh.net/api/text2image"


class ImageGenError(Exception):
    pass


def generate_image(prompt: str, negative_prompt: str = "") -> bytes:
    """Returns PNG/JPEG bytes. Raises ImageGenError on failure — including
    the 2-req/min rate limit, which callers should treat as a real,
    disclosed failure rather than something to silently retry."""
    payload = {"prompt": prompt}
    if negative_prompt:
        payload["negative_prompt"] = negative_prompt

    try:
        resp = requests.post(BASE_URL, json=payload, timeout=60)
    except requests.RequestException as e:
        raise ImageGenError(f"Image generation request failed: {e}")

    if resp.status_code == 429:
        raise ImageGenError("Rate limited (2 requests/minute on the free anonymous tier)")
    if resp.status_code >= 400:
        raise ImageGenError(f"Image generation error {resp.status_code}: {resp.text[:300]}")

    return resp.content
