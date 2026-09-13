"""One way to ask an Ollama model for text.

Every text-generation call goes through here so thinking models (Qwen3 and similar) work. Left to themselves they
put their answer in a separate "thinking" field and leave "response" empty, which looks to a caller like the model
returned nothing: chunking comes back empty, query expansion silently falls back to the raw prompt, and so on.
"""

import logging
from typing import Any

import requests

from . import config

logger = logging.getLogger(__name__)


def generate(
    prompt: str,
    model: str,
    *,
    url: str | None = None,
    timeout: float | None = None,
    json_format: bool = False,
    options: dict[str, Any] | None = None,
) -> str:
    """The model's reply text, or "" if it returned nothing.

    Network and HTTP errors are raised exactly as requests.post raises them, so callers keep their own handling.
    """
    base = (url or config.OLLAMA_HOST or "").rstrip("/")
    payload: dict[str, Any] = {"model": model, "prompt": prompt, "stream": False, "think": False}
    if json_format:
        payload["format"] = "json"
    if options:
        payload["options"] = options
    timeout = timeout or config.CHUNK_LLM_TIMEOUT

    r = requests.post(f"{base}/api/generate", json=payload, timeout=timeout)
    if r.status_code >= 400 and "think" in r.text.lower():
        # Ollama versions from before thinking support reject the field; their models can't think anyway
        logger.info("Ollama rejected think=false for %s; retrying without it", model)
        payload.pop("think")
        r = requests.post(f"{base}/api/generate", json=payload, timeout=timeout)
    r.raise_for_status()

    data = r.json()
    text = (data.get("response") or "").strip()
    if not text:
        text = (data.get("thinking") or "").strip()
        if text:
            logger.info("Model %s answered in the thinking field", model)
    return text
