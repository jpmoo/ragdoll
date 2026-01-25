"""Embed text via Ollama nomic-embed-text."""

import logging

import requests

from . import config
from .action_log import log as action_log

logger = logging.getLogger(__name__)


def embed(texts: list[str], base_url: str | None = None) -> list[list[float]]:
    """
    Embed a list of texts. Returns list of embedding vectors.
    Batches into a single API call when possible (Ollama accepts input as array).
    """
    url = (base_url or config.OLLAMA_HOST).rstrip("/")
    model = config.EMBED_MODEL
    if not texts:
        return []

    try:
        r = requests.post(
            f"{url}/api/embed",
            json={"model": model, "input": texts},
            timeout=300,
        )
        r.raise_for_status()
        data = r.json()
        embs = data.get("embeddings", [])
        dim = len(embs[0]) if embs else None
        action_log("embed", model=model, num_inputs=len(texts), num_outputs=len(embs), dim=dim)
        return embs
    except requests.RequestException as e:
        action_log("embed_error", model=model, num_inputs=len(texts), error=str(e))
        logger.error("Embed request failed: %s", e)
        raise
