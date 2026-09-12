"""Embed text via Ollama nomic-embed-text."""

import logging
import math

import requests

from . import config
from .action_log import log as action_log

logger = logging.getLogger(__name__)


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two embedding vectors; 0.0 if either has no magnitude."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def build_text_to_embed(
    document_summary: str | None,
    primary_question_answered: str | None,
    chunk_text: str,
) -> str:
    """
    Build the single string we embed for a chunk: document summary + primary question + chunk body.
    Used at ingest and whenever summary, primary question, or chunk text changes (re-embed).
    """
    parts = []
    if document_summary and (s := (document_summary or "").strip()):
        parts.append(s)
    if primary_question_answered and (q := (primary_question_answered or "").strip()):
        parts.append("Primary question answered: " + q)
    parts.append((chunk_text or "").strip() or "")
    return "\n\n".join(parts)


def embed(texts: list[str], base_url: str | None = None, group: str = "_root") -> list[list[float]]:
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
        action_log("embed", model=model, num_inputs=len(texts), num_outputs=len(embs), dim=dim, group=group)
        return embs
    except requests.RequestException as e:
        action_log("embed_error", model=model, num_inputs=len(texts), error=str(e), group=group)
        logger.error("Embed request failed: %s", e)
        raise
