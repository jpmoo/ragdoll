"""Semantic chunking with LLM-assisted splitting for long paragraphs."""

import json
import logging
import re
from typing import Any

from . import config
from .action_log import log as action_log

logger = logging.getLogger(__name__)

# Rough: ~4 chars per token for English
CHARS_PER_TOKEN = 4


def _tokens_approx(text: str) -> int:
    return max(1, len(text) // CHARS_PER_TOKEN)


def _split_blocks(text: str) -> list[str]:
    """Split into paragraphs/blocks (by double newline or single when very long)."""
    text = text.strip()
    if not text:
        return []
    blocks = re.split(r"\n\s*\n", text)
    out: list[str] = []
    for b in blocks:
        b = b.strip()
        if not b:
            continue
        # If a block is huge, split by single newlines into smaller pieces
        if _tokens_approx(b) > config.MAX_CHUNK_TOKENS * 2:
            for line in b.split("\n"):
                line = line.strip()
                if line:
                    out.append(line)
        else:
            out.append(b)
    return out


def _llm_split_long(text: str, ollama_url: str) -> list[str]:
    """Use LLM to split a long block into 2-3 semantic chunks. Falls back to mid-split on error."""
    # Truncate if still too long for context (leave room for prompt + response)
    max_in = (config.MAX_CHUNK_TOKENS * 3) * CHARS_PER_TOKEN  # ~3x max chunk
    if len(text) > max_in:
        text = text[:max_in] + "\n[...truncated...]"

    prompt = (
        'Split the following text into 2 or 3 coherent semantic segments. '
        'Each segment should be self-contained. '
        'Return ONLY valid JSON in this exact format, no other text:\n'
        '{"chunks": ["first segment text", "second segment text"]}\n\n'
        'Text to split:\n'
    ) + text

    try:
        import requests

        r = requests.post(
            f"{ollama_url.rstrip('/')}/api/generate",
            json={
                "model": config.CHUNK_MODEL,
                "prompt": prompt,
                "stream": False,
                "format": "json",
            },
            timeout=120,
        )
        r.raise_for_status()
        data = r.json()
        resp = (data.get("response") or "").strip()
        # Handle markdown code block
        if "```" in resp:
            m = re.search(r"```(?:json)?\s*([\s\S]*?)```", resp)
            if m:
                resp = m.group(1).strip()
        obj: dict[str, Any] = json.loads(resp)
        chunks = obj.get("chunks")
        if isinstance(chunks, list) and all(isinstance(c, str) for c in chunks) and chunks:
            out = [c.strip() for c in chunks if c.strip()]
            action_log("chunk_llm", model=config.CHUNK_MODEL, input_len=len(text), num_chunks=len(out), fallback=False)
            return out
    except Exception as e:
        logger.warning("LLM split failed, using mid-split: %s", e)

    # Fallback: split near the middle at a sentence or newline
    mid = len(text) // 2
    for sep in (". ", ".\n", "\n", " "):
        i = text.find(sep, mid - 200, mid + 200)
        if i != -1:
            out = [text[: i + len(sep)].strip(), text[i + len(sep) :].strip()]
            action_log("chunk_llm", model=config.CHUNK_MODEL, input_len=len(text), num_chunks=len(out), fallback=True)
            return out
    out = [text[:mid].strip(), text[mid:].strip()]
    action_log("chunk_llm", model=config.CHUNK_MODEL, input_len=len(text), num_chunks=len(out), fallback=True)
    return out


def chunk_text(text: str, ollama_url: str | None = None) -> list[str]:
    """
    Split text into semantic chunks. Uses paragraph boundaries and, for very long
    paragraphs, llama3.2 to find semantic split points.
    """
    url = ollama_url or config.OLLAMA_HOST
    blocks = _split_blocks(text)
    if not blocks:
        return []

    chunks: list[str] = []
    current: list[str] = []
    current_tokens = 0

    for i, block in enumerate(blocks):
        block_tokens = _tokens_approx(block)

        if block_tokens > config.MAX_CHUNK_TOKENS:
            # Flush current before handling long block
            if current:
                chunks.append("\n\n".join(current))
                current = []
                current_tokens = 0
            sub = _llm_split_long(block, url)
            for s in sub:
                if _tokens_approx(s) <= config.MAX_CHUNK_TOKENS:
                    chunks.append(s)
                else:
                    # recurse would be safe but could be slow; mid-split again
                    chunks.extend(_llm_split_long(s, url))
            continue

        if current_tokens + block_tokens > config.TARGET_CHUNK_TOKENS and current:
            chunks.append("\n\n".join(current))
            # Optional: overlap by keeping last N sentences of previous chunk
            current = []
            current_tokens = 0

        current.append(block)
        current_tokens += block_tokens

    if current:
        chunks.append("\n\n".join(current))

    return [c for c in chunks if c.strip()]
