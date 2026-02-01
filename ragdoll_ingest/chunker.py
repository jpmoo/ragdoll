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

# Max characters to send to LLM in one request for semantic boundary detection (leave room for prompt + response)
SEMANTIC_CHUNK_WINDOW = 10000


def _clean_for_chunking(text: str) -> str:
    """Strip non-meaning-bearing characters: links, markdown formatting, normalize whitespace. Keeps substantive text."""
    if not text or not text.strip():
        return ""
    # Replace markdown links [text](url) with just the link text
    text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", text)
    # Remove bare URLs (http/https)
    text = re.sub(r"https?://\S+", " ", text)
    # Remove **bold** and __bold__ (keep inner text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)
    text = re.sub(r"__([^_]+)__", r"\1", text)
    # Remove *italic* and _italic_ (keep inner text; avoid breaking mid-word)
    text = re.sub(r"\*([^*]+)\*", r"\1", text)
    text = re.sub(r"(?<!\w)_([^_]+)_(?!\w)", r"\1", text)
    # Remove # at start of line (markdown headers) but keep the rest of the line
    text = re.sub(r"^#+\s*", "", text, flags=re.MULTILINE)
    # Normalize whitespace: collapse multiple spaces, normalize newlines to \n
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\r\n?", "\n", text)
    text = re.sub(r"\n\s*\n\s*\n+", "\n\n", text)
    return text.strip()


def _snap_to_boundaries(text: str, start: int, end: int) -> tuple[int, int]:
    """Snap start/end to paragraph or sentence boundaries so we don't cut mid-sentence."""
    n = len(text)
    if n == 0:
        return 0, 0
    start = max(0, min(start, n - 1))
    end = max(start, min(end, n))
    # Snap start backward to paragraph start (\n\n) or sentence start (after . or \n)
    for i in range(start, -1, -1):
        if i == 0:
            break
        if i >= 2 and text[i - 2 : i] == "\n\n":
            start = i
            break
        if i >= 2 and text[i - 1] in " \t" and text[i - 2] in ".!?":
            start = i
            break
        if text[i - 1] == "\n":
            start = i
            break
    else:
        start = 0
    # Snap end forward to paragraph end or sentence end
    for i in range(end, n + 1):
        if i >= n:
            end = n
            break
        if i <= n - 2 and text[i : i + 2] == "\n\n":
            end = i + 2
            break
        if i < n and text[i] in ".!?" and (i + 1 >= n or text[i + 1] in " \t\n"):
            end = i + 1
            break
        if i < n and text[i] == "\n":
            end = i + 1
            break
    else:
        end = n
    return start, end


def _get_semantic_chunk_texts_one(
    window_text: str, ollama_url: str, group: str = "_root", window_offset: int = 0
) -> list[tuple[str, int]]:
    """Ask LLM for semantic chunks as text. Returns list of (chunk_text, start_offset) by locating each chunk in window_text."""
    prompt = (
        "Split the following text into coherent semantic chunks. Prefer longer, self-contained sections where possible. "
        "Each chunk must be a self-contained unit that includes the full content: if a section has a heading (e.g. 'Key phases:'), "
        "include that heading AND all paragraphs and bullet lists under it in the same chunk. "
        "Never output a chunk that is only a heading or title without the content that follows. "
        "Copy the exact text for each chunk; do not paraphrase or omit. "
        "Return ONLY valid JSON in this exact format, no other text:\n"
        '{"chunks": ["first chunk full text", "second chunk full text", ...]}\n\n'
        "Text:\n\n"
    ) + window_text

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
            timeout=config.CHUNK_LLM_TIMEOUT,
        )
        r.raise_for_status()
        resp = (r.json().get("response") or "").strip()
        if not resp:
            return []
        if "```" in resp:
            m = re.search(r"```(?:json)?\s*([\s\S]*?)```", resp)
            if m:
                resp = m.group(1).strip()
        obj: dict[str, Any] = json.loads(resp)
        chunks_raw = obj.get("chunks")
        if not isinstance(chunks_raw, list):
            return []
        out: list[tuple[str, int]] = []
        search_start = 0
        for item in chunks_raw:
            if not isinstance(item, str):
                continue
            chunk_str = item.strip()
            if not chunk_str:
                continue
            # Locate this chunk in the window text (from search_start) to get start_offset for page mapping
            idx = window_text.find(chunk_str, search_start)
            if idx >= 0:
                start_offset = window_offset + idx
                search_start = idx + len(chunk_str)
            else:
                start_offset = window_offset + search_start
            out.append((chunk_str, start_offset))
        if not out:
            return []
        action_log(
            "chunk_semantic",
            model=config.CHUNK_MODEL,
            input_len=len(window_text),
            num_chunks=len(out),
            group=group,
        )
        return out
    except Exception as e:
        logger.warning("Semantic chunk (text) request failed: %s", e)
        return []


# Min window size when breaking at paragraph so we don't create tiny windows
_SEMANTIC_WINDOW_MIN = 2000


def _get_semantic_chunk_texts(
    full_text: str, ollama_url: str, group: str = "_root"
) -> list[tuple[str, int]]:
    """Get semantic chunks as text (and start_offset for page mapping), using windowing if too long.
    Windows are cut at paragraph boundaries (\\n\\n) when possible to avoid splitting a section heading from its content.
    """
    if not full_text.strip():
        return []
    if len(full_text) <= SEMANTIC_CHUNK_WINDOW:
        return _get_semantic_chunk_texts_one(full_text, ollama_url, group, window_offset=0)
    all_chunks: list[tuple[str, int]] = []
    offset = 0
    while offset < len(full_text):
        window_end = min(offset + SEMANTIC_CHUNK_WINDOW, len(full_text))
        # Prefer to break at last paragraph boundary so we don't cut mid-section (e.g. heading from bullets)
        break_at = window_end
        if window_end < len(full_text):
            search_region = full_text[offset:window_end]
            last_pp = search_region.rfind("\n\n")
            if last_pp >= _SEMANTIC_WINDOW_MIN:
                break_at = offset + last_pp
        window = full_text[offset:break_at]
        if not window.strip():
            offset = break_at
            continue
        part = _get_semantic_chunk_texts_one(window, ollama_url, group, window_offset=offset)
        all_chunks.extend(part)
        offset = break_at
    return all_chunks


def chunk_text_semantic(
    full_text: str,
    ollama_url: str | None = None,
    group: str = "_root",
    pre_cleaned: bool = False,
) -> list[tuple[str, int]]:
    """
    Clean full text (unless pre_cleaned=True), ask LLM to output chunk text directly (no indices).
    Locate each chunk in cleaned text to get start_offset for page mapping.
    Returns list of (chunk_string, start_offset).
    """
    url = ollama_url or config.OLLAMA_HOST
    cleaned = full_text if pre_cleaned else _clean_for_chunking(full_text)
    if not cleaned.strip():
        return []
    result = _get_semantic_chunk_texts(cleaned, url, group)
    if not result:
        # Fallback: one chunk if short enough, else legacy chunk_text
        if _tokens_approx(cleaned) <= config.MAX_CHUNK_TOKENS:
            return [(cleaned, 0)]
        legacy_chunks = chunk_text(cleaned, url, group)
        if not legacy_chunks:
            return [(cleaned.strip(), 0)] if cleaned.strip() else []
        return [(c, 0) for c in legacy_chunks]
    return result


def _tokens_approx(text: str) -> int:
    return max(1, len(text) // CHARS_PER_TOKEN)


def _looks_like_section_header(block: str) -> bool:
    """True if block is a short title/header that should stay with the following content."""
    block = block.strip()
    if not block:
        return False
    lines = [ln.strip() for ln in block.splitlines() if ln.strip()]
    # Markdown-style header (## Key Terms or # Overview)
    if len(lines) == 1 and re.match(r"^#+\s*\S", lines[0]):
        return True
    # Single line, short (e.g. "Key Terms", "Overview")
    if len(lines) == 1 and len(lines[0]) <= 80:
        line = lines[0]
        if line.endswith(":"):
            return True
        # Common section titles (case-insensitive)
        if line.lower() in (
            "key terms", "overview", "summary", "introduction", "background",
            "key concepts", "key points", "glossary", "definitions", "references",
        ):
            return True
        # Short line that looks like a title (no sentence-ending punctuation)
        if len(line) <= 60 and "." not in line and "!" not in line and "?" not in line:
            return True
    # Two short lines (e.g. "Key Terms" + blank or subtitle)
    if len(lines) == 2 and all(len(ln) <= 60 for ln in lines):
        return True
    return False


def _merge_header_blocks(blocks: list[str]) -> list[str]:
    """Merge header-like blocks with the next block so section headers stay with their content."""
    if not blocks:
        return []
    out: list[str] = []
    i = 0
    while i < len(blocks):
        b = blocks[i]
        if _looks_like_section_header(b) and i + 1 < len(blocks):
            # Merge header with next block
            out.append(b.strip() + "\n\n" + blocks[i + 1].strip())
            i += 2
            continue
        out.append(b)
        i += 1
    return out


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
    return _merge_header_blocks(out)


def _llm_split_long(text: str, ollama_url: str, group: str = "_root") -> list[str]:
    """Use LLM to split a long block into 2-3 semantic chunks. Falls back to mid-split on error."""
    # Truncate if still too long for context (leave room for prompt + response)
    # Ollama default context is 4096 tokens; use ~3500 tokens max to be safe (prompt + response overhead)
    max_in_tokens = 3500
    max_in = max_in_tokens * CHARS_PER_TOKEN  # ~3500 tokens = ~14000 chars
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
            timeout=config.CHUNK_LLM_TIMEOUT,
        )
        r.raise_for_status()
        data = r.json()
        resp = (data.get("response") or "").strip()
        
        # Check for empty response before parsing
        if not resp:
            logger.warning("LLM returned empty response (input_len=%d chars, ~%d tokens), using mid-split", 
                          len(text), _tokens_approx(text))
            # Fall through to fallback
        else:
            # Handle markdown code block
            if "```" in resp:
                m = re.search(r"```(?:json)?\s*([\s\S]*?)```", resp)
                if m:
                    resp = m.group(1).strip()
            
            try:
                obj: dict[str, Any] = json.loads(resp)
                chunks = obj.get("chunks")
                
                # Validate chunks structure
                if isinstance(chunks, list) and chunks:
                    # Check all items are strings
                    if all(isinstance(c, str) for c in chunks):
                        out = [c.strip() for c in chunks if c.strip()]
                        if out:  # Ensure we have at least one non-empty chunk
                            action_log("chunk_llm", model=config.CHUNK_MODEL, input_len=len(text), num_chunks=len(out), fallback=False, group=group)
                            return out
                    else:
                        logger.warning("LLM returned chunks with non-string items (input_len=%d chars, ~%d tokens), using mid-split", 
                                      len(text), _tokens_approx(text))
                else:
                    logger.warning("LLM returned invalid chunks format (expected list, got %s) (input_len=%d chars, ~%d tokens), using mid-split", 
                                  type(chunks).__name__, len(text), _tokens_approx(text))
            except json.JSONDecodeError as e:
                logger.warning("LLM returned invalid JSON (input_len=%d chars, ~%d tokens): %s, using mid-split", 
                              len(text), _tokens_approx(text), e)
                # Fall through to fallback
    except requests.exceptions.Timeout as e:
        logger.warning("LLM split timed out after %d seconds (input_len=%d chars, ~%d tokens), using mid-split", 
                      config.CHUNK_LLM_TIMEOUT, len(text), _tokens_approx(text))
        # Fall through to fallback
    except requests.exceptions.RequestException as e:
        logger.warning("LLM request failed (input_len=%d chars, ~%d tokens): %s, using mid-split", 
                      len(text), _tokens_approx(text), e)
        # Fall through to fallback
    except Exception as e:
        logger.warning("LLM split failed unexpectedly (input_len=%d chars, ~%d tokens): %s, using mid-split", 
                      len(text), _tokens_approx(text), e)

    # Fallback: split near the middle at a sentence or newline
    mid = len(text) // 2
    for sep in (". ", ".\n", "\n", " "):
        i = text.find(sep, mid - 200, mid + 200)
        if i != -1:
            out = [text[: i + len(sep)].strip(), text[i + len(sep) :].strip()]
            action_log("chunk_llm", model=config.CHUNK_MODEL, input_len=len(text), num_chunks=len(out), fallback=True, group=group)
            return out
    out = [text[:mid].strip(), text[mid:].strip()]
    action_log("chunk_llm", model=config.CHUNK_MODEL, input_len=len(text), num_chunks=len(out), fallback=True, group=group)
    return out


def chunk_text(text: str, ollama_url: str | None = None, group: str = "_root") -> list[str]:
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
            sub = _llm_split_long(block, url, group)
            for s in sub:
                if _tokens_approx(s) <= config.MAX_CHUNK_TOKENS:
                    chunks.append(s)
                else:
                    # recurse would be safe but could be slow; mid-split again
                    chunks.extend(_llm_split_long(s, url, group))
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
