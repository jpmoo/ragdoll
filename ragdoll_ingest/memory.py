"""Memory collection: MCP-only write and semantic search over structured notes.

Memories are stored in a dedicated "memory" group. Each memory has:
- topic, date, tags (stored in source summary as JSON)
- conclusion, reasoning, open_threads (each embedded separately)
- full text (embedded and stored)
"""

import json
import re
from typing import Any

from . import config
from .embedder import embed
from .storage import (
    _connect,
    add_chunks,
    clean_text,
    get_group_paths,
    init_db,
    set_source_summary,
)

MEMORY_GROUP = "memory"
MEMORY_SOURCE_TYPE = "memory"
# Chunk roles for memory sections (searchable separately and as full text)
MEMORY_SECTION_ROLES = ("conclusion", "reasoning", "open_threads", "full")


def parse_memory_text(raw: str) -> dict[str, Any] | None:
    """Parse memory text in the standard format. Returns dict with topic, date, tags, conclusion, reasoning, open_threads, full_text; or None if too sparse.

    Expected format:
        Topic: ...
        Date: ...
        Tags: tag1, tag2, ...
        Conclusion: ...
        Reasoning: ...
        Open threads: ...
    """
    if not (raw or "").strip():
        return None
    text = raw.strip()
    # Section headers (case-insensitive); capture rest of line or multiline body
    patterns = [
        (r"^\s*Topic:\s*(.+?)(?=\n\s*(?:Date|Tags|Conclusion|Reasoning|Open\s+threads):|\Z)", "topic", True),
        (r"^\s*Date:\s*(.+?)(?=\n\s*(?:Topic|Tags|Conclusion|Reasoning|Open\s+threads):|\Z)", "date", True),
        (r"^\s*Tags:\s*(.+?)(?=\n\s*(?:Topic|Date|Conclusion|Reasoning|Open\s+threads):|\Z)", "tags", True),
        (r"^\s*Conclusion:\s*(.+?)(?=\n\s*(?:Topic|Date|Tags|Reasoning|Open\s+threads):|\Z)", "conclusion", False),
        (r"^\s*Reasoning:\s*(.+?)(?=\n\s*(?:Topic|Date|Tags|Conclusion|Open\s+threads):|\Z)", "reasoning", False),
        (r"^\s*Open\s+threads:\s*(.+?)(?=\n\s*(?:Topic|Date|Tags|Conclusion|Reasoning):|\Z)", "open_threads", False),
    ]
    out: dict[str, Any] = {
        "topic": "",
        "date": "",
        "tags": [],
        "conclusion": "",
        "reasoning": "",
        "open_threads": "",
        "full_text": text,
    }
    for pattern, key, single_line in patterns:
        m = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
        if m:
            val = m.group(1).strip()
            if single_line:
                val = val.split("\n")[0].strip()
            if key == "tags":
                out[key] = [t.strip() for t in val.split(",") if t.strip()]
            else:
                out[key] = val
    if not out["topic"] and not out["conclusion"] and not out["reasoning"]:
        return None
    return out


def _memory_source_path(parsed: dict[str, Any]) -> str:
    """Unique source_path for this memory (within memory group)."""
    date = (parsed.get("date") or "").strip() or "undated"
    # Slug from topic: alphanumeric and hyphens
    topic = (parsed.get("topic") or "untitled").strip()
    slug = re.sub(r"[^\w\s-]", "", topic)[:50].strip()
    slug = re.sub(r"[-\s]+", "-", slug).lower() or "memory"
    return f"memory/{date}_{slug}.memory"


def _memory_doc_summary(parsed: dict[str, Any]) -> str:
    """JSON summary for source: topic, date, tags, full_text."""
    return json.dumps({
        "topic": parsed.get("topic") or "",
        "date": parsed.get("date") or "",
        "tags": parsed.get("tags") or [],
        "full_text": parsed.get("full_text") or "",
    }, ensure_ascii=False)


def store_memory(parsed: dict[str, Any]) -> dict[str, Any]:
    """Store one parsed memory in the memory group. Creates group/DB if needed. Returns summary dict with source_path, topic, date, chunks_created."""
    group = MEMORY_GROUP
    gp = get_group_paths(group)
    gp.group_dir.mkdir(parents=True, exist_ok=True)
    conn = _connect(group)
    try:
        init_db(conn)
        source_path = _memory_source_path(parsed)
        doc_summary = _memory_doc_summary(parsed)

        # Build (role, text) for each non-empty section in fixed order
        section_texts = [
            (parsed.get("conclusion") or "").strip(),
            (parsed.get("reasoning") or "").strip(),
            (parsed.get("open_threads") or "").strip(),
            (parsed.get("full_text") or "").strip(),
        ]
        role_text_pairs = [(MEMORY_SECTION_ROLES[i], t) for i, t in enumerate(section_texts) if t]
        if not role_text_pairs:
            return {"ok": False, "error": "No substantive sections to store"}

        to_embed = [t for _, t in role_text_pairs]
        embs = embed(to_embed, group=group)
        if len(embs) != len(to_embed):
            return {"ok": False, "error": "Embedding count mismatch"}

        chunks_list: list[dict[str, Any]] = [
            {"text": t, "embedding": embs[i], "chunk_role": role}
            for i, (role, t) in enumerate(role_text_pairs)
        ]

        add_chunks(conn, source_path, MEMORY_SOURCE_TYPE, chunks_list, doc_summary=doc_summary)
        conn.commit()
        return {
            "ok": True,
            "source_path": source_path,
            "topic": parsed.get("topic") or "",
            "date": parsed.get("date") or "",
            "chunks_created": len(chunks_list),
        }
    finally:
        conn.close()


def parse_memory_summary(summary: str | None) -> dict[str, Any] | None:
    """Parse source summary JSON from a memory source. Returns dict with topic, date, tags, full_text or None."""
    if not (summary or "").strip():
        return None
    try:
        return json.loads(summary)
    except (json.JSONDecodeError, TypeError):
        return None
