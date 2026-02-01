#!/usr/bin/env python3
"""
Backfill one-sentence document summaries onto existing RAG samples.

Iterates over all collections (groups), then over each source in the collection's
DB. For each source, builds document text from current chunk texts, calls the
LLM to produce a "[docname].[extension] is a ..." summary, then prepends that
sentence to every chunk for that source and re-embeds.

Run from project root (so ragdoll_ingest is importable), e.g.:
  python backfill_doc_summaries.py

Uses the same config as ingest (env.ragdoll / RAGDOLL_*). Requires Ollama
(CHUNK_MODEL, EMBED_MODEL) to be available.
"""

import logging
import re
import sys
from pathlib import Path

# Run from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent))

from ragdoll_ingest import config
from ragdoll_ingest.embedder import embed
from ragdoll_ingest.interpreters import summarize_document
from ragdoll_ingest.storage import (
    _connect,
    _list_sync_groups,
    get_chunks_for_source,
    init_db,
    list_sources,
    update_chunk_text,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def _strip_leading_doc_summary(text: str, filename: str) -> str:
    """
    If text starts with "[filename] is a ... ." (and optional \\n\\n), return
    the rest (chunk body). Otherwise return text unchanged.
    """
    if not text or not filename:
        return text
    prefix = f"{filename} is a "
    if not text.strip().lower().startswith(prefix.lower()):
        return text
    rest = text[len(prefix) :].lstrip()
    # First sentence ends at ". " or ".\n"
    for end in (".\n\n", ". ", ".\n"):
        idx = rest.find(end)
        if idx >= 0:
            return rest[idx + len(end) :].lstrip()
    # No period found; rest is the single sentence, use empty body for prepend
    return ""


def _body_for_document_text(text: str, filename: str) -> str:
    """Return chunk content to include in document text (strip existing summary line)."""
    stripped = _strip_leading_doc_summary(text, filename)
    return stripped if stripped else text


def backfill_one_source(conn, group: str, source_id: int, source_path: str) -> int:
    """
    For one source: build doc text from chunks, get summary, prepend to each
    chunk and re-embed. Returns number of chunks updated.
    """
    init_db(conn)
    chunks = get_chunks_for_source(conn, source_id)
    if not chunks:
        return 0
    filename = Path(source_path).name
    # Build document text from chunk bodies (strip any existing summary line)
    parts = [_body_for_document_text(c["text"], filename) for c in chunks]
    document_text = "\n\n".join(p for p in parts if p.strip())
    if not document_text.strip():
        logger.warning("  [%s] source_id=%s no usable text, skipping", group, source_id)
        return 0
    summary = summarize_document(document_text, group=group, filename=filename)
    if not summary:
        logger.warning("  [%s] source_id=%s summary failed, skipping", group, source_id)
        return 0
    # Prepend summary to each chunk and re-embed
    texts_to_embed = []
    for c in chunks:
        body = _strip_leading_doc_summary(c["text"], filename)
        if not body:
            body = c["text"]
        new_text = f"{summary}\n\n{body}"
        texts_to_embed.append(new_text)
    try:
        embs = embed(texts_to_embed, group=group)
    except Exception as e:
        logger.warning("  [%s] source_id=%s embed failed: %s", group, source_id, e)
        return 0
    if len(embs) != len(chunks):
        logger.warning("  [%s] source_id=%s embed count mismatch", group, source_id)
        return 0
    for c, new_text, emb in zip(chunks, texts_to_embed, embs):
        update_chunk_text(conn, c["id"], new_text, emb)
    return len(chunks)


def main() -> None:
    groups = _list_sync_groups()
    if not groups:
        logger.info("No collections found (no ragdoll.db under DATA_DIR).")
        return
    logger.info("Collections: %s", groups)
    total_sources = 0
    total_chunks = 0
    for group in sorted(groups):
        conn = _connect(group)
        try:
            sources = list_sources(conn)
            if not sources:
                logger.info("[%s] no sources", group)
                continue
            logger.info("[%s] %d source(s)", group, len(sources))
            for source_id, source_path, count in sources:
                updated = backfill_one_source(conn, group, source_id, source_path)
                if updated:
                    conn.commit()
                    total_sources += 1
                    total_chunks += updated
                    logger.info("  [%s] %s -> %d chunks updated", group, source_path, updated)
        finally:
            conn.close()
    logger.info("Done: %d sources, %d chunks updated across all collections.", total_sources, total_chunks)


if __name__ == "__main__":
    main()
