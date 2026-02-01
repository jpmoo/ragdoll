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


SUMMARY_OF_PREFIX = "SUMMARY of "  # leading line or bracketed: "[SUMMARY of filename: ...]"


def _strip_trailing_bracketed_summary(text: str) -> str:
    """
    Strip trailing "[SUMMARY of ...]" from end of text (after \\n\\n or at end).
    Returns chunk body without the bracketed summary.
    """
    if not text or not text.strip():
        return text
    t = text.rstrip()
    # Look for \n\n[SUMMARY of ...] at the end
    marker = "\n\n[SUMMARY of "
    if marker.upper() in t.upper():
        idx = t.upper().rfind(marker.upper())
        if idx >= 0 and t[idx:].rstrip().endswith("]"):
            return t[:idx].rstrip()
    return text


def _strip_leading_summary_line(text: str, filename: str | None = None) -> str:
    """
    Strip a leading summary line and return the chunk body. Removes either:
    - "SUMMARY of filename: ..." (first line if it starts with SUMMARY of )
    - "[filename] is a ... ." (first sentence when filename is provided)
    Otherwise return text unchanged.
    """
    if not text or not text.strip():
        return text
    t = text.strip()
    # Strip "SUMMARY of filename: sentence." (first line up to newline)
    if t.upper().startswith(SUMMARY_OF_PREFIX.upper()):
        idx = t.find(": ")
        if idx >= 0:
            after = t[idx + 2 :].lstrip()
            nn = after.find("\n\n")
            if nn >= 0:
                return after[nn + 2 :].lstrip()
            n = after.find("\n")
            if n >= 0:
                return after[n + 1 :].lstrip()
            return ""  # whole chunk was just the summary line
    # Strip "[filename] is a ... ." when filename given
    if filename:
        prefix = f"{filename} is a "
        if t.lower().startswith(prefix.lower()):
            rest = t[len(prefix) :].lstrip()
            for end in (".\n\n", ". ", ".\n"):
                idx = rest.find(end)
                if idx >= 0:
                    return rest[idx + len(end) :].lstrip()
            return ""
    return text


def _chunk_body_only(text: str, filename: str | None = None) -> str:
    """Return chunk body with any leading or trailing summary stripped."""
    t = _strip_leading_summary_line(text, filename)
    t = _strip_trailing_bracketed_summary(t)
    return t


def _body_for_document_text(text: str, filename: str | None) -> str:
    """Return chunk content to include in document text (strip existing summary)."""
    return _chunk_body_only(text, filename) or text


def backfill_one_source(conn, group: str, source_id: int, source_path: str) -> int:
    """
    For one source: build doc text from chunks, get summary, append bracketed
    summary to each chunk and re-embed. Returns number of chunks updated.
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
    # Append "[SUMMARY of filename: (model sentence)]" to each chunk and re-embed
    suffix = f"\n\n[SUMMARY of {filename}: {summary}]"
    texts_to_embed = []
    for c in chunks:
        body = _chunk_body_only(c["text"], filename)
        if not body:
            body = c["text"]
        new_text = body + suffix
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
