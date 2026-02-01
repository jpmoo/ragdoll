#!/usr/bin/env python3
"""
Backfill one-sentence document summaries onto existing RAG samples.

Iterates over all collections (groups), then over each source in the collection's
DB. For each source, builds document text from chunk bodies, calls the LLM for
a short summary, then appends "[SUMMARY of filename: ...]" to each chunk.

At start you choose:
  do over  - Strip all existing trailing [SUMMARY of ...] and re-run (full redo).
  continue - Leave chunks that already have a summary untouched; only update
             chunks that don't, picking up where the script last left off.

Run from project root (so ragdoll_ingest is importable), e.g.:
  python backfill_doc_summaries.py

Uses the same config as ingest (env.ragdoll / RAGDOLL_*). Requires Ollama
(CHUNK_MODEL, EMBED_MODEL) to be available.
"""

import logging
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


def _strip_trailing_bracketed_summary(text: str) -> str:
    """
    Strip trailing "[SUMMARY of ...]" from end of text (after \\n\\n or at end).
    Returns chunk body without the bracketed summary.
    """
    if not text or not text.strip():
        return text
    t = text.rstrip()
    marker = "\n\n[SUMMARY of "
    if marker.upper() in t.upper():
        idx = t.upper().rfind(marker.upper())
        if idx >= 0 and t[idx:].rstrip().endswith("]"):
            return t[:idx].rstrip()
    return text


def _has_trailing_bracketed_summary(text: str) -> bool:
    """True if text ends with \\n\\n[SUMMARY of ...]."""
    if not text or not text.strip():
        return False
    t = text.rstrip()
    marker = "\n\n[SUMMARY of "
    if marker.upper() not in t.upper():
        return False
    idx = t.upper().rfind(marker.upper())
    return idx >= 0 and t[idx:].rstrip().endswith("]")


def _chunk_body_only(text: str) -> str:
    """Return chunk body with trailing [SUMMARY of ...] stripped."""
    return _strip_trailing_bracketed_summary(text)


def backfill_one_source(
    conn, group: str, source_id: int, source_path: str, do_over: bool
) -> int:
    """
    For one source: build doc text from chunk bodies, get summary, append
    bracketed summary to each chunk (or only chunks that don't have one if
    continue mode). Returns number of chunks updated.
    """
    init_db(conn)
    chunks = get_chunks_for_source(conn, source_id)
    if not chunks:
        return 0
    filename = Path(source_path).name
    # Build document text from all chunk bodies (strip trailing summary)
    parts = [_chunk_body_only(c["text"]) for c in chunks]
    document_text = "\n\n".join(p for p in parts if p.strip())
    if not document_text.strip():
        logger.warning("  [%s] source_id=%s no usable text, skipping", group, source_id)
        return 0
    summary = summarize_document(document_text, group=group, filename=filename)
    if not summary:
        logger.warning("  [%s] source_id=%s summary failed, skipping", group, source_id)
        return 0
    suffix = f"\n\n[SUMMARY of {filename}: {summary}]"

    if do_over:
        # Update all chunks: strip trailing summary, append new one
        to_update = [(c, _chunk_body_only(c["text"]) or c["text"]) for c in chunks]
    else:
        # Continue: only chunks that don't already have trailing summary
        to_update = []
        for c in chunks:
            if _has_trailing_bracketed_summary(c["text"]):
                continue
            to_update.append((c, c["text"]))
        if not to_update:
            return 0

    texts_to_embed = [body + suffix for _, body in to_update]
    try:
        embs = embed(texts_to_embed, group=group)
    except Exception as e:
        logger.warning("  [%s] source_id=%s embed failed: %s", group, source_id, e)
        return 0
    if len(embs) != len(to_update):
        logger.warning("  [%s] source_id=%s embed count mismatch", group, source_id)
        return 0
    for (c, _), new_text, emb in zip(to_update, texts_to_embed, embs):
        update_chunk_text(conn, c["id"], new_text, emb)
    return len(to_update)


def main() -> None:
    print(
        "Do over (strip all existing [SUMMARY of ...] and re-run) or Continue "
        "(skip chunks that already have summary, pick up where left off)?"
    )
    choice = input("[do over / continue]: ").strip().lower()
    if choice in ("do over", "doover", "do_over", "over", "redo"):
        do_over = True
    elif choice in ("continue", "cont", "c"):
        do_over = False
    else:
        logger.error("Choose 'do over' or 'continue'.")
        return
    logger.info("Mode: %s", "do over" if do_over else "continue")

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
                updated = backfill_one_source(conn, group, source_id, source_path, do_over)
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
