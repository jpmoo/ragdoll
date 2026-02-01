#!/usr/bin/env python3
"""
Strip all trailing "[SUMMARY of ...]" blocks from every chunk across all collections.

Removes every such block at the end of each sample, including multiple in a row.
Updates chunk text and re-embeds. No LLM calls.

Run from project root (so ragdoll_ingest is importable), e.g.:
  python strip_summaries.py

Uses the same config as ingest (env.ragdoll / RAGDOLL_*). Requires Ollama
(EMBED_MODEL) for re-embedding.
"""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ragdoll_ingest.embedder import embed
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

MARKER = "[SUMMARY of "


def _strip_one_trailing(text: str) -> str:
    """Remove the last trailing '[SUMMARY of ...]' block. Returns stripped text."""
    if not text or not text.strip():
        return text
    t = text.rstrip()
    idx = t.upper().rfind(MARKER.upper())
    if idx < 0:
        return text
    last_rb = t.rfind("]")
    if last_rb < idx:
        return text
    return t[:idx].rstrip()


def strip_all_trailing_summaries(text: str) -> str:
    """Remove all trailing '[SUMMARY of ...]' blocks (one or many). Returns body only."""
    if not text or not text.strip():
        return text
    current = text.strip()
    while True:
        next_ = _strip_one_trailing(current)
        if next_ == current:
            break
        current = next_
    return current


def main() -> None:
    groups = _list_sync_groups()
    if not groups:
        logger.info("No collections found (no ragdoll.db under DATA_DIR).")
        return
    logger.info("Collections: %s", groups)
    total_chunks = 0
    total_updated = 0
    for group in sorted(groups):
        conn = _connect(group)
        try:
            init_db(conn)
            sources = list_sources(conn)
            if not sources:
                logger.info("[%s] no sources", group)
                continue
            logger.info("[%s] %d source(s)", group, len(sources))
            for source_id, source_path, _ in sources:
                chunks = get_chunks_for_source(conn, source_id)
                if not chunks:
                    continue
                to_update = []
                for c in chunks:
                    total_chunks += 1
                    body = strip_all_trailing_summaries(c["text"])
                    if body != c["text"]:
                        to_update.append((c["id"], body))
                if not to_update:
                    continue
                bodies = [body for _, body in to_update]
                try:
                    embs = embed(bodies, group=group)
                except Exception as e:
                    logger.warning("  [%s] source_id=%s embed failed: %s", group, source_id, e)
                    continue
                if len(embs) != len(bodies):
                    logger.warning("  [%s] source_id=%s embed count mismatch", group, source_id)
                    continue
                for (cid, body), emb in zip(to_update, embs):
                    update_chunk_text(conn, cid, body, emb)
                conn.commit()
                total_updated += len(to_update)
                logger.info("  [%s] %s -> %d chunks stripped", group, source_path, len(to_update))
        finally:
            conn.close()
    logger.info("Done: %d chunks scanned, %d updated across all collections.", total_chunks, total_updated)


if __name__ == "__main__":
    main()
