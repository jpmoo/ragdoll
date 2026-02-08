#!/usr/bin/env python3
"""
Re-embed all chunks in every collection using the current rule: document summary + primary question + chunk body.

Use after changing the embed formula (e.g. to summary + primary_question + body only). Safe to run multiple times.

Run from project root (ragdoll_ingest on path). Uses RAGDOLL env/config for DATA_DIR and Ollama for embedding.
"""

from __future__ import annotations

import sys

from ragdoll_ingest.embedder import build_text_to_embed, embed
from ragdoll_ingest.storage import (
    _connect,
    _list_sync_groups,
    get_chunks_for_source,
    init_db,
    list_sources,
    update_chunk_embedding,
)

BATCH_SIZE = 100  # Max chunks per embed API call


def rembed_group(group: str) -> tuple[int, int]:
    """Re-embed all chunks in one group. Returns (sources_processed, chunks_updated)."""
    conn = _connect(group)
    try:
        init_db(conn)
        raw = list_sources(conn)
        sources_processed = 0
        chunks_updated = 0
        for source_id, _source_path, _count, summary in raw:
            summary = (summary or "").strip() or ""
            chunks = get_chunks_for_source(conn, source_id)
            if not chunks:
                continue
            sources_processed += 1
            to_embed_list = [
                build_text_to_embed(
                    summary,
                    c.get("primary_question_answered"),
                    c.get("text") or "",
                )
                for c in chunks
            ]
            for i in range(0, len(to_embed_list), BATCH_SIZE):
                batch_texts = to_embed_list[i : i + BATCH_SIZE]
                batch_chunks = chunks[i : i + BATCH_SIZE]
                embs = embed(batch_texts, group=group)
                if len(embs) != len(batch_chunks):
                    raise RuntimeError(
                        f"Group {group} source_id {source_id}: embed returned {len(embs)}, expected {len(batch_chunks)}"
                    )
                for c, emb in zip(batch_chunks, embs):
                    update_chunk_embedding(conn, c["id"], emb)
                    chunks_updated += 1
        conn.commit()
        return sources_processed, chunks_updated
    finally:
        conn.close()


def main() -> int:
    groups = _list_sync_groups()
    if not groups:
        print("No RAG groups found (no ragdoll.db under DATA_DIR).")
        return 0
    total_sources = 0
    total_chunks = 0
    for group in sorted(groups):
        try:
            n_sources, n_chunks = rembed_group(group)
            total_sources += n_sources
            total_chunks += n_chunks
            if n_chunks:
                print(f"{group}: {n_sources} sources, {n_chunks} chunks re-embedded")
        except Exception as e:
            print(f"{group}: ERROR {e}", file=sys.stderr)
            raise
    if total_chunks:
        print(f"Total: {total_sources} sources, {total_chunks} chunks re-embedded")
    else:
        print("No chunks found to re-embed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
