#!/usr/bin/env python3
"""
Migrate document summaries from chunk text into the source's summary field.

- For each source in each collection: look for "[SUMMARY of filename: ...]" in any chunk text.
- If found: extract the summary, remove that bracketed block from the chunk, re-embed and save
  the chunk, and set the source's summary to the extracted text.
- If no chunk had a summary: combine chunk texts, call the LLM to generate a 1–3 sentence
  summary, and set the source's summary.

Run from project root: python migrate_summaries_to_sources.py
Uses env.ragdoll / RAGDOLL_* (Ollama CHUNK_MODEL, embedder).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ragdoll_ingest import config
from ragdoll_ingest.embedder import embed
from ragdoll_ingest.interpreters import summarize_document
from ragdoll_ingest.storage import (
    _connect,
    _list_sync_groups,
    get_chunk_by_id,
    get_chunks_for_source,
    init_db,
    list_sources,
    set_source_summary,
    update_chunk_full,
)

_MARKER = "[SUMMARY of "


def _strip_summary_from_text(text: str) -> tuple[str, str | None]:
    """Remove trailing [SUMMARY of ...: ...] from text. Returns (stripped_text, summary or None)."""
    if not text or not text.strip():
        return (text or "", None)
    t = text.rstrip()
    idx = t.upper().rfind(_MARKER.upper())
    if idx < 0:
        return (text, None)
    after_marker = t[idx + len(_MARKER) :]
    colon = after_marker.find(": ")
    if colon < 0:
        return (text, None)
    summary = after_marker[colon + 2 :].rstrip()
    if summary.endswith("]"):
        summary = summary[:-1].rstrip()
    stripped = t[:idx].rstrip()
    return (stripped, summary if summary else None)


def _text_to_embed(text: str, concept: str = "", decision_context: str = "", primary_question_answered: str = "", key_signals: list | None = None, chunk_role: str = "") -> str:
    parts = [text or ""]
    if concept:
        parts.append("Concept: " + concept)
    if decision_context:
        parts.append("Decision context: " + decision_context)
    if primary_question_answered:
        parts.append("Primary question answered: " + primary_question_answered)
    if key_signals:
        parts.append("Key signals: " + ", ".join(key_signals))
    if chunk_role:
        parts.append("Chunk role: " + chunk_role)
    return "\n\n".join(parts)


def migrate_one_source(conn, group: str, source_id: int, source_path: str, source_summary: str | None) -> tuple[int, str | None]:
    """
    For one source: extract summary from chunks if present, else generate via LLM.
    Returns (chunks_updated, summary_set).
    """
    init_db(conn)
    chunks = get_chunks_for_source(conn, source_id)
    if not chunks:
        return (0, None)
    filename = Path(source_path).name
    extracted_summary: str | None = None
    chunks_updated = 0

    for c in chunks:
        stripped, summary = _strip_summary_from_text(c.get("text") or "")
        if summary is None:
            continue
        extracted_summary = summary
        if stripped == (c.get("text") or "").strip():
            continue
        row = get_chunk_by_id(conn, c["id"])
        if not row:
            continue
        concept = (row.get("concept") or "").strip() or ""
        decision_context = (row.get("decision_context") or "").strip() or ""
        primary_question_answered = (row.get("primary_question_answered") or "").strip() or ""
        key_signals = row.get("key_signals") or []
        chunk_role = (row.get("chunk_role") or "").strip() or ""
        to_embed_str = _text_to_embed(stripped, concept, decision_context, primary_question_answered, key_signals, chunk_role)
        embs = embed([to_embed_str], group=group)
        if not embs:
            continue
        update_chunk_full(
            conn, c["id"], stripped, embs[0],
            concept=concept or None, decision_context=decision_context or None,
            primary_question_answered=primary_question_answered or None,
            key_signals=key_signals if key_signals else None, chunk_role=chunk_role or None,
        )
        chunks_updated += 1

    if extracted_summary:
        set_source_summary(conn, source_id, extracted_summary)
        return (chunks_updated, extracted_summary)

    if source_summary and source_summary.strip():
        return (chunks_updated, None)

    combined = "\n\n".join((c.get("text") or "").strip() for c in chunks)
    new_summary = summarize_document(combined, group=group, filename=filename)
    if new_summary:
        set_source_summary(conn, source_id, new_summary)
        return (chunks_updated, new_summary)
    return (chunks_updated, None)


def main() -> int:
    groups = _list_sync_groups()
    if not groups:
        print("No RAG groups found.")
        return 0
    total_stripped = 0
    total_sources_with_summary = 0
    for group in sorted(groups):
        conn = _connect(group)
        try:
            init_db(conn)
            raw = list_sources(conn)
            for source_id, source_path, count, summary in raw:
                updated, set_summary = migrate_one_source(conn, group, source_id, source_path, summary)
                total_stripped += updated
                if set_summary:
                    total_sources_with_summary += 1
                    print(f"  [{group}] source_id={source_id} {Path(source_path).name}: set summary ({updated} chunk(s) stripped)")
            conn.commit()
        finally:
            conn.close()
    print(f"Done: {total_stripped} chunk(s) stripped, {total_sources_with_summary} source(s) with summary set.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
