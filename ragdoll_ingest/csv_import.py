"""Shared CSV chunk import (Review export / Claude handoff format). Used by CLI and Review web."""

from __future__ import annotations

import csv
import io
import json
from collections import defaultdict
from dataclasses import dataclass, field

from . import config
from .chunk_csv import CHUNK_CSV_HEADERS
from .embedder import build_text_to_embed, embed
from .memory import MEMORY_GROUP
from .storage import (
    _connect,
    _list_sync_groups,
    add_chunks,
    delete_source_by_id,
    init_db,
    set_source_display_title,
    set_source_external_url,
)

REQUIRED_CSV_FIELDS = frozenset({"chunk_index", "text"})


def _strip_csv_row(d: dict[str, str | None]) -> dict[str, str]:
    out: dict[str, str] = {}
    for k, v in d.items():
        key = (k or "").strip()
        if not key:
            continue
        if isinstance(v, str):
            out[key] = v.strip()
        elif v is None:
            out[key] = ""
        else:
            out[key] = str(v)
    return out


def _parse_int(val: str | None, _field: str) -> int | None:
    s = (val or "").strip()
    if s == "":
        return None
    try:
        return int(s)
    except ValueError:
        return None


def _parse_key_signals(raw: str | None) -> list[str] | str | None:
    s = (raw or "").strip()
    if not s:
        return None
    if s.startswith("["):
        try:
            data = json.loads(s)
            if isinstance(data, list):
                return [str(x).strip() for x in data if str(x).strip()]
        except json.JSONDecodeError:
            pass
    return s


def ensure_collection_db(group: str) -> None:
    """Create group directory, sources/, and SQLite schema if new."""
    gp = config.get_group_paths(group)
    gp.group_dir.mkdir(parents=True, exist_ok=True)
    gp.sources_dir.mkdir(parents=True, exist_ok=True)
    conn = _connect(group)
    try:
        init_db(conn)
        conn.commit()
    finally:
        conn.close()


def _effective_source_path(row: dict[str, str]) -> str | None:
    """Stable per-document id for grouping and DB uniqueness.

    Prefer explicit ``source_path`` (filesystem path from export). If blank, use
    ``canonical_url`` (typical for web/Claude handoffs) or ``import:source_key:{source_key}``.
    """
    sp = (row.get("source_path") or "").strip()
    if sp:
        return sp
    url = (row.get("canonical_url") or "").strip()
    if url:
        return url
    sk = (row.get("source_key") or "").strip()
    if sk:
        return f"import:source_key:{sk}"
    return None


def parse_csv_bytes(data: bytes) -> list[dict[str, str]]:
    """Parse CSV body (UTF-8 with optional BOM). Raises ValueError on bad header or empty."""
    text = data.decode("utf-8-sig")
    f = io.StringIO(text)
    reader = csv.DictReader(f)
    if not reader.fieldnames:
        raise ValueError("CSV has no header row.")
    fields = {(h or "").strip() for h in reader.fieldnames if h}
    missing = REQUIRED_CSV_FIELDS - fields
    if missing:
        raise ValueError(
            f"CSV missing required column(s): {', '.join(sorted(missing))}. "
            f"Expected header includes: {', '.join(CHUNK_CSV_HEADERS)}"
        )
    return [_strip_csv_row(dict(r)) for r in reader]


@dataclass
class CsvImportSummary:
    group: str
    created_collection: bool
    total_chunks: int
    messages: list[str] = field(default_factory=list)


def run_csv_import(
    raw_collection_name: str,
    rows: list[dict[str, str]],
    *,
    replace_sources: bool,
) -> CsvImportSummary:
    """Import rows into collection ``raw_collection_name`` (sanitized). Raises ValueError if nothing to import."""
    name = (raw_collection_name or "").strip()
    if not name:
        raise ValueError("Collection name is required.")

    group = config._sanitize_group(name)
    if group == MEMORY_GROUP:
        raise ValueError("Do not import into the 'memory' collection; use MCP write_memory.")

    by_source: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for r in rows:
        sp = _effective_source_path(r)
        st = (r.get("source_type") or "").strip() or ".txt"
        if not sp:
            continue
        by_source[(sp, st)].append(r)

    if not by_source:
        raise ValueError(
            "No importable rows. Each row needs non-empty text and a document id: "
            "set source_path, or leave it blank and set canonical_url (or source_key)."
        )

    existed_before = group in _list_sync_groups()
    ensure_collection_db(group)
    gp = config.get_group_paths(group)

    messages: list[str] = []
    if not existed_before:
        messages.append(f"Created collection '{group}' at {gp.group_dir}")
    else:
        messages.append(f"Using existing collection '{group}'.")

    conn = _connect(group)
    total_chunks = 0
    try:
        init_db(conn)
        for (source_path, source_type), src_rows in sorted(by_source.items(), key=lambda x: x[0][0]):

            def row_chunk_index(row: dict[str, str]) -> int:
                v = _parse_int(row.get("chunk_index", ""), "chunk_index")
                return v if v is not None else 0

            src_rows.sort(key=row_chunk_index)

            doc_summary = ""
            for r in src_rows:
                if (r.get("doc_summary") or "").strip():
                    doc_summary = r["doc_summary"].strip()
                    break
            canonical = ""
            for r in src_rows:
                if (r.get("canonical_url") or "").strip():
                    canonical = r["canonical_url"].strip()
                    break
            source_title = ""
            for r in src_rows:
                if (r.get("source_title") or "").strip():
                    source_title = r["source_title"].strip()
                    break

            existing = conn.execute(
                "SELECT id FROM sources WHERE source_path = ?", (source_path,)
            ).fetchone()
            if existing:
                if replace_sources:
                    delete_source_by_id(conn, int(existing["id"]))
                    messages.append(f"Replaced existing source: {source_path!r}")
                else:
                    messages.append(f"Skipped existing source: {source_path!r}")
                    continue

            valid_rows = [r for r in src_rows if (r.get("text") or "").strip()]
            if not valid_rows:
                messages.append(f"Skipped {source_path!r}: no rows with non-empty text.")
                continue

            embed_inputs = [
                build_text_to_embed(
                    doc_summary or None,
                    ((r.get("primary_question_answered") or "").strip() or None),
                    (r.get("text") or "").strip(),
                )
                for r in valid_rows
            ]

            batch_size = 100
            all_embs: list[list[float]] = []
            for i in range(0, len(embed_inputs), batch_size):
                batch = embed_inputs[i : i + batch_size]
                all_embs.extend(embed(batch, group=group))

            bodies: list[dict] = []
            for r, emb in zip(valid_rows, all_embs, strict=True):
                text = (r.get("text") or "").strip()
                pqa = (r.get("primary_question_answered") or "").strip() or None
                ks = _parse_key_signals(r.get("key_signals", ""))
                page_v = _parse_int(r.get("page", ""), "page")
                art = (r.get("artifact_type") or "text").strip() or "text"
                apath = (r.get("artifact_path") or "").strip() or None
                role = (r.get("chunk_role") or "").strip() or None
                concept = (r.get("concept") or "").strip() or None
                dctx = (r.get("decision_context") or "").strip() or None
                chunk: dict = {
                    "text": text,
                    "embedding": emb,
                    "artifact_type": art,
                    "artifact_path": apath,
                    "page": page_v,
                    "concept": concept,
                    "decision_context": dctx,
                    "primary_question_answered": pqa,
                    "chunk_role": role,
                }
                if isinstance(ks, list):
                    chunk["key_signals"] = ks
                elif ks:
                    chunk["key_signals"] = ks
                bodies.append(chunk)

            add_chunks(
                conn,
                source_path,
                source_type,
                bodies,
                doc_summary=doc_summary or None,
            )
            sid_row = conn.execute(
                "SELECT id FROM sources WHERE source_path = ?", (source_path,)
            ).fetchone()
            sid = int(sid_row["id"]) if sid_row else None
            if sid is not None:
                if canonical:
                    set_source_external_url(conn, sid, canonical)
                if source_title:
                    set_source_display_title(conn, sid, source_title)
            conn.commit()
            total_chunks += len(bodies)
            messages.append(f"Imported {len(bodies)} chunk(s) for {source_path!r}")

        messages.append(f"Done. {total_chunks} chunk(s) written.")
        return CsvImportSummary(
            group=group,
            created_collection=not existed_before,
            total_chunks=total_chunks,
            messages=messages,
        )
    finally:
        conn.close()
