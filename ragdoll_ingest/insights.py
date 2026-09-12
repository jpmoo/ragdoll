"""Insights collection: learnings RAGDoll builds from how its collections are used.

Each insight is one source in the "insights" group (source_path "insight/<id>") with embedded chunks for its
statement and rationale, so normal retrieval finds it. Metadata, lineage (the chunks, queries, and insights it
came from), and a revision history live in extra tables in the same DB.

Insights are never hard-deleted. Retiring one removes its chunks from retrieval but keeps the row and its
embedding, so later runs can recognize it instead of regenerating it. Changes made by a person (chat or CLI)
pin the insight so automated runs leave it alone.

Replaces the old "memory" collection; see migrate_memory_collection().
"""

import json
import os
import re
import sqlite3
import uuid
from datetime import datetime, timezone
from typing import Any

from . import config
from .embedder import build_text_to_embed, cosine_similarity, embed
from .storage import _connect, add_chunks, init_db

INSIGHTS_GROUP = "insights"
INSIGHT_SOURCE_TYPE = "insight"
ORIGINS = ("stew", "chat", "asserted", "agent")
STATUSES = ("active", "retired", "superseded")
# Actors whose changes pin an insight against automated edits
HUMAN_ACTORS = ("chat", "user")
EDITABLE_FIELDS = ("statement", "rationale", "question", "topic", "tags", "open_questions", "confidence")
# Fields that change what gets embedded
EMBEDDED_FIELDS = ("statement", "rationale", "question")
LINEAGE_KINDS = ("chunk", "query", "insight")
LEGACY_MEMORY_GROUP = "memory"


class InsightError(ValueError):
    """Invalid insight operation (not found, bad field, wrong status, version conflict)."""


def init_insights_db(conn: sqlite3.Connection) -> None:
    init_db(conn)
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS insights (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source_path TEXT NOT NULL UNIQUE,
            statement TEXT NOT NULL,
            rationale TEXT,
            question TEXT,
            topic TEXT,
            tags TEXT,
            open_questions TEXT,
            origin TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'active',
            status_reason TEXT,
            superseded_by INTEGER REFERENCES insights(id),
            confidence REAL,
            pinned INTEGER NOT NULL DEFAULT 0,
            run_id TEXT,
            legacy_source_path TEXT UNIQUE,
            embedding TEXT,
            version INTEGER NOT NULL DEFAULT 1,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS ix_insights_status ON insights(status);
        CREATE INDEX IF NOT EXISTS ix_insights_run ON insights(run_id);

        CREATE TABLE IF NOT EXISTS insight_lineage (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            insight_id INTEGER NOT NULL REFERENCES insights(id),
            kind TEXT NOT NULL,
            relation TEXT,
            ref_group TEXT,
            ref_source_path TEXT,
            ref_chunk_id INTEGER,
            ref_chunk_index INTEGER,
            ref_query_id INTEGER,
            ref_insight_id INTEGER,
            similarity REAL,
            text_snapshot TEXT,
            created_at TEXT DEFAULT (datetime('now'))
        );
        CREATE INDEX IF NOT EXISTS ix_insight_lineage_insight ON insight_lineage(insight_id);

        CREATE TABLE IF NOT EXISTS insight_revisions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            insight_id INTEGER NOT NULL REFERENCES insights(id),
            ts TEXT DEFAULT (datetime('now')),
            actor TEXT NOT NULL,
            action TEXT NOT NULL,
            before TEXT,
            after TEXT,
            reason TEXT,
            run_id TEXT,
            session_id TEXT
        );
        CREATE INDEX IF NOT EXISTS ix_insight_revisions_insight ON insight_revisions(insight_id);
    """)


def _connect_insights() -> sqlite3.Connection:
    conn = _connect(INSIGHTS_GROUP)
    init_insights_db(conn)
    return conn


def _now() -> str:
    """UTC timestamp in SQLite datetime('now') format."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def _normalize_fields(fields: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in fields.items():
        if k not in EDITABLE_FIELDS:
            raise InsightError(f"Field is not editable: {k}")
        if k == "tags":
            if isinstance(v, str):
                v = v.split(",")
            out[k] = [str(t).strip() for t in (v or []) if str(t).strip()]
        elif k == "confidence":
            if v is None or v == "":
                out[k] = None
                continue
            try:
                c = float(v)
            except (TypeError, ValueError) as e:
                raise InsightError(f"confidence must be a number between 0 and 1, got {v!r}") from e
            if not 0.0 <= c <= 1.0:
                raise InsightError(f"confidence must be between 0 and 1, got {c}")
            out[k] = c
        else:
            out[k] = (str(v).strip() or None) if v is not None else None
    return out


def _row_to_insight(row: sqlite3.Row) -> dict[str, Any]:
    d = dict(row)
    d.pop("embedding", None)
    try:
        d["tags"] = json.loads(d["tags"]) if d.get("tags") else []
    except (TypeError, json.JSONDecodeError):
        d["tags"] = []
    d["pinned"] = bool(d.get("pinned"))
    return d


def _snapshot(insight: dict[str, Any]) -> dict[str, Any]:
    return {k: insight.get(k) for k in EDITABLE_FIELDS}


def _fetch_row(conn: sqlite3.Connection, insight_id: int) -> sqlite3.Row:
    row = conn.execute("SELECT * FROM insights WHERE id = ?", (insight_id,)).fetchone()
    if not row:
        raise InsightError(f"Insight {insight_id} not found")
    return row


def _display_title(fields: dict[str, Any]) -> str:
    topic = (fields.get("topic") or "").strip()
    if topic:
        return topic
    statement = (fields.get("statement") or "").strip()
    return statement if len(statement) <= 80 else statement[:77] + "..."


def _embed_insight(fields: dict[str, Any]) -> list[dict[str, Any]]:
    """Chunks for the insight's statement (and rationale, if any), embedded. The statement chunk is first.

    Call before writing to the DB so the network round-trip doesn't hold SQLite's write lock.
    """
    statement = fields["statement"]
    question = fields.get("question")
    rationale = fields.get("rationale")
    parts = [("statement", statement, build_text_to_embed(None, question, statement))]
    if rationale:
        parts.append(("rationale", rationale, build_text_to_embed(statement, question, rationale)))
    embs = embed([p[2] for p in parts], group=INSIGHTS_GROUP)
    if len(embs) != len(parts):
        raise InsightError("Embedding count mismatch")
    return [
        {"text": text, "embedding": embs[i], "chunk_role": role, "primary_question_answered": question}
        for i, (role, text, _) in enumerate(parts)
    ]


def _index_insight(conn: sqlite3.Connection, source_path: str, chunks: list[dict[str, Any]]) -> str:
    """Replace the insight's retrievable chunks. Returns the statement embedding as JSON for the insights row."""
    conn.execute("DELETE FROM chunks WHERE source_path = ?", (source_path,))
    add_chunks(conn, source_path, INSIGHT_SOURCE_TYPE, chunks)
    return json.dumps(chunks[0]["embedding"])


def _set_display_title(conn: sqlite3.Connection, source_path: str, fields: dict[str, Any]) -> None:
    conn.execute("UPDATE sources SET display_title = ? WHERE source_path = ?", (_display_title(fields), source_path))


def _record_revision(
    conn: sqlite3.Connection,
    insight_id: int,
    *,
    actor: str,
    action: str,
    before: dict[str, Any] | None,
    after: dict[str, Any] | None,
    reason: str | None,
    run_id: str | None,
    session_id: str | None,
) -> None:
    conn.execute(
        "INSERT INTO insight_revisions (insight_id, actor, action, before, after, reason, run_id, session_id) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (
            insight_id, actor, action,
            json.dumps(before, ensure_ascii=False) if before is not None else None,
            json.dumps(after, ensure_ascii=False) if after is not None else None,
            reason, run_id, session_id,
        ),
    )


def _add_lineage(conn: sqlite3.Connection, insight_id: int, entries: list[dict[str, Any]]) -> None:
    """entries: {kind: chunk|query|insight, relation?, group?, source_path?, chunk_id?, chunk_index?, query_id?, insight_id?, similarity?, text_snapshot?}."""
    for e in entries:
        kind = e.get("kind")
        if kind not in LINEAGE_KINDS:
            raise InsightError(f"Lineage kind must be one of {LINEAGE_KINDS}, got {kind!r}")
        conn.execute(
            "INSERT INTO insight_lineage (insight_id, kind, relation, ref_group, ref_source_path, ref_chunk_id, "
            "ref_chunk_index, ref_query_id, ref_insight_id, similarity, text_snapshot) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                insight_id, kind, e.get("relation"), e.get("group"), e.get("source_path"), e.get("chunk_id"),
                e.get("chunk_index"), e.get("query_id"), e.get("insight_id"), e.get("similarity"), e.get("text_snapshot"),
            ),
        )


def create_insight(
    statement: str,
    *,
    origin: str,
    actor: str,
    rationale: str | None = None,
    question: str | None = None,
    topic: str | None = None,
    tags: list[str] | str | None = None,
    open_questions: str | None = None,
    confidence: float | None = None,
    lineage: list[dict[str, Any]] | None = None,
    run_id: str | None = None,
    reason: str | None = None,
    session_id: str | None = None,
    created_at: str | None = None,
    legacy_source_path: str | None = None,
) -> dict[str, Any]:
    """Create an active insight, embed it, and record its lineage and a 'create' revision."""
    if origin not in ORIGINS:
        raise InsightError(f"origin must be one of {ORIGINS}, got {origin!r}")
    fields = _normalize_fields({
        "statement": statement, "rationale": rationale, "question": question, "topic": topic,
        "tags": tags, "open_questions": open_questions, "confidence": confidence,
    })
    if not fields["statement"]:
        raise InsightError("statement is required")
    pinned = actor in HUMAN_ACTORS or origin in ("chat", "asserted")
    chunks = _embed_insight(fields)
    now = _now()
    conn = _connect_insights()
    try:
        # source_path is derived from the id, so insert under a placeholder first
        cur = conn.execute(
            "INSERT INTO insights (source_path, statement, rationale, question, topic, tags, open_questions, origin, "
            "confidence, pinned, run_id, legacy_source_path, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                f"insight/pending-{uuid.uuid4().hex}", fields["statement"], fields["rationale"], fields["question"],
                fields["topic"], json.dumps(fields["tags"], ensure_ascii=False), fields["open_questions"], origin,
                fields["confidence"], int(pinned), run_id, legacy_source_path, created_at or now, now,
            ),
        )
        insight_id = cur.lastrowid
        source_path = f"insight/{insight_id}"
        emb_json = _index_insight(conn, source_path, chunks)
        _set_display_title(conn, source_path, fields)
        conn.execute(
            "UPDATE insights SET source_path = ?, embedding = ? WHERE id = ?",
            (source_path, emb_json, insight_id),
        )
        if lineage:
            _add_lineage(conn, insight_id, lineage)
        _record_revision(
            conn, insight_id, actor=actor, action="create", before=None, after=fields,
            reason=reason, run_id=run_id, session_id=session_id,
        )
        conn.commit()
        return _row_to_insight(_fetch_row(conn, insight_id))
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def update_insight(
    insight_id: int,
    changes: dict[str, Any],
    *,
    actor: str,
    reason: str | None = None,
    expected_version: int | None = None,
    run_id: str | None = None,
    session_id: str | None = None,
) -> dict[str, Any]:
    """Edit an active insight's fields, re-embedding if its text changed. Pass expected_version to fail on concurrent edits."""
    fields = _normalize_fields(changes)
    conn = _connect_insights()
    try:
        row = _fetch_row(conn, insight_id)
        if expected_version is not None and row["version"] != expected_version:
            raise InsightError(f"Insight {insight_id} is at version {row['version']}, expected {expected_version}")
        if row["status"] != "active":
            raise InsightError(f"Insight {insight_id} is {row['status']}; restore it before editing")
        before = _snapshot(_row_to_insight(row))
        after = {**before, **fields}
        if not after["statement"]:
            raise InsightError("statement is required")
        if after == before:
            return _row_to_insight(row)
        if any(before[k] != after[k] for k in EMBEDDED_FIELDS):
            emb_json = _index_insight(conn, row["source_path"], _embed_insight(after))
        else:
            emb_json = row["embedding"]
        _set_display_title(conn, row["source_path"], after)
        pinned = bool(row["pinned"]) or actor in HUMAN_ACTORS
        conn.execute(
            "UPDATE insights SET statement = ?, rationale = ?, question = ?, topic = ?, tags = ?, open_questions = ?, "
            "confidence = ?, embedding = ?, pinned = ?, version = version + 1, updated_at = ? WHERE id = ?",
            (
                after["statement"], after["rationale"], after["question"], after["topic"],
                json.dumps(after["tags"], ensure_ascii=False), after["open_questions"], after["confidence"],
                emb_json, int(pinned), _now(), insight_id,
            ),
        )
        changed = [k for k in EDITABLE_FIELDS if before[k] != after[k]]
        _record_revision(
            conn, insight_id, actor=actor, action="update",
            before={k: before[k] for k in changed}, after={k: after[k] for k in changed},
            reason=reason, run_id=run_id, session_id=session_id,
        )
        conn.commit()
        return _row_to_insight(_fetch_row(conn, insight_id))
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def retire_insight(
    insight_id: int,
    *,
    actor: str,
    reason: str,
    superseded_by: int | None = None,
    run_id: str | None = None,
    session_id: str | None = None,
) -> dict[str, Any]:
    """Remove an insight from retrieval, keeping it (and the reason) so later runs don't regenerate it."""
    if not (reason or "").strip():
        raise InsightError("A reason is required when retiring an insight")
    conn = _connect_insights()
    try:
        row = _fetch_row(conn, insight_id)
        if row["status"] != "active":
            raise InsightError(f"Insight {insight_id} is already {row['status']}")
        if superseded_by is not None:
            if superseded_by == insight_id:
                raise InsightError("An insight cannot supersede itself")
            if _fetch_row(conn, superseded_by)["status"] != "active":
                raise InsightError(f"Superseding insight {superseded_by} is not active")
        status = "superseded" if superseded_by is not None else "retired"
        pinned = bool(row["pinned"]) or actor in HUMAN_ACTORS
        conn.execute("DELETE FROM chunks WHERE source_path = ?", (row["source_path"],))
        conn.execute(
            "UPDATE insights SET status = ?, status_reason = ?, superseded_by = ?, pinned = ?, "
            "version = version + 1, updated_at = ? WHERE id = ?",
            (status, reason.strip(), superseded_by, int(pinned), _now(), insight_id),
        )
        _record_revision(
            conn, insight_id, actor=actor, action="supersede" if superseded_by is not None else "retire",
            before={"status": "active"}, after={"status": status, "superseded_by": superseded_by},
            reason=reason.strip(), run_id=run_id, session_id=session_id,
        )
        conn.commit()
        return _row_to_insight(_fetch_row(conn, insight_id))
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def restore_insight(
    insight_id: int,
    *,
    actor: str,
    reason: str | None = None,
    session_id: str | None = None,
) -> dict[str, Any]:
    """Make a retired or superseded insight active and retrievable again."""
    conn = _connect_insights()
    try:
        row = _fetch_row(conn, insight_id)
        if row["status"] == "active":
            raise InsightError(f"Insight {insight_id} is already active")
        insight = _row_to_insight(row)
        emb_json = _index_insight(conn, row["source_path"], _embed_insight(insight))
        _set_display_title(conn, row["source_path"], insight)
        pinned = bool(row["pinned"]) or actor in HUMAN_ACTORS
        conn.execute(
            "UPDATE insights SET status = 'active', status_reason = NULL, superseded_by = NULL, embedding = ?, "
            "pinned = ?, version = version + 1, updated_at = ? WHERE id = ?",
            (emb_json, int(pinned), _now(), insight_id),
        )
        _record_revision(
            conn, insight_id, actor=actor, action="restore",
            before={"status": row["status"], "superseded_by": row["superseded_by"]}, after={"status": "active"},
            reason=reason, run_id=None, session_id=session_id,
        )
        conn.commit()
        return _row_to_insight(_fetch_row(conn, insight_id))
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def get_insight(insight_id: int) -> dict[str, Any]:
    """One insight with its lineage and revision history."""
    conn = _connect_insights()
    try:
        insight = _row_to_insight(_fetch_row(conn, insight_id))
        insight["lineage"] = [
            dict(r) for r in conn.execute(
                "SELECT * FROM insight_lineage WHERE insight_id = ? ORDER BY id", (insight_id,)
            ).fetchall()
        ]
        revisions = []
        for r in conn.execute("SELECT * FROM insight_revisions WHERE insight_id = ? ORDER BY id", (insight_id,)).fetchall():
            rev = dict(r)
            for k in ("before", "after"):
                rev[k] = json.loads(rev[k]) if rev[k] else None
            revisions.append(rev)
        insight["revisions"] = revisions
        return insight
    finally:
        conn.close()


def list_insights(
    status: str | None = "active",
    origin: str | None = None,
    run_id: str | None = None,
    limit: int = 100,
) -> list[dict[str, Any]]:
    """Insights newest-updated first. status=None lists every status."""
    where, params = [], []
    if status is not None:
        where.append("status = ?")
        params.append(status)
    if origin is not None:
        where.append("origin = ?")
        params.append(origin)
    if run_id is not None:
        where.append("run_id = ?")
        params.append(run_id)
    sql = "SELECT * FROM insights" + (" WHERE " + " AND ".join(where) if where else "") + " ORDER BY updated_at DESC, id DESC LIMIT ?"
    conn = _connect_insights()
    try:
        return [_row_to_insight(r) for r in conn.execute(sql, (*params, limit)).fetchall()]
    finally:
        conn.close()


def find_similar_insights(
    embedding: list[float],
    *,
    statuses: tuple[str, ...] = ("active",),
    top_k: int = 5,
    min_similarity: float = 0.0,
) -> list[dict[str, Any]]:
    """Insights whose statement embedding is closest to `embedding`, each with a "similarity" field.

    Retired and superseded insights keep their embedding, so pass those statuses to check whether something
    was already considered and rejected.
    """
    conn = _connect_insights()
    try:
        placeholders = ",".join("?" * len(statuses))
        rows = conn.execute(
            f"SELECT * FROM insights WHERE status IN ({placeholders}) AND embedding IS NOT NULL",
            tuple(statuses),
        ).fetchall()
        scored: list[dict[str, Any]] = []
        for r in rows:
            try:
                sim = cosine_similarity(embedding, json.loads(r["embedding"]))
            except (TypeError, json.JSONDecodeError):
                continue
            if sim >= min_similarity:
                d = _row_to_insight(r)
                d["similarity"] = round(sim, 4)
                scored.append(d)
        scored.sort(key=lambda d: d["similarity"], reverse=True)
        return scored[:top_k]
    finally:
        conn.close()


def reinforce_insight(
    insight_id: int,
    lineage: list[dict[str, Any]],
    *,
    actor: str,
    reason: str | None = None,
    run_id: str | None = None,
    session_id: str | None = None,
) -> dict[str, Any]:
    """Attach new supporting lineage to an existing insight without changing its text.

    This is the merge case: a later run found the same conclusion again. The statement stays as written (so a
    person's wording is never overwritten) and the version is unchanged; only lineage and history grow.
    """
    conn = _connect_insights()
    try:
        row = _fetch_row(conn, insight_id)
        if row["status"] != "active":
            raise InsightError(f"Insight {insight_id} is {row['status']}; only active insights can be reinforced")
        if lineage:
            _add_lineage(conn, insight_id, lineage)
        conn.execute("UPDATE insights SET updated_at = ? WHERE id = ?", (_now(), insight_id))
        _record_revision(
            conn, insight_id, actor=actor, action="reinforce", before=None,
            after={"lineage_added": len(lineage or [])}, reason=reason, run_id=run_id, session_id=session_id,
        )
        conn.commit()
        return _row_to_insight(_fetch_row(conn, insight_id))
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def get_insights_by_source_paths(source_paths: list[str]) -> dict[str, dict[str, Any]]:
    """Brief metadata for query results, keyed by source_path."""
    paths = sorted(set(source_paths))
    if not paths:
        return {}
    conn = _connect_insights()
    try:
        rows = conn.execute(
            "SELECT id, source_path, topic, tags, question, origin, status, confidence, pinned, created_at, updated_at "
            f"FROM insights WHERE source_path IN ({','.join('?' * len(paths))})",
            paths,
        ).fetchall()
        out = {}
        for r in rows:
            d = _row_to_insight(r)
            out[d.pop("source_path")] = d
        return out
    finally:
        conn.close()


_SECTION_HEADERS = {
    "topic": "topic",
    "date": "date",
    "tags": "tags",
    "insight": "statement",
    "statement": "statement",
    "conclusion": "statement",
    "question": "question",
    "reasoning": "rationale",
    "rationale": "rationale",
    "open threads": "open_questions",
    "open questions": "open_questions",
    "confidence": "confidence",
}
_SINGLE_LINE_SECTIONS = {"topic", "date", "tags", "confidence"}
_HEADER_RE = re.compile(
    r"^[ \t]*(" + "|".join(h.replace(" ", r"\s+") for h in sorted(_SECTION_HEADERS, key=len, reverse=True)) + r")[ \t]*:",
    re.IGNORECASE | re.MULTILINE,
)


def parse_insight_text(raw: str) -> dict[str, Any] | None:
    """Parse header-formatted text into create_insight fields. Returns None if there is no statement.

    Headers (case-insensitive, each at the start of a line): Topic, Date, Tags, Insight (or Statement /
    Conclusion), Question, Reasoning (or Rationale), Open questions (or Open threads), Confidence (0-1).
    The legacy memory format (Topic, Date, Tags, Conclusion, Reasoning, Open threads) parses as-is.
    """
    text = (raw or "").strip()
    matches = list(_HEADER_RE.finditer(text))
    sections: dict[str, str] = {}
    for i, m in enumerate(matches):
        key = _SECTION_HEADERS[re.sub(r"\s+", " ", m.group(1).lower())]
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[m.end():end].strip()
        if key in _SINGLE_LINE_SECTIONS:
            body = body.split("\n")[0].strip()
        if body and key not in sections:
            sections[key] = body
    if not sections.get("statement"):
        return None
    out: dict[str, Any] = {k: sections.get(k) for k in ("statement", "rationale", "question", "topic", "open_questions")}
    out["tags"] = [t.strip() for t in sections.get("tags", "").split(",") if t.strip()]
    try:
        conf = float(sections["confidence"]) if "confidence" in sections else None
        out["confidence"] = conf if conf is not None and 0.0 <= conf <= 1.0 else None
    except ValueError:
        out["confidence"] = None
    date = sections.get("date", "")
    out["created_at"] = f"{date[:10]} 00:00:00" if re.match(r"^\d{4}-\d{2}-\d{2}", date) else None
    return out


def submit_insight_text(raw: str) -> dict[str, Any]:
    """Create an agent-origin insight from header-formatted text (the MCP submission path)."""
    parsed = parse_insight_text(raw)
    if parsed is None:
        return {"ok": False, "error": "Could not parse insight: an 'Insight:' (or 'Conclusion:') section is required"}
    insight = create_insight(origin="agent", actor="agent", reason="Submitted via MCP", **parsed)
    return {
        "ok": True,
        "insight_id": insight["id"],
        "source_path": insight["source_path"],
        "topic": insight["topic"] or "",
        "statement": insight["statement"],
    }


def migrate_memory_collection(*, archive: bool = True, dry_run: bool = False) -> dict[str, Any]:
    """Copy memories from the legacy "memory" collection into insights (origin "agent"), then archive the memory dir.

    Safe to re-run: memories already migrated (matched on their old source_path) are skipped. The memory dir is
    moved to {DATA_DIR}/_archive/ only after every memory has been migrated.
    """
    result: dict[str, Any] = {
        "found": 0, "migrated": 0, "skipped_existing": 0, "skipped_empty": 0, "archived_to": None, "archive_error": None,
    }
    gp = config.get_group_paths(LEGACY_MEMORY_GROUP)
    if not gp.rag_db_path.exists():
        return result

    src = sqlite3.connect(str(gp.rag_db_path))
    src.row_factory = sqlite3.Row
    try:
        memories = []
        for s in src.execute("SELECT source_path, summary, created_at FROM sources ORDER BY id").fetchall():
            chunks = {
                r["chunk_role"]: r["text"]
                for r in src.execute(
                    "SELECT chunk_role, text FROM chunks WHERE source_path = ? ORDER BY chunk_index", (s["source_path"],)
                ).fetchall()
                if r["chunk_role"]
            }
            memories.append((dict(s), chunks))
    finally:
        src.close()
    result["found"] = len(memories)

    already: set[str] = set()
    if config.get_group_paths(INSIGHTS_GROUP).rag_db_path.exists():
        conn = _connect_insights()
        try:
            already = {
                r[0] for r in conn.execute("SELECT legacy_source_path FROM insights WHERE legacy_source_path IS NOT NULL")
            }
        finally:
            conn.close()

    for s, chunks in memories:
        if s["source_path"] in already:
            result["skipped_existing"] += 1
            continue
        try:
            meta = json.loads(s["summary"]) if s["summary"] else {}
        except json.JSONDecodeError:
            meta = {}
        # Prefer the stored full text (keeps line breaks); fall back to the per-section chunks
        parsed = parse_insight_text(meta.get("full_text") or "") or {}
        statement = (
            parsed.get("statement") or chunks.get("conclusion") or chunks.get("reasoning")
            or chunks.get("open_threads") or meta.get("topic")
        )
        if not statement:
            result["skipped_empty"] += 1
            continue
        rationale = parsed.get("rationale") or chunks.get("reasoning")
        open_questions = parsed.get("open_questions") or chunks.get("open_threads")
        if not dry_run:
            create_insight(
                statement,
                origin="agent",
                actor="migration",
                rationale=rationale if rationale != statement else None,
                open_questions=open_questions if open_questions != statement else None,
                topic=meta.get("topic") or parsed.get("topic"),
                tags=meta.get("tags") or parsed.get("tags"),
                reason="Migrated from the memory collection",
                created_at=parsed.get("created_at") or s["created_at"],
                legacy_source_path=s["source_path"],
            )
        result["migrated"] += 1

    if archive and not dry_run:
        dest = config.DATA_DIR / "_archive" / f"memory-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
        try:
            dest.parent.mkdir(parents=True, exist_ok=True)
            # Rename only (same filesystem). shutil.move falls back to copy-then-delete, which leaves a
            # duplicate copy behind when the delete isn't permitted.
            os.rename(gp.group_dir, dest)
            result["archived_to"] = str(dest)
        except OSError as e:
            result["archive_error"] = f"Could not move {gp.group_dir} to {dest}: {e}"
    return result
