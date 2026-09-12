"""Query log: API and MCP queries with their embeddings and the chunks they returned (input for the insights stew).

Lives at {DATA_DIR}/_querylog/querylog.db. Deliberately not named ragdoll.db, so it is never listed as a collection.
Logging failures are swallowed: a query must never fail because it couldn't be logged.
"""

import json
import logging
import sqlite3
from typing import Any

from . import config
from .storage import SQLITE_TIMEOUT

logger = logging.getLogger(__name__)

QUERYLOG_DIR = "_querylog"
SNAPSHOT_MAX_CHARS = 2000


def _connect_log() -> sqlite3.Connection:
    d = config.DATA_DIR / QUERYLOG_DIR
    d.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(d / "querylog.db"), timeout=SQLITE_TIMEOUT)
    conn.row_factory = sqlite3.Row
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS queries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT DEFAULT (datetime('now')),
            transport TEXT NOT NULL,
            prompt TEXT NOT NULL,
            expanded_query TEXT,
            history TEXT,
            collections TEXT,
            threshold REAL,
            embedding TEXT,
            result_count INTEGER NOT NULL,
            top_similarity REAL
        );
        CREATE INDEX IF NOT EXISTS ix_queries_ts ON queries(ts);

        CREATE TABLE IF NOT EXISTS query_hits (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            query_id INTEGER NOT NULL REFERENCES queries(id),
            rank INTEGER NOT NULL,
            group_name TEXT NOT NULL,
            source_path TEXT NOT NULL,
            chunk_id INTEGER,
            chunk_index INTEGER,
            chunk_role TEXT,
            similarity REAL NOT NULL,
            text_snapshot TEXT
        );
        CREATE INDEX IF NOT EXISTS ix_query_hits_query ON query_hits(query_id);
    """)
    return conn


def log_query(
    *,
    transport: str,
    prompt: str,
    expanded_query: str,
    history: str | None,
    groups: list[str],
    threshold: float,
    embedding: list[float],
    results: list[dict[str, Any]],
) -> int | None:
    """Record one query and its top hits (results must be sorted by similarity). Returns the query id, or None if not logged."""
    if not config.QUERY_LOG_ENABLED:
        return None
    try:
        conn = _connect_log()
        try:
            cur = conn.execute(
                "INSERT INTO queries (transport, prompt, expanded_query, history, collections, threshold, embedding, "
                "result_count, top_similarity) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    transport, prompt, expanded_query,
                    history if config.QUERY_LOG_HISTORY else None,
                    json.dumps(groups), threshold, json.dumps(embedding), len(results),
                    results[0]["similarity"] if results else None,
                ),
            )
            query_id = cur.lastrowid
            for rank, r in enumerate(results[: config.QUERY_LOG_TOP_K], 1):
                conn.execute(
                    "INSERT INTO query_hits (query_id, rank, group_name, source_path, chunk_id, chunk_index, chunk_role, "
                    "similarity, text_snapshot) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        query_id, rank, r["group"], r["source_path"], r.get("chunk_id"), r.get("chunk_index"),
                        r.get("chunk_role"), r["similarity"], (r.get("text") or "")[:SNAPSHOT_MAX_CHARS],
                    ),
                )
            conn.commit()
            return query_id
        finally:
            conn.close()
    except Exception as e:
        logger.warning("Query log write failed: %s", e)
        return None


def queries_since(since: str | None = None, limit: int | None = None) -> list[dict[str, Any]]:
    """Logged queries from `since` (SQLite timestamp) onward, oldest first, with their embeddings.

    This is the stew's input, so unlike recent_queries it keeps the embedding vector.
    """
    conn = _connect_log()
    try:
        sql = (
            "SELECT id, ts, transport, prompt, expanded_query, collections, threshold, embedding, result_count, "
            "top_similarity FROM queries"
        )
        params: list[Any] = []
        if since:
            sql += " WHERE ts >= ?"
            params.append(since)
        sql += " ORDER BY id"
        if limit:
            sql += " LIMIT ?"
            params.append(limit)
        out = []
        for r in conn.execute(sql, params).fetchall():
            d = dict(r)
            d["collections"] = json.loads(d["collections"] or "[]")
            try:
                d["embedding"] = json.loads(d["embedding"]) if d["embedding"] else None
            except json.JSONDecodeError:
                d["embedding"] = None
            out.append(d)
        return out
    finally:
        conn.close()


def hits_for_queries(query_ids: list[int]) -> dict[int, list[dict[str, Any]]]:
    """Logged hits for each query id, best first."""
    if not query_ids:
        return {}
    conn = _connect_log()
    try:
        placeholders = ",".join("?" * len(query_ids))
        rows = conn.execute(
            f"SELECT * FROM query_hits WHERE query_id IN ({placeholders}) ORDER BY query_id, rank",
            tuple(query_ids),
        ).fetchall()
        out: dict[int, list[dict[str, Any]]] = {}
        for r in rows:
            out.setdefault(r["query_id"], []).append(dict(r))
        return out
    finally:
        conn.close()


def recent_queries(limit: int = 20) -> list[dict[str, Any]]:
    """Most recent queries first, without embeddings."""
    conn = _connect_log()
    try:
        rows = conn.execute(
            "SELECT id, ts, transport, prompt, expanded_query, collections, threshold, result_count, top_similarity "
            "FROM queries ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [{**dict(r), "collections": json.loads(r["collections"] or "[]")} for r in rows]
    finally:
        conn.close()


def get_query(query_id: int) -> dict[str, Any] | None:
    """One query (without embedding) and its logged hits."""
    conn = _connect_log()
    try:
        row = conn.execute(
            "SELECT id, ts, transport, prompt, expanded_query, history, collections, threshold, result_count, top_similarity "
            "FROM queries WHERE id = ?",
            (query_id,),
        ).fetchone()
        if not row:
            return None
        hits = conn.execute("SELECT * FROM query_hits WHERE query_id = ? ORDER BY rank", (query_id,)).fetchall()
        return {**dict(row), "collections": json.loads(row["collections"] or "[]"), "hits": [dict(h) for h in hits]}
    finally:
        conn.close()
