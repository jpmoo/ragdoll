"""Storage: SQLite chunks DB, JSONL samples, processed-file dedup. Sync keeps DB and JSONL identical. Per-group."""

import json
import logging
import re
import shutil
import sqlite3
import threading
from pathlib import Path

from . import config
from .action_log import log as action_log

logger = logging.getLogger(__name__)


def clean_text(text: str) -> str:
    """Strip newlines, normalize whitespace, and clean characters that interfere with meaning."""
    if not text:
        return ""
    # Replace newlines, carriage returns, tabs with spaces
    text = text.replace("\n", " ").replace("\r", " ").replace("\t", " ")
    # Normalize multiple spaces to single space
    text = re.sub(r" +", " ", text)
    # Strip leading/trailing whitespace
    return text.strip()


def extract_key_phrases_from_filename(filename: str) -> list[str]:
    """Extract descriptive phrases from filename (multi-word terms)."""
    if not filename:
        return []
    # Remove extension
    stem = Path(filename).stem
    # Split on common delimiters but preserve meaningful sequences
    # First, try to preserve camelCase and TitleCase
    parts = re.split(r"[_\-\s\.]+", stem)
    phrases = []
    
    # Extract 2-3 word phrases from filename parts
    stop_words = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by", "from"}
    meaningful_parts = [p.strip().lower() for p in parts if p and p.strip() and len(p.strip()) >= 3 and p.strip().lower() not in stop_words and not p.strip().isdigit()]
    
    if not meaningful_parts:
        return []
    
    # Create 2-word phrases
    for i in range(len(meaningful_parts) - 1):
        if meaningful_parts[i] and meaningful_parts[i+1]:
            phrase = f"{meaningful_parts[i]} {meaningful_parts[i+1]}"
            if len(phrase) >= 6:  # At least 6 chars total
                phrases.append(phrase)
    
    # Also include single meaningful words if they're substantial
    for part in meaningful_parts:
        if part and len(part) >= 5:  # Only longer single words
            phrases.append(part)
    
    # Filter out any None or empty strings
    phrases = [p for p in phrases if p and isinstance(p, str)]
    return phrases[:10]  # Limit to 10 phrases


def extract_key_phrases_from_text(text: str, max_phrases: int = 10) -> list[str]:
    """Extract descriptive phrases from text (2-3 word n-grams, excluding stop words)."""
    if not text:
        return []
    
    stop_words = {
        "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by", "from",
        "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "do", "does", "did",
        "will", "would", "should", "could", "may", "might", "must", "can", "this", "that", "these", "those",
        "it", "its", "they", "them", "their", "there", "then", "than", "what", "which", "who", "when", "where", "why", "how",
        "as", "if", "so", "not", "no", "yes", "up", "down", "out", "off", "over", "under", "again", "further",
    }
    
    # Extract words (alphanumeric, at least 3 chars)
    words = [w.lower() for w in re.findall(r"\b[a-zA-Z]{3,}\b", text.lower()) if w.lower() not in stop_words]
    
    if len(words) < 2:
        return []
    
    # Extract 2-word phrases
    phrase_counts: dict[str, int] = {}
    for i in range(len(words) - 1):
        phrase = f"{words[i]} {words[i+1]}"
        if len(phrase) >= 6:  # At least 6 chars
            phrase_counts[phrase] = phrase_counts.get(phrase, 0) + 1
    
    # Extract 3-word phrases
    for i in range(len(words) - 2):
        phrase = f"{words[i]} {words[i+1]} {words[i+2]}"
        if len(phrase) >= 10:  # At least 10 chars
            phrase_counts[phrase] = phrase_counts.get(phrase, 0) + 1
    
    # Sort by frequency, return top phrases
    sorted_phrases = sorted(phrase_counts.items(), key=lambda x: x[1], reverse=True)
    phrases = [phrase for phrase, _ in sorted_phrases[:max_phrases]]
    # Filter out any None or empty strings (defensive)
    return [p for p in phrases if p and isinstance(p, str)]

_processed_cache: dict[str, set[tuple[str, float, int]]] = {}
_processed_lock = threading.Lock()
_jsonl_lock = threading.Lock()


# --- Migration: flat DATA_DIR layout -> DATA_DIR/_root/ ---

def migrate_flat_to_root() -> None:
    """If DATA_DIR has ragdoll.db at top level and _root/ does not, move into _root/ for group layout."""
    d = Path(config.DATA_DIR)
    flat_db = d / "ragdoll.db"
    root_dir = d / "_root"
    if not flat_db.exists() or (root_dir / "ragdoll.db").exists():
        return
    root_dir.mkdir(parents=True, exist_ok=True)
    for name in ["ragdoll.db", "rag_samples.jsonl", "processed.jsonl", "action.log"]:
        f = d / name
        if f.exists():
            shutil.move(str(f), str(root_dir / name))
    sources = d / "sources"
    if sources.is_dir():
        shutil.move(str(sources), str(root_dir / "sources"))
    logger.info("Migrated flat layout to %s/_root/", d)


# --- Processed (file-level dedup, per group) ---

def _ensure_processed_loaded(group: str) -> None:
    global _processed_cache
    with _processed_lock:
        if group in _processed_cache:
            return
        gp = config.get_group_paths(group)
        _processed_cache[group] = set()
        if gp.processed_path.exists():
            with open(gp.processed_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    rec = json.loads(line)
                    _processed_cache[group].add((rec["path"], rec["mtime"], rec["size"]))
        logger.debug("Loaded %d processed records for group %s from %s", len(_processed_cache[group]), group, gp.processed_path)


def already_processed(path: str, mtime: float, size: int, group: str) -> bool:
    _ensure_processed_loaded(group)
    with _processed_lock:
        return (path, mtime, size) in _processed_cache[group]


def mark_processed(path: str, mtime: float, size: int, group: str) -> None:
    _ensure_processed_loaded(group)
    gp = config.get_group_paths(group)
    gp.group_dir.mkdir(parents=True, exist_ok=True)
    line = json.dumps({"path": path, "mtime": mtime, "size": size}, ensure_ascii=False) + "\n"
    with open(gp.processed_path, "a", encoding="utf-8") as f:
        f.write(line)
    with _processed_lock:
        _processed_cache[group].add((path, mtime, size))
    logger.debug("Marked processed: %s (group=%s)", path, group)


# --- DB (chunks, per group) ---

def _connect(group: str) -> sqlite3.Connection:
    gp = config.get_group_paths(group)
    gp.group_dir.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(gp.rag_db_path))
    conn.row_factory = sqlite3.Row
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS chunks (
            id INTEGER PRIMARY KEY,
            source_path TEXT NOT NULL,
            source_type TEXT NOT NULL,
            chunk_index INTEGER NOT NULL,
            text TEXT NOT NULL,
            embedding TEXT NOT NULL,
            created_at TEXT DEFAULT (datetime('now'))
        );
        CREATE INDEX IF NOT EXISTS ix_chunks_source ON chunks(source_path);
    """)
    for col, defn in [
        ("artifact_type", "TEXT DEFAULT 'text'"),
        ("artifact_path", "TEXT"),
        ("page", "INTEGER"),
        ("key_terms", "TEXT"),  # JSON array of key terms
    ]:
        try:
            conn.execute(f"ALTER TABLE chunks ADD COLUMN {col} {defn}")
        except sqlite3.OperationalError as e:
            if "duplicate" not in str(e).lower():
                raise


def add_chunks(
    conn: sqlite3.Connection,
    source_path: str,
    source_type: str,
    chunks: list[dict],
) -> None:
    """
    chunks: list of {text, embedding, artifact_type?, artifact_path?, page?}.
    Defaults: artifact_type='text', artifact_path=None, page=None.
    Note: key phrases are now embedded in the text field itself (appended as "Key terms: ...").
    """
    init_db(conn)
    for i, c in enumerate(chunks):
        text = clean_text(c.get("text", ""))
        emb = c.get("embedding", [])
        atype = c.get("artifact_type", "text")
        apath = c.get("artifact_path")
        page = c.get("page")
        # key_terms column exists for backward compatibility but we no longer write to it
        conn.execute(
            "INSERT INTO chunks (source_path, source_type, chunk_index, text, embedding, artifact_type, artifact_path, page) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (source_path, source_type, i, text, json.dumps(emb), atype, apath, page),
        )


# --- JSONL (samples, per group) ---

def append_samples_jsonl(chunks: list[dict], source_path: str, source_type: str, group: str) -> None:
    """Append new chunk samples to the group's JSONL. chunks: list of {text, embedding, artifact_type?, artifact_path?, page?}."""
    gp = config.get_group_paths(group)
    gp.group_dir.mkdir(parents=True, exist_ok=True)
    with _jsonl_lock:
        with open(gp.samples_path, "a", encoding="utf-8") as f:
            for i, c in enumerate(chunks):
                rec = {
                    "text": clean_text(c.get("text", "")),
                    "embedding": c.get("embedding", []),
                    "source": source_path,
                    "source_type": source_type,
                    "chunk_index": i,
                    "artifact_type": c.get("artifact_type", "text"),
                    "artifact_path": c.get("artifact_path"),
                    "page": c.get("page"),
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    logger.info("Appended %d samples to %s (group=%s)", len(chunks), gp.samples_path, group)


def build_jsonl_from_db(conn: sqlite3.Connection, group: str) -> None:
    """Overwrite the group's JSONL with all chunks from the DB. Holds _jsonl_lock."""
    init_db(conn)
    rows = conn.execute(
        "SELECT source_path, source_type, chunk_index, text, embedding, artifact_type, artifact_path, page FROM chunks ORDER BY source_path, chunk_index"
    ).fetchall()
    gp = config.get_group_paths(group)
    gp.group_dir.mkdir(parents=True, exist_ok=True)
    with _jsonl_lock:
        with open(gp.samples_path, "w", encoding="utf-8") as f:
            for r in rows:
                rec = {
                    "text": clean_text(r["text"]),
                    "embedding": json.loads(r["embedding"]),
                    "source": r["source_path"],
                    "source_type": r["source_type"],
                    "chunk_index": r["chunk_index"],
                    "artifact_type": r["artifact_type"] or "text",
                    "artifact_path": r["artifact_path"],
                    "page": r["page"],
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    logger.info("Built %s from DB (%d chunks, group=%s)", gp.samples_path, len(rows), group)


def run_dedup(conn: sqlite3.Connection) -> int:
    """Remove duplicate (source_path, chunk_index), keeping min(id). Returns number of rows deleted."""
    init_db(conn)
    n_before = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    conn.execute(
        "DELETE FROM chunks WHERE id NOT IN (SELECT MIN(id) FROM chunks GROUP BY source_path, chunk_index)"
    )
    n_after = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    return n_before - n_after


def _list_sync_groups() -> list[str]:
    """Group dirs under DATA_DIR that have ragdoll.db or rag_samples.jsonl."""
    d = Path(config.DATA_DIR)
    if not d.exists():
        return []
    return [
        x.name for x in d.iterdir()
        if x.is_dir() and ((x / "ragdoll.db").exists() or (x / "rag_samples.jsonl").exists())
    ]


def run_sync_pass(group: str | None = None) -> None:
    """
    Keep DB and JSONL identical for one or all groups:
    - If group is None: run for each existing group (discovered from DATA_DIR).
    - If JSONL missing: build from DB.
    - Run dedup on DB (removes duplicate source_path+chunk_index).
    - If counts differ (or dedup removed rows): rebuild JSONL from DB.
    Skips rebuild when DB is empty and JSONL has data (migration from pre-DB).
    """
    groups = [group] if group is not None else _list_sync_groups()
    for g in groups:
        _run_sync_pass_one(g)


def _run_sync_pass_one(group: str) -> None:
    conn = _connect(group)
    try:
        init_db(conn)
        gp = config.get_group_paths(group)
        samples_path = gp.samples_path

        if not samples_path.exists():
            build_jsonl_from_db(conn, group)
            action_log("sync_rebuild", reason="jsonl_missing", count=conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0], group=group)
            return

        n_deleted = run_dedup(conn)
        conn.commit()
        if n_deleted > 0:
            logger.info("Dedup removed %d duplicate chunk(s) (group=%s)", n_deleted, group)
            action_log("sync_dedup", n_deleted=n_deleted, group=group)

        count_db = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]

        with open(samples_path, "r", encoding="utf-8") as f:
            count_jsonl = sum(1 for _ in f)

        need_rebuild = n_deleted > 0 or (
            count_db != count_jsonl and not (count_db == 0 and count_jsonl > 0)
        )
        if need_rebuild:
            build_jsonl_from_db(conn, group)
            reason = "dedup" if n_deleted > 0 else "count_mismatch"
            action_log("sync_rebuild", reason=reason, count_db=count_db, count_jsonl=count_jsonl, group=group)
            logger.info("Reconciled JSONL with DB (dedup=%d, count_db=%d, count_jsonl=%d, group=%s)", n_deleted, count_db, count_jsonl, group)
    finally:
        conn.close()
