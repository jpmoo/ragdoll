"""Storage: SQLite chunks DB, processed-file dedup. Per-group."""

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


# Max characters sent to LLM for key-term extraction (avoid timeouts)
_KEY_TERMS_LLM_MAX_CHARS = 4000


def extract_key_phrases_llm(
    text: str,
    max_phrases: int = 10,
    ollama_url: str | None = None,
    group: str = "_root",
) -> list[str]:
    """Ask LLM to extract key terms/phrases from text. Returns [] on failure or empty response."""
    if not (text or "").strip():
        return []
    url = (ollama_url or config.OLLAMA_HOST or "").rstrip("/")
    if not url:
        return []
    input_text = text.strip()
    if len(input_text) > _KEY_TERMS_LLM_MAX_CHARS:
        input_text = input_text[: _KEY_TERMS_LLM_MAX_CHARS] + "..."
    prompt = (
        "From the following text, extract up to 10 key terms or short phrases (2-4 words) that best describe the content. "
        "Return ONLY valid JSON in this exact format, no other text:\n"
        '{"key_terms": ["term1", "term2", ...]}\n\n'
        "Text:\n\n"
    ) + input_text
    try:
        import requests

        r = requests.post(
            f"{url}/api/generate",
            json={
                "model": config.INTERPRET_MODEL,
                "prompt": prompt,
                "stream": False,
                "format": "json",
            },
            timeout=config.CHUNK_LLM_TIMEOUT,
        )
        r.raise_for_status()
        resp = (r.json().get("response") or "").strip()
        if not resp:
            return []
        if "```" in resp:
            m = re.search(r"```(?:json)?\s*([\s\S]*?)```", resp)
            if m:
                resp = m.group(1).strip()
        obj = json.loads(resp)
        raw = obj.get("key_terms")
        if not isinstance(raw, list):
            return []
        phrases = [str(x).strip() for x in raw if x and str(x).strip()][:max_phrases]
        phrases = [p for p in phrases if p and isinstance(p, str)]
        if phrases:
            action_log("key_terms_llm", model=config.INTERPRET_MODEL, num_terms=len(phrases), group=group)
        return phrases
    except Exception as e:
        logger.warning("Key terms LLM request failed: %s", e)
        return []


def get_key_phrases_for_content(
    text: str,
    filename: str | None = None,
    max_phrases: int = 10,
    ollama_url: str | None = None,
    group: str = "_root",
) -> list[str]:
    """Get key phrases from content: try LLM first, fall back to heuristic (filename + text n-grams)."""
    llm_phrases = extract_key_phrases_llm(text, max_phrases=max_phrases, ollama_url=ollama_url, group=group)
    if llm_phrases:
        return llm_phrases
    filename_phrases = extract_key_phrases_from_filename(filename or "")
    text_phrases = extract_key_phrases_from_text(text or "", max_phrases=max_phrases)
    combined = [p for p in set(filename_phrases + text_phrases) if p and isinstance(p, str)][:max_phrases]
    return combined


_processed_cache: dict[str, set[tuple[str, float, int]]] = {}
_processed_lock = threading.Lock()


# --- Migration: flat DATA_DIR layout -> DATA_DIR/_root/ ---

def migrate_flat_to_root() -> None:
    """If DATA_DIR has ragdoll.db at top level and _root/ does not, move into _root/ for group layout."""
    d = Path(config.DATA_DIR)
    flat_db = d / "ragdoll.db"
    root_dir = d / "_root"
    if not flat_db.exists() or (root_dir / "ragdoll.db").exists():
        return
    root_dir.mkdir(parents=True, exist_ok=True)
    for name in ["ragdoll.db", "processed.jsonl", "action.log"]:
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


def _processed_path_matches(stored_path: str, match: str) -> bool:
    """True if stored_path should be considered a match for match (full path or filename)."""
    if not match:
        return False
    if stored_path == match:
        return True
    # Match by filename (stored path ends with /filename or \filename)
    return stored_path.endswith("/" + match) or stored_path.endswith("\\" + match)


def unmark_processed(match: str, group: str) -> int:
    """
    Remove processed record(s) for a file so it will be re-ingested when seen again.
    match: full ingest path or just the filename (e.g. "Issue Briefing - Key PLC Protocols.pdf").
    Returns number of entries removed.
    """
    global _processed_cache
    gp = config.get_group_paths(group)
    gp.group_dir.mkdir(parents=True, exist_ok=True)
    if not gp.processed_path.exists():
        return 0
    with open(gp.processed_path, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    kept: list[str] = []
    removed = 0
    for line in lines:
        try:
            rec = json.loads(line)
            path = rec.get("path", "")
            if _processed_path_matches(path, match):
                removed += 1
                continue
        except (json.JSONDecodeError, TypeError):
            pass
        kept.append(line)
    if removed > 0:
        with open(gp.processed_path, "w", encoding="utf-8") as f:
            for line in kept:
                f.write(line + "\n")
        with _processed_lock:
            if group in _processed_cache:
                _processed_cache[group] = {
                    (p, m, s) for (p, m, s) in _processed_cache[group]
                    if not _processed_path_matches(p, match)
                }
        logger.info("Unmarked %d processed record(s) for match=%s (group=%s)", removed, match, group)
    return removed


# --- DB (chunks, per group) ---

# SQLite busy timeout (seconds): wait for lock instead of failing with "database is locked"
SQLITE_TIMEOUT = 15


def _connect(group: str) -> sqlite3.Connection:
    gp = config.get_group_paths(group)
    gp.group_dir.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(gp.rag_db_path), timeout=SQLITE_TIMEOUT)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    # Create sources table first
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS sources (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source_path TEXT NOT NULL UNIQUE,
            source_type TEXT NOT NULL,
            created_at TEXT DEFAULT (datetime('now'))
        );
        CREATE INDEX IF NOT EXISTS ix_sources_path ON sources(source_path);
    """)
    
    # Check if chunks table exists
    table_exists = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='chunks'"
    ).fetchone() is not None
    
    if table_exists:
        # Table exists - check if source_id column exists and add it if needed
        try:
            conn.execute("SELECT source_id FROM chunks LIMIT 1")
            # Column exists, nothing to do
        except sqlite3.OperationalError:
            # Column doesn't exist, add it
            try:
                conn.execute("ALTER TABLE chunks ADD COLUMN source_id INTEGER")
                logger.info("Added source_id column to existing chunks table")
            except sqlite3.OperationalError as e:
                if "duplicate" not in str(e).lower() and "already exists" not in str(e).lower():
                    logger.warning("Could not add source_id column: %s", e)
    else:
        # Table doesn't exist, create it with source_id
        conn.executescript("""
            CREATE TABLE chunks (
                id INTEGER PRIMARY KEY,
                source_id INTEGER,
                source_path TEXT NOT NULL,
                source_type TEXT NOT NULL,
                chunk_index INTEGER NOT NULL,
                text TEXT NOT NULL,
                embedding TEXT NOT NULL,
                created_at TEXT DEFAULT (datetime('now')),
                FOREIGN KEY (source_id) REFERENCES sources(id)
            );
            CREATE INDEX IF NOT EXISTS ix_chunks_source ON chunks(source_path);
            CREATE INDEX IF NOT EXISTS ix_chunks_source_id ON chunks(source_id);
        """)
    
    # Ensure indexes exist (in case table was created before indexes were added)
    conn.executescript("""
        CREATE INDEX IF NOT EXISTS ix_chunks_source ON chunks(source_path);
        CREATE INDEX IF NOT EXISTS ix_chunks_source_id ON chunks(source_id);
    """)
    
    for col, defn in [
        ("artifact_type", "TEXT DEFAULT 'text'"),
        ("artifact_path", "TEXT"),
        ("page", "INTEGER"),
    ]:
        try:
            conn.execute(f"ALTER TABLE chunks ADD COLUMN {col} {defn}")
        except sqlite3.OperationalError as e:
            if "duplicate" not in str(e).lower():
                raise


def _migrate_sources_table(conn: sqlite3.Connection) -> None:
    """Migrate existing chunks to populate sources table and set source_id."""
    init_db(conn)
    
    # Check if source_id column exists
    try:
        conn.execute("SELECT source_id FROM chunks LIMIT 1")
        has_source_id = True
    except sqlite3.OperationalError:
        has_source_id = False
        # Try to add it
        try:
            conn.execute("ALTER TABLE chunks ADD COLUMN source_id INTEGER")
        except sqlite3.OperationalError:
            pass  # Might already exist or table doesn't exist yet
    
    # Check if sources table needs migration (has chunks but no sources)
    try:
        source_count = conn.execute("SELECT COUNT(*) FROM sources").fetchone()[0]
    except sqlite3.OperationalError:
        # Sources table doesn't exist yet, init_db should have created it
        source_count = 0
    
    if source_count > 0:
        # Check if there are chunks without source_id that need updating
        if has_source_id:
            rows = conn.execute("""
                SELECT DISTINCT c.source_path, c.source_type 
                FROM chunks c 
                WHERE c.source_id IS NULL
            """).fetchall()
            if rows:
                for row in rows:
                    source_path = row["source_path"]
                    source_type = row["source_type"]
                    # Get or create source
                    src_row = conn.execute("SELECT id FROM sources WHERE source_path = ?", (source_path,)).fetchone()
                    if src_row:
                        source_id = src_row["id"]
                    else:
                        cursor = conn.execute(
                            "INSERT INTO sources (source_path, source_type) VALUES (?, ?)",
                            (source_path, source_type)
                        )
                        source_id = cursor.lastrowid
                    # Update chunks with source_id
                    conn.execute(
                        "UPDATE chunks SET source_id = ? WHERE source_path = ? AND source_id IS NULL",
                        (source_id, source_path)
                    )
                logger.info("Updated %d source paths with source_id", len(rows))
        return  # Already has sources
    
    # Get all unique source_paths from chunks
    try:
        rows = conn.execute("SELECT DISTINCT source_path, source_type FROM chunks").fetchall()
    except sqlite3.OperationalError:
        return  # No chunks table yet
    
    if not rows:
        return  # No chunks to migrate
    
    # Create source records and update chunks
    for row in rows:
        source_path = row["source_path"]
        source_type = row["source_type"]
        cursor = conn.execute(
            "INSERT INTO sources (source_path, source_type) VALUES (?, ?)",
            (source_path, source_type)
        )
        source_id = cursor.lastrowid
        # Update chunks with source_id (if column exists)
        if has_source_id:
            conn.execute(
                "UPDATE chunks SET source_id = ? WHERE source_path = ?",
                (source_id, source_path)
            )
    logger.info("Migrated %d sources to sources table", len(rows))


def _get_or_create_source(conn: sqlite3.Connection, source_path: str, source_type: str) -> int:
    """Get or create a source record and return its ID."""
    init_db(conn)
    _migrate_sources_table(conn)  # Ensure migration is done
    # Try to get existing source
    row = conn.execute("SELECT id FROM sources WHERE source_path = ?", (source_path,)).fetchone()
    if row:
        return row["id"]
    # Create new source
    cursor = conn.execute(
        "INSERT INTO sources (source_path, source_type) VALUES (?, ?)",
        (source_path, source_type)
    )
    return cursor.lastrowid


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
    source_id = _get_or_create_source(conn, source_path, source_type)
    for i, c in enumerate(chunks):
        text = clean_text(c.get("text", ""))
        emb = c.get("embedding", [])
        atype = c.get("artifact_type", "text")
        apath = c.get("artifact_path")
        page = c.get("page")
        conn.execute(
            "INSERT INTO chunks (source_id, source_path, source_type, chunk_index, text, embedding, artifact_type, artifact_path, page) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (source_id, source_path, source_type, i, text, json.dumps(emb), atype, apath, page),
        )


def run_dedup(conn: sqlite3.Connection) -> int:
    """Remove duplicate (source_path, chunk_index), keeping min(id). Returns number of rows deleted."""
    init_db(conn)
    n_before = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    conn.execute(
        "DELETE FROM chunks WHERE id NOT IN (SELECT MIN(id) FROM chunks GROUP BY source_path, chunk_index)"
    )
    n_after = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    return n_before - n_after


def delete_source_by_id(conn: sqlite3.Connection, source_id: int) -> int:
    """Delete all chunks for a given source_id. Returns number of rows deleted."""
    init_db(conn)
    n_before = conn.execute("SELECT COUNT(*) FROM chunks WHERE source_id = ?", (source_id,)).fetchone()[0]
    conn.execute("DELETE FROM chunks WHERE source_id = ?", (source_id,))
    # Also delete the source record if no chunks remain
    conn.execute("DELETE FROM sources WHERE id = ? AND NOT EXISTS (SELECT 1 FROM chunks WHERE chunks.source_id = sources.id)", (source_id,))
    n_after = conn.execute("SELECT COUNT(*) FROM chunks WHERE source_id = ?", (source_id,)).fetchone()[0]
    return n_before - n_after


def get_source_by_id(conn: sqlite3.Connection, source_id: int) -> tuple[str, str] | None:
    """Get source_path and source_type for a given source_id. Returns (source_path, source_type) or None."""
    init_db(conn)
    _migrate_sources_table(conn)  # Ensure migration is done
    row = conn.execute("SELECT source_path, source_type FROM sources WHERE id = ?", (source_id,)).fetchone()
    if row:
        return (row["source_path"], row["source_type"])
    return None


def list_sources(conn: sqlite3.Connection) -> list[tuple[int, str, int]]:
    """List all sources with their IDs and chunk counts. Returns list of (source_id, source_path, count) tuples."""
    init_db(conn)
    _migrate_sources_table(conn)  # Ensure migration is done
    
    # Use sources table (should be populated after migration)
    rows = conn.execute("""
        SELECT s.id, s.source_path, COUNT(c.id) as count
        FROM sources s
        LEFT JOIN chunks c ON c.source_id = s.id
        GROUP BY s.id, s.source_path
        ORDER BY s.id
    """).fetchall()
    
    if rows:
        return [(row["id"], row["source_path"], row["count"]) for row in rows]
    
    # Fallback: no sources found (empty database)
    return []


def get_chunks_for_source(
    conn: sqlite3.Connection, source_id: int, page: int | None = None
) -> list[dict]:
    """
    List chunks for a source. Returns list of dicts with id, chunk_index, text, page, artifact_type.
    If page is not None, only chunks with that page (or None for page-agnostic) are returned.
    """
    init_db(conn)
    _migrate_sources_table(conn)
    if page is not None:
        rows = conn.execute(
            "SELECT id, chunk_index, text, page, artifact_type FROM chunks WHERE source_id = ? AND (page IS NULL OR page = ?) ORDER BY chunk_index",
            (source_id, page),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT id, chunk_index, text, page, artifact_type FROM chunks WHERE source_id = ? ORDER BY chunk_index",
            (source_id,),
        ).fetchall()
    return [
        {
            "id": row["id"],
            "chunk_index": row["chunk_index"],
            "text": row["text"],
            "page": row["page"],
            "artifact_type": row["artifact_type"] or "text",
        }
        for row in rows
    ]


def get_chunk_by_id(conn: sqlite3.Connection, chunk_id: int) -> dict | None:
    """Get a single chunk by id. Returns dict with id, source_id, source_path, source_type, chunk_index, text, page, artifact_type, or None."""
    init_db(conn)
    row = conn.execute(
        "SELECT id, source_id, source_path, source_type, chunk_index, text, page, artifact_type FROM chunks WHERE id = ?",
        (chunk_id,),
    ).fetchone()
    if not row:
        return None
    return dict(row)


def update_chunk_text(
    conn: sqlite3.Connection, chunk_id: int, new_text: str, new_embedding: list[float]
) -> None:
    """Update chunk text and embedding by id."""
    init_db(conn)
    text = clean_text(new_text)
    conn.execute(
        "UPDATE chunks SET text = ?, embedding = ? WHERE id = ?",
        (text, json.dumps(new_embedding), chunk_id),
    )


def insert_chunk_at(
    conn: sqlite3.Connection,
    source_id: int,
    source_path: str,
    source_type: str,
    at_index: int,
    text: str,
    embedding: list[float],
    page: int | None = None,
    artifact_type: str = "text",
    artifact_path: str | None = None,
) -> int:
    """
    Insert a new chunk at chunk_index = at_index. Existing chunks with chunk_index >= at_index are shifted by 1.
    Returns the new chunk's id.
    """
    init_db(conn)
    text = clean_text(text)
    conn.execute(
        "UPDATE chunks SET chunk_index = chunk_index + 1 WHERE source_id = ? AND chunk_index >= ?",
        (source_id, at_index),
    )
    cursor = conn.execute(
        "INSERT INTO chunks (source_id, source_path, source_type, chunk_index, text, embedding, artifact_type, artifact_path, page) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (source_id, source_path, source_type, at_index, text, json.dumps(embedding), artifact_type, artifact_path, page),
    )
    return cursor.lastrowid


def delete_chunk(conn: sqlite3.Connection, chunk_id: int) -> bool:
    """Delete a chunk by id. Returns True if a row was deleted."""
    init_db(conn)
    cursor = conn.execute("DELETE FROM chunks WHERE id = ?", (chunk_id,))
    return cursor.rowcount > 0


def _list_sync_groups() -> list[str]:
    """Group dirs under DATA_DIR that have ragdoll.db."""
    d = Path(config.DATA_DIR)
    if not d.exists():
        return []
    return [
        x.name for x in d.iterdir()
        if x.is_dir() and (x / "ragdoll.db").exists()
    ]


def run_sync_pass(group: str | None = None) -> None:
    """
    Run dedup on DB for one or all groups:
    - If group is None: run for each existing group (discovered from DATA_DIR).
    - Run dedup on DB (removes duplicate source_path+chunk_index).
    """
    groups = [group] if group is not None else _list_sync_groups()
    for g in groups:
        _run_sync_pass_one(g)


def _run_sync_pass_one(group: str) -> None:
    try:
        conn = _connect(group)
    except sqlite3.OperationalError as e:
        if "locked" in str(e).lower() or "busy" in str(e).lower():
            logger.warning("Sync pass skipped for %s: database busy (%s); will retry next interval", group, e)
            return
        raise
    try:
        init_db(conn)
        n_deleted = run_dedup(conn)
        conn.commit()
        if n_deleted > 0:
            logger.info("Dedup removed %d duplicate chunk(s) (group=%s)", n_deleted, group)
            action_log("sync_dedup", n_deleted=n_deleted, group=group)
    except sqlite3.OperationalError as e:
        if "locked" in str(e).lower() or "busy" in str(e).lower():
            logger.warning("Sync pass skipped for %s: database busy during dedup (%s); will retry next interval", group, e)
            return
        raise
    finally:
        conn.close()
