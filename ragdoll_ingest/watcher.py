"""File watcher for the ingest folder."""

import logging
import queue
import shutil
import threading
import time
from pathlib import Path

from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer

from . import config
from .action_log import log as action_log
from .chunker import chunk_text
from .embedder import embed
from .extractors import extract_text
from .storage import (
    _connect,
    add_chunks,
    already_processed,
    mark_processed,
    append_samples_jsonl,
    migrate_flat_to_root,
    run_sync_pass,
)

logger = logging.getLogger(__name__)


def _is_supported(p: Path) -> bool:
    return p.suffix.lower() in config.SUPPORTED_EXT


def _should_ignore(p: Path, root: Path) -> bool:
    # macOS resource-fork / AppleDouble files (._*) are not real documents; PyMuPDF etc. fail on them
    if p.name.startswith("._"):
        return True
    try:
        r = p.resolve().relative_to(root.resolve())
    except ValueError:
        return True
    s = str(r)
    return config.PROCESSED_SUBDIR in s or config.FAILED_SUBDIR in s


def _group_from_path(p: Path) -> str:
    try:
        rel = p.resolve().relative_to(Path(config.INGEST_PATH).resolve())
    except ValueError:
        return "_root"
    parts = rel.parts
    return "_root" if len(parts) == 1 else parts[0]


def _rel_within_group(p: Path) -> Path:
    """Path for the file inside the group's sources/. Only one level of grouping: the first subfolder under ingest is the group. Any deeper nesting (e.g. reports/2024/x.pdf) is flattened into a single filename (e.g. 2024_x.pdf)."""
    try:
        rel = p.resolve().relative_to(Path(config.INGEST_PATH).resolve())
    except ValueError:
        return Path(p.name)
    parts = rel.parts
    if len(parts) == 1:
        return rel  # _root: single file at top level
    # Group = first segment. Flatten parts[1:] into one name so sources/ has no nested dirs.
    return Path("_".join(parts[1:]))


def _move_to(f: Path, root: Path, subdir: str, group: str) -> Path:
    """Move file to root/subdir/rel (e.g. ingest/failed/). Only the file is moved; ingest subfolders are left as-is (even if empty)."""
    try:
        rel = f.resolve().relative_to(root.resolve())
    except ValueError:
        rel = f.name
    dest = root / subdir / rel
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        dest.unlink()
    shutil.move(str(f), str(dest))
    action_log("move", src=str(f), to=str(dest), reason=subdir, group=group)
    return dest


def _move_to_sources(f: Path, dest: Path, group: str) -> Path:
    """Move file to the group's sources folder inside the RAG output directory. Only the file is moved; we never remove empty subfolders under the ingest root."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        dest.unlink()
    shutil.move(str(f), str(dest))
    action_log("move", src=str(f), to=str(dest), reason="sources", group=group)
    return dest


def _process_one(fpath: Path) -> None:
    p = Path(fpath)
    if not p.is_file():
        return
    stat = p.stat()
    group = _group_from_path(p)
    if already_processed(str(p), stat.st_mtime, stat.st_size, group):
        action_log("already_processed", file=str(p), group=group)
        logger.info("Already processed: %s", p)
        return

    root = Path(config.INGEST_PATH)
    action_log("process_start", file=str(p), group=group)
    try:
        text = extract_text(p)
    except Exception as e:
        action_log("extract_fail", file=str(p), error=str(e), group=group)
        logger.exception("Extract failed for %s: %s", p, e)
        _move_to(p, root, config.FAILED_SUBDIR, group)
        return

    if not (text and text.strip()):
        action_log("extract_empty", file=str(p), group=group)
        logger.warning("No text extracted from %s, moving to failed", p)
        _move_to(p, root, config.FAILED_SUBDIR, group)
        return

    action_log("extract_ok", file=str(p), chars=len(text), group=group)
    try:
        chunks = chunk_text(text, group=group)
    except Exception as e:
        action_log("chunk_fail", file=str(p), error=str(e), group=group)
        logger.exception("Chunking failed for %s: %s", p, e)
        _move_to(p, root, config.FAILED_SUBDIR, group)
        return

    if not chunks:
        action_log("chunk_empty", file=str(p), group=group)
        logger.warning("No chunks from %s, moving to failed", p)
        _move_to(p, root, config.FAILED_SUBDIR, group)
        return

    action_log("chunk_ok", file=str(p), num_chunks=len(chunks), group=group)
    try:
        embs = embed(chunks, group=group)
    except Exception as e:
        action_log("embed_fail", file=str(p), error=str(e), group=group)
        logger.exception("Embed failed for %s: %s", p, e)
        _move_to(p, root, config.FAILED_SUBDIR, group)
        return

    if len(embs) != len(chunks):
        action_log("embed_mismatch", file=str(p), group=group)
        logger.error("Embed count mismatch for %s", p)
        _move_to(p, root, config.FAILED_SUBDIR, group)
        return

    # Dest: {DATA_DIR}/{group}/sources/{rel_within_group}
    rel_within = _rel_within_group(p)
    dest = config.get_group_paths(group).sources_dir / rel_within

    conn = _connect(group)
    try:
        add_chunks(conn, str(dest), p.suffix.lower(), list(zip(chunks, embs)))
        conn.commit()
    finally:
        conn.close()
    append_samples_jsonl(list(zip(chunks, embs)), str(dest), p.suffix.lower(), group)
    mark_processed(str(p), stat.st_mtime, stat.st_size, group)
    action_log("store", source=str(dest), num_chunks=len(chunks), group=group)

    _move_to_sources(p, dest, group)
    action_log("process_done", file=str(p), dest=str(dest), num_chunks=len(chunks), group=group)
    logger.info("Processed %s -> %d chunks -> %s (group=%s)", p, len(chunks), dest, group)


def _worker(q: queue.Queue, stop: threading.Event) -> None:
    while not stop.is_set():
        try:
            path = q.get(timeout=0.5)
        except queue.Empty:
            continue
        if path is None:
            break
        # Allow writes to settle
        time.sleep(2)
        if not Path(path).exists():
            continue
        try:
            _process_one(Path(path))
        except Exception as e:
            grp = _group_from_path(Path(path))
            action_log("worker_error", file=path, error=str(e), group=grp)
            logger.exception("Worker error for %s: %s", path, e)
        q.task_done()


class IngestHandler(FileSystemEventHandler):
    def __init__(self, root: Path, q: queue.Queue):
        self.root = Path(root)
        self.queue = q

    def _enqueue(self, path: str) -> None:
        p = Path(path)
        if not _is_supported(p) or _should_ignore(p, self.root):
            return
        self.queue.put(path)

    def on_created(self, event: FileSystemEvent) -> None:
        if event.is_directory:
            return
        self._enqueue(event.src_path)

    def on_moved(self, event: FileSystemEvent) -> None:
        if event.is_directory:
            return
        self._enqueue(event.dest_path)


def _scan_existing(root: Path, q: queue.Queue) -> None:
    for p in root.rglob("*"):
        if p.is_file() and _is_supported(p) and not _should_ignore(p, root):
            q.put(str(p))


def _sync_loop() -> None:
    while True:
        time.sleep(config.SYNC_INTERVAL)
        try:
            run_sync_pass()
        except Exception as e:
            logger.exception("Sync pass failed: %s", e)


def run_watcher(process_existing: bool = True) -> None:
    if not config.INGEST_PATH or not config.INGEST_PATH.is_dir():
        raise SystemExit("RAGDOLL_INGEST_PATH must be set to an existing directory")

    migrate_flat_to_root()
    action_log("watcher_start", ingest_path=str(config.INGEST_PATH), group="_root")
    q: queue.Queue = queue.Queue()
    stop = threading.Event()
    t = threading.Thread(target=_worker, args=(q, stop), daemon=False)
    t.start()

    try:
        run_sync_pass()
    except Exception as e:
        logger.exception("Initial sync pass failed: %s", e)
    if config.SYNC_INTERVAL > 0:
        sync_thread = threading.Thread(target=_sync_loop, daemon=True)
        sync_thread.start()

    if process_existing:
        _scan_existing(Path(config.INGEST_PATH), q)

    observer = Observer()
    observer.schedule(IngestHandler(config.INGEST_PATH, q), str(config.INGEST_PATH), recursive=True)
    observer.start()
    logger.info("Watching %s", config.INGEST_PATH)

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()
        q.put(None)
        t.join(timeout=5)
        observer.stop()
        observer.join(timeout=5)
