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
    already_processed,
    mark_processed,
    append_samples_jsonl,
)

logger = logging.getLogger(__name__)


def _is_supported(p: Path) -> bool:
    return p.suffix.lower() in config.SUPPORTED_EXT


def _should_ignore(p: Path, root: Path) -> bool:
    try:
        r = p.resolve().relative_to(root.resolve())
    except ValueError:
        return True
    s = str(r)
    return config.PROCESSED_SUBDIR in s or config.FAILED_SUBDIR in s


def _move_to(f: Path, root: Path, subdir: str) -> Path:
    try:
        rel = f.resolve().relative_to(root.resolve())
    except ValueError:
        rel = f.name
    dest = root / subdir / rel
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        dest.unlink()
    shutil.move(str(f), str(dest))
    action_log("move", src=str(f), to=str(dest), reason=subdir)
    return dest


def _move_to_sources(f: Path, dest: Path) -> Path:
    """Move file to the sources folder inside the RAG output directory."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        dest.unlink()
    shutil.move(str(f), str(dest))
    action_log("move", src=str(f), to=str(dest), reason="sources")
    return dest


def _process_one(fpath: Path) -> None:
    p = Path(fpath)
    if not p.is_file():
        return
    stat = p.stat()
    if already_processed(str(p), stat.st_mtime, stat.st_size):
        action_log("already_processed", file=str(p))
        logger.info("Already processed: %s", p)
        return

    root = Path(config.INGEST_PATH)
    action_log("process_start", file=str(p))
    try:
        text = extract_text(p)
    except Exception as e:
        action_log("extract_fail", file=str(p), error=str(e))
        logger.exception("Extract failed for %s: %s", p, e)
        _move_to(p, root, config.FAILED_SUBDIR)
        return

    if not (text and text.strip()):
        action_log("extract_empty", file=str(p))
        logger.warning("No text extracted from %s, moving to failed", p)
        _move_to(p, root, config.FAILED_SUBDIR)
        return

    action_log("extract_ok", file=str(p), chars=len(text))
    try:
        chunks = chunk_text(text)
    except Exception as e:
        action_log("chunk_fail", file=str(p), error=str(e))
        logger.exception("Chunking failed for %s: %s", p, e)
        _move_to(p, root, config.FAILED_SUBDIR)
        return

    if not chunks:
        action_log("chunk_empty", file=str(p))
        logger.warning("No chunks from %s, moving to failed", p)
        _move_to(p, root, config.FAILED_SUBDIR)
        return

    action_log("chunk_ok", file=str(p), num_chunks=len(chunks))
    try:
        embs = embed(chunks)
    except Exception as e:
        action_log("embed_fail", file=str(p), error=str(e))
        logger.exception("Embed failed for %s: %s", p, e)
        _move_to(p, root, config.FAILED_SUBDIR)
        return

    if len(embs) != len(chunks):
        action_log("embed_mismatch", file=str(p))
        logger.error("Embed count mismatch for %s", p)
        _move_to(p, root, config.FAILED_SUBDIR)
        return

    # Final path: {OUTPUT}/sources/{rel}; rel preserves structure under ingest
    try:
        rel = p.resolve().relative_to(Path(config.INGEST_PATH).resolve())
    except ValueError:
        rel = Path(p.name)
    dest = config.DATA_DIR / config.SOURCES_SUBDIR / rel

    append_samples_jsonl(list(zip(chunks, embs)), str(dest), p.suffix.lower())
    mark_processed(str(p), stat.st_mtime, stat.st_size)
    action_log("store", source=str(dest), num_chunks=len(chunks))

    _move_to_sources(p, dest)
    action_log("process_done", file=str(p), dest=str(dest), num_chunks=len(chunks))
    logger.info("Processed %s -> %d chunks -> %s", p, len(chunks), dest)


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
            action_log("worker_error", file=path, error=str(e))
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


def run_watcher(process_existing: bool = True) -> None:
    if not config.INGEST_PATH or not config.INGEST_PATH.is_dir():
        raise SystemExit("RAGDOLL_INGEST_PATH must be set to an existing directory")

    action_log("watcher_start", ingest_path=str(config.INGEST_PATH))
    q: queue.Queue = queue.Queue()
    stop = threading.Event()
    t = threading.Thread(target=_worker, args=(q, stop), daemon=False)
    t.start()

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
