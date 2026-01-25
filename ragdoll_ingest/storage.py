"""JSONL storage: RAG samples and processed-file dedup (no DB)."""

import json
import logging
import threading
from pathlib import Path

from . import config

logger = logging.getLogger(__name__)

_processed_cache: set[tuple[str, float, int]] | None = None
_processed_lock = threading.Lock()


def _ensure_processed_loaded() -> None:
    global _processed_cache
    with _processed_lock:
        if _processed_cache is not None:
            return
        p = Path(config.PROCESSED_PATH)
        _processed_cache = set()
        if p.exists():
            with open(p, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    rec = json.loads(line)
                    _processed_cache.add((rec["path"], rec["mtime"], rec["size"]))
        logger.debug("Loaded %d processed records from %s", len(_processed_cache), p)


def already_processed(path: str, mtime: float, size: int) -> bool:
    _ensure_processed_loaded()
    with _processed_lock:
        return (path, mtime, size) in _processed_cache


def mark_processed(path: str, mtime: float, size: int) -> None:
    _ensure_processed_loaded()
    p = Path(config.PROCESSED_PATH)
    p.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps({"path": path, "mtime": mtime, "size": size}, ensure_ascii=False) + "\n"
    with open(p, "a", encoding="utf-8") as f:
        f.write(line)
    with _processed_lock:
        _processed_cache.add((path, mtime, size))
    logger.debug("Marked processed: %s", path)


def append_samples_jsonl(chunks: list[tuple[str, list[float]]], source_path: str, source_type: str) -> None:
    """Append new chunk samples to the master RAG samples JSONL file."""
    Path(config.SAMPLES_PATH).parent.mkdir(parents=True, exist_ok=True)
    with open(config.SAMPLES_PATH, "a", encoding="utf-8") as f:
        for i, (text, emb) in enumerate(chunks):
            rec = {
                "text": text,
                "embedding": emb,
                "source": source_path,
                "source_type": source_type,
                "chunk_index": i,
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    logger.info("Appended %d samples to %s", len(chunks), config.SAMPLES_PATH)
