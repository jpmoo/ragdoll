"""Action log: AI calls, file moves, extract/chunk/store. No embeddings or long text."""

import json
import threading
from datetime import datetime, timezone
from pathlib import Path

from . import config

_lock = threading.Lock()


def log(action: str, **kwargs: object) -> None:
    """
    Append one JSONL record to the action log. Keys and values must be JSON-serializable.
    Do not pass 'embedding', 'embeddings', or raw embedding vectors.
    """
    path = Path(config.ACTION_LOG_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    rec = {"ts": datetime.now(timezone.utc).isoformat(), "action": action, **kwargs}
    line = json.dumps(rec, ensure_ascii=False) + "\n"
    with _lock:
        with open(path, "a", encoding="utf-8") as f:
            f.write(line)
