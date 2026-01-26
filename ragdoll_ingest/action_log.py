"""Action log: AI calls, file moves, extract/chunk/store. No embeddings or long text. Per-group."""

import json
import threading
from datetime import datetime, timezone
from pathlib import Path

from . import config

_lock = threading.Lock()


def log(action: str, group: str = "_root", **kwargs: object) -> None:
    """
    Append one JSONL record to the group's action log. Keys and values must be JSON-serializable.
    Do not pass 'embedding', 'embeddings', or raw embedding vectors.
    """
    gp = config.get_group_paths(group or "_root")
    gp.group_dir.mkdir(parents=True, exist_ok=True)
    rec = {"ts": datetime.now(timezone.utc).isoformat(), "action": action, **kwargs}
    line = json.dumps(rec, ensure_ascii=False) + "\n"
    with _lock:
        with open(gp.action_log_path, "a", encoding="utf-8") as f:
            f.write(line)
