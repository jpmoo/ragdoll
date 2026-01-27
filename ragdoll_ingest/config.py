"""Configuration from environment variables and optional env.ragdoll file."""

import os
from pathlib import Path


def _load_env_file() -> None:
    """Load KEY=value lines from env.ragdoll (project root) or path in RAGDOLL_ENV. setdefault so env overrides."""
    path = os.environ.get("RAGDOLL_ENV")
    path = Path(path).expanduser().resolve() if path else Path(__file__).resolve().parents[1] / "env.ragdoll"
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                eq = line.find("=")
                if eq >= 0:
                    k, v = line[:eq].strip(), line[eq + 1 :].strip()
                    if k:
                        os.environ.setdefault(k, v)


_load_env_file()


def get_env(key: str, default: str | None = None) -> str | None:
    return os.environ.get(key, default)


def get_env_path(key: str, default: Path | None = None) -> Path | None:
    v = get_env(key)
    if v is None or v == "":
        return default
    return Path(v).expanduser().resolve()


# Required: user-provided ingest folder
INGEST_PATH = get_env_path("RAGDOLL_INGEST_PATH")

# Optional: output folder for RAG DB and sources (user-specified)
# RAGDOLL_OUTPUT_PATH or RAGDOLL_DATA_DIR; each group gets {DATA_DIR}/{group}/ with its own
# ragdoll.db, processed.jsonl, action.log, sources/
DATA_DIR = get_env_path("RAGDOLL_OUTPUT_PATH") or get_env_path("RAGDOLL_DATA_DIR") or (Path(__file__).resolve().parents[1] / "data")

# Sync: dedup DB; run every N seconds (0 = disabled)
SYNC_INTERVAL = int(get_env("RAGDOLL_SYNC_INTERVAL") or "300")

# Sources: original documents are moved here (inside each group dir) after successful ingest
SOURCES_SUBDIR = "sources"

# Artifacts: stored chart images, table JSON; under {group}/artifacts/charts|tables/
ARTIFACTS_SUBDIR = "artifacts"


class GroupPaths:
    def __init__(
        self,
        *,
        group_dir: Path,
        rag_db_path: Path,
        processed_path: Path,
        action_log_path: Path,
        sources_dir: Path,
    ):
        self.group_dir = group_dir
        self.rag_db_path = rag_db_path
        self.processed_path = processed_path
        self.action_log_path = action_log_path
        self.sources_dir = sources_dir
        self.artifacts_dir = group_dir / ARTIFACTS_SUBDIR


def _sanitize_group(g: str) -> str:
    if not g or g in (".", ".."):
        return "_root"
    return "".join(c if (c.isalnum() or c in "_.-") else "_" for c in g) or "_root"


def get_group_paths(group: str) -> GroupPaths:
    """Paths for one output group. group='_root' for top-level ingest files; else first subfolder name."""
    s = _sanitize_group(group)
    d = DATA_DIR / s
    return GroupPaths(
        group_dir=d,
        rag_db_path=d / "ragdoll.db",
        processed_path=d / "processed.jsonl",
        action_log_path=d / "action.log",
        sources_dir=d / SOURCES_SUBDIR,
    )

# Subfolders inside ingest: we don't watch these; failed files go to failed/
PROCESSED_SUBDIR = "processed"
FAILED_SUBDIR = "failed"

# Ollama
OLLAMA_HOST = get_env("RAGDOLL_OLLAMA_HOST") or get_env("OLLAMA_HOST") or "http://localhost:11434"
EMBED_MODEL = get_env("RAGDOLL_EMBED_MODEL") or "nomic-embed-text:latest"
CHUNK_MODEL = get_env("RAGDOLL_CHUNK_MODEL") or "llama3.2:3b"
# LLM for chart/table/figure interpretation (qualitative summaries; no numeric guessing)
INTERPRET_MODEL = get_env("RAGDOLL_INTERPRET_MODEL") or CHUNK_MODEL
# LLM for query expansion (standalone description of information need)
QUERY_MODEL = get_env("RAGDOLL_QUERY_MODEL") or "llama3.2:3b"

# API server
API_PORT = int(get_env("RAGDOLL_API_PORT") or "9042")

# Garbage control
GARBAGE_MIN_CHARS = int(get_env("RAGDOLL_GARBAGE_MIN_CHARS") or "20")
GARBAGE_MIN_TOKENS = int(get_env("RAGDOLL_GARBAGE_MIN_TOKENS") or "10")
GARBAGE_MIN_DIVERSITY = float(get_env("RAGDOLL_GARBAGE_MIN_DIVERSITY") or "0.3")
GARBAGE_MAX_STOPWORD_RATIO = float(get_env("RAGDOLL_GARBAGE_MAX_STOPWORD_RATIO") or "0.7")
GARBAGE_MIN_SCORE = float(get_env("RAGDOLL_GARBAGE_MIN_SCORE") or "0.3")
GARBAGE_LLM_VALIDATION = (get_env("RAGDOLL_GARBAGE_LLM_VALIDATION") or "false").lower() in ("true", "1", "yes")

# Chunking
TARGET_CHUNK_TOKENS = int(get_env("RAGDOLL_TARGET_CHUNK_TOKENS") or "400")
MAX_CHUNK_TOKENS = int(get_env("RAGDOLL_MAX_CHUNK_TOKENS") or "600")
OVERLAP_SENTENCES = int(get_env("RAGDOLL_OVERLAP_SENTENCES") or "1")
CHUNK_LLM_TIMEOUT = int(get_env("RAGDOLL_CHUNK_LLM_TIMEOUT") or "300")

# Supported extensions (lowercase)
TEXT_EXT = {".txt", ".md", ".markdown"}
WORD_EXT = {".docx"}  # .doc would need LibreOffice/antiword
EXCEL_EXT = {".xlsx", ".xls"}
PDF_EXT = {".pdf"}
IMAGE_EXT = {".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp", ".gif"}

SUPPORTED_EXT = TEXT_EXT | WORD_EXT | EXCEL_EXT | PDF_EXT | IMAGE_EXT
