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

# Optional: output folder for RAG JSONL and sources (user-specified)
# RAGDOLL_OUTPUT_PATH or RAGDOLL_DATA_DIR; sources/ will be created inside it
DATA_DIR = get_env_path("RAGDOLL_OUTPUT_PATH") or get_env_path("RAGDOLL_DATA_DIR") or (Path(__file__).resolve().parents[1] / "data")
SAMPLES_PATH = get_env_path("RAGDOLL_SAMPLES") or (DATA_DIR / "rag_samples.jsonl")
PROCESSED_PATH = get_env_path("RAGDOLL_PROCESSED") or (DATA_DIR / "processed.jsonl")

# Sources: original documents are moved here (inside DATA_DIR) after successful ingest
SOURCES_SUBDIR = "sources"

# Action log: AI calls, file moves, extract/chunk/store (no embeddings or long text)
ACTION_LOG_PATH = get_env_path("RAGDOLL_ACTION_LOG") or (DATA_DIR / "action.log")

# Subfolders inside ingest: we don't watch these; failed files go to failed/
PROCESSED_SUBDIR = "processed"
FAILED_SUBDIR = "failed"

# Ollama
OLLAMA_HOST = get_env("RAGDOLL_OLLAMA_HOST") or get_env("OLLAMA_HOST") or "http://localhost:11434"
EMBED_MODEL = get_env("RAGDOLL_EMBED_MODEL") or "nomic-embed-text:latest"
CHUNK_MODEL = get_env("RAGDOLL_CHUNK_MODEL") or "llama3.2:3b"

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
