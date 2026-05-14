"""Shared CSV column order for chunk export (Review UI) and ``ragdoll import-csv``."""

# One row per chunk; source metadata repeated. Keep in sync with export/import tooling.
CHUNK_CSV_HEADERS: tuple[str, ...] = (
    "source_key",
    "canonical_url",
    "source_title",
    "fetched_at",
    "source_type",
    "source_path",
    "doc_summary",
    "chunk_index",
    "text",
    "page",
    "chunk_role",
    "primary_question_answered",
    "key_signals",
    "artifact_type",
    "artifact_path",
    "concept",
    "decision_context",
)
