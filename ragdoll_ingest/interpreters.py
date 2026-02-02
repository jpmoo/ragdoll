"""LLM interpreters for charts and tables. Produce qualitative summaries only; no numeric guessing. Anti-hallucination."""

import json
import logging
import re

from . import config
from .action_log import log as action_log

logger = logging.getLogger(__name__)

CHUNK_ROLES = (
    "description",
    "application",
    "implication",
)

AUTH = (
    "Do not invent values, steps, or relationships. If something is unclear, say so. "
    "Be neutral and descriptive rather than evaluative. Focus on what is present, not what is missing or problematic."
)


def _ollama_text(prompt: str, model: str, group: str = "_root", timeout: int | None = None) -> str | None:
    """Call Ollama and return raw response text (no JSON required). Returns None on request failure or empty response."""
    timeout = timeout or config.CHUNK_LLM_TIMEOUT
    url = (config.OLLAMA_HOST or "").rstrip("/")
    try:
        import requests

        r = requests.post(
            f"{url}/api/generate",
            json={"model": model, "prompt": prompt, "stream": False},
            timeout=timeout,
        )
        r.raise_for_status()
        out = (r.json().get("response") or "").strip() or None
        if out is None:
            logger.info("Ollama returned empty or whitespace-only response (model=%s)", model)
        return out
    except Exception as e:
        logger.warning("Ollama interpret request failed: %s", e)
        return None


# Max chars of document to send for one-sentence summary (avoids huge prompts)
DOC_SUMMARY_MAX_CHARS = 12_000
# Max chars for the summary sentence (enforced as fallback; prompt asks model to stay under this)
DOC_SUMMARY_RESPONSE_MAX_CHARS = 200


def summarize_document(
    document_text: str,
    group: str = "_root",
    filename: str | None = None,
) -> str:
    """
    Produce a one-sentence summary of the document via LLM.
    Returns the summary string only (no prefix). Caller appends bracketed
    "[SUMMARY of filename: ...]" when storing. Returns empty string on failure.
    """
    if not (document_text and document_text.strip()):
        return ""
    text = document_text.strip()
    if len(text) > DOC_SUMMARY_MAX_CHARS:
        text = text[:DOC_SUMMARY_MAX_CHARS] + "\n\n[... document truncated ...]"
    model = config.CHUNK_MODEL
    filename_context = f"Document filename: {filename.strip()}\n\n" if filename and filename.strip() else ""
    prompt = (
        "Summarize the following document in exactly one short sentence. "
        "Keep it as short as possible. "
        "Be factual and descriptive. Reply with only that one sentence, no preamble or labels.\n\n"
        f"{filename_context}"
        "Document:\n\n"
    ) + text
    summary = _ollama_text(prompt, model, group)
    if not summary:
        return ""
    summary = summary.strip()
    # Use first sentence only if model produced multiple
    first_sentence = summary.split(". ")[0].strip()
    if first_sentence and not first_sentence.endswith("."):
        first_sentence += "."
    summary = first_sentence or summary
    # Enforce character cap as fallback
    if len(summary) > DOC_SUMMARY_RESPONSE_MAX_CHARS:
        summary = summary[: DOC_SUMMARY_RESPONSE_MAX_CHARS - 1].rsplit(" ", 1)[0]
        if summary and not summary.endswith("."):
            summary += "."
    action_log("summarize_document", model=model, group=group)
    return summary


# Max chars of chunk text to send for semantic labels (avoid huge prompts)
CHUNK_SEMANTIC_MAX_CHARS = 8000


def extract_chunk_semantic_labels(chunk_text: str, group: str = "_root") -> dict:
    """
    Ask LLM for semantic labels for a chunk. Returns dict with: concept, decision_context,
    primary_question_answered, key_signals (list), chunk_role. Any field may be empty if
    LLM returns garbage or omits. chunk_role must be one of CHUNK_ROLES or empty.
    """
    out = {
        "concept": "",
        "decision_context": "",
        "primary_question_answered": "",
        "key_signals": [],
        "chunk_role": "",
    }
    if not (chunk_text and chunk_text.strip()):
        return out
    text = chunk_text.strip()
    if len(text) > CHUNK_SEMANTIC_MAX_CHARS:
        text = text[:CHUNK_SEMANTIC_MAX_CHARS] + "\n\n[... truncated ...]"
    roles_str = ", ".join(CHUNK_ROLES)
    prompt = (
        "For the following text chunk, extract semantic labels. "
        "Reply with ONLY valid JSON in this exact format, no other text:\n"
        '{"concept": "main concept as a short label", '
        '"decision_context": "main area of work where this chunk would be useful (e.g. organizational improvement planning)", '
        '"primary_question_answered": "single main question this chunk might answer", '
        '"key_signals": ["condition 1", "condition 2", "..."], '
        '"chunk_role": "exactly one of: ' + roles_str + '"}\n\n'
        "Use empty string or empty array for any field you cannot determine. "
        "chunk_role must be exactly one of: " + roles_str + ". "
        "key_signals: 3-5 short conditions that would make this chunk relevant (e.g. teachers are exhausted).\n\n"
        "Chunk text:\n\n"
    ) + text
    model = config.CHUNK_MODEL
    resp = _ollama_text(prompt, model, group)
    if not resp:
        return out
    resp = resp.strip()
    if "```" in resp:
        m = re.search(r"```(?:json)?\s*([\s\S]*?)```", resp)
        if m:
            resp = m.group(1).strip()
    try:
        obj = json.loads(resp)
    except json.JSONDecodeError:
        return out
    if not isinstance(obj, dict):
        return out
    concept = obj.get("concept")
    if isinstance(concept, str) and concept.strip():
        out["concept"] = concept.strip()
    decision_context = obj.get("decision_context")
    if isinstance(decision_context, str) and decision_context.strip():
        out["decision_context"] = decision_context.strip()
    primary_question_answered = obj.get("primary_question_answered")
    if isinstance(primary_question_answered, str) and primary_question_answered.strip():
        out["primary_question_answered"] = primary_question_answered.strip()
    key_signals = obj.get("key_signals")
    if isinstance(key_signals, list):
        out["key_signals"] = [str(s).strip() for s in key_signals if str(s).strip()][:10]
    chunk_role = obj.get("chunk_role")
    if isinstance(chunk_role, str):
        chunk_role = chunk_role.strip().lower()
        if chunk_role in CHUNK_ROLES:
            out["chunk_role"] = chunk_role
    action_log("extract_chunk_semantic_labels", model=model, group=group)
    return out


def interpret_chart(ocr_text: str, group: str = "_root", filename: str | None = None) -> str:
    """
    Qualitative chart summary from OCR of titles, labels, legends. No numeric guessing.
    Store: image + OCR; Embed: this summary only.
    """
    model = config.INTERPRET_MODEL
    filename_context = f"Source filename: {filename}\n\n" if filename else ""
    prompt = (
        "You are summarizing a chart or graph. Use ONLY the OCR text from the chart (titles, axis labels, legends, annotations).\n"
        f"{filename_context}"
        "Output a short qualitative summary: what is being compared, major trends, outliers, and any annotations. "
        "Include relevant context from the filename if it provides useful information. "
        "Do NOT guess or invent specific numbers from bars or lines. "
        f"{AUTH}\n\n"
        "Reply with only your summary, no JSON and no preamble.\n\n"
        "OCR text:\n"
    ) + (ocr_text.strip() or "(no text detected)")

    summary = _ollama_text(prompt, model, group)
    if summary:
        action_log("interpret_chart", model=model, group=group)
        return summary
    fallback = f"Chart: {ocr_text[:500].strip() or 'no OCR text'}." if ocr_text else "Chart: no OCR text."
    action_log("interpret_chart", model=model, fallback=True, group=group)
    return fallback


def interpret_figure(ocr_text: str, group: str = "_root", filename: str | None = None) -> tuple[str, dict]:
    """
    Infer process: steps, decisions, conditions, actors, end states. State uncertainty if unclear.
    Returns (summary: str for embedding, process_dict for storage).
    Store: process JSON + OCR + image ref; Embed: summary only.
    """
    model = config.INTERPRET_MODEL
    filename_context = f"Source filename: {filename}\n\n" if filename else ""
    prompt = (
        "You are analyzing a figure or process diagram. Use ONLY the OCR text from the diagram.\n"
        f"{filename_context}"
        "Infer: steps, decisions (with conditions), actors, and end states. If order or branching is unclear, state the uncertainty. "
        "Include relevant context from the filename if it provides useful information. "
        f"{AUTH}\n\n"
        "Reply with only your process summary, no JSON and no preamble.\n\n"
        "OCR text:\n"
    ) + (ocr_text.strip() or "(no text detected)")

    summary = _ollama_text(prompt, model, group)
    if summary:
        action_log("interpret_figure", model=model, group=group)
        return summary, {"steps": [], "decisions": [], "actors": [], "end_states": []}
    fallback = f"Figure: {ocr_text[:500].strip() or 'no OCR text'}." if ocr_text else "Figure: no OCR text."
    action_log("interpret_figure", model=model, fallback=True, group=group)
    return fallback, {"steps": [], "decisions": [], "actors": [], "end_states": []}


def interpret_table(table_data: list[list[str]], group: str = "_root", filename: str | None = None) -> str:
    """
    Summarize table purpose, metrics, key comparisons, rankings, trends. No inventing.
    Store: full table (JSON/CSV); Embed: this summary only.
    """
    model = config.INTERPRET_MODEL
    # Subset if huge: first 20 rows
    rows = table_data[:20]
    tbl = "\n".join("\t".join(str(c) for c in row) for row in rows)
    if len(table_data) > 20:
        tbl += f"\n... ({len(table_data) - 20} more rows)"

    filename_context = f"Source filename: {filename}\n\n" if filename else ""
    prompt = (
        "You are summarizing a table. Use only the provided cells.\n"
        f"{filename_context}"
        "Output: purpose of the table, main metrics, key comparisons or rankings, and any trends or notes. "
        "Include relevant context from the filename if it provides useful information. "
        "Do not invent or guess values that are not in the table. "
        f"{AUTH}\n\n"
        "Reply with only your summary, no JSON and no preamble.\n\n"
        "Table (tab-separated):\n"
    ) + tbl

    summary = _ollama_text(prompt, model, group)
    if summary:
        action_log("interpret_table", model=model, rows=len(table_data), group=group)
        return summary
    fallback = "Table: " + (tbl[:400].replace("\n", " ") or "empty")
    action_log("interpret_table", model=model, fallback=True, group=group)
    return fallback
