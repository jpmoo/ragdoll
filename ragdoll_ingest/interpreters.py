"""LLM interpreters for charts and tables. Produce qualitative summaries only; no numeric guessing. Anti-hallucination."""

import json
import logging
import re
from typing import Any

from . import config
from .action_log import log as action_log

logger = logging.getLogger(__name__)

AUTH = (
    "Do not invent values, steps, or relationships. If something is unclear, say so."
)


def _ollama_json(prompt: str, model: str, group: str = "_root", timeout: int | None = None) -> dict[str, Any] | None:
    timeout = timeout or config.CHUNK_LLM_TIMEOUT
    url = (config.OLLAMA_HOST or "").rstrip("/")
    try:
        import requests

        r = requests.post(
            f"{url}/api/generate",
            json={"model": model, "prompt": prompt, "stream": False, "format": "json"},
            timeout=timeout,
        )
        r.raise_for_status()
        resp = (r.json().get("response") or "").strip()
        if "```" in resp:
            m = re.search(r"```(?:json)?\s*([\s\S]*?)```", resp)
            if m:
                resp = m.group(1).strip()
        return json.loads(resp)
    except Exception as e:
        logger.warning("Ollama interpret request failed: %s", e)
        return None


def interpret_chart(ocr_text: str, group: str = "_root") -> str:
    """
    Qualitative chart summary from OCR of titles, labels, legends. No numeric guessing.
    Store: image + OCR; Embed: this summary only.
    """
    model = config.INTERPRET_MODEL
    prompt = (
        "You are summarizing a chart or graph for a RAG system. Use ONLY the OCR text from the chart (titles, axis labels, legends, annotations).\n"
        "Output a short qualitative summary: what is being compared, major trends, outliers, and any annotations. "
        "Do NOT guess or invent specific numbers from bars or lines. "
        f"{AUTH}\n\n"
        "Return valid JSON: {\"summary\": \"your summary here\"}\n\n"
        "OCR text:\n"
    ) + (ocr_text.strip() or "(no text detected)")

    obj = _ollama_json(prompt, model, group)
    if isinstance(obj, dict) and isinstance(obj.get("summary"), str) and obj["summary"].strip():
        action_log("interpret_chart", model=model, group=group)
        return obj["summary"].strip()
    fallback = f"Chart: {ocr_text[:500].strip() or 'no OCR text'}." if ocr_text else "Chart: no OCR text."
    action_log("interpret_chart", model=model, fallback=True, group=group)
    return fallback


def interpret_flowchart(ocr_text: str, group: str = "_root") -> tuple[str, dict]:
    """
    Infer process: steps, decisions, conditions, actors, end states. State uncertainty if unclear.
    Returns (summary: str for embedding, process_dict for storage).
    Store: process JSON + OCR + image ref; Embed: summary only.
    """
    model = config.INTERPRET_MODEL
    prompt = (
        "You are analyzing a flowchart or process diagram for a RAG system. Use ONLY the OCR text from the diagram.\n"
        "Infer: steps, decisions (with conditions), actors, and end states. If order or branching is unclear, state the uncertainty. "
        f"{AUTH}\n\n"
        'Return valid JSON: {"summary": "natural-language process summary", "steps": ["step1", ...], "decisions": [{"label": "...", "condition": "..."}], "actors": [], "end_states": []}\n\n'
        "OCR text:\n"
    ) + (ocr_text.strip() or "(no text detected)")

    obj = _ollama_json(prompt, model, group)
    if isinstance(obj, dict) and obj.get("summary"):
        summary = str(obj["summary"]).strip()
        process = {k: obj[k] for k in ("steps", "decisions", "actors", "end_states") if k in obj}
        action_log("interpret_flowchart", model=model, group=group)
        return summary, process
    fallback = f"Flowchart: {ocr_text[:500].strip() or 'no OCR text'}." if ocr_text else "Flowchart: no OCR text."
    action_log("interpret_flowchart", model=model, fallback=True, group=group)
    return fallback, {"steps": [], "decisions": [], "actors": [], "end_states": []}


def interpret_table(table_data: list[list[str]], group: str = "_root") -> str:
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

    prompt = (
        "You are summarizing a table for a RAG system. Use only the provided cells.\n"
        "Output: purpose of the table, main metrics, key comparisons or rankings, and any trends or notes. "
        "Do not invent or guess values that are not in the table. "
        f"{AUTH}\n\n"
        "Return valid JSON: {\"summary\": \"your summary here\"}\n\n"
        "Table (tab-separated):\n"
    ) + tbl

    obj = _ollama_json(prompt, model, group)
    if isinstance(obj, dict) and isinstance(obj.get("summary"), str) and obj["summary"].strip():
        action_log("interpret_table", model=model, rows=len(table_data), group=group)
        return obj["summary"].strip()
    fallback = "Table: " + (tbl[:400].replace("\n", " ") or "empty")
    action_log("interpret_table", model=model, fallback=True, group=group)
    return fallback
