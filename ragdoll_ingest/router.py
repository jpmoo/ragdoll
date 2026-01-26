"""Route and classify free-standing or embedded images: text, table, chart, or figure."""

import re

from . import config
from .artifacts import store_chart_image, store_figure, store_table
from .chunker import chunk_text
from .extractors import ocr_image_bytes
from .interpreters import interpret_chart, interpret_figure, interpret_table
from .storage import extract_key_phrases_from_filename, extract_key_phrases_from_text


def classify_image(ocr_text: str) -> str:
    """
    Heuristic classification from OCR. Returns "text" | "table" | "chart" | "figure".
    """
    t = (ocr_text or "").strip()
    lines = [ln.strip() for ln in t.splitlines() if ln.strip()]

    # Figure: arrows, decision words, many short lines
    arrows = bool(re.search(r"->|=>|→|←|↓|↑|yes|no|start|end|decision", t, re.I))
    if arrows and len(lines) >= 3 and sum(len(l) for l in lines) / max(1, len(lines)) < 60:
        return "figure"

    # Table: many lines with 3+ columns (tabs or 2+ spaces)
    if len(lines) >= 3:
        cols = [len(re.split(r"\t|\s{2,}", ln)) for ln in lines]
        if max(cols, default=0) >= 3 and sum(1 for c in cols if c >= 2) >= len(lines) // 2:
            return "table"

    # Long prose -> text
    if len(t) > 500:
        return "text"

    # Default: chart (labels, short annotations)
    return "chart"


def parse_ocr_to_table(ocr_text: str) -> list[list[str]]:
    """Heuristic: split lines by tabs or 2+ spaces into rows of cells."""
    lines = [ln for ln in (ocr_text or "").splitlines() if ln.strip()]
    if not lines:
        return []
    # Prefer tabs; else 2+ spaces
    if any("\t" in ln for ln in lines):
        return [[c.strip() for c in ln.split("\t")] for ln in lines]
    return [[c.strip() for c in re.split(r"\s{2,}", ln)] for ln in lines]


def route_image(
    image_bytes: bytes,
    ext: str,
    page_or_idx: int | None,
    group: str,
    source_stem: str,
    idx: int,
) -> list[dict]:
    """
    OCR, classify, and route to text/chart/table/figure. Returns list of chunk dicts
    (text, artifact_type, artifact_path, page) without embedding.
    """
    ocr = ocr_image_bytes(image_bytes)
    kind = classify_image(ocr)

    if kind == "text":
        chunks = chunk_text(ocr, group=group)
        if not chunks:
            chunks = [ocr[:2000] or "(no text)"]
        return [
            {"text": c, "artifact_type": "text", "artifact_path": None, "page": page_or_idx}
            for c in chunks
        ]

    if kind == "chart":
        summary = interpret_chart(ocr, group=group)
        ap = store_chart_image(group, source_stem, page_or_idx or 0, idx, image_bytes, ext or "png")
        filename_phrases = extract_key_phrases_from_filename(str(source_stem) if source_stem else "")
        ocr_phrases = extract_key_phrases_from_text(ocr or "")
        all_phrases = [p for p in set(filename_phrases + ocr_phrases) if p and isinstance(p, str)][:10]
        # Append key phrases to summary text
        if all_phrases:
            summary = f"{summary} Key terms: {', '.join(all_phrases)}."
        return [{"text": summary, "artifact_type": "chart_summary", "artifact_path": ap, "page": page_or_idx}]

    if kind == "table":
        data = parse_ocr_to_table(ocr)
        if not data:
            data = [[ocr[:500] or "(no structure)"]]
        summary = interpret_table(data, group=group)
        ap = store_table(group, source_stem, page_or_idx, idx, data)
        filename_phrases = extract_key_phrases_from_filename(str(source_stem) if source_stem else "")
        table_text = " ".join(" ".join(str(c) if c is not None else "" for c in row) for row in (data or []) if row)
        table_phrases = extract_key_phrases_from_text(table_text)
        all_phrases = [p for p in set(filename_phrases + table_phrases) if p and isinstance(p, str)][:10]
        # Append key phrases to summary text
        if all_phrases:
            summary = f"{summary} Key terms: {', '.join(all_phrases)}."
        return [{"text": summary, "artifact_type": "table_summary", "artifact_path": ap, "page": page_or_idx}]

    # figure
    summary, process = interpret_figure(ocr, group=group)
    ap = store_figure(group, source_stem, page_or_idx or 0, idx, image_bytes, process, ocr)
    filename_phrases = extract_key_phrases_from_filename(str(source_stem) if source_stem else "")
    ocr_phrases = extract_key_phrases_from_text(ocr or "")
    all_phrases = [p for p in set(filename_phrases + ocr_phrases) if p and isinstance(p, str)][:10]
    # Append key phrases to summary text
    if all_phrases:
        summary = f"{summary} Key terms: {', '.join(all_phrases)}."
    return [{"text": summary, "artifact_type": "figure_summary", "artifact_path": ap, "page": page_or_idx}]
