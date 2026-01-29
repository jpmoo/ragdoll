"""Docling-based document extraction. Converts Docling output to RAGDoll Document. Optional: install with pip install -e '.[docling]'."""

import io
import logging
from pathlib import Path

from . import config
from .extractors import (
    ChartRegion,
    Document,
    FigureRegion,
    TableRegion,
    TextBlock,
)

logger = logging.getLogger(__name__)

# Extensions that Docling can ingest (subset of SUPPORTED_EXT)
DOCLING_EXT = config.PDF_EXT | config.WORD_EXT | config.EXCEL_EXT | {".pptx"} | config.IMAGE_EXT


def _page_from_prov(item) -> int | None:
    """Get page number from a Docling item's provenance, if any."""
    prov = getattr(item, "prov", None) or []
    if prov and len(prov) > 0:
        p = prov[0]
        return getattr(p, "page_no", None)
    return getattr(item, "page_no", None)


def _docling_to_document(conv_res, path: Path) -> Document:
    """Map Docling ConversionResult to RAGDoll Document."""
    doc = Document()
    dd = conv_res.document

    # Text: use the richer source. Markdown can be sparse for layout-heavy PDFs (tables/lists);
    # document.texts can under-represent for other docs. Prefer whichever has more prose.
    texts_items = getattr(dd, "texts", []) or []
    texts_combined_len = sum(
        len(str((getattr(item, "text", None) or getattr(item, "orig", "") or "")).strip())
        for item in texts_items
    )
    full_md = None
    if hasattr(dd, "export_to_markdown") and callable(getattr(dd, "export_to_markdown")):
        try:
            full_md = dd.export_to_markdown()
        except Exception as e:
            logger.debug("Docling export_to_markdown failed: %s", e)
    full_md = (full_md and isinstance(full_md, str) and full_md.strip()) or None
    # Use markdown only when it's clearly richer (avoids sparse markdown on table-heavy PDFs)
    use_markdown = full_md and len(full_md) >= max(texts_combined_len, 400)
    if use_markdown:
        doc.text_blocks.append(TextBlock(page=None, text=full_md))
    else:
        for item in texts_items:
            text = getattr(item, "text", None) or getattr(item, "orig", "") or ""
            if not (text and str(text).strip()):
                continue
            page = _page_from_prov(item)
            doc.text_blocks.append(TextBlock(page=page, text=str(text).strip()))
        # If we had no text items, use markdown anyway when available
        if not doc.text_blocks and full_md:
            doc.text_blocks.append(TextBlock(page=None, text=full_md))

    # Tables
    for idx, item in enumerate(getattr(dd, "tables", []) or []):
        page = _page_from_prov(item)
        try:
            df = item.export_to_dataframe(doc=dd)
            if df is not None and not df.empty:
                # Fill NaN, convert to list[list[str]]
                data = df.fillna("").astype(str).values.tolist()
                if data:
                    doc.table_regions.append(TableRegion(page=page, data=data))
                    # Also add table cell text as prose so protocol names/lists are searchable (not just the LLM summary)
                    table_text_lines = [" ".join(str(c or "").strip() for c in row if (c or "").strip()) for row in data]
                    table_text = "\n".join(ln for ln in table_text_lines if ln)
                    if len(table_text) > 80:
                        doc.text_blocks.append(TextBlock(page=page, text=table_text))
        except Exception as e:
            logger.debug("Docling table export failed for table %s: %s", idx, e)

    # Pictures: chart vs figure by label
    for idx, item in enumerate(getattr(dd, "pictures", []) or []):
        page = _page_from_prov(item)
        if page is None:
            page = 1
        try:
            pil_img = item.get_image(dd)
            if pil_img is None:
                continue
            buf = io.BytesIO()
            pil_img.save(buf, format="PNG")
            image_bytes = buf.getvalue()
            label = getattr(item, "label", None)
            if label == "chart":
                doc.chart_regions.append(ChartRegion(page=page, image_bytes=image_bytes, image_ext="png"))
            else:
                doc.figure_regions.append(FigureRegion(page=page, image_bytes=image_bytes))
        except Exception as e:
            logger.debug("Docling picture export failed for picture %s: %s", idx, e)

    return doc


def extract_document_with_docling(path: Path) -> Document | None:
    """
    Convert a file with Docling and return a RAGDoll Document, or None if format unsupported or conversion failed.
    Requires: pip install '.[docling]'
    """
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix not in DOCLING_EXT:
        return None

    try:
        from docling.document_converter import DocumentConverter
        from docling.datamodel.base_models import InputFormat
    except ImportError:
        logger.debug("Docling not installed; install with pip install '.[docling]'")
        return None

    # Map suffix to Docling InputFormat
    fmt = None
    if suffix in config.PDF_EXT:
        fmt = InputFormat.PDF
    elif suffix in config.WORD_EXT:
        fmt = InputFormat.DOCX
    elif suffix in config.EXCEL_EXT:
        fmt = InputFormat.XLSX
    elif suffix == ".pptx":
        fmt = InputFormat.PPTX
    elif suffix in config.IMAGE_EXT:
        fmt = InputFormat.IMAGE
    if fmt is None:
        return None

    try:
        # For PDF, enable picture images so we get figures/charts as images
        if fmt == InputFormat.PDF:
            from docling.document_converter import PdfFormatOption
            from docling.datamodel.pipeline_options import PdfPipelineOptions
            opts = PdfPipelineOptions()
            opts.generate_picture_images = True
            opts.generate_page_images = False  # we don't need full page images for existing flow
            converter = DocumentConverter(
                format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)}
            )
        else:
            converter = DocumentConverter()

        conv_res = converter.convert(path, raises_on_error=True)
    except Exception as e:
        logger.debug("Docling convert failed for %s: %s", path, e)
        return None

    status = getattr(conv_res, "status", None)
    if status is not None and str(status) not in ("success", "partial_success"):
        logger.debug("Docling conversion status for %s: %s", path, status)
        return None

    doc = _docling_to_document(conv_res, path)
    if not doc.has_embeddable():
        return None
    # For PDFs, if Docling produced very little prose, fall back to legacy (PyMuPDF) for better coverage
    if suffix in config.PDF_EXT:
        prose_len = sum(len(blk.text) for blk in doc.text_blocks)
        if prose_len < 500:
            logger.info("Docling produced sparse prose for PDF (%d chars), falling back to legacy: %s", prose_len, path.name)
            return None
    return doc
