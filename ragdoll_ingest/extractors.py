"""Extract plain text from supported file types."""

import logging
from pathlib import Path

from . import config

logger = logging.getLogger(__name__)


def extract_text(path: Path) -> str:
    """Extract text from a file. Raises ValueError if unsupported or on error."""
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix in config.TEXT_EXT:
        return _extract_plain(path)
    if suffix in config.WORD_EXT:
        return _extract_docx(path)
    if suffix in config.EXCEL_EXT:
        return _extract_excel(path)
    if suffix in config.PDF_EXT:
        return _extract_pdf(path)
    if suffix in config.IMAGE_EXT:
        return _extract_image(path)

    raise ValueError(f"Unsupported extension: {suffix}")


def _extract_plain(path: Path) -> str:
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return f.read()


def _extract_docx(path: Path) -> str:
    from docx import Document

    doc = Document(path)
    return "\n".join(p.text for p in doc.paragraphs)


def _extract_excel(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".xlsx":
        import openpyxl

        wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    else:
        import xlrd

        wb = xlrd.open_workbook(path)

    parts: list[str] = []
    if hasattr(wb, "sheetnames"):
        for name in wb.sheetnames:
            sh = wb[name]
            for row in sh.iter_rows(values_only=True):
                line = "\t".join(str(c) if c is not None else "" for c in row).strip()
                if line:
                    parts.append(line)
    else:
        for i in range(wb.nsheets):
            sh = wb.sheet_by_index(i)
            for r in range(sh.nrows):
                row = [sh.cell_value(r, c) for c in range(sh.ncols)]
                line = "\t".join(str(c) if c else "" for c in row).strip()
                if line:
                    parts.append(line)

    if hasattr(wb, "close"):
        wb.close()
    return "\n".join(parts)


def _extract_pdf(path: Path) -> str:
    import fitz  # PyMuPDF

    doc = fitz.open(path)
    try:
        return "\n".join(page.get_text() for page in doc)
    finally:
        doc.close()


def _extract_image(path: Path) -> str:
    import pytesseract
    from PIL import Image

    img = Image.open(path)
    return pytesseract.image_to_string(img)
