"""Store chart images and table JSON under {group}/artifacts/. Embed only interpretations; keep raw here."""

import json
import re
from pathlib import Path

from . import config


def _safe_stem(s: str) -> str:
    return re.sub(r"[^\w\-.]", "_", s)[:80]


def store_chart_image(group: str, source_stem: str, page: int, idx: int, image_bytes: bytes, ext: str = "png") -> str:
    """Save chart image to {group}/artifacts/charts/{stem}_p{page}_{idx}.{ext}. Returns absolute path."""
    gp = config.get_group_paths(group)
    d = gp.artifacts_dir / "charts"
    d.mkdir(parents=True, exist_ok=True)
    stem = _safe_stem(source_stem)
    ext = (ext or "png").lstrip(".")
    if ext.lower() not in {"png", "jpg", "jpeg", "gif", "bmp", "tiff"}:
        ext = "png"
    p = d / f"{stem}_p{page}_{idx}.{ext}"
    p.write_bytes(image_bytes)
    return str(p)


def store_figure(
    group: str, source_stem: str, page: int, idx: int,
    image_bytes: bytes, process_dict: dict, ocr_text: str,
) -> str:
    """Save figure image and process JSON to {group}/artifacts/figures/. Returns path to the JSON."""
    gp = config.get_group_paths(group)
    d = gp.artifacts_dir / "figures"
    d.mkdir(parents=True, exist_ok=True)
    stem = _safe_stem(source_stem)
    base = f"{stem}_p{page}_{idx}"
    (d / f"{base}.png").write_bytes(image_bytes)
    j = d / f"{base}.json"
    j.write_text(
        json.dumps({"process": process_dict, "ocr": ocr_text}, ensure_ascii=False, indent=0),
        encoding="utf-8",
    )
    return str(j)


def store_table(group: str, source_stem: str, page: int | None, idx: int, data: list[list[str]]) -> str:
    """Save table as JSON to {group}/artifacts/tables/{stem}_p{page}_{idx}.json. Returns absolute path. page can be 0 when unknown."""
    gp = config.get_group_paths(group)
    d = gp.artifacts_dir / "tables"
    d.mkdir(parents=True, exist_ok=True)
    stem = _safe_stem(source_stem)
    pp = f"p{page}" if page is not None else "p0"
    p = d / f"{stem}_{pp}_{idx}.json"
    p.write_text(json.dumps(data, ensure_ascii=False, indent=0), encoding="utf-8")
    return str(p)
