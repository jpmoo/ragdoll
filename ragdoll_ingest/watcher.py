"""File watcher for the ingest folder."""

import logging
import queue
import shutil
import threading
import time
from pathlib import Path

from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer

from . import config
from .action_log import log as action_log
from .artifacts import store_chart_image, store_figure, store_table
from .chunker import _clean_for_chunking, chunk_text, chunk_text_semantic
from .embedder import embed
from .garbage_control import filter_chunks
from .extractors import extract_document, extract_text, ocr_image_bytes
from .interpreters import interpret_chart, interpret_figure, interpret_table, summarize_document
from .router import route_image
from .storage import (
    _connect,
    add_chunks,
    already_processed,
    get_key_phrases_for_content,
    mark_processed,
    migrate_flat_to_root,
    run_sync_pass,
)

logger = logging.getLogger(__name__)


def _page_for_offset(offset_to_page: list[tuple[int, int | None]], start_offset: int) -> int | None:
    """Given (offset, page) pairs for segment starts, return page for character at start_offset."""
    if not offset_to_page:
        return None
    page = offset_to_page[0][1]
    for offset, p in offset_to_page:
        if offset <= start_offset:
            page = p
        else:
            break
    return page


def _is_supported(p: Path) -> bool:
    return p.suffix.lower() in config.SUPPORTED_EXT


def _should_ignore(p: Path, root: Path) -> bool:
    # macOS resource-fork / AppleDouble files (._*) are not real documents; PyMuPDF etc. fail on them
    if p.name.startswith("._"):
        return True
    try:
        r = p.resolve().relative_to(root.resolve())
    except ValueError:
        return True
    s = str(r)
    return config.PROCESSED_SUBDIR in s or config.FAILED_SUBDIR in s


def _group_from_path(p: Path) -> str:
    try:
        ingest = Path(str(config.INGEST_PATH)).resolve() if config.INGEST_PATH else None
        if ingest is None:
            return "_root"
        rel = p.resolve().relative_to(ingest)
    except ValueError:
        return "_root"
    parts = rel.parts
    return "_root" if len(parts) == 1 else parts[0]


def _rel_within_group(p: Path) -> Path:
    """Path for the file inside the group's sources/. Only one level of grouping: the first subfolder under ingest is the group. Any deeper nesting (e.g. reports/2024/x.pdf) is flattened into a single filename (e.g. 2024_x.pdf)."""
    try:
        ingest = Path(str(config.INGEST_PATH)).resolve() if config.INGEST_PATH else None
        if ingest is None:
            return Path(p.name)
        rel = p.resolve().relative_to(ingest)
    except ValueError:
        return Path(p.name)
    parts = rel.parts
    if len(parts) == 1:
        return rel  # _root: single file at top level
    # Group = first segment. Flatten parts[1:] into one name so sources/ has no nested dirs.
    return Path("_".join(parts[1:]))


def _move_to(f: Path, root: Path, subdir: str, group: str) -> Path:
    """Move file to root/subdir/rel (e.g. ingest/failed/). Only the file is moved; ingest subfolders are left as-is (even if empty)."""
    try:
        rel = f.resolve().relative_to(root.resolve())
    except ValueError:
        rel = f.name
    dest = root / subdir / rel
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        dest.unlink()
    shutil.move(str(f), str(dest))
    action_log("move", src=str(f), to=str(dest), reason=subdir, group=group)
    return dest


def _move_to_sources(f: Path, dest: Path, group: str) -> Path:
    """Move file to the group's sources folder inside the RAG output directory. Only the file is moved; we never remove empty subfolders under the ingest root."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        dest.unlink()
    shutil.move(str(f), str(dest))
    action_log("move", src=str(f), to=str(dest), reason="sources", group=group)
    return dest


def _process_one(fpath: Path) -> None:
    p = Path(fpath)
    if not p.is_file():
        return
    stat = p.stat()
    group = _group_from_path(p)
    if already_processed(str(p), stat.st_mtime, stat.st_size, group):
        action_log("already_processed", file=str(p), group=group)
        logger.info("Already processed: %s", p)
        return

    root = Path(str(config.INGEST_PATH)).resolve() if config.INGEST_PATH else None
    if root is None or not root.is_dir():
        action_log("process_skip", file=str(p), error="INGEST_PATH not set or not a directory", group=group)
        return

    # Empty or still-copying (e.g. network mount): wait and recheck to avoid "Cannot open empty file"
    if stat.st_size == 0:
        time.sleep(10)
        try:
            stat = p.stat()
        except OSError:
            stat = None
        if not stat or stat.st_size == 0:
            action_log("file_empty", file=str(p), group=group)
            logger.warning("Skipping empty file (or still copying): %s", p)
            _move_to(p, root, config.FAILED_SUBDIR, group)
            return

    action_log("process_start", file=str(p), group=group)

    chunks_list: list[dict] = []
    try:
        doc = extract_document(p)
        if doc and doc.has_embeddable():
            # Structured: prose -> chunk; charts -> OCR + interpret + store; tables -> interpret + store. Embed summaries only.
            if config.SEMANTIC_CHUNKING and doc.text_blocks:
                # Ensure page mapping matches the original document: sort blocks by page so the
                # concatenated string is in file page order (page 1, then page 2, ...). Then
                # offset_to_page maps each character offset to the correct file page.
                _PAGE_SENTINEL = 99999  # blocks with no page go after known pages
                sorted_blocks = sorted(
                    enumerate(doc.text_blocks),
                    key=lambda ix_blk: (
                        ix_blk[1].page if ix_blk[1].page is not None else _PAGE_SENTINEL,
                        ix_blk[0],
                    ),
                )
                cleaned_parts = [_clean_for_chunking(blk.text) for _, blk in sorted_blocks]
                cleaned = "\n\n".join(cleaned_parts)
                offset_to_page = [(0, sorted_blocks[0][1].page)]
                for i in range(1, len(sorted_blocks)):
                    offset_to_page.append(
                        (offset_to_page[-1][0] + len(cleaned_parts[i - 1]) + 2, sorted_blocks[i][1].page)
                    )
                # Fill None pages so every offset maps to a file page (1-based)
                fill_page = 1
                new_otp: list[tuple[int, int | None]] = []
                for off, pg in offset_to_page:
                    if pg is not None:
                        fill_page = pg
                    new_otp.append((off, fill_page))
                offset_to_page = new_otp
                semantic_chunks = chunk_text_semantic(cleaned, group=group, pre_cleaned=True)
                for chunk_str, start_offset in semantic_chunks:
                    page = _page_for_offset(offset_to_page, start_offset)
                    chunks_list.append({"text": chunk_str, "artifact_type": "text", "artifact_path": None, "page": page})
            else:
                for blk in doc.text_blocks:
                    for c in chunk_text(blk.text, group=group):
                        chunks_list.append({"text": c, "artifact_type": "text", "artifact_path": None, "page": blk.page})
            for idx, cr in enumerate(doc.chart_regions):
                ocr = ocr_image_bytes(cr.image_bytes)
                summary = interpret_chart(ocr, group=group, filename=str(p.stem) if p.stem else None)
                ap = store_chart_image(group, p.stem, cr.page, idx, cr.image_bytes, cr.image_ext)
                content = f"{summary}\n{ocr or ''}"
                all_phrases = get_key_phrases_for_content(content, filename=str(p.stem) if p.stem else None, group=group)
                if all_phrases:
                    summary = f"{summary} Key terms: {', '.join(all_phrases)}."
                chunks_list.append({"text": summary, "artifact_type": "chart_summary", "artifact_path": ap, "page": cr.page})
            for idx, tr in enumerate(doc.table_regions):
                summary = interpret_table(tr.data, group=group, filename=str(p.stem) if p.stem else None)
                ap = store_table(group, p.stem, tr.page, idx, tr.data)
                table_text = " ".join(" ".join(str(c) if c is not None else "" for c in row) for row in (tr.data or []) if row)
                content = f"{summary}\n{table_text}"
                all_phrases = get_key_phrases_for_content(content, filename=str(p.stem) if p.stem else None, group=group)
                if all_phrases:
                    summary = f"{summary} Key terms: {', '.join(all_phrases)}."
                chunks_list.append({"text": summary, "artifact_type": "table_summary", "artifact_path": ap, "page": tr.page})
            for idx, fr in enumerate(doc.figure_regions):
                ocr = ocr_image_bytes(fr.image_bytes)
                summary, process = interpret_figure(ocr, group=group, filename=str(p.stem) if p.stem else None)
                ap = store_figure(group, p.stem, fr.page, idx, fr.image_bytes, process, ocr)
                content = f"{summary}\n{ocr or ''}"
                all_phrases = get_key_phrases_for_content(content, filename=str(p.stem) if p.stem else None, group=group)
                if all_phrases:
                    summary = f"{summary} Key terms: {', '.join(all_phrases)}."
                chunks_list.append({"text": summary, "artifact_type": "figure_summary", "artifact_path": ap, "page": fr.page})
            for idx, ir in enumerate(doc.image_regions):
                chunks_list.extend(route_image(ir.image_bytes, ir.ext, ir.page_or_idx, group, p.stem, idx))
            action_log("extract_ok", file=str(p), text_blocks=len(doc.text_blocks), charts=len(doc.chart_regions), tables=len(doc.table_regions), figures=len(doc.figure_regions), images=len(doc.image_regions), group=group)
        else:
            # Fallback: .txt, .md, or extract_document returned nothing. Standalone images: classify and route.
            if p.suffix.lower() in config.IMAGE_EXT:
                b = p.read_bytes()
                ext = (p.suffix or ".png").lstrip(".").lower() or "png"
                chunks_list = route_image(b, ext, None, group, p.stem, 0)
                action_log("extract_ok", file=str(p), kind="image_routed", group=group)
            else:
                text = extract_text(p)
                if not (text and text.strip()):
                    action_log("extract_empty", file=str(p), group=group)
                    logger.warning("No text extracted from %s, moving to failed", p)
                    _move_to(p, root, config.FAILED_SUBDIR, group)
                    return
                action_log("extract_ok", file=str(p), chars=len(text), group=group)
                if config.SEMANTIC_CHUNKING:
                    semantic_chunks = chunk_text_semantic(text, group=group)
                    if not semantic_chunks:
                        action_log("chunk_empty", file=str(p), group=group)
                        _move_to(p, root, config.FAILED_SUBDIR, group)
                        return
                    chunks_list = [{"text": c, "artifact_type": "text", "artifact_path": None, "page": None} for c, _ in semantic_chunks]
                else:
                    chunks = chunk_text(text, group=group)
                    if not chunks:
                        action_log("chunk_empty", file=str(p), group=group)
                        _move_to(p, root, config.FAILED_SUBDIR, group)
                        return
                    chunks_list = [{"text": c, "artifact_type": "text", "artifact_path": None, "page": None} for c in chunks]
    except Exception as e:
        action_log("extract_fail", file=str(p), error=str(e), group=group)
        logger.exception("Extract failed for %s: %s", p, e)
        _move_to(p, root, config.FAILED_SUBDIR, group)
        return

    if not chunks_list:
        action_log("chunk_empty", file=str(p), group=group)
        _move_to(p, root, config.FAILED_SUBDIR, group)
        return

    # Garbage control: filter chunks before embedding
    chunks_list = filter_chunks(chunks_list, str(p), group)
    
    if not chunks_list:
        action_log("chunk_all_rejected", file=str(p), group=group)
        logger.warning("All chunks rejected by garbage control for %s, moving to failed", p)
        _move_to(p, root, config.FAILED_SUBDIR, group)
        return

    # One-sentence document summary (25-35 words) via LLM; prepend to every chunk
    document_text = "\n\n".join(c["text"] for c in chunks_list)
    doc_summary = summarize_document(document_text, group=group, filename=p.name)
    if doc_summary:
        for c in chunks_list:
            c["text"] = doc_summary + "\n\n" + c["text"]

    action_log("chunk_ok", file=str(p), num_chunks=len(chunks_list), group=group)
    try:
        embs = embed([c["text"] for c in chunks_list], group=group)
    except Exception as e:
        action_log("embed_fail", file=str(p), error=str(e), group=group)
        logger.exception("Embed failed for %s: %s", p, e)
        _move_to(p, root, config.FAILED_SUBDIR, group)
        return

    if len(embs) != len(chunks_list):
        action_log("embed_mismatch", file=str(p), group=group)
        _move_to(p, root, config.FAILED_SUBDIR, group)
        return

    for i, e in enumerate(embs):
        chunks_list[i]["embedding"] = e

    rel_within = _rel_within_group(p)
    dest = config.get_group_paths(group).sources_dir / rel_within

    conn = _connect(group)
    try:
        add_chunks(conn, str(dest), p.suffix.lower(), chunks_list)
        conn.commit()
    finally:
        conn.close()
    mark_processed(str(p), stat.st_mtime, stat.st_size, group)
    action_log("store", source=str(dest), num_chunks=len(chunks_list), group=group)

    _move_to_sources(p, dest, group)
    action_log("process_done", file=str(p), dest=str(dest), num_chunks=len(chunks_list), group=group)
    logger.info("Processed %s -> %d chunks -> %s (group=%s)", p, len(chunks_list), dest, group)


def _worker(q: queue.Queue, stop: threading.Event) -> None:
    while not stop.is_set():
        try:
            path = q.get(timeout=0.5)
        except queue.Empty:
            continue
        if path is None:
            break
        # Allow writes to settle
        time.sleep(2)
        if not Path(path).exists():
            continue
        try:
            _process_one(Path(path))
        except Exception as e:
            grp = _group_from_path(Path(path))
            action_log("worker_error", file=path, error=str(e), group=grp)
            logger.exception("Worker error for %s: %s", path, e)
        q.task_done()


class IngestHandler(FileSystemEventHandler):
    def __init__(self, root: Path, q: queue.Queue):
        self.root = Path(root)
        self.queue = q

    def _enqueue(self, path: str) -> None:
        p = Path(path)
        if not _is_supported(p) or _should_ignore(p, self.root):
            return
        self.queue.put(path)

    def on_created(self, event: FileSystemEvent) -> None:
        if event.is_directory:
            return
        self._enqueue(event.src_path)

    def on_moved(self, event: FileSystemEvent) -> None:
        if event.is_directory:
            return
        self._enqueue(event.dest_path)


def _scan_existing(root: Path, q: queue.Queue) -> None:
    for p in root.rglob("*"):
        if p.is_file() and _is_supported(p) and not _should_ignore(p, root):
            q.put(str(p))


def _sync_loop() -> None:
    while True:
        time.sleep(config.SYNC_INTERVAL)
        try:
            run_sync_pass()
        except Exception as e:
            logger.exception("Sync pass failed: %s", e)


def run_watcher(process_existing: bool = True) -> None:
    ingest_path = Path(str(config.INGEST_PATH)).resolve() if config.INGEST_PATH else None
    if not ingest_path or not ingest_path.is_dir():
        raise SystemExit("RAGDOLL_INGEST_PATH must be set to an existing directory")

    migrate_flat_to_root()
    action_log("watcher_start", ingest_path=str(ingest_path), group="_root")
    q: queue.Queue = queue.Queue()
    stop = threading.Event()
    t = threading.Thread(target=_worker, args=(q, stop), daemon=False)
    t.start()

    try:
        run_sync_pass()
    except Exception as e:
        logger.exception("Initial sync pass failed: %s", e)
    if config.SYNC_INTERVAL > 0:
        sync_thread = threading.Thread(target=_sync_loop, daemon=True)
        sync_thread.start()

    if process_existing:
        _scan_existing(ingest_path, q)

    observer = Observer()
    observer.schedule(IngestHandler(ingest_path, q), str(ingest_path), recursive=True)
    observer.start()
    logger.info("Watching %s", ingest_path)

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()
        q.put(None)
        t.join(timeout=5)
        observer.stop()
        observer.join(timeout=5)
