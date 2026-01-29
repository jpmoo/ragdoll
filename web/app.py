"""Review web service: side-by-side source and samples (chunks), port 9043."""

import json
import logging
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Run from project root so ragdoll_ingest is on path
from ragdoll_ingest import config
from ragdoll_ingest.config import get_env
from ragdoll_ingest.embedder import embed
from ragdoll_ingest.storage import (
    _connect,
    _list_sync_groups,
    delete_chunk,
    get_chunk_by_id,
    get_chunks_for_source,
    get_source_by_id,
    init_db,
    insert_chunk_at,
    list_sources,
    update_chunk_text,
)

logger = logging.getLogger(__name__)

REVIEW_PORT = 9043
WEB_ROOT = Path(__file__).resolve().parent

app = FastAPI(title="RAGDoll Review", version="1.0.0")


# --- API ---

class ChunkUpdate(BaseModel):
    text: str


class ChunkCreate(BaseModel):
    text: str
    after_index: int | None = None  # insert after this index (new chunk_index = after_index + 1)
    before_index: int | None = None  # insert before this index (new chunk_index = before_index)


@app.get("/api/groups")
def api_list_groups():
    """List all RAG groups (collections)."""
    groups = _list_sync_groups()
    groups.sort()
    return {"groups": groups}


@app.get("/api/groups/{group}/sources")
def api_list_sources(group: str):
    """List sources in a group with chunk counts and fetch path."""
    safe_group = config._sanitize_group(group)
    conn = _connect(safe_group)
    try:
        init_db(conn)
        raw = list_sources(conn)
        gp = config.get_group_paths(safe_group)
        sources_dir = gp.sources_dir.resolve()
        out = []
        for source_id, source_path, count in raw:
            try:
                p = Path(source_path)
                if p.is_absolute() and str(p).startswith(str(sources_dir)):
                    fetch_path = p.relative_to(sources_dir)
                else:
                    fetch_path = Path(source_path).name if source_path else Path("")
                fetch_path = str(fetch_path).replace("\\", "/")
            except Exception:
                fetch_path = Path(source_path).name if source_path else ""
            out.append({
                "source_id": source_id,
                "source_path": source_path,
                "display_name": Path(source_path).name if source_path else f"Source {source_id}",
                "fetch_path": fetch_path,
                "chunk_count": count,
            })
        return {"sources": out}
    finally:
        conn.close()


@app.get("/api/groups/{group}/sources/{source_id}/chunks")
def api_list_chunks(group: str, source_id: int, page: int | None = None):
    """List chunks (samples) for a source. Optional query: page=N to filter by page."""
    safe_group = config._sanitize_group(group)
    conn = _connect(safe_group)
    try:
        chunks = get_chunks_for_source(conn, source_id, page=page)
        return {"chunks": chunks}
    finally:
        conn.close()


@app.get("/api/groups/{group}/fetch/{path:path}")
def api_fetch_source(group: str, path: str):
    """Serve a source file from the group's sources directory."""
    safe_group = config._sanitize_group(group)
    gp = config.get_group_paths(safe_group)
    sources_dir = gp.sources_dir.resolve()
    file_path = (sources_dir / path).resolve()
    if not str(file_path).startswith(str(sources_dir)) or not file_path.is_file():
        raise HTTPException(status_code=404, detail="Source file not found")
    ext = file_path.suffix.lower()
    media_types = {
        ".pdf": "application/pdf",
        ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        ".txt": "text/plain",
        ".md": "text/markdown",
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".gif": "image/gif",
        ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    }
    media_type = media_types.get(ext, "application/octet-stream")
    return FileResponse(
        path=str(file_path),
        media_type=media_type,
        filename=file_path.name,
        headers={"Content-Disposition": f'inline; filename="{file_path.name}"'},
    )


@app.patch("/api/groups/{group}/chunks/{chunk_id}")
def api_update_chunk(group: str, chunk_id: int, body: ChunkUpdate):
    """Update chunk text and re-embed."""
    safe_group = config._sanitize_group(group)
    conn = _connect(safe_group)
    try:
        row = get_chunk_by_id(conn, chunk_id)
        if not row:
            raise HTTPException(status_code=404, detail="Chunk not found")
        embs = embed([body.text], group=safe_group)
        if not embs:
            raise HTTPException(status_code=500, detail="Embedding failed")
        update_chunk_text(conn, chunk_id, body.text, embs[0])
        conn.commit()
        return {"ok": True, "chunk_id": chunk_id}
    finally:
        conn.close()


@app.post("/api/groups/{group}/sources/{source_id}/chunks")
def api_create_chunk(group: str, source_id: int, body: ChunkCreate):
    """Insert a new chunk above or below an index."""
    safe_group = config._sanitize_group(group)
    src = get_source_by_id(_connect(safe_group), source_id)
    if not src:
        raise HTTPException(status_code=404, detail="Source not found")
    source_path, source_type = src
    if body.after_index is not None:
        at_index = body.after_index + 1
    elif body.before_index is not None:
        at_index = body.before_index
    else:
        at_index = 0
    embs = embed([body.text], group=safe_group)
    if not embs:
        raise HTTPException(status_code=500, detail="Embedding failed")
    conn = _connect(safe_group)
    try:
        new_id = insert_chunk_at(
            conn, source_id, source_path, source_type, at_index, body.text, embs[0],
            page=None, artifact_type="text", artifact_path=None,
        )
        conn.commit()
        return {"ok": True, "chunk_id": new_id, "chunk_index": at_index}
    finally:
        conn.close()


@app.delete("/api/groups/{group}/chunks/{chunk_id}")
def api_delete_chunk(group: str, chunk_id: int):
    """Delete a chunk."""
    safe_group = config._sanitize_group(group)
    conn = _connect(safe_group)
    try:
        ok = delete_chunk(conn, chunk_id)
        conn.commit()
        if not ok:
            raise HTTPException(status_code=404, detail="Chunk not found")
        return {"ok": True}
    finally:
        conn.close()


# --- Static frontend ---

static_dir = WEB_ROOT / "static"
if static_dir.is_dir():
    app.mount("/", StaticFiles(directory=str(static_dir), html=True), name="static")


@app.get("/api/health")
def health():
    return {"status": "ok", "port": REVIEW_PORT}


if __name__ == "__main__":
    import uvicorn
    port = int(get_env("RAGDOLL_REVIEW_PORT") or str(REVIEW_PORT))
    uvicorn.run(app, host="0.0.0.0", port=port)
