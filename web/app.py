"""Review web service: side-by-side source and samples (chunks), port 9043."""

import base64
import logging
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

# Run from project root so ragdoll_ingest is on path
from ragdoll_ingest import config
from ragdoll_ingest.config import get_env
from ragdoll_ingest.embedder import embed
from ragdoll_ingest.interpreters import extract_chunk_semantic_labels
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
    reindex_chunks_after_delete,
    update_chunk_full,
    update_chunk_text,
)

logger = logging.getLogger(__name__)

REVIEW_PORT = 9043
WEB_ROOT = Path(__file__).resolve().parent

# Optional HTTP Basic Auth: set both to enable
REVIEW_USER = get_env("RAGDOLL_REVIEW_USER") or ""
REVIEW_PASSWORD = get_env("RAGDOLL_REVIEW_PASSWORD") or ""


class BasicAuthMiddleware(BaseHTTPMiddleware):
    """Require HTTP Basic Auth when REVIEW_USER and REVIEW_PASSWORD are set."""

    async def dispatch(self, request: Request, call_next):
        if not REVIEW_USER or not REVIEW_PASSWORD:
            return await call_next(request)
        auth = request.headers.get("Authorization")
        if not auth or not auth.startswith("Basic "):
            return Response(
                status_code=401,
                content="Authentication required",
                headers={"WWW-Authenticate": 'Basic realm="RAGDoll Review"'},
            )
        try:
            raw = base64.b64decode(auth[6:].strip()).decode("utf-8")
            user, _, password = raw.partition(":")
            if user != REVIEW_USER or password != REVIEW_PASSWORD:
                return Response(
                    status_code=401,
                    content="Invalid credentials",
                    headers={"WWW-Authenticate": 'Basic realm="RAGDoll Review"'},
                )
        except Exception:
            return Response(
                status_code=401,
                content="Invalid Authorization header",
                headers={"WWW-Authenticate": 'Basic realm="RAGDoll Review"'},
            )
        return await call_next(request)


app = FastAPI(title="RAGDoll Review", version="1.0.0")
if REVIEW_USER and REVIEW_PASSWORD:
    app.add_middleware(BasicAuthMiddleware)


# --- API ---

def _text_to_embed(text: str, concept: str = "", decision_context: str = "", primary_question_answered: str = "", key_signals: list | None = None, chunk_role: str = "") -> str:
    """Build string to embed: chunk text plus semantic fields (same as ingest)."""
    parts = [text or ""]
    if concept:
        parts.append("Concept: " + concept)
    if decision_context:
        parts.append("Decision context: " + decision_context)
    if primary_question_answered:
        parts.append("Primary question answered: " + primary_question_answered)
    if key_signals:
        parts.append("Key signals: " + ", ".join(key_signals))
    if chunk_role:
        parts.append("Chunk role: " + chunk_role)
    return "\n\n".join(parts)


class ChunkUpdate(BaseModel):
    text: str
    concept: str | None = None
    decision_context: str | None = None
    primary_question_answered: str | None = None
    key_signals: list[str] | None = None
    chunk_role: str | None = None


class ChunkCreate(BaseModel):
    text: str
    after_index: int | None = None  # insert after this index (new chunk_index = after_index + 1)
    before_index: int | None = None  # insert before this index (new chunk_index = before_index)
    concept: str | None = None
    decision_context: str | None = None
    primary_question_answered: str | None = None
    key_signals: list[str] | None = None
    chunk_role: str | None = None


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
    """Update chunk text; re-run LLM for semantic labels and re-embed."""
    safe_group = config._sanitize_group(group)
    conn = _connect(safe_group)
    try:
        row = get_chunk_by_id(conn, chunk_id)
        if not row:
            raise HTTPException(status_code=404, detail="Chunk not found")
        labels = extract_chunk_semantic_labels(body.text, group=safe_group)
        concept = (labels.get("concept") or "").strip() or None
        decision_context = (labels.get("decision_context") or "").strip() or None
        primary_question_answered = (labels.get("primary_question_answered") or "").strip() or None
        key_signals = labels.get("key_signals") or []
        chunk_role = (labels.get("chunk_role") or "").strip() or None
        to_embed = _text_to_embed(body.text, concept or "", decision_context or "", primary_question_answered or "", key_signals, chunk_role or "")
        embs = embed([to_embed], group=safe_group)
        if not embs:
            raise HTTPException(status_code=500, detail="Embedding failed")
        update_chunk_full(
            conn, chunk_id, body.text, embs[0],
            concept=concept, decision_context=decision_context,
            primary_question_answered=primary_question_answered,
            key_signals=key_signals if key_signals else None, chunk_role=chunk_role,
        )
        conn.commit()
        return {"ok": True, "chunk_id": chunk_id}
    finally:
        conn.close()


def _join_chunk_with_neighbor(
    conn, safe_group: str, chunk_id: int, direction: str
) -> dict:
    """Join current chunk with above (direction='above') or below (direction='below'). Merges text, re-runs LLM semantic labels, re-embeds, deletes neighbor, reindexes. Returns updated chunk info or raises HTTPException."""
    current = get_chunk_by_id(conn, chunk_id)
    if not current:
        raise HTTPException(status_code=404, detail="Chunk not found")
    source_id = current["source_id"]
    idx = current["chunk_index"]
    chunks = get_chunks_for_source(conn, source_id)
    if direction == "above":
        neighbor = next((c for c in chunks if c["chunk_index"] == idx - 1), None)
        if not neighbor:
            raise HTTPException(status_code=404, detail="No chunk above to join")
        merged_text = (neighbor["text"] or "").strip() + "\n\n" + (current["text"] or "").strip()
        to_delete_id = neighbor["id"]
        deleted_index = neighbor["chunk_index"]
    else:
        neighbor = next((c for c in chunks if c["chunk_index"] == idx + 1), None)
        if not neighbor:
            raise HTTPException(status_code=404, detail="No chunk below to join")
        merged_text = (current["text"] or "").strip() + "\n\n" + (neighbor["text"] or "").strip()
        to_delete_id = neighbor["id"]
        deleted_index = neighbor["chunk_index"]
    labels = extract_chunk_semantic_labels(merged_text, group=safe_group)
    concept = (labels.get("concept") or "").strip() or None
    decision_context = (labels.get("decision_context") or "").strip() or None
    primary_question_answered = (labels.get("primary_question_answered") or "").strip() or None
    key_signals = labels.get("key_signals") or []
    chunk_role = (labels.get("chunk_role") or "").strip() or None
    to_embed = _text_to_embed(merged_text, concept or "", decision_context or "", primary_question_answered or "", key_signals, chunk_role or "")
    embs = embed([to_embed], group=safe_group)
    if not embs:
        raise HTTPException(status_code=500, detail="Embedding failed")
    delete_chunk(conn, to_delete_id)
    reindex_chunks_after_delete(conn, source_id, deleted_index)
    update_chunk_full(
        conn, chunk_id, merged_text, embs[0],
        concept=concept, decision_context=decision_context,
        primary_question_answered=primary_question_answered,
        key_signals=key_signals if key_signals else None, chunk_role=chunk_role,
    )
    return {"ok": True, "chunk_id": chunk_id, "merged_text": merged_text}


@app.post("/api/groups/{group}/chunks/{chunk_id}/join-above")
def api_join_chunk_above(group: str, chunk_id: int):
    """Merge the chunk above into this chunk (above text + this text), re-run semantic labels and embedding, delete the above chunk."""
    safe_group = config._sanitize_group(group)
    conn = _connect(safe_group)
    try:
        result = _join_chunk_with_neighbor(conn, safe_group, chunk_id, "above")
        conn.commit()
        return result
    finally:
        conn.close()


@app.post("/api/groups/{group}/chunks/{chunk_id}/join-below")
def api_join_chunk_below(group: str, chunk_id: int):
    """Merge the chunk below into this chunk (this text + below text), re-run semantic labels and embedding, delete the below chunk."""
    safe_group = config._sanitize_group(group)
    conn = _connect(safe_group)
    try:
        result = _join_chunk_with_neighbor(conn, safe_group, chunk_id, "below")
        conn.commit()
        return result
    finally:
        conn.close()


@app.post("/api/groups/{group}/sources/{source_id}/chunks")
def api_create_chunk(group: str, source_id: int, body: ChunkCreate):
    """Insert a new chunk above or below an index, with optional semantic fields."""
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
    concept = (body.concept or "").strip()
    decision_context = (body.decision_context or "").strip()
    primary_question_answered = (body.primary_question_answered or "").strip()
    key_signals = body.key_signals or []
    chunk_role = (body.chunk_role or "").strip()
    # Fill in any semantic fields the user left empty via LLM
    if not concept or not decision_context or not primary_question_answered or not key_signals or not chunk_role:
        labels = extract_chunk_semantic_labels(body.text, group=safe_group)
        if not concept and labels.get("concept"):
            concept = (labels["concept"] or "").strip()
        if not decision_context and labels.get("decision_context"):
            decision_context = (labels["decision_context"] or "").strip()
        if not primary_question_answered and labels.get("primary_question_answered"):
            primary_question_answered = (labels["primary_question_answered"] or "").strip()
        if not key_signals and labels.get("key_signals"):
            key_signals = labels["key_signals"] or []
        if not chunk_role and labels.get("chunk_role"):
            chunk_role = (labels["chunk_role"] or "").strip()
    to_embed = _text_to_embed(body.text, concept, decision_context, primary_question_answered, key_signals, chunk_role)
    embs = embed([to_embed], group=safe_group)
    if not embs:
        raise HTTPException(status_code=500, detail="Embedding failed")
    conn = _connect(safe_group)
    try:
        new_id = insert_chunk_at(
            conn, source_id, source_path, source_type, at_index, body.text, embs[0],
            page=None, artifact_type="text", artifact_path=None,
            concept=concept or None, decision_context=decision_context or None,
            primary_question_answered=primary_question_answered or None,
            key_signals=key_signals if key_signals else None, chunk_role=chunk_role or None,
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
