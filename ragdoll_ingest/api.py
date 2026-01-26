"""HTTP API server for RAG queries."""

import json
import logging
import math
from pathlib import Path
from typing import Any

import requests
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from . import config
from .embedder import embed
from .storage import _connect, _list_sync_groups, clean_text, init_db, _sanitize_group
from .config import get_group_paths

logger = logging.getLogger(__name__)

app = FastAPI(title="RAGDoll API", version="1.0.0")


class QueryRequest(BaseModel):
    prompt: str
    history: str | None = None
    threshold: float = 0.45
    group: str | None = None  # Optional: specific collection/group to query; if None, searches all


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _expand_query(prompt: str, history: str | None) -> str:
    """Use LLM to produce a standalone description of the user's information need."""
    model = config.QUERY_MODEL
    url = (config.OLLAMA_HOST or "").rstrip("/")
    
    if history:
        # Has conversation history - include it
        prompt_text = (
            "Produce a single, standalone description of the user's current information need.\n\n"
            f"Conversation context:\n{history}\n\nUser: {prompt}\n\n"
            "Standalone description:"
        )
    else:
        # No history - just the current question
        prompt_text = (
            "Produce a single, standalone description of the user's information need based on this question.\n\n"
            f"Question: {prompt}\n\n"
            "Standalone description:"
        )
    
    try:
        r = requests.post(
            f"{url}/api/generate",
            json={"model": model, "prompt": prompt_text, "stream": False},
            timeout=config.CHUNK_LLM_TIMEOUT,
        )
        r.raise_for_status()
        response = r.json().get("response", "").strip()
        if not response:
            # Fallback to original prompt if LLM fails
            logger.warning("Query expansion returned empty, using original prompt")
            return prompt
        return response
    except Exception as e:
        logger.warning("Query expansion failed: %s, using original prompt", e)
        return prompt


@app.get("/rags")
def list_rags() -> dict[str, Any]:
    """Return all recognized RAG collections (groups)."""
    groups = _list_sync_groups()
    # Sort for consistent output
    groups.sort()
    return {"collections": groups}


@app.get("/fetch/{group}/{filename:path}")
def fetch_source(group: str, filename: str) -> FileResponse:
    """
    Fetch a source document by group and filename.
    
    The filename should match the relative path within the group's sources/ directory.
    For example, if source_path is "sources/report.pdf", use: /fetch/{group}/report.pdf
    
    Security: Only files within the group's sources directory are accessible.
    """
    # Sanitize group name
    safe_group = _sanitize_group(group)
    
    # Get the group's sources directory
    gp = get_group_paths(safe_group)
    sources_dir = gp.sources_dir
    
    # Build the file path
    file_path = sources_dir / filename
    
    # Security: Ensure the file is within the sources directory (prevent path traversal)
    try:
        file_path = file_path.resolve()
        sources_dir = sources_dir.resolve()
        if not str(file_path).startswith(str(sources_dir)):
            raise HTTPException(status_code=403, detail="Access denied: path outside sources directory")
    except (ValueError, OSError):
        raise HTTPException(status_code=400, detail="Invalid path")
    
    # Check if file exists
    if not file_path.exists() or not file_path.is_file():
        raise HTTPException(status_code=404, detail=f"Source file not found: {filename}")
    
    # Determine content type based on extension
    ext = file_path.suffix.lower()
    media_types = {
        ".pdf": "application/pdf",
        ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        ".doc": "application/msword",
        ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        ".xls": "application/vnd.ms-excel",
        ".txt": "text/plain",
        ".md": "text/markdown",
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".gif": "image/gif",
        ".bmp": "image/bmp",
        ".tiff": "image/tiff",
        ".tif": "image/tiff",
    }
    media_type = media_types.get(ext, "application/octet-stream")
    
    return FileResponse(
        path=str(file_path),
        media_type=media_type,
        filename=file_path.name,
        headers={"Content-Disposition": f'inline; filename="{file_path.name}"'}
    )


def _do_query(prompt: str, history: str | None, threshold: float, group: str | None = None) -> dict[str, Any]:
    """Shared query logic for GET and POST endpoints.
    
    Args:
        prompt: User's query/question
        history: Optional conversation history
        threshold: Minimum similarity score (0.0-1.0)
        group: Optional specific collection/group to query; if None, searches all collections
    """
    # Combine prompt and history, expand via LLM
    expanded = _expand_query(prompt, history)
    logger.info("Query expansion: %s -> %s", prompt[:100], expanded[:100])
    
    # Embed the expanded query
    try:
        query_emb = embed([expanded], group="_api")[0]
    except Exception as e:
        logger.error("Embedding failed: %s", e)
        raise HTTPException(status_code=500, detail=f"Embedding failed: {e}")
    
    # Determine which groups to search
    all_results: list[dict[str, Any]] = []
    if group:
        # Query specific group only
        groups = [group]
        # Validate group exists
        all_groups = _list_sync_groups()
        if group not in all_groups:
            raise HTTPException(status_code=404, detail=f"Collection '{group}' not found. Available collections: {all_groups}")
    else:
        # Query all groups
        groups = _list_sync_groups()
    
    for group_name in groups:
        conn = _connect(group_name)
        try:
            init_db(conn)
            rows = conn.execute(
                "SELECT source_path, source_type, chunk_index, text, embedding, artifact_type, artifact_path, page FROM chunks"
            ).fetchall()
            
            for row in rows:
                try:
                    chunk_emb = json.loads(row["embedding"])
                    similarity = _cosine_similarity(query_emb, chunk_emb)
                    
                    if similarity >= threshold:
                        source_path = row["source_path"]
                        source_name = Path(source_path).name
                        # Build URL for fetching the source file
                        # source_path stored in DB is the path within the group's sources_dir
                        gp = get_group_paths(group_name)
                        try:
                            full_source_path = Path(source_path)
                            # If source_path is absolute, get relative to sources_dir
                            if full_source_path.is_absolute():
                                try:
                                    rel_path = full_source_path.relative_to(gp.sources_dir)
                                except ValueError:
                                    # Fallback: just use filename
                                    rel_path = Path(source_name)
                            else:
                                # source_path is relative, might be just filename or include subdirs
                                rel_path = Path(source_path)
                                # Remove "sources/" prefix if present
                                parts = rel_path.parts
                                if len(parts) > 0 and parts[0] == "sources":
                                    rel_path = Path(*parts[1:])
                            
                            # URL-encode the path segments and build fetch URL
                            # Use forward slashes and URL-encode special characters
                            path_str = str(rel_path).replace("\\", "/")
                            from urllib.parse import quote
                            encoded_path = "/".join(quote(part, safe="") for part in path_str.split("/"))
                            fetch_url = f"/fetch/{group_name}/{encoded_path}"
                        except Exception as e:
                            logger.warning("Could not build fetch URL for %s: %s", source_path, e)
                            fetch_url = None
                        
                        all_results.append({
                            "group": group_name,
                            "source_path": source_path,
                            "source_type": row["source_type"],
                            "source_name": source_name,
                            "source_url": fetch_url,  # URL to fetch the source file
                            "chunk_index": row["chunk_index"],
                            "text": clean_text(row["text"]),
                            "artifact_type": row["artifact_type"] or "text",
                            "artifact_path": row["artifact_path"],
                            "page": row["page"],
                            "similarity": round(similarity, 4),
                        })
                except (json.JSONDecodeError, ValueError) as e:
                    logger.warning("Invalid embedding for chunk %s/%s/%d: %s", group_name, row["source_path"], row["chunk_index"], e)
                    continue
        finally:
            conn.close()
    
    # Sort by similarity (highest first)
    all_results.sort(key=lambda x: x["similarity"], reverse=True)
    
    return {
        "query": prompt,
        "expanded_query": expanded,
        "threshold": threshold,
        "results": all_results,
        "count": len(all_results),
    }


@app.get("/query")
def query_rag_get(prompt: str, history: str | None = None, threshold: float = 0.45, group: str | None = None) -> dict[str, Any]:
    """Query RAG collections via GET (simple URL format).
    
    Query parameters:
    - prompt: User's query/question (required)
    - history: Optional conversation history
    - threshold: Minimum similarity score (default: 0.45)
    - group: Optional specific collection/group to query; if absent, searches all collections
    """
    return _do_query(prompt, history, threshold, group)


@app.post("/query")
def query_rag(request: QueryRequest) -> dict[str, Any]:
    """Query RAG collections with semantic similarity search.
    
    Request body:
    - prompt: User's query/question (required)
    - history: Optional conversation history
    - threshold: Minimum similarity score (default: 0.45)
    - group: Optional specific collection/group to query; if absent, searches all collections
    """
    return _do_query(request.prompt, request.history, request.threshold, request.group)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=config.API_PORT)
