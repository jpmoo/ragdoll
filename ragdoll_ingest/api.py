"""HTTP API server for RAG queries."""

import json
import logging
import math
from pathlib import Path
from typing import Any

import requests
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from . import config
from .embedder import embed
from .storage import _connect, _list_sync_groups, init_db

logger = logging.getLogger(__name__)

app = FastAPI(title="RAGDoll API", version="1.0.0")


class QueryRequest(BaseModel):
    prompt: str
    history: str | None = None
    threshold: float = 0.60


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
    
    combined = prompt
    if history:
        combined = f"{history}\n\nUser: {prompt}"
    
    prompt_text = (
        "Produce a single, standalone description of the user's current information need.\n\n"
        f"Conversation context:\n{combined}\n\n"
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


def _do_query(prompt: str, history: str | None, threshold: float) -> dict[str, Any]:
    """Shared query logic for GET and POST endpoints."""
    # Combine prompt and history, expand via LLM
    expanded = _expand_query(prompt, history)
    logger.info("Query expansion: %s -> %s", prompt[:100], expanded[:100])
    
    # Embed the expanded query
    try:
        query_emb = embed([expanded], group="_api")[0]
    except Exception as e:
        logger.error("Embedding failed: %s", e)
        raise HTTPException(status_code=500, detail=f"Embedding failed: {e}")
    
    # Search all groups
    all_results: list[dict[str, Any]] = []
    groups = _list_sync_groups()
    
    for group in groups:
        conn = _connect(group)
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
                        all_results.append({
                            "group": group,
                            "source_path": row["source_path"],
                            "source_type": row["source_type"],
                            "source_name": Path(row["source_path"]).name,
                            "chunk_index": row["chunk_index"],
                            "text": row["text"],
                            "artifact_type": row["artifact_type"] or "text",
                            "artifact_path": row["artifact_path"],
                            "page": row["page"],
                            "similarity": round(similarity, 4),
                        })
                except (json.JSONDecodeError, ValueError) as e:
                    logger.warning("Invalid embedding for chunk %s/%s/%d: %s", group, row["source_path"], row["chunk_index"], e)
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
def query_rag_get(prompt: str, history: str | None = None, threshold: float = 0.60) -> dict[str, Any]:
    """Query RAG collections via GET (simple URL format)."""
    return _do_query(prompt, history, threshold)


@app.post("/query")
def query_rag(request: QueryRequest) -> dict[str, Any]:
    """Query RAG collections with semantic similarity search."""
    return _do_query(request.prompt, request.history, request.threshold)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=config.API_PORT)
