"""MCP server for RAGDoll: exposes query_rag and list_collections to MCP clients (stdio or HTTP/SSE)."""

import asyncio
import json
import logging
from pathlib import Path

from fastapi import HTTPException

from . import config
from .api import _do_query
from .storage import _connect, _list_sync_groups, init_db, list_sources

logger = logging.getLogger(__name__)

try:
    from mcp.server.fastmcp import FastMCP
except ImportError:
    FastMCP = None  # type: ignore[misc, assignment]


def _make_mcp() -> "FastMCP":
    if FastMCP is None:
        raise RuntimeError("MCP support requires: pip install -e '.[mcp]'")
    mcp = FastMCP(
        name="ragdoll",
        instructions=(
            "RAGDoll gives you semantic search over ingested document collections. "
            "Use query_rag to find relevant content. Use list_collections to discover "
            "what collections are available before querying a specific one."
        ),
    )

    @mcp.tool()
    def list_collections() -> dict:
        """List all available RAGDoll document collections. Call this before query_rag if you are not sure which collections exist."""
        coll = _list_sync_groups()
        coll.sort()
        return {"collections": coll}

    @mcp.tool()
    async def query_rag(
        prompt: str,
        history: str | None = None,
        threshold: float = 0.45,
        collections: list[str] | None = None,
        limit_chunk_role: bool = False,
        max_results: int = 20,
    ) -> dict:
        """Semantic similarity search over one or more RAGDoll document collections. Returns matching document chunks sorted by relevance.

        Args:
            prompt: Your question or information need.
            history: Optional prior conversation turns as plain text, used for query expansion.
            threshold: Minimum cosine similarity (0.0–1.0). Lower = more results, less precise. Default 0.45.
            collections: Collection names to search. If omitted or empty, searches all collections.
            limit_chunk_role: When true, infer up to 2 chunk roles from the prompt and restrict retrieval to those roles.
            max_results: Maximum number of chunks to return in the flat results list. Default 20. Does not cap the grouped documents view.
        """
        try:
            result = await asyncio.to_thread(
                _do_query,
                prompt,
                history,
                threshold,
                collections if collections else None,
                limit_chunk_role,
            )
        except HTTPException as e:
            raise ValueError(f"{e.detail}") from e
        except Exception as e:
            logger.exception("query_rag failed")
            raise ValueError(f"Query failed: {e}") from e

        # Cap flat results for large responses (spec: max_results parameter)
        results = result.get("results") or []
        if len(results) > max_results:
            result = dict(result)
            result["results"] = results[:max_results]
            result["count"] = len(result["results"])
            result["_truncated"] = True
            result["_total_matching"] = len(results)

        return result

    # Optional resources (ragdoll://collections and ragdoll://collections/{group}/sources)
    @mcp.resource("ragdoll://collections")
    def resource_collections() -> str:
        """List all collections. Same as list_collections tool."""
        coll = _list_sync_groups()
        coll.sort()
        return json.dumps({"collections": coll})

    @mcp.resource("ragdoll://collections/{group}/sources")
    def resource_collection_sources(group: str) -> str:
        """List sources in a collection with source_id, source_name, source_path, chunk_count, summary."""
        safe_group = config._sanitize_group(group)
        conn = _connect(safe_group)
        try:
            init_db(conn)
            raw = list_sources(conn)
            gp = config.get_group_paths(safe_group)
            sources_dir = gp.sources_dir.resolve()
            out = []
            for source_id, source_path, count, summary in raw:
                try:
                    p = Path(source_path)
                    if p.is_absolute() and str(p).startswith(str(sources_dir)):
                        name = p.name
                    else:
                        name = Path(source_path).name if source_path else f"Source {source_id}"
                except Exception:
                    name = source_path or f"Source {source_id}"
                out.append({
                    "source_id": source_id,
                    "source_name": name,
                    "source_path": source_path,
                    "chunk_count": count,
                    "summary": summary or "",
                })
            return json.dumps(out)
        finally:
            conn.close()

    return mcp


def main() -> None:
    """Entry point: run MCP server in stdio or SSE mode from RAGDOLL_MCP_TRANSPORT."""
    mcp = _make_mcp()
    transport = config.MCP_TRANSPORT
    if transport == "sse":
        # Official MCP SDK run() does not accept host/port; serve SSE app with uvicorn instead.
        # Pass mount path to sse_app so it serves at /mcp/sse and /mcp/messages and returns correct endpoint URL (no double path).
        import uvicorn
        sse_app = mcp.sse_app("/mcp")
        uvicorn.run(
            sse_app,
            host=config.MCP_HOST,
            port=config.MCP_PORT,
            log_level="info",
        )
    else:
        mcp.run()


if __name__ == "__main__":
    main()
