"""MCP server for RAGDoll: exposes query_rag and list_collections to MCP clients (stdio or HTTP/SSE)."""

import asyncio
import json
import logging
from pathlib import Path

from fastapi import HTTPException

from . import config
from .api import _do_query
from .memory import parse_memory_text, store_memory
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
            "RAGDoll gives you semantic search over ingested document collections and a dedicated memory collection. "
            "Use list_collections to discover collections. Use query_rag to search (when no collections are specified, the memory collection is included). "
            "Use write_memory to store a structured memory (Topic, Date, Tags, Conclusion, Reasoning, Open threads); memories are then searchable via query_rag."
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
        synthesize: bool = False,
        synthesis_mode: str = "instructions",
    ) -> dict:
        """Semantic similarity search over one or more RAGDoll document collections. Returns matching document chunks sorted by relevance.
        When synthesize=true, RAGDoll also uses its LLM to turn prompt+history+chunks into instructions or an answer (research-assistant style).

        Args:
            prompt: Your question or information need.
            history: Optional prior conversation turns as plain text, used for query expansion.
            threshold: Minimum cosine similarity (0.0–1.0). Lower = more results, less precise. Default 0.45.
            collections: Collection names to search. If omitted or empty, searches all collections.
            limit_chunk_role: When true, infer up to 2 chunk roles from the prompt and restrict retrieval to those roles.
            max_results: Maximum number of chunks to return in the flat results list. Default 20. Does not cap the grouped documents view.
            synthesize: When true, LLM synthesizes prompt+history+RAG into instructions for an assistant or a direct answer.
            synthesis_mode: "instructions" (default) = instructions for the caller to use; "answer" = direct summary/answer.
        """
        try:
            result = await asyncio.to_thread(
                _do_query,
                prompt,
                history,
                threshold,
                collections if collections else None,
                limit_chunk_role,
                synthesize,
                synthesis_mode,
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

    async def write_memory(content: str) -> dict:
        """Write a structured memory to the RAGDoll memory collection (MCP-only). Memories are then included in query_rag when searching all collections.

        Input format (plain text with these section headers):
        - Topic: ...
        - Date: YYYY-MM-DD
        - Tags: comma-separated list
        - Conclusion: ...
        - Reasoning: ...
        - Open threads: ...

        The memory is stored with embeddings for each section and the full text, and will appear in semantic search results with memory_topic, memory_date, and memory_tags in the result metadata.
        """
        parsed = parse_memory_text(content)
        if not parsed:
            return {"ok": False, "error": "Could not parse memory: need at least Topic or one of Conclusion/Reasoning/Open threads"}
        try:
            out = await asyncio.to_thread(store_memory, parsed)
            return out
        except Exception as e:
            logger.exception("write_memory failed")
            return {"ok": False, "error": str(e)}

    # Register write_memory in the MCP tool manifest (must be callable so clients see it in tools/list)
    mcp.tool()(write_memory)

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
    """Entry point: run MCP server in stdio, SSE, or streamable-http mode from RAGDOLL_MCP_TRANSPORT."""
    mcp = _make_mcp()
    transport = config.MCP_TRANSPORT
    if transport in ("sse", "streamable-http", "http"):
        import uvicorn
        if transport == "streamable-http" or transport == "http":
            # Streamable HTTP: POST at /mcp for session (mcp-remote, Claude). When mounted at /mcp, sub-app receives path / (streamable_http_path="/" in FastMCP).
            try:
                from starlette.applications import Starlette
                from starlette.routing import Mount, Route
                from starlette.responses import JSONResponse

                def _mcp_base_ok(_request):
                    return JSONResponse({"protocol": "mcp", "server": "ragdoll"})

                app = Starlette(
                    routes=[
                        Route("/mcp", endpoint=_mcp_base_ok, methods=["GET"]),
                        Route("/mcp/", endpoint=_mcp_base_ok, methods=["GET"]),
                        Mount("/mcp", app=mcp.streamable_http_app()),
                    ]
                )
            except ImportError:
                app = mcp.streamable_http_app()
        else:
            # SSE: /mcp/sse and /mcp/messages
            sse_app = mcp.sse_app("/mcp")
            try:
                from starlette.applications import Starlette
                from starlette.routing import Mount
                app = Starlette(routes=[Mount("/mcp", app=sse_app)])
            except ImportError:
                app = sse_app
        uvicorn.run(
            app,
            host=config.MCP_HOST,
            port=config.MCP_PORT,
            log_level="info",
        )
    else:
        mcp.run()


if __name__ == "__main__":
    main()
