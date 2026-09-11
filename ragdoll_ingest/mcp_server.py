"""MCP server for RAGDoll: exposes query_rag and list_collections to MCP clients (stdio, SSE, or Streamable HTTP)."""

import asyncio
import json
import logging
from pathlib import Path

from fastapi import HTTPException

from . import config
from .api import _do_query, _group_results_by_document, _number_context
from .insights import submit_insight_text
from .storage import _connect, _list_sync_groups, init_db, list_sources

logger = logging.getLogger(__name__)

try:
    from mcp.server.fastmcp import FastMCP
    from mcp.server.transport_security import TransportSecuritySettings
except ImportError:
    FastMCP = None  # type: ignore[misc, assignment]
    TransportSecuritySettings = None  # type: ignore[misc, assignment]


def _make_mcp() -> "FastMCP":
    if FastMCP is None:
        raise RuntimeError("MCP support requires: pip install -e '.[mcp]'")
    # Loopback bind + reverse proxy: clients send the public Host header; don't reject it.
    transport_security = None
    if config.MCP_HOST in ("127.0.0.1", "localhost", "::1"):
        transport_security = TransportSecuritySettings(enable_dns_rebinding_protection=False)
    mcp = FastMCP(
        name="ragdoll",
        instructions=(
            "RAGDoll gives you semantic search over ingested document collections plus an insights collection: learnings built from how the collections are used. "
            "Use list_collections to discover collections. Use query_rag to search; insights are included by default, even when you name specific collections "
            "(set include_insights=false to leave them out). "
            "Use submit_insight to contribute a learning (Topic, Tags, Insight, Question, Reasoning, Open questions, Confidence); it is searchable immediately via query_rag."
        ),
        host=config.MCP_HOST,
        port=config.MCP_PORT,
        transport_security=transport_security,
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
        threshold: float | None = None,
        collections: list[str] | None = None,
        limit_chunk_role: bool = False,
        max_results: int = 20,
        synthesize: bool = False,
        synthesis_mode: str = "instructions",
        include_insights: bool = True,
    ) -> dict:
        """Semantic similarity search over one or more RAGDoll document collections. Returns matching document chunks sorted by relevance.
        When synthesize=true, RAGDoll also uses its LLM to turn prompt+history+chunks into instructions or an answer (research-assistant style).

        Args:
            prompt: Your question or information need.
            history: Optional prior conversation turns as plain text, used for query expansion.
            threshold: Minimum cosine similarity (0.0–1.0). Lower = more results, less precise. Omit to use RAGDOLL_QUERY_THRESHOLD (default 0.45).
            collections: Collection names to search. If omitted or empty, searches all collections.
            limit_chunk_role: When true, infer up to 2 chunk roles from the prompt and restrict retrieval to those roles.
            max_results: Maximum number of chunks to return. Default 20. Applies to both the flat results list and the grouped documents view (_total_matching reports how many matched).
            synthesize: When true, LLM synthesizes prompt+history+RAG into instructions for an assistant or a direct answer.
            synthesis_mode: "instructions" (default) = instructions for the caller to use; "answer" = direct summary/answer.
            include_insights: When true (default), the insights collection is searched too, even if collections names others. Insight hits carry an "insight" object (id, topic, tags, origin, confidence).
        """
        try:
            use_threshold = config.QUERY_THRESHOLD if threshold is None else threshold
            result = await asyncio.to_thread(
                _do_query,
                prompt,
                history,
                use_threshold,
                collections if collections else None,
                limit_chunk_role,
                synthesize,
                synthesis_mode,
                include_insights=include_insights,
                log_as="mcp",
            )
        except HTTPException as e:
            raise ValueError(f"{e.detail}") from e
        except Exception as e:
            logger.exception("query_rag failed")
            raise ValueError(f"Query failed: {e}") from e

        # Cap results for large responses (spec: max_results parameter). Rebuild the documents view from the kept
        # chunks too; otherwise it still carries every match (hundreds of chunks, megabytes of JSON).
        results = result.get("results") or []
        if len(results) > max_results:
            kept = results[:max_results]
            _number_context(kept)
            result = dict(result)
            result["results"] = kept
            result["documents"] = _group_results_by_document(kept)
            result["count"] = len(kept)
            result["_truncated"] = True
            result["_total_matching"] = len(results)

        return result

    async def submit_insight(content: str) -> dict:
        """Add an insight to the RAGDoll insights collection (MCP-only). It is searchable via query_rag immediately.

        Use this for a durable learning: a conclusion worth finding again, with the reasoning behind it.

        Input format (plain text with these section headers; only Insight is required):
        - Topic: short title
        - Tags: comma-separated list
        - Insight: the learning itself, stated plainly (Conclusion also works)
        - Question: the question this insight answers
        - Reasoning: why it holds; evidence and how you got there
        - Open questions: what's still unresolved (Open threads also works)
        - Confidence: 0-1

        The statement and reasoning are embedded; results from the insights collection include an "insight" object with id, topic, tags, origin, and confidence.
        """
        try:
            return await asyncio.to_thread(submit_insight_text, content)
        except Exception as e:
            logger.exception("submit_insight failed")
            return {"ok": False, "error": str(e)}

    async def write_memory(content: str) -> dict:
        """Deprecated alias for submit_insight: the memory collection has been replaced by insights.

        Accepts the old memory format (Topic, Date, Tags, Conclusion, Reasoning, Open threads); Conclusion becomes the insight.
        """
        return await submit_insight(content)

    # Register in the MCP tool manifest (must be callable so clients see them in tools/list)
    mcp.tool()(submit_insight)
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
            for source_id, source_path, count, summary, external_url, display_title in raw:
                try:
                    p = Path(source_path)
                    if p.is_absolute() and str(p).startswith(str(sources_dir)):
                        name = p.name
                    else:
                        name = Path(source_path).name if source_path else f"Source {source_id}"
                except Exception:
                    name = source_path or f"Source {source_id}"
                disp = (display_title or "").strip() if display_title else ""
                pretty = disp or name
                out.append({
                    "source_id": source_id,
                    "source_name": pretty,
                    "display_title": disp or None,
                    "source_path": source_path,
                    "chunk_count": count,
                    "summary": summary or "",
                    "external_url": external_url or "",
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
            # FastMCP serves Streamable HTTP at /mcp (GET+POST). Do not add GET-only routes on
            # the same path — they block POST and clients report "endpoint not found".
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
