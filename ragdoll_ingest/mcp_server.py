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

# Per-chunk fields that repeat their document's metadata; query_rag returns them once, on the document entry
_DOCUMENT_LEVEL_FIELDS = ("source_url", "source_type", "source_summary", "insight")


def _compact_query_result(result: dict) -> dict:
    """Return each chunk's text and each document's metadata once, to keep MCP responses readable by clients.

    results keeps the chunks in relevance order without document-level fields; documents keeps one entry per source
    (summary, URL, type, insight metadata) with the chunk_ids it contributed instead of repeating those chunks.
    """
    out = dict(result)
    out["results"] = [
        {k: v for k, v in r.items() if k not in _DOCUMENT_LEVEL_FIELDS} for r in result.get("results") or []
    ]
    out["documents"] = [
        {**{k: v for k, v in d.items() if k != "samples"}, "chunk_ids": [s["chunk_id"] for s in d.get("samples") or []]}
        for d in result.get("documents") or []
    ]
    return out


def _limit_per_document(results: list[dict], max_per_document: int) -> list[dict]:
    """Keep at most max_per_document chunks from each source (results are sorted by similarity). 0 = no limit."""
    if max_per_document <= 0:
        return results
    counts: dict[tuple[str, str], int] = {}
    kept = []
    for r in results:
        k = (r["group"], r["source_path"])
        if counts.get(k, 0) < max_per_document:
            counts[k] = counts.get(k, 0) + 1
            kept.append(r)
    return kept


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
        max_per_document: int = 3,
        synthesize: bool = False,
        synthesis_mode: str = "instructions",
        include_insights: bool = True,
    ) -> dict:
        """Semantic similarity search over one or more RAGDoll document collections. Returns matching document chunks sorted by relevance.
        When synthesize=true, RAGDoll also uses its LLM to turn prompt+history+chunks into instructions or an answer (research-assistant style).

        Response: "results" lists the matching chunks in relevance order (group, source_name, source_path, chunk_id, text,
        similarity, chunk_role, page, context_index/context_total). "documents" has one entry per source, ordered by its best
        chunk: source_summary, source_url, sample_count, the chunk_ids it contributed, and for the insights collection an
        "insight" object. Each chunk's text and each document's summary appear once. Chunk ids are unique only within a collection, so join on group + chunk_id.

        Args:
            prompt: Your question or information need.
            history: Optional prior conversation turns as plain text, used for query expansion.
            threshold: Minimum cosine similarity (0.0–1.0). Lower = more results, less precise. Omit to use RAGDOLL_QUERY_THRESHOLD (default 0.45).
            collections: Collection names to search. If omitted or empty, searches all collections.
            limit_chunk_role: When true, infer up to 2 chunk roles from the prompt and restrict retrieval to those roles.
            max_results: Maximum number of chunks to return. Default 20. Applies to both the flat results list and the grouped documents view (_total_matching reports how many matched).
            max_per_document: Maximum chunks returned from any one document, applied before max_results so the slots spread across sources. Default 3; 0 = no limit.
            synthesize: When true, LLM synthesizes prompt+history+RAG into instructions for an assistant or a direct answer.
            synthesis_mode: "instructions" (default) = instructions for the caller to use; "answer" = direct summary/answer.
            include_insights: When true (default), the insights collection is searched too, even if collections names others. Insight documents carry an "insight" object (id, topic, tags, origin, confidence). Insights are held to about a quarter of max_results so they don't crowd out the documents they were drawn from.
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
                # Insight statements closely match the questions that produced them, so they outrank the
                # documents behind them. Keep them to a quarter of the slots the caller asked for.
                insights_cap=min(config.INSIGHTS_MAX_RESULTS, max(1, max_results // 4)),
            )
        except HTTPException as e:
            raise ValueError(f"{e.detail}") from e
        except Exception as e:
            logger.exception("query_rag failed")
            raise ValueError(f"Query failed: {e}") from e

        # Cap results for large responses (spec: max_results), after limiting chunks per document so the slots spread
        # across sources. Rebuild the documents view from the kept chunks too; otherwise it still carries every match.
        results = result.get("results") or []
        kept = _limit_per_document(results, max_per_document)[:max_results]
        if len(kept) < len(results):
            _number_context(kept)
            result = dict(result)
            result["results"] = kept
            result["documents"] = _group_results_by_document(kept)
            result["count"] = len(kept)
            result["_truncated"] = True
            result["_total_matching"] = len(results)

        return _compact_query_result(result)

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


class _NormalizeTrailingSlash:
    """Serve "/mcp/" as "/mcp" instead of redirecting to it.

    Starlette answers the trailing-slash form with a 307 whose Location it builds from the request's Host
    header. Behind a reverse proxy that doesn't forward the external host, that Location is the server's own
    loopback address (http://127.0.0.1:9044/mcp), and clients that follow it hang on their own localhost.
    Rewriting the path here means either form just works, however the proxy is configured.
    """

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope.get("type") == "http":
            path = scope.get("path", "")
            if len(path) > 1 and path.endswith("/"):
                scope = dict(scope)
                scope["path"] = path.rstrip("/")
                if scope.get("raw_path"):
                    scope["raw_path"] = scope["path"].encode("utf-8")
        await self.app(scope, receive, send)


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
            _NormalizeTrailingSlash(app),
            host=config.MCP_HOST,
            port=config.MCP_PORT,
            log_level="info",
        )
    else:
        mcp.run()


if __name__ == "__main__":
    main()
