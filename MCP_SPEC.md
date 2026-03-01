# RAGDoll MCP Server — Implementation Spec

## Overview

Add a **Model Context Protocol (MCP) server** to RAGDoll so that MCP-capable clients (Claude Desktop, Claude Code, Cursor, Zed, etc.) can query RAGDoll collections as a first-class tool — no HTTP wiring required on the client side.

The MCP server wraps the existing `ragdoll_ingest` package directly (same in-process calls, no round-trip through port 9042) and is delivered as a standalone script that can run in **stdio mode** (standard MCP local transport) or optionally in **HTTP/SSE mode** (remote transport).

---

## Goals

- Expose RAGDoll's semantic search to any MCP client with zero extra configuration beyond a single entry in the client's MCP settings.
- Reuse all existing logic from `ragdoll_ingest` (embedder, storage, query expansion, role inference) — no duplication.
- Ship as a lightweight addition: one new file (`ragdoll_ingest/mcp_server.py`), one new entry point in `pyproject.toml`, one new optional systemd service file for HTTP/SSE mode.
- No breaking changes to the existing API server (port 9042) or review app (port 9043).

---

## Transport Options

### Option A — stdio (recommended for local / Claude Desktop / Claude Code)

The MCP server is launched as a subprocess by the client. Communication is over stdin/stdout using the MCP JSON-RPC framing. This is the default and requires no open port.

```
claude-desktop / claude-code
  └─ spawns: ragdoll-mcp   (reads RAGDOLL_ENV / env.ragdoll for config)
       └─ stdin/stdout JSON-RPC
```

### Option B — HTTP + SSE (for remote clients or multi-user deployments)

The MCP server also supports SSE transport, exposed as a FastAPI route mounted at a configurable path (default `/mcp`). This can either be mounted onto the **existing API app** (port 9042) or run as a **separate process** on a new port (default `9044`).

For simplicity, the spec recommends running it as a **separate process** so the existing API is not changed. Mounting onto the existing app is noted as an alternative.

---

## New File: `ragdoll_ingest/mcp_server.py`

### Dependencies

Add to `pyproject.toml` under `[project.optional-dependencies]`:

```toml
[project.optional-dependencies]
mcp = ["mcp[cli]>=1.0.0"]
```

Install with:

```bash
pip install -e '.[mcp]'
```

The `mcp` package is the official Anthropic MCP Python SDK. It provides `FastMCP`, the decorator-based server API used below.

### Server Bootstrap

```python
# ragdoll_ingest/mcp_server.py  (pseudocode — structure only)

from mcp.server.fastmcp import FastMCP
from ragdoll_ingest import config
from ragdoll_ingest.api import _do_query        # existing shared query logic
from ragdoll_ingest.storage import _list_sync_groups

mcp = FastMCP(
    name="ragdoll",
    instructions=(
        "RAGDoll gives you semantic search over ingested document collections. "
        "Use query_rag to find relevant content. Use list_collections to discover "
        "what collections are available before querying a specific one."
    ),
)
```

---

## MCP Tools

### Tool 1: `query_rag`

**Description:** Semantic similarity search over one or more RAGDoll document collections. Returns matching document chunks sorted by relevance.

**Input schema:**

| Field | Type | Required | Default | Description |
|---|---|---|---|---|
| `prompt` | string | yes | — | The user's question or information need |
| `history` | string | no | `null` | Prior conversation turns as a plain-text string, used for query expansion |
| `threshold` | number | no | `0.45` | Minimum cosine similarity (0.0–1.0). Lower = more results, less precise |
| `collections` | array of strings | no | `null` | Collection names to search. If omitted or empty, searches all collections |
| `limit_chunk_role` | boolean | no | `false` | When true, asks the LLM to infer up to 2 chunk roles and restricts retrieval to those roles |

**Output:** The full `_do_query()` response dict, which includes:

- `query` — original prompt
- `expanded_query` — LLM-expanded standalone information need
- `threshold` — threshold used
- `count` — total matching chunks
- `documents` — array of document blocks, each containing:
  - `group`, `source_name`, `source_path`, `source_type`, `source_url`, `source_summary`
  - `sample_count` — how many chunks from this document matched
  - `samples` — array of chunk objects (see below)
- `results` — flat array of all matching chunks (same data, sorted by similarity)
- `inferred_roles` / `limit_chunk_role` — present when role filtering was used

Each chunk in `samples` / `results`:

| Field | Description |
|---|---|
| `group` | Collection name |
| `source_name` | Filename of the source document |
| `source_path` | Full path in the sources directory |
| `source_url` | Relative URL to fetch the source via the HTTP API (`/fetch/{group}/...`) |
| `source_type` | File extension (`.pdf`, `.docx`, etc.) |
| `source_summary` | LLM-generated document-level summary |
| `chunk_index` | Position of the chunk within its source document |
| `text` | The chunk text |
| `primary_question_answered` | Semantic label: what question this chunk answers |
| `chunk_role` | Semantic role: `description`, `application`, or `implication` |
| `artifact_type` | `text`, `chart_summary`, `table_summary`, or `figure_summary` |
| `artifact_path` | Path to stored artifact (chart image, table JSON) if applicable |
| `page` | Source page number if known |
| `similarity` | Cosine similarity score (0.0–1.0) |
| `context_index` | Rank of this chunk among chunks from the same document (1 of N) |
| `context_total` | Total chunks returned from the same document |

**Implementation note:** Call `_do_query(prompt, history, threshold, collections, limit_chunk_role)` directly — this is the same function already used by the HTTP API endpoints. The MCP layer is just a thin adapter.

---

### Tool 2: `list_collections`

**Description:** List all available RAGDoll document collections. Call this before `query_rag` if you are not sure which collections exist, or to discover what knowledge bases are available.

**Input schema:** No inputs.

**Output:**

```json
{
  "collections": ["_root", "reports", "legal", "edleadership"]
}
```

**Implementation note:** Call `_list_sync_groups()` directly.

---

## MCP Resources (optional, implement if time allows)

Resources let MCP clients browse and read content without calling a tool. These are lower priority than the tools above but add significant value for clients that support resource browsing.

### Resource: `ragdoll://collections`

Lists all collections. URI: `ragdoll://collections`
MIME type: `application/json`
Returns the same payload as `list_collections`.

### Resource: `ragdoll://collections/{group}/sources`

Lists sources in a specific collection.
URI template: `ragdoll://collections/{group}/sources`
MIME type: `application/json`
Returns an array of `{ source_id, source_name, source_path, chunk_count, summary }` — same data as `GET /api/groups/{group}/sources` on the review app.

---

## Entry Point

Add to `pyproject.toml` `[project.scripts]`:

```toml
ragdoll-mcp = "ragdoll_ingest.mcp_server:main"
```

The `main()` function should:

1. Load config (env.ragdoll / environment variables) — this already happens at import time via `ragdoll_ingest.config`.
2. Read the transport mode from the environment: `RAGDOLL_MCP_TRANSPORT` = `stdio` (default) or `sse`.
3. If `stdio`: call `mcp.run(transport="stdio")`.
4. If `sse`: call `mcp.run(transport="sse", host=..., port=...)` using `RAGDOLL_MCP_HOST` (default `127.0.0.1`) and `RAGDOLL_MCP_PORT` (default `9044`).

---

## New Config Variables

Add to `ragdoll_ingest/config.py` and `env.ragdoll.example`:

| Variable | Default | Description |
|---|---|---|
| `RAGDOLL_MCP_TRANSPORT` | `stdio` | `stdio` for local subprocess mode, `sse` for HTTP/SSE server mode |
| `RAGDOLL_MCP_HOST` | `127.0.0.1` | Bind host when `RAGDOLL_MCP_TRANSPORT=sse` |
| `RAGDOLL_MCP_PORT` | `9044` | Bind port when `RAGDOLL_MCP_TRANSPORT=sse` |

---

## systemd Service (HTTP/SSE mode only)

Create `ragdoll-mcp.service` following the same pattern as `ragdoll-api.service`:

```ini
[Unit]
Description=RAGDoll MCP Server (HTTP/SSE)
After=network.target ragdoll-api.service

[Service]
Type=simple
EnvironmentFile=-/opt/ragdoll/env.ragdoll
EnvironmentFile=-/etc/default/ragdoll-ingest
Environment=RAGDOLL_MCP_TRANSPORT=sse
WorkingDirectory=/opt/ragdoll
ExecStart=/opt/ragdoll/.venv/bin/python -m ragdoll_ingest.mcp_server
Restart=on-failure

[Install]
WantedBy=multi-user.target
```

---

## Claude Desktop Configuration (stdio mode)

Once installed, users add RAGDoll to `~/Library/Application Support/Claude/claude_desktop_config.json` (macOS) or the equivalent on their OS:

```json
{
  "mcpServers": {
    "ragdoll": {
      "command": "/opt/ragdoll/.venv/bin/ragdoll-mcp",
      "env": {
        "RAGDOLL_ENV": "/opt/ragdoll/env.ragdoll"
      }
    }
  }
}
```

If using a venv activation approach instead of the direct binary path, the command can also be:

```json
{
  "command": "/opt/ragdoll/.venv/bin/python",
  "args": ["-m", "ragdoll_ingest.mcp_server"]
}
```

---

## Claude Code / Cowork Configuration (stdio mode)

Add to `.mcp.json` in the project root, or configure globally:

```json
{
  "mcpServers": {
    "ragdoll": {
      "command": "/opt/ragdoll/.venv/bin/ragdoll-mcp",
      "env": {
        "RAGDOLL_ENV": "/opt/ragdoll/env.ragdoll"
      }
    }
  }
}
```

---

## Example MCP Interaction

An MCP client (e.g. Claude Desktop) would call the tools like this:

**List collections:**
```
Tool: list_collections
Input: {}
Output: {"collections": ["_root", "edleadership", "reports"]}
```

**Query:**
```
Tool: query_rag
Input: {
  "prompt": "What is double-loop learning?",
  "collections": ["edleadership"],
  "threshold": 0.45
}
Output: {
  "query": "What is double-loop learning?",
  "expanded_query": "A description of double-loop learning as a concept in organizational theory...",
  "count": 3,
  "documents": [
    {
      "group": "edleadership",
      "source_name": "argyris_1977.pdf",
      "source_summary": "...",
      "sample_count": 2,
      "samples": [
        {
          "text": "Double-loop learning occurs when...",
          "similarity": 0.81,
          "chunk_role": "description",
          "page": 4,
          ...
        }
      ]
    }
  ],
  ...
}
```

---

## File Checklist for Vibe-Coder

- [x] `ragdoll_ingest/mcp_server.py` — new MCP server module
- [x] `pyproject.toml` — add `mcp` optional dependency and `ragdoll-mcp` entry point
- [x] `ragdoll_ingest/config.py` — add `RAGDOLL_MCP_TRANSPORT`, `RAGDOLL_MCP_HOST`, `RAGDOLL_MCP_PORT`
- [x] `env.ragdoll.example` — document the three new MCP variables
- [x] `ragdoll-mcp.service` — systemd unit for SSE mode
- [x] `README.md` — add MCP section (transport options, config, client setup examples)
- [x] `install-service.sh` — optionally include `ragdoll-mcp.service` in the install script

---

## Implementation Notes and Gotchas

**Import path:** `_do_query` in `ragdoll_ingest/api.py` currently lives as a module-level function. Import it as `from ragdoll_ingest.api import _do_query`. If the leading underscore makes the vibe-coder nervous, rename it to `do_query` at the same time (it's only called internally).

**Threading / async:** FastMCP handles its own event loop. The existing `_do_query` path uses synchronous `requests` calls to Ollama and synchronous SQLite. This is fine for stdio mode (one user, sequential calls). For SSE mode with concurrent clients, consider wrapping the blocking `_do_query` call in `asyncio.to_thread()` inside an async tool handler.

**Error handling:** If Ollama is unreachable, `_do_query` raises an `HTTPException`. The MCP layer should catch this and raise an `McpError` with a human-readable message rather than letting a raw FastAPI exception bubble up.

**Output size:** The full `results` array can be large for broad queries. Consider adding a `max_results` parameter (default 20) to `query_rag` to cap the flat results list. The `documents` grouped view already provides a natural summary.

**Security:** In stdio mode, the MCP server inherits the file permissions of the launching process. In SSE mode, bind to `127.0.0.1` by default (not `0.0.0.0`) and document that users should put it behind a reverse proxy with auth if exposing externally, consistent with how the review app is documented.

**Testing the server:** After install, verify with the MCP CLI inspector:
```bash
npx @modelcontextprotocol/inspector ragdoll-mcp
```
This opens a browser UI to test tools interactively without a full client.
