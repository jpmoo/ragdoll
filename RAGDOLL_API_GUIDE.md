# RAGDoll API Integration Guide

This guide explains how to integrate your chatbot or application with RAGDoll's HTTP API for semantic search across document collections.

## Overview

RAGDoll provides an HTTP API server (default port `9042`) that enables:
- **Semantic search** across ingested documents using natural language queries
- **Collection management** to discover available document collections (including the optional **`memory`** collection when it exists; memories are written via MCP `write_memory`, not the HTTP API)
- **Query expansion** via LLM to improve search accuracy
- **Flexible querying** across all collections or specific ones (repeat `group` on GET, or JSON array on POST)
- **Optional synthesis** (`synthesize`) and **role-limited retrieval** (`limit_chunk_role`); see §3

**Server-side default similarity threshold:** Set **`RAGDOLL_QUERY_THRESHOLD`** in `env.ragdoll` (default `0.45` if unset). Used when a request omits `threshold` (GET) or when POST JSON omits `threshold` (POST body default).

## Base URL

```
http://localhost:9042
```

(Or your server's hostname/IP if accessing remotely)

## Authentication

Currently, RAGDoll API has no authentication. Ensure the API server is only accessible on trusted networks or behind a firewall/proxy.

---

## 1. Discovering Collections

Before querying, discover what document collections are available.

### Endpoint: `GET /rags`

Returns a list of all available RAG collections (groups).

**Request:**
```bash
curl http://localhost:9042/rags
```

**Response:**
```json
{
  "collections": ["_root", "edleadership", "reports", "legal"]
}
```

**In your code:**
```python
import requests

response = requests.get("http://localhost:9042/rags")
collections = response.json()["collections"]
print(f"Available collections: {collections}")
```

**Collection naming:**
- `_root`: Documents ingested directly into the root ingest folder
- Other names: First-level subfolders in the ingest directory (e.g., `edleadership`, `reports`)

---

## 2. Fetching Source Documents

RAGDoll provides a `/fetch` endpoint to retrieve the original source documents referenced in query results.

### Endpoint: `GET /fetch/{group}/{filename}`

Fetches a source document from a specific collection.

**URL Format:**
```
http://localhost:9042/fetch/{group}/{filename}
```

The `filename` should match the relative path within the group's `sources/` directory. For nested paths, use forward slashes (they will be URL-encoded automatically).

**Examples:**
```bash
# Simple filename
curl http://localhost:9042/fetch/edleadership/Visualizing%20Double-loop%20Learning.pdf

# Nested path (if file was in a subdirectory)
curl http://localhost:9042/fetch/reports/2024/Annual%20Report.pdf
```

**Response:**
- Returns the file with appropriate `Content-Type` headers
- PDFs, images, and documents are served inline (browsers can display them)
- Returns `404` if file not found
- Returns `403` if path traversal attempt detected (security)

**Python example:**
```python
import requests

# Get source_url from query result
result = query_ragdoll("double-loop learning")[0]
source_url = result["source_url"]  # e.g., "/fetch/edleadership/document.pdf"

# Fetch the document
base_url = "http://localhost:9042"
response = requests.get(f"{base_url}{source_url}")

if response.status_code == 200:
    # Save or display the file
    with open("document.pdf", "wb") as f:
        f.write(response.content)
    print(f"Downloaded: {result['source_name']}")
```

**Security:**
- Only files within the group's `sources/` directory are accessible
- Path traversal attacks (e.g., `../../../etc/passwd`) are blocked
- Group names are sanitized to prevent directory traversal

---

## 3. Querying Collections

RAGDoll supports two query methods: **GET** (simple URL) and **POST** (JSON body). Both support querying all collections or a specific one.

### Method 1: GET Request (Simple URL)

Best for simple queries without conversation history.

**Query all collections:**
```bash
curl "http://localhost:9042/query?prompt=What%20is%20double-loop%20learning&threshold=0.45"
```

**Query all collections using server default threshold** (omit `threshold`; uses `RAGDOLL_QUERY_THRESHOLD` from `env.ragdoll`, or `0.45`):
```bash
curl "http://localhost:9042/query?prompt=What%20is%20double-loop%20learning"
```

**Query specific collection:**
```bash
curl "http://localhost:9042/query?prompt=What%20is%20double-loop%20learning&group=edleadership&threshold=0.45"
```

**Query multiple collections** (repeat `group`):
```bash
curl "http://localhost:9042/query?prompt=...&group=edleadership&group=memory"
```

**Parameters:**
- `prompt` (required): Your natural language query/question
- `history` (optional): Previous conversation context for better query expansion
- `threshold` (optional): Minimum similarity score (0.0–1.0). Lower = more results, higher = more precise. **If omitted**, the server uses **`RAGDOLL_QUERY_THRESHOLD`** from environment / `env.ragdoll`, or **`0.45`**.
- `group` (optional): One or more collection names; repeat the query parameter (`?group=a&group=b`). If absent, searches **all** collections (including `memory` when that collection exists)
- `limit_chunk_role` (optional, default: false): If true, the server runs your prompt and context through an LLM to infer up to two chunk roles (from the same roles used at ingest), then limits retrieval to chunks matching those roles. If false or absent, retrieval is not limited by role.
- `synthesize` (optional, default: false): If true, after retrieval the server uses the same LLM (query model) to turn prompt+history+top chunks into **instructions for an assistant** or a **direct answer**, so the API can act as a research assistant. The response includes a `synthesis` field.
- `synthesis_mode` (optional, default: "instructions"): When `synthesize=true`, use `"instructions"` (summarize context into instructions for the caller) or `"answer"` (produce a direct answer from the passages).

**Python example:**
```python
import requests
from urllib.parse import quote

prompt = "What is double-loop learning?"
# Omit &threshold=... to use RAGDOLL_QUERY_THRESHOLD on the server
url = f"http://localhost:9042/query?prompt={quote(prompt)}&threshold=0.45"

response = requests.get(url)
results = response.json()
```

### Method 2: POST Request (JSON Body)

Best for complex queries with conversation history and better control.

**Query all collections:**
```bash
curl -X POST http://localhost:9042/query \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "What is double-loop learning?",
    "threshold": 0.45
  }'
```

**Query specific collection with history:**
```bash
curl -X POST http://localhost:9042/query \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Tell me more about that",
    "history": "User: What is double-loop learning?\nAssistant: Double-loop learning is...",
    "group": ["edleadership"],
    "threshold": 0.45
  }'
```

**Python example:**
```python
import requests

payload = {
    "prompt": "What is double-loop learning?",
    "history": "Previous conversation context...",  # Optional
    "threshold": 0.45,  # Optional; default from RAGDOLL_QUERY_THRESHOLD or 0.45
    "group": ["edleadership"],  # Optional: one or more collections; omit or [] to search all
    "limit_chunk_role": False,  # Optional; if True, infer roles from prompt+context and limit retrieval
    "synthesize": False,        # Optional; if True, LLM produces instructions or answer from RAG context
    "synthesis_mode": "instructions"  # "instructions" or "answer" when synthesize=True
}

response = requests.post("http://localhost:9042/query", json=payload)
results = response.json()
```

**Limit retrieval by inferred chunk role (POST):**
```bash
curl -X POST http://localhost:9042/query \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "How do I diagnose low morale?",
    "history": "User: We have turnover issues.\nAssistant: ...",
    "limit_chunk_role": true,
    "threshold": 0.45
  }'
```

When `limit_chunk_role` is true, the server sends your prompt and any `history` to the same LLM used for query expansion, which picks up to two chunk roles (e.g. `description`, `application`, `implication`) from the ingest role list. Retrieval is then limited to chunks with those roles. If the LLM returns no valid roles, retrieval is not limited. The response may include `limit_chunk_role: true` and `inferred_roles: ["role1", "role2"]` when role filtering was applied.

**Synthesis (research-assistant style):** When `synthesize` is true, after retrieval the server calls the same query LLM with the top chunks (up to 15) and asks it to produce either **instructions** for the caller (so the caller can answer using that context) or a **direct answer**. The response includes `synthesis` (the LLM text) and `synthesis_mode`. This makes RAGDoll act as a thinking/synthesis layer: the client receives both the raw RAG results and a digested version.

---

## 4. Understanding Query Results

### Response Structure

Results are returned in two forms:

1. **`documents`** — Grouped by source: each document has its summary and metadata, then its samples (1 of X, 2 of X). Use this to display “X pieces of context from a document with this summary” and “Chunk 1 of X”, “Chunk 2 of X”.
2. **`results`** — Flat list of all chunks sorted by similarity (same as before, for backward compatibility).

**Example (`documents` structure):**

```json
{
  "query": "What is double-loop learning?",
  "expanded_query": "The user is seeking information about double-loop learning...",
  "threshold": 0.45,
  "count": 5,
  "documents": [
    {
      "group": "edleadership",
      "source_path": "/mnt/.../Visualizing Double-loop Learning.pdf",
      "source_name": "Visualizing Double-loop Learning.pdf",
      "source_type": ".pdf",
      "source_url": "/fetch/edleadership/Visualizing%20Double-loop%20Learning.pdf",
      "source_summary": "This PDF explains double-loop learning and its role in organizational change.",
      "sample_count": 2,
      "samples": [
        {
          "chunk_index": 12,
          "context_index": 1,
          "context_total": 2,
          "text": "Double-loop learning involves questioning underlying assumptions...",
          "artifact_type": "text",
          "page": 3,
          "chunk_role": "description",
          "similarity": 0.8234
        },
        {
          "chunk_index": 14,
          "context_index": 2,
          "context_total": 2,
          "text": "In contrast to single-loop learning...",
          "similarity": 0.8012
        }
      ]
    },
    {
      "group": "edleadership",
      "source_name": "SEQuity Teacher Reference.pdf",
      "source_summary": null,
      "sample_count": 1,
      "samples": [
        {
          "context_index": 1,
          "context_total": 1,
          "text": "...",
          "similarity": 0.7123
        }
      ]
    }
  ],
  "results": [ ... ]
}
```

### Field Descriptions

- **`query`**: Your original prompt
- **`expanded_query`**: LLM-expanded standalone description (used for embedding)
- **`threshold`**: Similarity threshold that was applied (reflects the request value or server default from `RAGDOLL_QUERY_THRESHOLD`)
- **`count`**: Total number of chunks returned
- **`documents`**: Array of document blocks. Each block has: **`group`**, **`source_path`**, **`source_name`**, **`source_url`**, **`source_type`**, **`source_summary`** (document summary or null), **`sample_count`**, **`samples`** (array of chunks from that document with **`context_index`** (1 of X), **`context_total`** (X), **`text`**, **`similarity`**, etc.). Documents are ordered by best similarity in that document.
- **`results`**: Flat array of all chunks, sorted by similarity (for backward compatibility)

### Result Object Fields

Each result in the `results` array contains:

- **`group`**: Collection name the chunk belongs to
- **`source_path`**: Full path to the source document (filesystem path)
- **`source_name`**: Filename of the source document
- **`source_url`**: HTTP URL to fetch the source document (e.g., `/fetch/edleadership/document.pdf`)
- **`source_type`**: File extension (e.g., `.pdf`, `.docx`)
- **`chunk_index`**: Index of this chunk within the source document
- **`source_summary`**: The document's 1–3 sentence summary (if set during ingest); `null` if none
- **`context_index`**: 1-based position of this chunk among results from the same document (e.g. 1, 2, 3)
- **`context_total`**: Total number of result chunks from this document (e.g. 3 → "1 of 3", "2 of 3", "3 of 3")
- **`text`**: The actual text content (cleaned, no newlines)
- **`artifact_type`**: Type of content:
  - `"text"`: Regular prose text
  - `"chart_summary"`: LLM summary of a chart/graph
  - `"table_summary"`: LLM summary of a table
  - `"figure_summary"`: LLM summary of a figure/diagram
- **`artifact_path`**: Path to stored artifact (image/JSON) if applicable, `null` for text
- **`page`**: Page number (for PDFs), `null` for non-paginated documents
- **`chunk_role`**: For document chunks: role from ingest (e.g. `description`, `application`, `implication`), or `null`. For **memory** chunks: `conclusion`, `reasoning`, `open_threads`, or `full`.
- **`similarity`**: Cosine similarity score (0.0-1.0), higher = more relevant
- **`memory_topic`**, **`memory_date`**, **`memory_tags`**: Present when **`group`** is **`memory`** (and summary JSON parses); empty string / `[]` if missing
- **`primary_question_answered`**: May be set on document chunks from ingest; typically unused for memory

**Presenting results:** Prefer the **`documents`** array for a document-first UX. For each document, show its summary and metadata, then list its samples with labels like *"Sample 1 of 3"*, *"Sample 2 of 3"* using each sample’s `context_index` and `context_total`. Documents are ordered by relevance (best similarity in that document). If you use the flat **`results`** list instead, group by `(group, source_path)` and use `source_summary`, `context_index`, and `context_total` on each result for the same labels.

When `limit_chunk_role` was true and role filtering was applied, the response also includes:
- **`limit_chunk_role`**: `true`
- **`inferred_roles`**: Array of the one or two roles inferred from the user input (e.g. `["description", "application"]`)

When `synthesize` was true, the response also includes:
- **`synthesis`**: The LLM-produced text (instructions for the caller or a direct answer, depending on `synthesis_mode`)
- **`synthesis_mode`**: `"instructions"` or `"answer"`

If no chunks matched the inferred roles (e.g. most chunks have no role set), the server falls back to unfiltered retrieval and adds **`role_filter_relaxed`**: `true` and **`inferred_roles`** so you still get results and know the filter was relaxed.

### Memory collection (`group` = `memory`)

When the **`memory`** collection exists (created after at least one MCP `write_memory` call), it is included in **search-all** queries the same as document collections. You can also target it with `group=memory` (GET) or `"group": ["memory"]` (POST).

- **Result metadata:** Each hit from `memory` may include **`memory_topic`**, **`memory_date`**, and **`memory_tags`** (parsed from the stored memory). Document-level entries in **`documents`** may include the same fields on the block.
- **`chunk_role`:** For memory chunks, values are **`conclusion`**, **`reasoning`**, **`open_threads`**, or **`full`** (not the document ingest roles `description` / `application` / `implication`).
- **`source_url`:** Usually **`null`** for memory (there is no file to fetch).
- **`limit_chunk_role`:** Role filtering applies only to **document** collections; the **`memory`** group is always searched without that filter so memories are not excluded.

**Writing memories** is **not** available on the HTTP API; use the MCP tool **`write_memory`**.

---

## 5. Integration Patterns for Chatbots

### Pattern 1: Simple Question-Answer

```python
import requests

def query_ragdoll(prompt: str, collection: str = None, threshold: float | None = None):
    """Query RAGDoll and return top results. Omit threshold to use server default (RAGDOLL_QUERY_THRESHOLD)."""
    url = "http://localhost:9042/query"
    payload = {"prompt": prompt}
    if threshold is not None:
        payload["threshold"] = threshold
    if collection:
        payload["group"] = [collection]
    
    response = requests.post(url, json=payload)
    if response.status_code == 200:
        data = response.json()
        return data["results"]
    else:
        raise Exception(f"Query failed: {response.status_code}")

# Usage
results = query_ragdoll("What is double-loop learning?", collection="edleadership")
if results:
    top_result = results[0]
    print(f"Found: {top_result['text'][:200]}...")
    print(f"Source: {top_result['source_name']}")
    print(f"Similarity: {top_result['similarity']}")
```

### Pattern 2: Conversational Context

```python
def query_with_history(prompt: str, conversation_history: str, collection: str = None):
    """Query with conversation context for better query expansion."""
    url = "http://localhost:9042/query"
    payload = {
        "prompt": prompt,
        "history": conversation_history,
        "threshold": 0.45,
    }
    if collection:
        payload["group"] = [collection]
    
    response = requests.post(url, json=payload)
    return response.json()["results"]

# Usage in chatbot
conversation = "User: What is double-loop learning?\nAssistant: Double-loop learning is..."
results = query_with_history("Tell me more about that", conversation, "edleadership")
```

### Pattern 3: Multi-Collection Search

```python
def search_all_collections(prompt: str, threshold: float = 0.45):
    """Search across all collections and group results by collection."""
    url = "http://localhost:9042/query"
    payload = {
        "prompt": prompt,
        "threshold": threshold
    }
    # Don't specify 'group' to search all
    
    response = requests.post(url, json=payload)
    data = response.json()
    
    # Group results by collection
    by_collection = {}
    for result in data["results"]:
        group = result["group"]
        if group not in by_collection:
            by_collection[group] = []
        by_collection[group].append(result)
    
    return by_collection

# Usage
results_by_collection = search_all_collections("strategic planning")
for collection, results in results_by_collection.items():
    print(f"\n{collection}: {len(results)} results")
    for r in results[:3]:  # Top 3 per collection
        print(f"  - {r['source_name']} (similarity: {r['similarity']})")
```

### Pattern 4: Adaptive Threshold

```python
def adaptive_search(prompt: str, collection: str = None, min_results: int = 3):
    """Search with adaptive threshold to ensure minimum results."""
    thresholds = [0.6, 0.5, 0.45, 0.4, 0.35, 0.3]  # Try higher to lower
    
    for threshold in thresholds:
        url = "http://localhost:9042/query"
        payload = {"prompt": prompt, "threshold": threshold}
        if collection:
            payload["group"] = [collection]
        
        response = requests.post(url, json=payload)
        data = response.json()
        
        if len(data["results"]) >= min_results:
            return data["results"]
    
    # Return whatever we got at lowest threshold
    return data["results"]
```

---

## 6. Error Handling

### HTTP Status Codes

- **200 OK**: Query successful
- **404 Not Found**: Collection name doesn't exist (when `group` is specified)
- **500 Internal Server Error**: Server error (embedding failure, etc.)

### Error Response Format

```json
{
  "detail": "Collection 'invalid_collection' not found. Available collections: ['_root', 'edleadership']"
}
```

### Robust Query Function

```python
import requests
from typing import Optional, List, Dict, Any

def safe_query_ragdoll(
    prompt: str,
    collection: Optional[str] = None,
    threshold: float = 0.45,
    history: Optional[str] = None,
    timeout: int = 60
) -> Dict[str, Any]:
    """
    Safely query RAGDoll with error handling.
    
    Returns:
        dict with 'success', 'results', 'error' keys
    """
    url = "http://localhost:9042/query"
    payload = {
        "prompt": prompt,
        "threshold": threshold
    }
    if collection:
        payload["group"] = [collection]
    if history:
        payload["history"] = history
    
    try:
        response = requests.post(url, json=payload, timeout=timeout)
        response.raise_for_status()
        data = response.json()
        return {
            "success": True,
            "results": data.get("results", []),
            "count": data.get("count", 0),
            "expanded_query": data.get("expanded_query", "")
        }
    except requests.exceptions.Timeout:
        return {
            "success": False,
            "error": "Request timeout",
            "results": []
        }
    except requests.exceptions.ConnectionError:
        return {
            "success": False,
            "error": "Cannot connect to RAGDoll API",
            "results": []
        }
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 404:
            return {
                "success": False,
                "error": f"Collection not found: {collection}",
                "results": []
            }
        return {
            "success": False,
            "error": f"HTTP error: {e.response.status_code}",
            "results": []
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"Unexpected error: {str(e)}",
            "results": []
        }

# Usage
result = safe_query_ragdoll("What is double-loop learning?", collection="edleadership")
if result["success"]:
    print(f"Found {result['count']} results")
    for r in result["results"]:
        print(f"- {r['text'][:100]}...")
else:
    print(f"Error: {result['error']}")
```

---

## 7. Best Practices

### 1. **Choose Appropriate Thresholds**

- **0.6-0.7**: Very precise, fewer results (good for specific facts)
- **0.45-0.55**: Balanced (default, good for most queries)
- **0.3-0.4**: More results, may include less relevant content

### 2. **Use Conversation History**

When users ask follow-up questions, include previous conversation context:

```python
# Bad: No context
query_ragdoll("Tell me more about that")

# Good: With context
history = "User: What is double-loop learning?\nAssistant: Double-loop learning is..."
query_ragdoll("Tell me more about that", history=history)
```

### 3. **Handle Empty Results**

```python
results = query_ragdoll("very specific query", threshold=0.7)
if not results:
    # Try lower threshold
    results = query_ragdoll("very specific query", threshold=0.45)
    if not results:
        return "I couldn't find relevant information in the documents."
```

### 4. **Respect Result Types**

Different `artifact_type` values indicate different content:

- **`text`**: Use directly in responses
- **`chart_summary`**, **`table_summary`**, **`figure_summary`**: These are LLM-generated summaries. Consider mentioning "According to a chart/table in..." when citing

### 5. **Cite Sources with Links**

Always include source information and provide links to original documents:

```python
def format_response(result, base_url: str = "http://localhost:9042"):
    """Format a RAG result for chatbot display with source link."""
    text = result["text"]
    source = result["source_name"]
    source_url = result.get("source_url")
    page = result.get("page")
    
    citation = f"Source: {source}"
    if page:
        citation += f" (page {page})"
    
    if source_url:
        full_url = f"{base_url}{source_url}"
        citation += f" | [View document]({full_url})"
    
    return f"{text}\n\n{citation}"
```

### 6. **Provide Document Access**

When users ask to see the original document, use the `source_url`:

```python
def get_document_link(result, base_url: str = "http://localhost:9042"):
    """Get full URL to fetch the source document."""
    source_url = result.get("source_url")
    if source_url:
        return f"{base_url}{source_url}"
    return None

# Usage
result = query_ragdoll("double-loop learning")[0]
doc_link = get_document_link(result)
if doc_link:
    print(f"View full document: {doc_link}")
```

### 7. **Batch Similar Queries**

If you need to query multiple related topics, consider batching or caching:

```python
# Cache results for similar queries
query_cache = {}

def cached_query(prompt: str, collection: str = None):
    cache_key = f"{prompt}:{collection}"
    if cache_key in query_cache:
        return query_cache[cache_key]
    
    results = query_ragdoll(prompt, collection)
    query_cache[cache_key] = results
    return results
```

---

## 8. Complete Chatbot Integration Example

```python
import requests
from typing import List, Dict, Optional

class RAGDollClient:
    """Simple client for RAGDoll API."""
    
    def __init__(self, base_url: str = "http://localhost:9042"):
        self.base_url = base_url.rstrip("/")
    
    def list_collections(self) -> List[str]:
        """Get list of available collections."""
        response = requests.get(f"{self.base_url}/rags")
        return response.json()["collections"]
    
    def query(
        self,
        prompt: str,
        collection: Optional[str] = None,
        threshold: Optional[float] = None,
        history: Optional[str] = None
    ) -> Dict:
        """Query RAGDoll. Omit threshold to use server default (RAGDOLL_QUERY_THRESHOLD)."""
        payload: Dict = {"prompt": prompt}
        if threshold is not None:
            payload["threshold"] = threshold
        if collection:
            payload["group"] = [collection]  # API expects a list of collection names
        if history:
            payload["history"] = history
        
        response = requests.post(f"{self.base_url}/query", json=payload)
        response.raise_for_status()
        return response.json()
    
    def format_response(self, results: List[Dict], max_results: int = 3) -> str:
        """Format results for chatbot display."""
        if not results:
            return "I couldn't find relevant information in the documents."
        
        formatted = []
        for i, result in enumerate(results[:max_results], 1):
            text = result["text"]
            source = result["source_name"]
            similarity = result["similarity"]
            
            # Truncate long text
            if len(text) > 500:
                text = text[:500] + "..."
            
            formatted.append(
                f"[{i}] {text}\n"
                f"   Source: {source} (relevance: {similarity:.2f})"
            )
        
        return "\n\n".join(formatted)


# Usage in chatbot
ragdoll = RAGDollClient()

# Discover collections
collections = ragdoll.list_collections()
print(f"Available collections: {collections}")

# Query
results = ragdoll.query(
    "What is double-loop learning?",
    collection="edleadership",
    threshold=0.45
)

# Format for user
response_text = ragdoll.format_response(results["results"])
print(response_text)
```

---

## 9. Troubleshooting

### No Results Returned

1. **Lower the threshold**: Try `0.35` or `0.3` in the request, or set **`RAGDOLL_QUERY_THRESHOLD`** lower in `env.ragdoll` for the default
2. **Check collection name**: Verify with `GET /rags`
3. **Broaden query**: Make the prompt more general
4. **Check if collection has data**: Verify the collection's database exists and has chunks

### Slow Responses

1. **Check Ollama**: Ensure Ollama is running and responsive
2. **Reduce threshold**: Lower thresholds may return more results but shouldn't be slower
3. **Query specific collection**: Use `group` parameter to limit search scope

### Connection Errors

1. **Verify API is running**: `systemctl status ragdoll-api`
2. **Check port**: Default is `9042`, verify `RAGDOLL_API_PORT` setting
3. **Check firewall**: Ensure port is accessible
4. **Test locally**: `curl http://localhost:9042/rags`

---

## 10. Quick Reference

### Environment variables (see `env.ragdoll.example`)

| Variable | Role for the HTTP API |
|----------|------------------------|
| `RAGDOLL_API_PORT` | API listen port (default `9042`) |
| `RAGDOLL_QUERY_THRESHOLD` | Default minimum similarity when a request omits `threshold` |
| `RAGDOLL_ENV` | Path to `env.ragdoll` if not at project root |

Other paths (`RAGDOLL_INGEST_PATH`, `RAGDOLL_OUTPUT_PATH`, Ollama host/model, etc.) affect ingest and query behavior; see the example env file and main README.

### Endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/rags` | GET | List all collections |
| `/fetch/{group}/{filename}` | GET | Fetch source document |
| `/query` | GET | Query (URL parameters) |
| `/query` | POST | Query (JSON body) |

### Query parameters (GET and POST body)

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `prompt` | string | Yes | - | Natural language query |
| `history` | string | No | null | Conversation context for query expansion |
| `threshold` | float | No | `RAGDOLL_QUERY_THRESHOLD` or `0.45` | Minimum cosine similarity (0.0–1.0). Omit on GET to use server default. |
| `group` | string[] | No | null | One or more collections. GET: repeat `?group=a&group=b`. POST: JSON array, e.g. `["edleadership"]` or `["a","memory"]`. Omit to search all (including `memory` if present). |
| `limit_chunk_role` | boolean | No | false | Infer ingest chunk roles and filter document chunks; **memory** group is never filtered this way. |
| `synthesize` | boolean | No | false | If true, LLM produces `synthesis` from top chunks. |
| `synthesis_mode` | string | No | `"instructions"` | With `synthesize`: `"instructions"` or `"answer"`. |

### Result fields (per chunk in `results` / samples in `documents`)

| Field | Type | Description |
|-------|------|-------------|
| `text` | string | Chunk text content |
| `similarity` | float | Relevance score (0.0–1.0) |
| `source_name` | string | Source filename or memory pseudo-path segment |
| `source_url` | string / null | `/fetch/...` for files; **null** for memory |
| `group` | string | Collection name (`memory` for memories) |
| `page` | int / null | Page number (PDFs); often null for memory |
| `artifact_type` | string | Content type (e.g. `text`, chart/table/figure summaries) |
| `chunk_role` | string / null | Document: `description` / `application` / `implication`. Memory: `conclusion` / `reasoning` / `open_threads` / `full`. |
| `memory_topic` | string | Memory collection only (when set) |
| `memory_date` | string | Memory collection only (when set) |
| `memory_tags` | array | Memory collection only (when set) |

---

## Support

For issues or questions:
- Check RAGDoll logs: `/path/to/data/{group}/action.log`
- Check API server logs: `journalctl -u ragdoll-api`
- Verify Ollama is running: `curl http://localhost:11434/api/tags`
