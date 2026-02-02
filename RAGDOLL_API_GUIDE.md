# RAGDoll API Integration Guide

This guide explains how to integrate your chatbot or application with RAGDoll's HTTP API for semantic search across document collections.

## Overview

RAGDoll provides an HTTP API server (default port `9042`) that enables:
- **Semantic search** across ingested documents using natural language queries
- **Collection management** to discover available document collections
- **Query expansion** via LLM to improve search accuracy
- **Flexible querying** across all collections or specific ones

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

**Query specific collection:**
```bash
curl "http://localhost:9042/query?prompt=What%20is%20double-loop%20learning&group=edleadership&threshold=0.45"
```

**Parameters:**
- `prompt` (required): Your natural language query/question
- `history` (optional): Previous conversation context for better query expansion
- `threshold` (optional, default: 0.45): Minimum similarity score (0.0-1.0). Lower = more results, higher = more precise
- `group` (optional): Specific collection name. If absent, searches all collections
- `limit_chunk_role` (optional, default: false): If true, the server runs your prompt and context through an LLM to infer up to two chunk roles (from the same roles used at ingest), then limits retrieval to chunks matching those roles. If false or absent, retrieval is not limited by role.

**Python example:**
```python
import requests
from urllib.parse import quote

prompt = "What is double-loop learning?"
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
    "group": "edleadership",
    "threshold": 0.45
  }'
```

**Python example:**
```python
import requests

payload = {
    "prompt": "What is double-loop learning?",
    "history": "Previous conversation context...",  # Optional
    "threshold": 0.45,  # Optional, default 0.45
    "group": "edleadership",  # Optional, searches all if absent
    "limit_chunk_role": False  # Optional; if True, infer roles from prompt+context and limit retrieval
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

---

## 4. Understanding Query Results

### Response Structure

```json
{
  "query": "What is double-loop learning?",
  "expanded_query": "The user is seeking information about double-loop learning, a concept in organizational learning theory...",
  "threshold": 0.45,
  "count": 5,
  "results": [
    {
      "group": "edleadership",
      "source_path": "/mnt/media/ragdoll/data/edleadership/sources/Visualizing Double-loop Learning.pdf",
      "source_name": "Visualizing Double-loop Learning.pdf",
      "source_type": ".pdf",
      "chunk_index": 12,
      "text": "Double-loop learning involves questioning underlying assumptions...",
      "artifact_type": "text",
      "artifact_path": null,
      "page": 3,
      "chunk_role": "description",
      "similarity": 0.8234
    },
    {
      "group": "edleadership",
      "source_path": "/mnt/media/ragdoll/data/edleadership/sources/SEQuity Teacher Reference.pdf",
      "source_name": "SEQuity Teacher Reference.pdf",
      "source_type": ".pdf",
      "chunk_index": 5,
      "text": "In contrast to single-loop learning, double-loop learning...",
      "artifact_type": "text",
      "page": 1,
      "similarity": 0.7123
    }
  ]
}
```

### Field Descriptions

- **`query`**: Your original prompt
- **`expanded_query`**: LLM-expanded standalone description (used for embedding)
- **`threshold`**: Similarity threshold that was applied
- **`count`**: Number of results returned
- **`results`**: Array of matching chunks, sorted by similarity (highest first)

### Result Object Fields

Each result in the `results` array contains:

- **`group`**: Collection name the chunk belongs to
- **`source_path`**: Full path to the source document (filesystem path)
- **`source_name`**: Filename of the source document
- **`source_url`**: HTTP URL to fetch the source document (e.g., `/fetch/edleadership/document.pdf`)
- **`source_type`**: File extension (e.g., `.pdf`, `.docx`)
- **`chunk_index`**: Index of this chunk within the source document
- **`text`**: The actual text content (cleaned, no newlines)
- **`artifact_type`**: Type of content:
  - `"text"`: Regular prose text
  - `"chart_summary"`: LLM summary of a chart/graph
  - `"table_summary"`: LLM summary of a table
  - `"figure_summary"`: LLM summary of a figure/diagram
- **`artifact_path`**: Path to stored artifact (image/JSON) if applicable, `null` for text
- **`page`**: Page number (for PDFs), `null` for non-paginated documents
- **`chunk_role`**: Role assigned during ingest (e.g. `description`, `application`, `implication`), or `null` if none
- **`similarity`**: Cosine similarity score (0.0-1.0), higher = more relevant

When `limit_chunk_role` was true and role filtering was applied, the response also includes:
- **`limit_chunk_role`**: `true`
- **`inferred_roles`**: Array of the one or two roles inferred from the user input (e.g. `["description", "application"]`)

If no chunks matched the inferred roles (e.g. most chunks have no role set), the server falls back to unfiltered retrieval and adds **`role_filter_relaxed`**: `true` and **`inferred_roles`** so you still get results and know the filter was relaxed.

---

## 5. Integration Patterns for Chatbots

### Pattern 1: Simple Question-Answer

```python
import requests

def query_ragdoll(prompt: str, collection: str = None, threshold: float = 0.45):
    """Query RAGDoll and return top results."""
    url = "http://localhost:9042/query"
    payload = {
        "prompt": prompt,
        "threshold": threshold
    }
    if collection:
        payload["group"] = collection
    
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
        "threshold": 0.45
    }
    if collection:
        payload["group"] = collection
    
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
            payload["group"] = collection
        
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
        payload["group"] = collection
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
        threshold: float = 0.45,
        history: Optional[str] = None
    ) -> Dict:
        """Query RAGDoll."""
        payload = {
            "prompt": prompt,
            "threshold": threshold
        }
        if collection:
            payload["group"] = collection
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

1. **Lower the threshold**: Try `0.35` or `0.3`
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

### Endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/rags` | GET | List all collections |
| `/fetch/{group}/{filename}` | GET | Fetch source document |
| `/query` | GET | Query (URL parameters) |
| `/query` | POST | Query (JSON body) |

### Query Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `prompt` | string | Yes | - | Natural language query |
| `history` | string | No | null | Conversation context |
| `threshold` | float | No | 0.45 | Similarity threshold (0.0-1.0) |
| `group` | string | No | null | Specific collection (searches all if absent) |

### Result Fields

| Field | Type | Description |
|-------|------|-------------|
| `text` | string | Chunk text content |
| `similarity` | float | Relevance score (0.0-1.0) |
| `source_name` | string | Source document filename |
| `source_url` | string | URL to fetch source document (`/fetch/{group}/{filename}`) |
| `group` | string | Collection name |
| `page` | int/null | Page number (PDFs) |
| `artifact_type` | string | Content type (text/chart_summary/table_summary/figure_summary) |

---

## Support

For issues or questions:
- Check RAGDoll logs: `/path/to/data/{group}/action.log`
- Check API server logs: `journalctl -u ragdoll-api`
- Verify Ollama is running: `curl http://localhost:11434/api/tags`
