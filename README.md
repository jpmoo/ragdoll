# RAGDoll Ingest Service

A Linux background service that watches an ingest folder and ingests complex documents into a RAG-ready form. It follows a **multimodal** design: **prose** is chunked and embedded; **charts** and **tables** are interpreted by an LLM into qualitative summaries (no numeric guessing), with raw images and table data stored separately. Only prose chunks and these summaries are embedded. Subfolders in the ingest folder become separate output groups, each with its own samples, DB, sources, and artifacts.

## Requirements

- **Linux** (inotify via `watchdog`)
- **Python 3.10+**
- **Ollama** with:
  - `nomic-embed-text:latest` (embeddings)
  - `llama3.2:3b` (semantic chunking of long paragraphs; chart/table interpretation)
- **Tesseract OCR** (for images and chart regions): `sudo apt install tesseract-ocr` (Debian/Ubuntu) or equivalent

## Install

```bash
cd /path/to/RAGDoll
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

**Optional: Docling ingester** — For better table extraction and layout understanding (PDF, DOCX, XLSX, PPTX, images), install the Docling extra:

```bash
pip install -e '.[docling]'
# In env.ragdoll: RAGDOLL_ALWAYS_USE_DOCLING=true to use Docling for every supported file; false = use Docling only for types RAGDoll doesn't cover (e.g. PPTX)
```

With `RAGDOLL_ALWAYS_USE_DOCLING=false` (default), Docling is used only for file types RAGDoll has no legacy extractor for (e.g. PPTX). With `true`, Docling is used for all supported types; legacy is used only on Docling exception. Chunking, LLM interpretation of figures/tables, and embedding are unchanged.

## Configuration

### One file: `env.ragdoll`

Copy `env.ragdoll.example` to `env.ragdoll` in the project root, edit it, and the app will load all variables from it at startup (environment variables override the file):

```bash
cp env.ragdoll.example env.ragdoll
# edit env.ragdoll: set RAGDOLL_INGEST_PATH and any optional paths/models
```

For systemd, point the unit at it: `EnvironmentFile=/opt/ragdoll/env.ragdoll` (or keep using `/etc/default/ragdoll-ingest`). To use a different path, set `RAGDOLL_ENV=/path/to/your.env` before starting.

### Or use environment variables

Set the **ingest path** (required):

```bash
export RAGDOLL_INGEST_PATH=/path/to/ingest/folder
```

Optional env vars:

| Variable | Default | Description |
|----------|---------|-------------|
| `RAGDOLL_OUTPUT_PATH` | — | Output folder (takes precedence over `RAGDOLL_DATA_DIR`) |
| `RAGDOLL_DATA_DIR` | `./data` | Output folder for per-group subdirs if `RAGDOLL_OUTPUT_PATH` is unset |
| `RAGDOLL_SYNC_INTERVAL` | `300` | Seconds between DB dedup sync. `0`=disabled |
| `RAGDOLL_OLLAMA_HOST` | `http://localhost:11434` | Ollama API base URL |
| `RAGDOLL_EMBED_MODEL` | `nomic-embed-text:latest` | Embedding model |
| `RAGDOLL_CHUNK_MODEL` | `llama3.2:3b` | Model for semantic splitting of long paragraphs |
| `RAGDOLL_INTERPRET_MODEL` | same as `RAGDOLL_CHUNK_MODEL` | Model for chart and table interpretation (qualitative summaries; anti-hallucination) |
| `RAGDOLL_SEMANTIC_CHUNKING` | `true` | When `true`, combine all document text, strip links/formatting, and ask the LLM to output the text of each semantic chunk (then locate in text for page mapping). When `false`, use paragraph-based splitting and LLM only for long paragraphs. |
| `RAGDOLL_TARGET_CHUNK_TOKENS` | `400` | Target size per chunk |
| `RAGDOLL_MAX_CHUNK_TOKENS` | `600` | Max before LLM-assisted split |
| `RAGDOLL_CHUNK_LLM_TIMEOUT` | `300` | Seconds to wait for Ollama (chunk split, chart/table interpret) |
| `RAGDOLL_ALWAYS_USE_DOCLING` | `false` | `true` = use [Docling](https://docling-project.github.io/docling/) for every supported file (PDF/DOCX/XLSX/PPTX/image); `false` = use Docling only for types RAGDoll doesn't cover (e.g. PPTX). Requires `pip install -e '.[docling]'`. |

## Run manually

```bash
export RAGDOLL_INGEST_PATH=/home/you/ingest
python -m ragdoll_ingest
# or: ragdoll-ingest
```

- On start, existing supported files in the ingest folder are processed.
- New or moved-in files are picked up automatically.
- Processed files are **moved** to the appropriate group’s `sources/` (see **Subfolders = separate groups** below); failures go to `ingest/failed/`.

## Run as a systemd service

1. Copy the service and env example:

   ```bash
   sudo cp ragdoll-ingest.service /etc/systemd/system/
   sudo cp etc-default-ragdoll-ingest.example /etc/default/ragdoll-ingest
   sudo nano /etc/default/ragdoll-ingest   # set RAGDOLL_INGEST_PATH=/your/ingest
   ```

2. If you use a venv or custom install path, override the service:

   ```bash
   sudo systemctl edit ragdoll-ingest
   ```

   Add and adjust:

   ```ini
   [Service]
   Environment="RAGDOLL_INGEST_PATH=/home/you/ingest"
   WorkingDirectory=/opt/ragdoll
   ExecStart=/opt/ragdoll/.venv/bin/python -m ragdoll_ingest
   User=you
   Group=you
   ```

3. Enable and start:

   ```bash
   sudo systemctl daemon-reload
   sudo systemctl enable ragdoll-ingest
   sudo systemctl start ragdoll-ingest
   sudo journalctl -u ragdoll-ingest -f
   ```

## Chunking

With **`RAGDOLL_SEMANTIC_CHUNKING=true`** (default), document text is combined into one string, stripped of links and markdown formatting, and the LLM **outputs the text of each semantic chunk** (no start/end indices). Each chunk is then located in the cleaned string to get a start offset for page mapping. Long documents are processed in windows (~10k chars per LLM call). Set to `false` to use paragraph-based splitting and LLM only for long paragraphs.

## Supported file types

- **Text**: `.txt`, `.md`, `.markdown` — plain extract and chunk.
- **Word**: `.docx` — paragraphs as prose; native tables as table regions; **embedded images** extracted, classified, and routed (chart/table/figure/text).
- **Excel**: `.xlsx`, `.xls` — each sheet as a table region (LLM summary + stored JSON).
- **PDF**: `.pdf` — text blocks; **tables** via pdfplumber; low-text + images → chart regions; low-text + drawings/short blocks → **figure** (OCR + LLM process summary, image + process JSON stored).
- **Images**: `.png`, `.jpg`, etc. — **classified** (text/table/chart/figure) from OCR and routed to the right handler; only summaries or prose are embedded.

## Subfolders = separate groups

Only **one level of grouping** is used: each **direct subfolder** of the ingest folder is a **separate group** with its own RAG outputs. Any deeper nesting under those subfolders is **flattened** into the group’s `sources/` (no nested dirs there):

- Files **directly in** the ingest folder → group `_root` → `{DATA_DIR}/_root/`
- Files in (example) `ingest/reports/` → group `reports` → `{DATA_DIR}/reports/`; e.g. `ingest/reports/a.pdf` → `reports/sources/a.pdf`
- Files in (example) `ingest/legal/2024/doc.pdf` → group `legal`; the path under `legal/` is flattened to a single filename → `legal/sources/2024_doc.pdf`

Each group gets its own:

- **ragdoll.db** — SQLite `chunks`: `source_path`, `source_type`, `chunk_index`, `text`, `embedding`, `artifact_type`, `artifact_path`, `page`.
- **processed.jsonl** — Dedup ledger: one `{path, mtime, size}` per successfully ingested file.
- **action.log** — JSONL of AI calls, moves, extract/chunk/interpret/store, and sync actions (`sync_dedup`) for that group.
- **sources/** — Ingested files moved here. Only one level of grouping: deeper paths are flattened to one filename in `sources/`.
- **artifacts/** — Chart images (`charts/`), table JSON (`tables/`), figure image+process JSON (`figures/`). Only interpretations are embedded; raw data is stored here.

If the output folder previously had a flat layout (`ragdoll.db`, `processed.jsonl`, etc. at the top level), it is **migrated once** on startup into `_root/`.

## Output

All outputs live under the **output folder** (`RAGDOLL_OUTPUT_PATH` or `RAGDOLL_DATA_DIR`) in **per-group subdirs** (see above). Use each group’s DB in your own RAG/vector tools.

- **SQLite** (`{group}/ragdoll.db`) — Chunks (source_path, source_type, chunk_index, text, embedding, artifact_type, artifact_path, page). A sync pass runs at startup and every `RAGDOLL_SYNC_INTERVAL` seconds **per group**: it deduplicates the DB (keeps one row per `source_path`+`chunk_index`).

  `artifact_type`: `text`, `chart_summary`, `table_summary`, or `figure_summary`. `artifact_path` points to `artifacts/charts/`, `artifacts/tables/`, or `artifacts/figures/` when present.

- **sources/** — Original documents are moved here from the ingest folder after successful processing. The `source_path` field in the DB points to these paths. Only files are moved; ingest subfolders are left in place (even when empty).

- **Processed** (`{group}/processed.jsonl`) — Dedup ledger: one `{path, mtime, size}` per successfully ingested file in that group.

- **Action log** (`{group}/action.log`) — JSONL of AI calls (embed, chunk_llm), file moves (src/to/reason), sync_dedup, and other actions for that group. Embedding vectors and long text are not written.

## CLI Tool

RAGDoll includes a CLI tool for managing collections and sources:

```bash
# List all collections
ragdoll collections

# List all sources in a collection (shows IDs)
ragdoll list <collection>

# Delete all chunks for a source by ID (with confirmation)
ragdoll delete <collection> <source_id>

# Delete without confirmation prompt
ragdoll delete <collection> <source_id> --yes
```

**Examples:**
```bash
# List all collections
ragdoll collections
# Output:
# Found 3 collection(s):
#   - _root
#   - edleadership
#   - reports

# List sources in a collection (with IDs)
ragdoll list edleadership
# Output:
# Found 5 source(s) in collection 'edleadership':
# ID     Source Path                                          Chunks    
# --------------------------------------------------------------------------------
# 1      sources/document1.pdf                                42        
# 2      sources/document2.docx                               18        
# ...
# --------------------------------------------------------------------------------
# Total: 156 chunks across 5 source(s)

# Delete chunks for a source by ID
ragdoll delete edleadership 1
# Prompts: "Are you sure? (yes/no):"
# Warning: This will delete 42 chunks from source ID 1:
#   Path: sources/document1.pdf
#   Type: .pdf
```

After installing with `pip install -e .`, the `ragdoll` command is available in your PATH.

## Updating the app (on your server)

From the install directory (e.g. `/opt/ragdoll`):

```bash
cd /opt/ragdoll
git pull origin main
.venv/bin/pip install -e .
sudo systemctl restart ragdoll-ingest ragdoll-api
```

This updates the code, installs any new dependencies, and restarts both services.

Your `env.ragdoll`, `data/`, and `sources/` are untouched by `git pull`. If `env.ragdoll.example` gains new variables, copy the new lines into your `env.ragdoll` as needed.

## API Server

The HTTP API server runs separately from the ingest watcher. To install it as a systemd service:

```bash
sudo cp ragdoll-api.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now ragdoll-api
```

**Note:** The service file assumes the venv is at `/opt/ragdoll/.venv/bin/python`. If your setup differs, edit the service file or use `systemctl edit ragdoll-api` to override `ExecStart`.

The API server listens on port `9042` by default (configurable via `RAGDOLL_API_PORT`).

**All three services (ingest, API, review web)** can be installed together so they start on boot and listen on their ports:

```bash
sudo cp ragdoll-ingest.service ragdoll-api.service ragdoll-web.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now ragdoll-ingest ragdoll-api ragdoll-web
```

### Endpoints

**`GET /rags`** — List all RAG collections (groups)
```bash
curl http://localhost:9042/rags
# Returns: {"collections": ["_root", "reports", "legal", ...]}
```

**`GET /query`** — Semantic similarity search (simple URL format)
```bash
# Query all collections
curl "http://localhost:9042/query?prompt=What%20is%20double-loop%20learning&threshold=0.45"

# Query specific collection
curl "http://localhost:9042/query?prompt=What%20is%20double-loop%20learning&group=edleadership&threshold=0.45"
```

Query parameters:
- `prompt` (required): User's query/question
- `history` (optional): Conversation history for context
- `threshold` (optional, default: 0.45): Minimum similarity score (0.0-1.0)
- `group` (optional): Specific collection/group to query; if absent, searches all collections
- `limit_chunk_role` (optional, default: false): If true, infer up to 2 chunk roles from prompt+context via LLM and limit retrieval to those roles

**`POST /query`** — Semantic similarity search (JSON body)
```bash
# Query all collections
curl -X POST http://localhost:9042/query \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "What is double-loop learning?",
    "history": "Previous conversation...",
    "threshold": 0.45
  }'

# Query specific collection
curl -X POST http://localhost:9042/query \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "What is double-loop learning?",
    "group": "edleadership",
    "threshold": 0.45
  }'
```

Request body:
- `prompt` (required): User's query/question
- `history` (optional): Conversation history for context
- `threshold` (optional, default: 0.45): Minimum similarity score (0.0-1.0)
- `group` (optional): Specific collection/group to query; if absent, searches all collections
- `limit_chunk_role` (optional, default: false): If true, infer up to 2 chunk roles from prompt+context via LLM and limit retrieval to those roles

Response includes:
- `query`: Original prompt
- `expanded_query`: LLM-expanded standalone description
- `threshold`: Used threshold
- `count`: Number of results
- `results`: Array of matching chunks (sorted by similarity, highest first), each with: `group`, `source_path`, `source_name`, `source_type`, `chunk_index`, `text`, `artifact_type`, `artifact_path`, `page`, `chunk_role`, `similarity`
- When role filtering was applied: `limit_chunk_role`: true, `inferred_roles`: list of inferred roles

## Review web app

A separate web service in `web/` lets you review **samples (chunks) side-by-side with their source** and edit them.

- **Port:** `9043` (configurable via `RAGDOLL_REVIEW_PORT`), bound to `0.0.0.0` so it’s reachable on the network.
- **Run manually:** From the project root:
  ```bash
  python run_web.py
  # or: python -m uvicorn web.app:app --host 0.0.0.0 --port 9043
  ```
- **Run as a service (starts with ingest/API and on reboot):**
  ```bash
  sudo cp ragdoll-web.service /etc/systemd/system/
  sudo systemctl daemon-reload
  sudo systemctl enable --now ragdoll-web
  ```
  The unit uses the same env files as the API (`/opt/ragdoll/env.ragdoll`, `/etc/default/ragdoll-ingest`) and the same venv at `/opt/ragdoll/.venv/bin/python`. Override with `systemctl edit ragdoll-web` if needed.
- **URL:** Open `http://localhost:9043` (or your host:9043) in a browser.

**Features:**
- Choose a **group** and **source**; the left panel shows the source document (PDF/image inline; other types open in a new tab). PDFs are **not** converted to scans—the file on disk is the original. PDFs open in the **browser’s native viewer** (iframe) so copy/paste works; use the “Page” filter and type the page number to see samples for that page.
- The right panel lists **samples (chunks)** for that source. You can:
  - **View all samples** or filter by **page** (samples for the current source page).
  - **Edit** a sample (inline; saves and re-embeds).
  - **Insert above** / **Insert below** (new sample at that index; re-embeds).
  - **Delete** a sample.

The review app uses the same RAG DB and `sources/` as the ingest and API; edits and inserts update the SQLite chunks and re-run the embedder so search stays in sync.

**Authentication (optional):**
- **App-level HTTP Basic Auth:** Set both `RAGDOLL_REVIEW_USER` and `RAGDOLL_REVIEW_PASSWORD` in `env.ragdoll` (or the environment). The app will require HTTP Basic Auth for all routes (UI and API). Leave either unset for no auth.
- **Server-inherited auth:** Run the app behind a reverse proxy (e.g. nginx or Apache) and add Basic Auth (or other auth) at the proxy. The app then receives requests only from the proxy; no env vars needed in the app.
  - Example (nginx): `auth_basic "RAGDoll Review"; auth_basic_user_file /etc/nginx/.htpasswd;` in the `location` that proxies to `http://127.0.0.1:9043`.

### Troubleshooting API access

If the API server isn't accessible from other machines on your network:

1. **Verify dependencies are installed:**
   ```bash
   /opt/ragdoll/.venv/bin/python -c "import uvicorn, fastapi; print('OK')"
   ```
   If this fails, run: `/opt/ragdoll/.venv/bin/pip install -e .`

2. **Verify service file uses venv Python:**
   ```bash
   sudo systemctl cat ragdoll-api | grep ExecStart
   ```
   Should show: `ExecStart=/opt/ragdoll/.venv/bin/python run_api.py`
   If not, update it:
   ```bash
   cd /opt/ragdoll
   sudo cp ragdoll-api.service /etc/systemd/system/
   sudo systemctl daemon-reload
   ```

3. **Check service status:**
   ```bash
   sudo systemctl status ragdoll-api
   ```

4. **Check if port is listening:**
   ```bash
   sudo netstat -tlnp | grep 9042
   # or
   sudo ss -tlnp | grep 9042
   ```
   Should show `0.0.0.0:9042` or `:::9042`.

5. **Check firewall (ufw on Ubuntu/Debian):**
   ```bash
   sudo ufw status
   sudo ufw allow 9042/tcp
   ```

6. **Check firewall (firewalld on RHEL/CentOS):**
   ```bash
   sudo firewall-cmd --list-ports
   sudo firewall-cmd --permanent --add-port=9042/tcp
   sudo firewall-cmd --reload
   ```

7. **Test locally first:**
   ```bash
   curl http://localhost:9042/rags
   ```

8. **Check service logs:**
   ```bash
   sudo journalctl -u ragdoll-api -f
   ```

9. **Verify the service is using the correct host:**
   The API server should bind to `0.0.0.0` (all interfaces). Check the service file uses `run_api.py` which sets `host="0.0.0.0"`.
