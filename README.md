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
| `RAGDOLL_SYNC_INTERVAL` | `300` | Seconds between DB↔JSONL sync (dedup, rebuild if needed). `0`=disabled |
| `RAGDOLL_OLLAMA_HOST` | `http://localhost:11434` | Ollama API base URL |
| `RAGDOLL_EMBED_MODEL` | `nomic-embed-text:latest` | Embedding model |
| `RAGDOLL_CHUNK_MODEL` | `llama3.2:3b` | Model for semantic splitting of long paragraphs |
| `RAGDOLL_INTERPRET_MODEL` | same as `RAGDOLL_CHUNK_MODEL` | Model for chart and table interpretation (qualitative summaries; anti-hallucination) |
| `RAGDOLL_TARGET_CHUNK_TOKENS` | `400` | Target size per chunk |
| `RAGDOLL_MAX_CHUNK_TOKENS` | `600` | Max before LLM-assisted split |
| `RAGDOLL_CHUNK_LLM_TIMEOUT` | `300` | Seconds to wait for Ollama (chunk split, chart/table interpret) |

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
- **rag_samples.jsonl** — One JSON per chunk: `text`, `embedding`, `source`, `source_type`, `chunk_index`, `artifact_type` (`text`|`chart_summary`|`table_summary`|`figure_summary`), `artifact_path`, `page`.
- **processed.jsonl** — Dedup ledger: one `{path, mtime, size}` per successfully ingested file.
- **action.log** — JSONL of AI calls, moves, extract/chunk/interpret/store, and sync actions (`sync_rebuild`, `sync_dedup`) for that group.
- **sources/** — Ingested files moved here. Only one level of grouping: deeper paths are flattened to one filename in `sources/`.
- **artifacts/** — Chart images (`charts/`), table JSON (`tables/`), figure image+process JSON (`figures/`). Only interpretations are embedded; raw data is stored here.

If the output folder previously had a flat layout (`ragdoll.db`, `rag_samples.jsonl`, etc. at the top level), it is **migrated once** on startup into `_root/`.

## Output

All outputs live under the **output folder** (`RAGDOLL_OUTPUT_PATH` or `RAGDOLL_DATA_DIR`) in **per-group subdirs** (see above). Use each group’s JSONL or DB in your own RAG/vector tools.

- **SQLite** (`{group}/ragdoll.db`) — Chunks (source_path, source_type, chunk_index, text, embedding, artifact_type, artifact_path, page). Kept in sync with JSONL by a background sync pass.

- **RAG samples** (`{group}/rag_samples.jsonl`): one JSON per chunk, e.g.:

  ```json
  {"text": "...", "embedding": [...], "source": "...", "source_type": ".pdf", "chunk_index": 0, "artifact_type": "text", "artifact_path": null, "page": 1}
  ```

  `artifact_type`: `text`, `chart_summary`, `table_summary`, or `figure_summary`. `artifact_path` points to `artifacts/charts/`, `artifacts/tables/`, or `artifacts/figures/` when present.

  If the file is missing, it is recreated from the DB. A sync pass runs at startup and every `RAGDOLL_SYNC_INTERVAL` seconds **per group**: it deduplicates the DB (keeps one row per `source_path`+`chunk_index`), compares DB and JSONL counts, and rebuilds the JSONL from the DB when they differ or after a dedup.

- **sources/** — Original documents are moved here from the ingest folder after successful processing. The `source` field in the JSONL/DB points to these paths. Only files are moved; ingest subfolders are left in place (even when empty).

- **Processed** (`{group}/processed.jsonl`) — Dedup ledger: one `{path, mtime, size}` per successfully ingested file in that group.

- **Action log** (`{group}/action.log`) — JSONL of AI calls (embed, chunk_llm), file moves (src/to/reason), sync_rebuild, sync_dedup, and other actions for that group. Embedding vectors and long text are not written.

## Updating the app (on your server)

From the install directory (e.g. `/opt/ragdoll`):

```bash
cd /opt/ragdoll
git pull origin main
.venv/bin/pip install -e .  # Install/update dependencies (required after adding FastAPI/uvicorn)
```

If you run as systemd services, restart both:

```bash
sudo systemctl restart ragdoll-ingest ragdoll-api
```

**Important:** If you see `ModuleNotFoundError: No module named 'uvicorn'` in the API service logs, the dependencies weren't installed. Run `.venv/bin/pip install -e .` to install FastAPI and uvicorn.

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

### Endpoints

**`GET /rags`** — List all RAG collections (groups)
```bash
curl http://localhost:9042/rags
# Returns: {"collections": ["_root", "reports", "legal", ...]}
```

**`POST /query`** — Semantic similarity search
```bash
curl -X POST http://localhost:9042/query \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "What is double-loop learning?",
    "history": "Previous conversation...",
    "threshold": 0.60
  }'
```

Request body:
- `prompt` (required): User's query/question
- `history` (optional): Conversation history for context
- `threshold` (optional, default: 0.60): Minimum similarity score (0.0-1.0)

Response includes:
- `query`: Original prompt
- `expanded_query`: LLM-expanded standalone description
- `threshold`: Used threshold
- `count`: Number of results
- `results`: Array of matching chunks (sorted by similarity, highest first), each with: `group`, `source_path`, `source_name`, `source_type`, `chunk_index`, `text`, `artifact_type`, `artifact_path`, `page`, `similarity`

### Troubleshooting API access

If the API server isn't accessible from other machines on your network:

1. **Check service status:**
   ```bash
   sudo systemctl status ragdoll-api
   ```

2. **Check if port is listening:**
   ```bash
   sudo netstat -tlnp | grep 9042
   # or
   sudo ss -tlnp | grep 9042
   ```
   Should show `0.0.0.0:9042` or `:::9042`.

3. **Check firewall (ufw on Ubuntu/Debian):**
   ```bash
   sudo ufw status
   sudo ufw allow 9042/tcp
   ```

4. **Check firewall (firewalld on RHEL/CentOS):**
   ```bash
   sudo firewall-cmd --list-ports
   sudo firewall-cmd --permanent --add-port=9042/tcp
   sudo firewall-cmd --reload
   ```

5. **Test locally first:**
   ```bash
   curl http://localhost:9042/rags
   ```

6. **Check service logs:**
   ```bash
   sudo journalctl -u ragdoll-api -f
   ```

7. **Verify the service is using the correct host:**
   The API server should bind to `0.0.0.0` (all interfaces). Check the service file uses `run_api.py` which sets `host="0.0.0.0"`.

8. **Missing dependencies (ModuleNotFoundError: No module named 'uvicorn'):**
   After pulling updates, install new dependencies:
   ```bash
   cd /opt/ragdoll
   .venv/bin/pip install -e .
   ```
   This installs FastAPI and uvicorn from `requirements.txt`.
