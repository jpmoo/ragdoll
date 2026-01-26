# RAGDoll Ingest Service

A Linux background service that watches an ingest folder, extracts text from incoming files (txt, md, Word, Excel, PDF, images with OCR), splits into **semantic chunks** using Ollama `llama3.2:3b`, embeds them with `nomic-embed-text:latest`, and appends to per-group **RAG samples** JSONL (for use in other tools). Subfolders in the ingest folder become separate output groups, each with its own samples, DB, and sources.

## Requirements

- **Linux** (inotify via `watchdog`)
- **Python 3.10+**
- **Ollama** with:
  - `nomic-embed-text:latest` (embeddings)
  - `llama3.2:3b` (semantic chunking of long paragraphs)
- **Tesseract OCR** (for images): `sudo apt install tesseract-ocr` (Debian/Ubuntu) or equivalent

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
| `RAGDOLL_TARGET_CHUNK_TOKENS` | `400` | Target size per chunk |
| `RAGDOLL_MAX_CHUNK_TOKENS` | `600` | Max before LLM-assisted split |
| `RAGDOLL_CHUNK_LLM_TIMEOUT` | `300` | Seconds to wait for Ollama when splitting long paragraphs (fallback: mid-split) |

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

- **Text**: `.txt`, `.md`, `.markdown`
- **Word**: `.docx`
- **Excel**: `.xlsx`, `.xls`
- **PDF**: `.pdf`
- **Images (OCR)**: `.png`, `.jpg`, `.jpeg`, `.tiff`, `.tif`, `.bmp`, `.gif`

## Subfolders = separate groups

Only **one level of grouping** is used: each **direct subfolder** of the ingest folder is a **separate group** with its own RAG outputs. Any deeper nesting under those subfolders is **flattened** into the group’s `sources/` (no nested dirs there):

- Files **directly in** the ingest folder → group `_root` → `{DATA_DIR}/_root/`
- Files in `ingest/reports/` → group `reports` → `{DATA_DIR}/reports/`; e.g. `ingest/reports/a.pdf` → `reports/sources/a.pdf`
- Files in `ingest/legal/2024/doc.pdf` → group `legal`; the path under `legal/` is flattened to a single filename → `legal/sources/2024_doc.pdf`

Each group gets its own:

- **ragdoll.db** — SQLite chunks for that group
- **rag_samples.jsonl** — RAG samples for that group
- **processed.jsonl** — Dedup ledger for that group
- **action.log** — AI calls, moves, extract/chunk/store for that group
- **sources/** — Ingested files moved here. Only one level of grouping: deeper paths (e.g. `legal/2024/q1/doc.pdf`) are flattened to one name (e.g. `2024_q1_doc.pdf`) so `sources/` has no subfolders.

If the output folder previously had a flat layout (`ragdoll.db`, `rag_samples.jsonl`, etc. at the top level), it is **migrated once** on startup into `_root/`.

## Output

All outputs live under the **output folder** (`RAGDOLL_OUTPUT_PATH` or `RAGDOLL_DATA_DIR`) in **per-group subdirs** (see above). Use each group’s JSONL or DB in your own RAG/vector tools.

- **SQLite** (`{group}/ragdoll.db`) — Chunks table (source_path, source_type, chunk_index, text, embedding). Written on every ingest; kept in sync with JSONL by a background sync pass.

- **RAG samples** (`{group}/rag_samples.jsonl`): one JSON object per chunk, e.g.:

  ```json
  {"text": "...", "embedding": [...], "source": "/output/reports/sources/file.pdf", "source_type": ".pdf", "chunk_index": 0}
  ```

  If the file is missing, it is recreated from the DB. A sync pass runs at startup and every `RAGDOLL_SYNC_INTERVAL` seconds **per group**: it deduplicates the DB (keeps one row per `source_path`+`chunk_index`), compares DB and JSONL counts, and rebuilds the JSONL from the DB when they differ or after a dedup.

- **sources/** — Original documents are moved here from the ingest folder after successful processing. The `source` field in the JSONL/DB points to these paths. Only files are moved; ingest subfolders are left in place (even when empty).

- **Processed** (`{group}/processed.jsonl`) — Dedup ledger: one `{path, mtime, size}` per successfully ingested file in that group.

- **Action log** (`{group}/action.log`) — JSONL of AI calls (embed, chunk_llm), file moves (src/to/reason), sync_rebuild, sync_dedup, and other actions for that group. Embedding vectors and long text are not written.

## Updating the app (on your server)

From the install directory (e.g. `/opt/ragdoll`):

```bash
cd /opt/ragdoll
git pull origin main
.venv/bin/pip install -e .
```

If you run as a systemd service, restart it:

```bash
sudo systemctl restart ragdoll-ingest
```

Your `env.ragdoll`, `data/`, and `sources/` are untouched by `git pull`. If `env.ragdoll.example` gains new variables, copy the new lines into your `env.ragdoll` as needed.
