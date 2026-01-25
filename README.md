# RAGDoll Ingest Service

A Linux background service that watches an ingest folder, extracts text from incoming files (txt, md, Word, Excel, PDF, images with OCR), splits into **semantic chunks** using Ollama `llama3.2:3b`, embeds them with `nomic-embed-text:latest`, and appends to a **master RAG samples** JSONL (for use in other tools).

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
| `RAGDOLL_OUTPUT_PATH` | — | Output folder for RAG JSONL and `sources/` (takes precedence over `RAGDOLL_DATA_DIR`) |
| `RAGDOLL_DATA_DIR` | `./data` | Output folder for JSONL and `sources/` if `RAGDOLL_OUTPUT_PATH` is unset |
| `RAGDOLL_SAMPLES` | `{DATA_DIR}/rag_samples.jsonl` | RAG samples (one JSON object per line) |
| `RAGDOLL_PROCESSED` | `{DATA_DIR}/processed.jsonl` | Dedup ledger: one `{path,mtime,size}` per ingested file |
| `RAGDOLL_OLLAMA_HOST` | `http://localhost:11434` | Ollama API base URL |
| `RAGDOLL_EMBED_MODEL` | `nomic-embed-text:latest` | Embedding model |
| `RAGDOLL_CHUNK_MODEL` | `llama3.2:3b` | Model for semantic splitting of long paragraphs |
| `RAGDOLL_TARGET_CHUNK_TOKENS` | `400` | Target size per chunk |
| `RAGDOLL_MAX_CHUNK_TOKENS` | `600` | Max before LLM-assisted split |
| `RAGDOLL_ACTION_LOG` | `{DATA_DIR}/action.log` | JSONL log of AI calls, file moves, extract/chunk/store (no embeddings) |

## Run manually

```bash
export RAGDOLL_INGEST_PATH=/home/you/ingest
python -m ragdoll_ingest
# or: ragdoll-ingest
```

- On start, existing supported files in the ingest folder are processed.
- New or moved-in files are picked up automatically.
- Processed files are **moved** to `{OUTPUT}/sources/` (inside the RAG output folder); failures go to `ingest/failed/`.

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

## Output

All outputs live in the **output folder** (`RAGDOLL_OUTPUT_PATH` or `RAGDOLL_DATA_DIR`). Use the JSONL in your own RAG/vector tools.

- **RAG samples** (`rag_samples.jsonl`): one JSON object per chunk, e.g.:

  ```json
  {"text": "...", "embedding": [...], "source": "/output/path/sources/file.pdf", "source_type": ".pdf", "chunk_index": 0}
  ```

- **sources/** — Original documents are moved here from the ingest folder after successful processing. The `source` field in the JSONL points to these paths.

- **Processed** (`processed.jsonl`) — Dedup ledger: one `{path, mtime, size}` per successfully ingested file.

- **Action log** (`action.log`, or `RAGDOLL_ACTION_LOG`) — JSONL of AI calls (embed, chunk_llm), file moves (src/to/reason), and other actions. Embedding vectors and long text are not written.
