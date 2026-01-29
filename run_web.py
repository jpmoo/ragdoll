"""Run the RAGDoll Review web server (port 9043)."""

import logging
import sys

import uvicorn

from ragdoll_ingest.config import get_env

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stdout,
)

if __name__ == "__main__":
    from web.app import app

    port = int(get_env("RAGDOLL_REVIEW_PORT") or "9043")
    logging.info("Starting RAGDoll Review on port %d", port)
    uvicorn.run(app, host="0.0.0.0", port=port)
