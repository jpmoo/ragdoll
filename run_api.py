"""Run the RAGDoll API server."""

import logging
import sys

import uvicorn

from ragdoll_ingest import config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stdout,
)

if __name__ == "__main__":
    from ragdoll_ingest.api import app
    
    port = config.API_PORT
    logging.info("Starting RAGDoll API server on port %d", port)
    uvicorn.run(app, host="0.0.0.0", port=port)
