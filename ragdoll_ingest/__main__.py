"""Entry point for python -m ragdoll_ingest."""

import logging
import sys

from .watcher import run_watcher

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stdout,
)


def main() -> None:
    run_watcher(process_existing=True)


if __name__ == "__main__":
    main()
