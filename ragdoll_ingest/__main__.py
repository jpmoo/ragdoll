"""Entry point for python -m ragdoll_ingest."""

import logging
import sys

from . import config
from .watcher import run_watcher

_level = getattr(logging, config.LOG_LEVEL, logging.INFO)
logging.basicConfig(
    level=_level,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stdout,
)


def main() -> None:
    run_watcher(process_existing=True)


if __name__ == "__main__":
    main()
