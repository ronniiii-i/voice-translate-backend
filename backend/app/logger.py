# backend/app/logger.py
"""
Centralized logging setup.
- Console: coloured, INFO level
- File: logs/app.log, rotates at 5MB, keeps 3 backups, DEBUG level

Usage (in any module):
    from app.logger import get_logger
    log = get_logger(__name__)
    log.info("hello")
"""

import logging
import logging.handlers
import os
from pathlib import Path

LOG_DIR  = Path(__file__).parent.parent / "logs"
LOG_FILE = LOG_DIR / "app.log"
LOG_DIR.mkdir(exist_ok=True)

_configured = False

def _setup():
    global _configured
    if _configured:
        return

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    fmt_file    = logging.Formatter(
        "%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    fmt_console = logging.Formatter(
        "%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    # ── Rotating file handler (5 MB × 3 backups ≈ 15 MB max) ────────────────
    fh = logging.handlers.RotatingFileHandler(
        LOG_FILE,
        maxBytes=5 * 1024 * 1024,
        backupCount=3,
        encoding="utf-8",
    )
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt_file)

    # ── Console handler ───────────────────────────────────────────────────────
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt_console)

    root.addHandler(fh)
    root.addHandler(ch)

    # Quieten noisy third-party loggers
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("faster_whisper").setLevel(logging.WARNING)

    _configured = True
    logging.getLogger(__name__).info(f"Logging to {LOG_FILE}")


def get_logger(name: str) -> logging.Logger:
    _setup()
    return logging.getLogger(name)