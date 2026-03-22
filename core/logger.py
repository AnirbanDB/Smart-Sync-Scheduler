"""
logger.py

Centralised logging setup for the application.

All modules obtain a logger via get_logger(). File-based logging is
configured once using the path from settings. If the log file cannot be
created or written to, the application continues without file logging
(graceful degradation per requirement 24).
"""

import logging
import os
from typing import Optional

_registry: dict = {}


def get_logger(
    name: str = "sync_scheduler",
    log_file: Optional[str] = None,
) -> logging.Logger:
    """
    Return a named logger, creating it on first call.

    Subsequent calls with the same name return the cached logger even if
    log_file is omitted, so callers do not need to re-supply the path.
    """
    if name in _registry:
        return _registry[name]

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    fmt = logging.Formatter(
        "%(asctime)s  [%(levelname)-8s]  %(name)s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if not logger.handlers:
        # Console: warnings and above only (keeps Streamlit output clean)
        ch = logging.StreamHandler()
        ch.setLevel(logging.WARNING)
        ch.setFormatter(fmt)
        logger.addHandler(ch)

        # File handler: full debug-level log
        if log_file:
            try:
                log_dir = os.path.dirname(log_file)
                if log_dir:
                    os.makedirs(log_dir, exist_ok=True)
                fh = logging.FileHandler(log_file, encoding="utf-8")
                fh.setLevel(logging.DEBUG)
                fh.setFormatter(fmt)
                logger.addHandler(fh)
            except Exception:
                # Logging failure must never crash the application
                pass

    _registry[name] = logger
    return logger