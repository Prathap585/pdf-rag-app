"""
utils/logger.py

Centralized logging configuration for the PDF RAG application.
Every module imports `get_logger(__name__)` from here.
Never use print() for application events — always use the logger.

Usage in any module:
    from utils.logger import get_logger
    logger = get_logger(__name__)
    logger.info("Something happened")
    logger.warning("Something looks wrong")
    logger.error("Something broke")
"""

import logging
import sys
from pathlib import Path


# ─────────────────────────────────────────
# LOG FORMAT
# ─────────────────────────────────────────

LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# ─────────────────────────────────────────
# LOG LEVEL
# Change to logging.DEBUG locally for verbose output
# In production this would come from config/settings.py
# ─────────────────────────────────────────

LOG_LEVEL = logging.INFO

# ─────────────────────────────────────────
# LOG FILE (optional — logs to both console and file)
# ─────────────────────────────────────────

LOG_DIR = Path("logs")
LOG_FILE = LOG_DIR / "rag_app.log"


def _setup_logger() -> logging.Logger:
    """
    Internal function — creates and configures the root application logger.
    Called once when this module is first imported.
    Returns the root 'rag_app' logger.
    """

    # Create logs directory if it doesn't exist
    LOG_DIR.mkdir(exist_ok=True)

    # Get (or create) the root logger for this app
    # All child loggers (e.g., 'rag_app.core.document_processor') inherit this config
    logger = logging.getLogger("rag_app")
    logger.setLevel(LOG_LEVEL)

    # Prevent duplicate log entries if this is called multiple times
    if logger.handlers:
        return logger

    # ── Console Handler ──────────────────────────────────────
    # Logs INFO and above to stdout (visible in terminal & Streamlit logs)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT))

    # ── File Handler ─────────────────────────────────────────
    # Logs everything (DEBUG and above) to a persistent log file
    # 'a' mode = append, so logs survive app restarts
    file_handler = logging.FileHandler(LOG_FILE, mode="a", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT))

    # Register both handlers
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger


# Initialize the root logger once at module import time
_root_logger = _setup_logger()


def get_logger(module_name: str) -> logging.Logger:
    """
    Public function — call this in every module to get a named logger.

    Usage:
        from utils.logger import get_logger
        logger = get_logger(__name__)

    Args:
        module_name: Pass __name__ so log entries show which module they came from.
                     e.g., 'rag_app.core.document_processor'

    Returns:
        A child logger that inherits all handlers from the root 'rag_app' logger.
    """
    # Strip the path prefix if module_name is something like 'core.document_processor'
    # We namespace it under 'rag_app' so all logs are grouped together
    return logging.getLogger(f"rag_app.{module_name}")