"""
Logging configuration for the Agriculture RAG system.

This module sets up structured logging with both file and console handlers.
"""

import logging
import logging.handlers
from pathlib import Path

from utils.config import settings


def setup_logger(name: str) -> logging.Logger:
    """
    Set up a logger with file and console handlers.

    Args:
        name: Logger name (typically __name__)

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)

    # Only configure if handlers haven't been set up yet
    if logger.hasHandlers():
        return logger

    logger.setLevel(getattr(logging, settings.log_level))

    # Create logs directory if it doesn't exist
    log_file_path = Path(settings.log_file)
    log_file_path.parent.mkdir(parents=True, exist_ok=True)

    # File handler - for persistent logging
    file_handler = logging.handlers.RotatingFileHandler(
        settings.log_file,
        maxBytes=10485760,  # 10MB
        backupCount=5,
    )
    file_handler.setLevel(getattr(logging, settings.log_level))

    # Console handler - for interactive feedback
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, settings.log_level))

    # Formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


# Create module-level logger
logger = setup_logger(__name__)
