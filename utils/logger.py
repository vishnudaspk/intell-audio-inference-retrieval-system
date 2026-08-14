"""
Centralized logging module providing structured application logging.
"""

import logging
import sys

from config.settings import settings


def setup_logger(name: str = "intell_audio") -> logging.Logger:
    """Configure and return a logger instance."""
    logger = logging.getLogger(name)

    if not logger.handlers:
        logger.setLevel(getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO))

        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


logger = setup_logger()
