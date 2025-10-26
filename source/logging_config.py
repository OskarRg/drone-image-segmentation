"""
Configures the root logger for the application.

This module provides a setup function to initialize
basic logging configuration, directing logs to stdout
with a standardized format.
"""

import logging
import sys


def setup_logging(level: int = logging.INFO) -> None:
    """
    Sets up the basic configuration for the root logger.

    Logs will be sent to stdout with a format that includes
    timestamp, log level, module name, and the message.

    :param level: The logging level to set (e.g., logging.INFO, logging.DEBUG).
    :type level: int
    """
    logging.basicConfig(
        level=level,
        # Format includes the module name [%(name)s]
        format="%(asctime)s - %(levelname)s - [%(name)s] - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
