"""
Logging configuration for MyAgent with structured logging support.
"""

import sys
from pathlib import Path
from loguru import logger

from .settings import settings


def setup_logging(level: str = "INFO", enable_file_logging: bool = True):
    """
    Configure structured logging for the application.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        enable_file_logging: Whether to enable file logging
    """

    # Remove default logger
    logger.remove()

    # Console logging with colors
    logger.add(
        sys.stdout,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level=level,
        colorize=True,
        backtrace=True,
        diagnose=True
    )

    if enable_file_logging:
        # Ensure logs directory exists
        settings.LOGS_DIR.mkdir(parents=True, exist_ok=True)

        # Main application log
        logger.add(
            settings.LOGS_DIR / "myagent.log",
            rotation="1 day",
            retention="30 days",
            level=level,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
            backtrace=True,
            diagnose=True,
            enqueue=True  # Thread-safe
        )

        # Error log (only errors and above)
        logger.add(
            settings.LOGS_DIR / "errors.log",
            rotation="1 week",
            retention="60 days",
            level="ERROR",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
            backtrace=True,
            diagnose=True,
            enqueue=True
        )

        # Orchestrator-specific log
        logger.add(
            settings.LOGS_DIR / "orchestrator.log",
            rotation="1 day",
            retention="30 days",
            level="DEBUG",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}",
            filter=lambda record: "orchestrator" in record["name"].lower(),
            enqueue=True
        )

        # Agent logs directory
        agent_logs_dir = settings.LOGS_DIR / "agents"
        agent_logs_dir.mkdir(parents=True, exist_ok=True)

        # Agent-specific logs
        for agent_name in ["coder", "tester", "debugger", "architect", "analyzer", "ui_refiner"]:
            logger.add(
                agent_logs_dir / f"{agent_name}_agent.log",
                rotation="1 day",
                retention="30 days",
                level="DEBUG",
                format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}",
                filter=lambda record, name=agent_name: name in record["name"].lower(),
                enqueue=True
            )

        # API request log
        logger.add(
            settings.LOGS_DIR / "api.log",
            rotation="1 day",
            retention="30 days",
            level="INFO",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}",
            filter=lambda record: "api" in record["name"].lower(),
            enqueue=True
        )

    logger.info(f"Logging configured: level={level}, file_logging={enable_file_logging}")


def get_logger(name: str):
    """
    Get a logger instance for a specific module.

    Args:
        name: Name of the module/component

    Returns:
        Logger instance
    """
    return logger.bind(name=name)


# Configure logging on module import
if not settings.DEBUG:
    setup_logging(level="INFO")
else:
    setup_logging(level="DEBUG")
