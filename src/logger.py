"""Structured logging via structlog."""

import structlog


def setup_logging() -> None:
    structlog.configure(
        processors=[
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.dev.ConsoleRenderer(),
        ],
    )


def get_logger(name: str) -> structlog.BoundLogger:
    return structlog.get_logger(name)
