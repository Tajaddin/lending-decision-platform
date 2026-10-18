"""Structured JSON logs (one object per line) for every process: Spark driver, API, sink, monitor."""

from __future__ import annotations

import logging
import sys

import structlog


def configure(level: str = "INFO", json: bool = True) -> None:
    logging.basicConfig(stream=sys.stdout, level=level, format="%(message)s")
    # Spark and py4j are chatty at INFO; their warnings still come through.
    for noisy in ("py4j", "urllib3", "google", "httpx"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    renderer = structlog.processors.JSONRenderer() if json else structlog.dev.ConsoleRenderer()
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            renderer,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(logging.getLevelName(level)),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    return structlog.get_logger(name)
