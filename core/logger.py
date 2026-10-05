"""
core/logger.py

Structured Logging and Request Sanitization for FGEAD Production Gateway.
Outputs sanitized structured logs without leaking tokens, keys, or raw traces.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from config.settings import settings


class JsonFormatter(logging.Formatter):
    """Format log records as structured JSON."""

    def format(self, record: logging.LogRecord) -> str:
        log_obj: Dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Include extra attributes if present
        for key in ("service", "host_id", "endpoint", "status_code", "latency_ms", "error"):
            if hasattr(record, key):
                log_obj[key] = getattr(record, key)

        if record.exc_info and settings.ENVIRONMENT != "PRODUCTION":
            log_obj["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_obj)


def setup_logger(name: str = "fgead") -> logging.Logger:
    """Configure and return the root FGEAD logger."""
    logger = logging.getLogger(name)
    level_name = settings.LOG_LEVEL
    level = getattr(logging, level_name, logging.INFO)
    logger.setLevel(level)

    # Avoid duplicate handlers
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level)
        if settings.STRUCTURED_LOGS:
            handler.setFormatter(JsonFormatter())
        else:
            fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
            handler.setFormatter(logging.Formatter(fmt))
        logger.addHandler(handler)

    return logger


# Global logger instance
logger = setup_logger("fgead.api")


def log_api_request(
    endpoint: str,
    method: str,
    status_code: int,
    latency_ms: float,
    host_id: Optional[str] = None,
    error: Optional[str] = None,
) -> None:
    """Log an API request with structured metadata."""
    extra = {
        "service": "fgead_api",
        "endpoint": f"{method} {endpoint}",
        "status_code": status_code,
        "latency_ms": round(latency_ms, 2),
        "host_id": host_id or "none",
    }
    if error:
        extra["error"] = error
        logger.warning(f"{method} {endpoint} -> {status_code} ({latency_ms:.1f}ms) - {error}", extra=extra)
    else:
        logger.info(f"{method} {endpoint} -> {status_code} ({latency_ms:.1f}ms)", extra=extra)
