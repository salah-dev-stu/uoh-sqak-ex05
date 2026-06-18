"""Logging setup: level from ``AIRBENCH_LOG_LEVEL`` (default INFO), single handler.

The base ``airbench`` logger is configured once (idempotent); child loggers
``airbench.<module>`` propagate to it, so there are never duplicate handlers.
"""

from __future__ import annotations

import logging
import os

_BASE = "airbench"
_configured = False


def _configure_base() -> None:
    global _configured
    if _configured:
        return
    level = os.environ.get("AIRBENCH_LOG_LEVEL", "INFO").upper()
    base = logging.getLogger(_BASE)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    base.addHandler(handler)
    base.setLevel(level)
    base.propagate = False
    _configured = True


def get_logger(name: str = _BASE) -> logging.Logger:
    """Return a configured logger; ``name`` becomes ``airbench.<name>`` if not already scoped."""
    _configure_base()
    if name != _BASE and not name.startswith(_BASE + "."):
        name = f"{_BASE}.{name}"
    return logging.getLogger(name)


def reset() -> None:
    """Tear down handlers + the configured flag (used by tests for a clean slate)."""
    global _configured
    base = logging.getLogger(_BASE)
    for handler in list(base.handlers):
        base.removeHandler(handler)
    _configured = False
