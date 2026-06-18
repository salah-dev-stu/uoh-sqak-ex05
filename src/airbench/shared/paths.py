"""Path guards. Model weights MUST live on the external SSD (under /Volumes) — never
on the nearly-full internal disk, and never committed (H10)."""

from __future__ import annotations

from pathlib import Path

EXTERNAL_PREFIX = "/Volumes/"


def assert_external(path: str | Path) -> Path:
    """Raise if ``path`` is not on an external volume; return it as a Path otherwise."""
    p = Path(path)
    if not str(p).startswith(EXTERNAL_PREFIX):
        raise ValueError(
            f"weights path must be on an external volume under {EXTERNAL_PREFIX}, got: {p}"
        )
    return p
