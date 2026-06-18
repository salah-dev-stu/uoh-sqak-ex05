"""Memory sampling: process RSS (psutil) + macOS unified-memory pressure (vm_stat).

On Apple Silicon CPU and GPU share one pool, so RSS + vm_stat free/active pages
together describe the memory picture. vm_stat is run through the Gatekeeper.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Apple Silicon vm_stat reports in 16 KiB pages.
PAGE_BYTES = 16384


@dataclass
class MemSample:
    rss_mb: float
    free_mb: float = 0.0
    active_mb: float = 0.0
    wired_mb: float = 0.0


def peak_rss_mb(proc=None) -> float:
    """Resident set size in MB. ``proc`` may be injected (a psutil.Process-like) for tests."""
    if proc is None:
        import psutil

        proc = psutil.Process()
    return proc.memory_info().rss / 1024 / 1024


def parse_vm_stat(text: str) -> dict[str, float]:
    """Parse ``vm_stat`` output into MB by category (free/active/wired/...)."""
    out: dict[str, float] = {}
    for line in text.splitlines():
        m = re.match(r'"?Pages ([^:"]+)"?:\s*([0-9]+)', line.strip())
        if m:
            key = m.group(1).strip().lower().replace(" ", "_")
            out[key] = int(m.group(2)) * PAGE_BYTES / 1024 / 1024
    return out


def sample_unified_memory(gatekeeper, rss_proc=None) -> MemSample:
    """One unified-memory snapshot: RSS + parsed vm_stat categories (vm_stat via Gatekeeper)."""
    res = gatekeeper.run_subprocess(["vm_stat"])
    cats = parse_vm_stat(res.stdout_tail)
    return MemSample(
        rss_mb=peak_rss_mb(rss_proc),
        free_mb=cats.get("free", 0.0),
        active_mb=cats.get("active", 0.0),
        wired_mb=cats.get("wired_down", 0.0),
    )
