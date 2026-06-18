"""Baseline 5.2a — the direct HF FP16 load that OOMs an 8 GB Mac (H2).

The loader (the transformers FP16 model-load closure, built by the experiment) is
run through the Gatekeeper. We EXPECT it to raise (memory exhaustion); the
exception + a memory trace become the documented failure and bottleneck evidence.
"""

from __future__ import annotations

from collections.abc import Callable

from airbench.runners.run_types import FailureReport


def diagnose(exc: BaseException) -> str:
    msg = str(exc).lower()
    if "mps" in msg or "metal" in msg:
        return "MPS/Metal allocation failure — model exceeds unified memory (memory-bound)"
    if any(s in msg for s in ("out of memory", "oom", "cannot allocate", "killed")):
        return "host OOM — FP16 weights exceed 8 GB RAM (memory-bound)"
    return f"load failed: {type(exc).__name__}"


def run_oom(
    gatekeeper,
    loader: Callable[[], object],
    peak_mem_fn: Callable[[], float] | None = None,
    label: str = "baseline FP16 load",
) -> FailureReport:
    """Attempt the oversized load; capture the failure (the expected outcome)."""
    try:
        gatekeeper.load_model(loader, label)
    except Exception as exc:  # the documented, expected failure
        peak = peak_mem_fn() if peak_mem_fn else None
        return FailureReport(type(exc).__name__, str(exc), peak, diagnose(exc))
    return FailureReport(
        "NoError",
        "model loaded without OOM",
        peak_mem_fn() if peak_mem_fn else None,
        "did not OOM — model unexpectedly fit; pick a larger model/precision",
    )
