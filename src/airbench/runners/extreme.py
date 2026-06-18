"""5.7 extension — the "extreme AirLLM" 70B-class run (H9).

Reuses the AirLLM attempt path with the 70B model; if the real library won't
run it streams via the layered demo. Tokens are capped (config) because each
token is ~80 s+ of pure SSD I/O at ~0.5 GB/s — the point is the I/O-bound number.
"""

from __future__ import annotations

from collections.abc import Callable

from airbench.runners.airllm import run_airllm
from airbench.runners.run_types import ConstraintReport, RunMetrics


def run_extreme(
    gatekeeper,
    loader: Callable[[], object],
    generate: Callable[[object], RunMetrics],
    timebox_min: int = 60,
) -> RunMetrics | ConstraintReport:
    result = run_airllm(gatekeeper, loader, generate, timebox_min, label="airllm-70b-extreme")
    if isinstance(result, RunMetrics):
        result.extra["extreme"] = True
    return result
