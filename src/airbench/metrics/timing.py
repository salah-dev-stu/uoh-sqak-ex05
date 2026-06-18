"""TTFT / TPOT / throughput from token-emission timestamps (H5).

Prefill vs Decode are split: TTFT is the time to the first token (Prefill +
KV-cache init); TPOT is the mean inter-token latency over the Decode phase. Pure
and fully mockable — no model lives here; a ``generate_fn`` callback supplies
``(start_time, [token_timestamps...])`` for one run.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from statistics import fmean, pstdev
from typing import Any


@dataclass
class RunTiming:
    ttft_s: float
    tpot_s: float | None
    throughput_tps: float
    n_tokens: int


@dataclass
class TimingStats:
    runs: list[RunTiming]

    def _agg(self, values: list[float]) -> dict[str, float]:
        clean = [v for v in values if v is not None]
        if not clean:
            return {"mean": float("nan"), "std": float("nan"), "n": 0}
        return {"mean": fmean(clean), "std": pstdev(clean), "n": len(clean)}

    def summary(self) -> dict[str, Any]:
        return {
            "ttft_s": self._agg([r.ttft_s for r in self.runs]),
            "tpot_s": self._agg([r.tpot_s for r in self.runs]),
            "throughput_tps": self._agg([r.throughput_tps for r in self.runs]),
            "n_runs": len(self.runs),
        }


def measure_once(start: float, token_times: list[float]) -> RunTiming:
    """Compute one run's timing from the submit time and per-token timestamps."""
    if not token_times:
        raise ValueError("token_times is empty — no tokens were generated")
    n = len(token_times)
    ttft = token_times[0] - start
    total = token_times[-1] - start
    tpot = (token_times[-1] - token_times[0]) / (n - 1) if n >= 2 else None
    throughput = n / total if total > 0 else float("inf")
    return RunTiming(ttft, tpot, throughput, n)


def measure_generation(
    generate_fn: Callable[[], tuple[float, list[float]]], n_runs: int, warmup: int = 1
) -> TimingStats:
    """Run ``generate_fn`` ``warmup + n_runs`` times; discard warmups; aggregate the rest."""
    runs: list[RunTiming] = []
    for i in range(warmup + n_runs):
        start, token_times = generate_fn()
        if i < warmup:
            continue
        runs.append(measure_once(start, token_times))
    return TimingStats(runs)
