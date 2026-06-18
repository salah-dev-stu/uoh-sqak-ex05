"""5.4 — serial GGUF quant sweep with the perplexity red-line (H4/H9).

For each level (Q8→Q5→Q4→Q2): download → benchmark → measure perplexity → DELETE
the weight before the next (so disk never holds more than one quant at a time).
The red line is the first level whose ΔPPL vs the Q8 baseline exceeds the
configured threshold — an objective accuracy cliff, not a vibe.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from airbench.runners.run_types import RunMetrics


@dataclass
class LevelResult:
    name: str
    metrics: RunMetrics | None
    ppl: float | None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "metrics": self.metrics.to_dict() if self.metrics else None,
            "ppl": self.ppl,
            "error": self.error,
        }


def detect_red_line(results: list[LevelResult], threshold: float) -> str | None:
    """First level whose PPL exceeds the best (lowest) PPL by more than ``threshold``."""
    ppls = [r.ppl for r in results if r.ppl is not None]
    if not ppls:
        return None
    baseline = min(ppls)
    for r in results:
        if r.ppl is not None and (r.ppl - baseline) > threshold:
            return r.name
    return None


def run_sweep(
    levels: list[dict],
    download: Callable[[dict], str],
    bench: Callable[[dict, str], RunMetrics],
    perplexity: Callable[[dict, str], float | None],
    cleanup: Callable[[str], None],
    threshold: float,
) -> dict[str, Any]:
    """Run the serial sweep; one level failing does not abort the rest."""
    results: list[LevelResult] = []
    for lvl in levels:
        path = download(lvl)
        try:
            metrics = bench(lvl, path)
            ppl = perplexity(lvl, path)
            results.append(LevelResult(lvl["name"], metrics, ppl))
        except Exception as exc:
            results.append(LevelResult(lvl["name"], None, None, error=str(exc)))
        finally:
            cleanup(path)  # always free disk before the next level
    return {
        "levels": [r.to_dict() for r in results],
        "red_line": detect_red_line(results, threshold),
    }
