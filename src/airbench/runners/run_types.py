"""Shared result types for runners (kept separate so each runner stays ≤150 lines)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class RunMetrics:
    """A successful (or partial) inference run's headline numbers."""

    label: str
    ttft_s: float | None = None
    tpot_s: float | None = None
    throughput_tps: float | None = None
    peak_mem_mb: float | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class FailureReport:
    """A baseline that failed to run — the documented bottleneck evidence (H2)."""

    error_type: str
    message: str
    peak_mem_mb: float | None
    diagnosis: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ConstraintReport:
    """A tool that could not run on this platform — the honest AirLLM path (H3)."""

    reason: str
    platform: str = "apple-silicon-mps"
    fallback: str = "layered"
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class LayeredMetrics:
    """Per-layer IO-vs-compute breakdown for the layer-streaming demo (H3/H8)."""

    n_layers: int
    io_s: list[float]
    compute_s: list[float]
    cumulative_io_bytes: int
    predicted_io_s: list[float] = field(default_factory=list)

    @property
    def total_io_s(self) -> float:
        return sum(self.io_s)

    @property
    def total_compute_s(self) -> float:
        return sum(self.compute_s)

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_layers": self.n_layers,
            "io_s": self.io_s,
            "compute_s": self.compute_s,
            "predicted_io_s": self.predicted_io_s,
            "cumulative_io_bytes": self.cumulative_io_bytes,
            "total_io_s": self.total_io_s,
            "total_compute_s": self.total_compute_s,
        }
