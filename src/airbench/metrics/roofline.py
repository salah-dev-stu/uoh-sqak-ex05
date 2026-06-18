"""Roofline model: arithmetic intensity, attainable performance, bound classification (H8).

For LLM **Decode**, each token re-reads all weights once, so bytes/token ≈
params·bytes_per_param and FLOPs/token ≈ 2·params. Intensity is then ≈
2/bytes_per_param (≈1 FLOP/byte at FP16) — far left of the ridge, i.e.
memory-bound. **Prefill** processes many tokens per weight read → higher
intensity → compute-bound. Ceilings come from ``config/hardware.json``.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RooflinePoint:
    intensity: float  # FLOP / byte
    achieved_gflops: float
    attainable_gflops: float
    bound: str  # "compute-bound" | "memory-bound"


def arithmetic_intensity(flops: float, bytes_moved: float) -> float:
    if bytes_moved <= 0:
        raise ValueError("bytes_moved must be > 0")
    return flops / bytes_moved


def attainable_gflops(intensity: float, peak_gflops: float, bw_gbps: float) -> float:
    """Roofline ceiling: min(compute peak, bandwidth · intensity). bw in GB/s, intensity FLOP/byte."""
    return min(peak_gflops, bw_gbps * intensity)


def ridge_intensity(peak_gflops: float, bw_gbps: float) -> float:
    return peak_gflops / bw_gbps


def classify(intensity: float, peak_gflops: float, bw_gbps: float) -> str:
    return "compute-bound" if intensity >= ridge_intensity(peak_gflops, bw_gbps) else "memory-bound"


def flops_per_token(params_b: float) -> float:
    """~2 FLOPs per parameter per token (one multiply-add across the weights)."""
    return 2.0 * params_b * 1e9


def bytes_per_token(params_b: float, bytes_per_param: float) -> float:
    """Decode re-reads every weight once per token."""
    return params_b * 1e9 * bytes_per_param


def evaluate(
    flops: float, bytes_moved: float, achieved_gflops: float, peak_gflops: float, bw_gbps: float
) -> RooflinePoint:
    intensity = arithmetic_intensity(flops, bytes_moved)
    return RooflinePoint(
        intensity=intensity,
        achieved_gflops=achieved_gflops,
        attainable_gflops=attainable_gflops(intensity, peak_gflops, bw_gbps),
        bound=classify(intensity, peak_gflops, bw_gbps),
    )
