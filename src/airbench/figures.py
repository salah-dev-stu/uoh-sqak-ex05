"""Matplotlib figure primitives + a tolerant build-all (H6/H8).

Uses the Agg backend so figures render headless (CI/grader). ``build_all`` reads
whatever metric JSONs exist under a results dir and emits the standard plot set;
it is tolerant of missing inputs so partial runs still produce what they can.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def bar_with_std(
    path, labels: list[str], means: list[float], stds: list[float], ylabel: str, title: str
) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(labels, means, yerr=stds, capsize=4, color="#4C72B0")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(p, dpi=130)
    plt.close(fig)
    return p


def line_chart(
    path, xs: list[float], series: dict[str, list[float]], xlabel: str, ylabel: str, title: str
) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4))
    for name, ys in series.items():
        ax.plot(xs, ys, marker="o", label=name)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(p, dpi=130)
    plt.close(fig)
    return p


def scatter_roofline(
    path, ridge: float, peak_gflops: float, bw_gbps: float, points: list[dict[str, Any]]
) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4))
    xs = [0.1, ridge, ridge * 100]
    ax.plot(xs, [min(peak_gflops, bw_gbps * x) for x in xs], "k-", label="roofline")
    for pt in points:
        ax.scatter(pt["intensity"], pt["achieved_gflops"], label=pt.get("label", ""))
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("arithmetic intensity (FLOP/byte)")
    ax.set_ylabel("achieved GFLOP/s")
    ax.set_title("Roofline — M2")
    ax.legend()
    fig.tight_layout()
    fig.savefig(p, dpi=130)
    plt.close(fig)
    return p


def build_all(results_dir, out_dir) -> list[Path]:
    """Best-effort: emit figures for whichever result files are present."""
    from airbench.metrics.aggregate import read_json

    results_dir, out_dir = Path(results_dir), Path(out_dir)
    written: list[Path] = []
    econ = results_dir / "economics.json"
    if econ.exists():
        data = read_json(econ)
        provs = data.get("providers", [])
        if provs:
            written.append(
                bar_with_std(
                    out_dir / "breakeven.png",
                    [p["provider"] for p in provs],
                    [p["break_even_req_per_day"] for p in provs],
                    [0] * len(provs),
                    "break-even (req/day)",
                    "On-Prem vs API break-even",
                )
            )
    return written
