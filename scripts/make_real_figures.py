#!/usr/bin/env python3
"""Build the real comparative figures from results/real/*.json into reports/figures/.

uv run python scripts/make_real_figures.py
"""

from __future__ import annotations

import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from airbench.metrics.aggregate import read_json  # noqa: E402

OUT = "reports/figures"
FLOPS_PER_TOKEN = 2 * 7.62e9  # ~2 FLOPs/param for Qwen2.5-7B


def _save(fig, name):
    fig.tight_layout()
    fig.savefig(f"{OUT}/{name}", dpi=130)
    plt.close(fig)
    print(f"wrote {OUT}/{name}")


def decode_chart(b):
    gpu = b["regimes"]["gpu_ram"]["decode_tps"]["mean"]
    cpu = 0.03  # <0.03 tok/s (6 tokens did not finish in 200 s)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(
        ["GPU + RAM\n(-ngl 99 -mmp 0)", "CPU + mmap\n(-ngl 0 -mmp 1)"],
        [gpu, cpu],
        color=["#C44E52", "#4C72B0"],
    )
    ax.set_yscale("log")
    ax.set_ylabel("decode throughput (tok/s, log)")
    ax.set_title("Decode on 8 GB M2 — the memory wall (Qwen2.5-7B Q4)")
    ax.text(0, gpu * 1.2, "17.6 tok/s\nbut FROZE the\n8 GB machine", ha="center", fontsize=9)
    ax.text(1, cpu * 1.2, "<0.03 tok/s\nstable but\ncrawls (USB paging)", ha="center", fontsize=9)
    _save(fig, "decode_throughput.png")


def prefill_chart(b):
    g = b["regimes"]["gpu_ram"]["prefill_tps"]
    c = b["regimes"]["cpu_mmap"]["prefill_tps"]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(
        ["GPU + RAM", "CPU + mmap"],
        [g["mean"], c["mean"]],
        yerr=[g["std"], c["std"]],
        capsize=5,
        color=["#C44E52", "#4C72B0"],
    )
    ax.set_yscale("log")
    ax.set_ylabel("prefill throughput (tok/s, log)")
    ax.set_title("Prefill on 8 GB M2 (Qwen2.5-7B Q4, mean±std)")
    _save(fig, "prefill_throughput.png")


def roofline_chart(b):
    peak, bw = 2840.0, 100.0  # M2 GFLOP/s, GB/s (config/hardware.json)
    ridge = peak / bw
    gpu_tps = b["regimes"]["gpu_ram"]["decode_tps"]["mean"]
    pts = [
        ("GPU decode", 3.5, gpu_tps * FLOPS_PER_TOKEN / 1e9, "#C44E52"),
        ("CPU decode", 3.5, 0.03 * FLOPS_PER_TOKEN / 1e9, "#4C72B0"),
    ]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    xs = [0.1, ridge, ridge * 50]
    ax.plot(xs, [min(peak, bw * x) for x in xs], "k-", label="roofline (M2)")
    ax.axvline(ridge, ls="--", color="gray", alpha=0.6, label=f"ridge ≈ {ridge:.0f}")
    for label, x, y, col in pts:
        ax.scatter(x, y, color=col, s=60, label=label, zorder=5)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("arithmetic intensity (FLOP/byte)")
    ax.set_ylabel("achieved GFLOP/s")
    ax.set_title("Roofline — decode is memory-bound (left of the ridge)")
    ax.legend(fontsize=8)
    _save(fig, "roofline.png")


def breakeven_chart(e):
    provs = e["providers"]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(
        [p["provider"] for p in provs],
        [p["break_even_req_per_day"] for p in provs],
        color="#55A868",
    )
    ax.set_ylabel("break-even (requests/day)")
    ax.set_title(f"On-Prem (${e['onprem_annual']:.0f}/yr) vs API break-even")
    ax.axhline(300, ls="--", color="red", label="~max req/day this 8 GB Mac can serve")
    ax.legend(fontsize=8)
    _save(fig, "breakeven.png")


def main() -> int:
    b = read_json("results/real/baseline_q4.json")
    e = read_json("results/real/economics.json")
    decode_chart(b)
    prefill_chart(b)
    roofline_chart(b)
    breakeven_chart(e)
    return 0


if __name__ == "__main__":
    sys.exit(main())
