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
FPT = 2 * 7.62e9  # FLOPs/token (Qwen2.5-7B)
RED, BLUE, GREEN = "#C44E52", "#4C72B0", "#55A868"


def _save(fig, name):
    fig.tight_layout()
    fig.savefig(f"{OUT}/{name}", dpi=130)
    plt.close(fig)
    print(f"wrote {OUT}/{name}")


def decode_chart(b):
    gpu = b["regimes"]["gpu_ram"]["decode_tps"]["mean"]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(["GPU + RAM", "CPU + mmap"], [gpu, 0.03], color=[RED, BLUE])
    ax.set_yscale("log")
    ax.set_ylabel("decode throughput (tok/s, log)")
    ax.set_title("Decode on 8 GB M2 — the memory wall (Qwen2.5-7B Q4)")
    ax.text(0, gpu * 1.2, "17.6 tok/s — but FROZE", ha="center", fontsize=8)
    ax.text(1, 0.036, "<0.03 tok/s — crawls (paging)", ha="center", fontsize=8)
    _save(fig, "decode_throughput.png")


def prefill_chart(b):
    g, c = b["regimes"]["gpu_ram"]["prefill_tps"], b["regimes"]["cpu_mmap"]["prefill_tps"]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(
        ["GPU + RAM", "CPU + mmap"], [g["mean"], c["mean"]], yerr=[g["std"], c["std"]], capsize=5
    )
    ax.set_yscale("log")
    ax.set_ylabel("prefill throughput (tok/s, log)")
    ax.set_title("Prefill on 8 GB M2 (Qwen2.5-7B Q4, mean±std)")
    _save(fig, "prefill_throughput.png")


def roofline_chart(b):
    peak, bw = 2840.0, 100.0
    ridge = peak / bw
    gpu = b["regimes"]["gpu_ram"]["decode_tps"]["mean"]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    xs = [0.1, ridge, ridge * 50]
    ax.plot(xs, [min(peak, bw * x) for x in xs], "k-", label="roofline (M2)")
    ax.axvline(ridge, ls="--", color="gray", alpha=0.6, label=f"ridge ≈ {ridge:.0f}")
    ax.scatter(3.5, gpu * FPT / 1e9, color=RED, s=60, label="GPU decode", zorder=5)
    ax.scatter(3.5, 0.03 * FPT / 1e9, color=BLUE, s=60, label="CPU decode", zorder=5)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("arithmetic intensity (FLOP/byte)")
    ax.set_ylabel("achieved GFLOP/s")
    ax.set_title("Roofline — decode is memory-bound (left of the ridge)")
    ax.legend(fontsize=8)
    _save(fig, "roofline.png")


def breakeven_chart(e):
    p = e["providers"]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar([x["provider"] for x in p], [x["break_even_req_per_day"] for x in p], color=GREEN)
    ax.set_ylabel("break-even (requests/day)")
    ax.set_title(f"On-Prem (${e['onprem_annual']:.0f}/yr) vs API break-even")
    ax.axhline(300, ls="--", color="red", label="~max req/day this 8 GB Mac serves")
    ax.legend(fontsize=8)
    _save(fig, "breakeven.png")


def memory_chart(m):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    safe, fast = m["safe_cpu_mmap"]["free_pct_series"], m["fast_gpu_resident"]["free_pct_series"]
    ax.plot([i * 20 for i in range(len(safe))], safe, "o-", color=BLUE, label="CPU + mmap — stable")
    ax.plot(
        [i * 20 for i in range(len(fast))], fast, "s--", color=RED, label="GPU resident — froze"
    )
    n = (len(fast) - 1) * 20
    ax.scatter([n], [0], marker="x", s=140, color=RED, zorder=6)
    ax.text(n, 4, "FREEZE (swap-death)", color=RED, fontsize=9, ha="center")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("system memory free (%)")
    ax.set_title("Memory during Q4 inference on 8 GB M2 — the wall")
    ax.legend(fontsize=8)
    _save(fig, "memory_pressure.png")


def tco_chart(e):
    onprem = e["onprem_annual"]
    reqs = list(range(0, 8001, 400))
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.axhline(onprem, color=GREEN, label=f"On-Prem (${onprem:.0f}/yr, flat)")
    for p, col in zip(e["providers"], [BLUE, RED], strict=False):
        ys = [p["api_per_request"] * r * 365 for r in reqs]
        ax.plot(reqs, ys, "-", color=col, label=f"API {p['provider']}")
    ax.axvspan(0, 300, alpha=0.12, color="red")
    ax.text(150, onprem * 1.5, "this Mac's capacity", color="red", fontsize=8, ha="center")
    ax.set_xlabel("requests / day")
    ax.set_ylabel("annual cost ($)")
    ax.set_title("TCO: On-Prem vs API — API wins far below the break-even")
    ax.legend(fontsize=8)
    _save(fig, "tco_curve.png")


def main() -> int:
    b = read_json("results/real/baseline_q4.json")
    e = read_json("results/real/economics.json")
    m = read_json("results/real/memory_regimes.json")
    decode_chart(b)
    prefill_chart(b)
    roofline_chart(b)
    breakeven_chart(e)
    memory_chart(m)
    tco_chart(e)
    return 0


if __name__ == "__main__":
    sys.exit(main())
