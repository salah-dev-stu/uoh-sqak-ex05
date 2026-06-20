#!/usr/bin/env python3
"""Drive the REAL llama.cpp benchmarks (baseline + quant sweep) with N-run mean±std.

    uv run python scripts/run_benchmarks.py baseline --model-path <gguf> [--n 5]
    uv run python scripts/run_benchmarks.py sweep [--n 3]

Each run shells out to llama-cli / llama-perplexity through the Gatekeeper; results
(with the gate ledger) are written under results/real/.
"""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

from airbench.metrics.aggregate import write_json
from airbench.runners import baseline_llamacpp
from airbench.runners.llamacpp_parser import parse_perplexity
from airbench.sdk import BenchSDK
from airbench.shared import config


def _agg(runs: list[dict], key: str) -> dict | None:
    vals = [r[key] for r in runs if r.get(key) is not None]
    if not vals:
        return None
    return {
        "mean": statistics.fmean(vals),
        "std": statistics.pstdev(vals),
        "n": len(vals),
        "values": vals,
    }


def bench_n(sdk: BenchSDK, model_path: str, n: int, warmup: int, label: str) -> dict:
    rt = config.get_runtime()
    runs: list[dict] = []
    for i in range(warmup + n):
        rm = baseline_llamacpp.run(sdk.gatekeeper, model_path, rt["prompts"][0], rt, label=label)
        print(
            f"  {label} run {i + 1}/{warmup + n}: ttft={rm.ttft_s} tpot={rm.tpot_s} tps={rm.throughput_tps}"
        )
        if i >= warmup:
            runs.append(rm.to_dict())
    return {
        "label": label,
        "n_runs": n,
        "model_path": model_path,
        "ttft_s": _agg(runs, "ttft_s"),
        "tpot_s": _agg(runs, "tpot_s"),
        "throughput_tps": _agg(runs, "throughput_tps"),
        "runs": runs,
    }


def perplexity_of(sdk: BenchSDK, model_path: str) -> float | None:
    rt = config.get_runtime()
    argv = [rt["llamacpp_perplexity_bin"], "-m", model_path, "-f", rt["perplexity_corpus"]]
    res = sdk.gatekeeper.run_subprocess(argv, timeout=rt.get("timeout_s"))
    return parse_perplexity(res.stdout_tail)


def cmd_baseline(sdk: BenchSDK, args) -> None:
    out = bench_n(sdk, args.model_path, args.n, config.get_runtime()["warmup"], "llama.cpp Q4_K_M")
    out["perplexity"] = perplexity_of(sdk, args.model_path)
    write_json(sdk.results_dir / "baseline" / "q4_aggregate.json", out)
    print(f"baseline → {sdk.results_dir / 'baseline' / 'q4_aggregate.json'}")


def cmd_sweep(sdk: BenchSDK, args) -> None:
    model = config.get_model("primary")
    repo, wdir = model["gguf_repo"], model["weights_dir"]
    results = []
    for lvl in config.get_quant_levels():
        path = Path(wdir) / lvl["gguf_file"]
        if not path.exists():
            print(f"downloading {lvl['name']} ...")
            path = sdk.gatekeeper.download(
                repo, lvl["gguf_file"], wdir, size_gb=lvl.get("approx_gb")
            )
        agg = bench_n(sdk, str(path), args.n, config.get_runtime()["warmup"], lvl["name"])
        agg["perplexity"] = perplexity_of(sdk, str(path))
        agg["approx_gb"] = lvl.get("approx_gb")
        results.append(agg)
        if lvl["name"] != "Q4_K_M":  # keep Q4 (the baseline anchor); free the rest
            Path(path).unlink(missing_ok=True)
            print(f"deleted {lvl['name']} to free disk")
    ppls = [r["perplexity"] for r in results if r["perplexity"] is not None]
    base = min(ppls) if ppls else None
    thr = config.get_runtime()["red_line_ppl_delta"]
    red = next(
        (
            r["label"]
            for r in results
            if r["perplexity"] and base is not None and r["perplexity"] - base > thr
        ),
        None,
    )
    write_json(sdk.results_dir / "quant" / "sweep.json", {"levels": results, "red_line": red})
    print(f"sweep → red_line={red}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("baseline")
    b.add_argument("--model-path", required=True)
    b.add_argument("--n", type=int, default=5)
    s = sub.add_parser("sweep")
    s.add_argument("--n", type=int, default=3)
    ap.add_argument("--run-id", default="real")
    args = ap.parse_args(argv)
    sdk = BenchSDK(run_id=args.run_id)
    {"baseline": cmd_baseline, "sweep": cmd_sweep}[args.cmd](sdk, args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
