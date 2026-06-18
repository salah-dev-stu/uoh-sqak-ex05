"""Baseline 5.2b — a runnable quantized run via llama.cpp (the comparison anchor).

Builds the ``llama-cli`` argv, runs it through the Gatekeeper subprocess wrapper,
and parses the timings block into TTFT (Prefill) / TPOT (Decode) / throughput.
"""

from __future__ import annotations

from airbench.runners.llamacpp_parser import parse_metrics
from airbench.runners.run_types import RunMetrics


def build_argv(
    binary: str, model_path: str, prompt: str, n_predict: int, ctx: int, seed: int
) -> list[str]:
    return [
        binary,
        "-m",
        model_path,
        "-p",
        prompt,
        "-n",
        str(n_predict),
        "-c",
        str(ctx),
        "-s",
        str(seed),
        "--no-display-prompt",
    ]


def run(
    gatekeeper,
    model_path: str,
    prompt: str,
    runtime: dict,
    ctx: int | None = None,
    label: str = "llama.cpp baseline",
) -> RunMetrics:
    argv = build_argv(
        runtime.get("llamacpp_bin", "llama-cli"),
        model_path,
        prompt,
        runtime["max_tokens"],
        ctx or (runtime.get("ctx_sizes") or [512])[0],
        runtime["seed"],
    )
    res = gatekeeper.run_subprocess(argv, timeout=runtime.get("timeout_s"))
    m = parse_metrics(res.stdout_tail)
    ttft = m["prefill_ms"] / 1000 if m["prefill_ms"] is not None else None
    tpot = m["tpot_ms"] / 1000 if m["tpot_ms"] is not None else None
    return RunMetrics(
        label=label,
        ttft_s=ttft,
        tpot_s=tpot,
        throughput_tps=m["decode_tps"],
        extra={"raw": m, "returncode": res.returncode, "timed_out": res.timed_out},
    )
