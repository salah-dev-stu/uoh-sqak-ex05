"""llama-bench JSON → TTFT/TPOT/throughput with mean±std (robust real-run engine).

llama-bench runs each test (prefill `pp`, decode `tg`) over R repetitions and reports
``avg_ts`` ± ``stddev_ts`` (tokens/s) — exactly the mean±std H5 wants, natively. We run
with ``-mmp 0`` (load weights into RAM, not mmap'd off the SSD) and ``-ngl 99`` (Metal GPU).
stdout (the JSON array) is mixed with stderr (Metal logs) by the Gatekeeper runner, so we
extract the bracketed array before parsing.
"""

from __future__ import annotations

import json
from typing import Any


def build_argv(
    binary: str,
    model_path: str,
    n_prompt: int,
    n_gen: int,
    reps: int,
    ngl: int = 99,
    mmap: bool = False,
) -> list[str]:
    return [
        binary,
        "-m",
        model_path,
        "-p",
        str(n_prompt),
        "-n",
        str(n_gen),
        "-ngl",
        str(ngl),
        "-r",
        str(reps),
        "-mmp",
        "1" if mmap else "0",
        "-o",
        "json",
    ]


def _extract_json_array(text: str) -> list[dict]:
    start, end = text.find("["), text.rfind("]")
    if start == -1 or end == -1 or end < start:
        raise ValueError("no JSON array found in llama-bench output")
    return json.loads(text[start : end + 1])


def parse_json(text: str) -> dict[str, Any]:
    """Extract prefill (Prefill/TTFT) and decode (Decode/TPOT) stats from llama-bench JSON."""
    data = _extract_json_array(text)
    pp = next((r for r in data if r.get("n_prompt", 0) > 0 and r.get("n_gen", 0) == 0), None)
    tg = next((r for r in data if r.get("n_gen", 0) > 0 and r.get("n_prompt", 0) == 0), None)
    out: dict[str, Any] = {}
    if pp:
        out["prefill_tps"] = {"mean": pp["avg_ts"], "std": pp["stddev_ts"]}
        out["n_prompt"] = pp["n_prompt"]
        out["ttft_s"] = pp["n_prompt"] / pp["avg_ts"] if pp["avg_ts"] else None
    if tg:
        out["decode_tps"] = {"mean": tg["avg_ts"], "std": tg["stddev_ts"]}
        out["throughput_tps"] = tg["avg_ts"]
        out["tpot_ms"] = 1000.0 / tg["avg_ts"] if tg["avg_ts"] else None
    return out
