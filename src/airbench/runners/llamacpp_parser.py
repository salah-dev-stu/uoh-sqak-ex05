"""Parse llama.cpp / llama-perplexity stdout into structured metrics.

llama.cpp prints a timings block: ``prompt eval time`` is the Prefill phase
(maps to TTFT) and ``eval time`` is the Decode phase (maps to TPOT). Tolerant of
missing lines (returns None) so a partial/failed run still yields a record.
"""

from __future__ import annotations

import re
from typing import Any

_PROMPT = re.compile(r"prompt eval time\s*=\s*([\d.]+)\s*ms\s*/\s*(\d+)\s*tokens")
_EVAL = re.compile(
    r"\beval time\s*=\s*([\d.]+)\s*ms\s*/\s*(\d+)\s*runs.*?([\d.]+)\s*ms per token"
    r".*?([\d.]+)\s*tokens per second"
)
_PPL = re.compile(r"PPL\s*=\s*([\d.]+)")


def parse_metrics(text: str) -> dict[str, Any]:
    """Extract prefill/decode timings. Values are None when the line is absent."""
    out: dict[str, Any] = {
        "prefill_ms": None,
        "prefill_tokens": None,
        "decode_ms": None,
        "decode_runs": None,
        "tpot_ms": None,
        "decode_tps": None,
    }
    if (m := _PROMPT.search(text)) is not None:
        out["prefill_ms"] = float(m.group(1))
        out["prefill_tokens"] = int(m.group(2))
    if (m := _EVAL.search(text)) is not None:
        out["decode_ms"] = float(m.group(1))
        out["decode_runs"] = int(m.group(2))
        out["tpot_ms"] = float(m.group(3))
        out["decode_tps"] = float(m.group(4))
    return out


def parse_perplexity(text: str) -> float | None:
    """Final perplexity estimate from ``llama-perplexity`` output (the red-line metric)."""
    matches = _PPL.findall(text)
    return float(matches[-1]) if matches else None
