"""Paging deltas from two ``vm_stat`` snapshots — the OS virtual-memory evidence (H8).

AirLLM/layer-streaming spills to disk; the swapins/pageins delta during a run is
the direct, measurable signature of paging overhead the lecture asks us to tie
back to the bottleneck analysis.
"""

from __future__ import annotations

import re

_COUNTERS = ("pageins", "pageouts", "swapins", "swapouts")


def parse_counters(text: str) -> dict[str, int]:
    """Extract cumulative paging counters from ``vm_stat`` output (tolerant of format drift)."""
    out: dict[str, int] = {}
    for line in text.splitlines():
        m = re.match(r'"?([A-Za-z ]+?)"?:\s*([0-9]+)\.?', line.strip())
        if not m:
            continue
        key = m.group(1).strip().lower().replace(" ", "")
        if key in _COUNTERS:
            out[key] = int(m.group(2))
    return out


def delta(before_text: str, after_text: str) -> dict[str, int]:
    """Per-counter (after - before). Missing counters are treated as 0."""
    before, after = parse_counters(before_text), parse_counters(after_text)
    return {k: after.get(k, 0) - before.get(k, 0) for k in _COUNTERS}
