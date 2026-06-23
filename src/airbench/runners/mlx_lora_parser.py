"""Parse mlx_lm.lora stdout into a loss curve + parameter-efficiency numbers (H5/H9).

Tolerant of missing lines (returns empties/None) so a partial run still yields a record.
"""

from __future__ import annotations

import re
from typing import Any

_TRAIN = re.compile(r"Iter (\d+): Train loss ([\d.]+)")
_VAL = re.compile(r"Iter (\d+): Val loss ([\d.]+)")
_TRAINABLE = re.compile(r"Trainable parameters:\s*([\d.]+)%\s*\(([\d.]+)M/([\d.]+)M\)")
_PEAK = re.compile(r"Peak mem ([\d.]+) GB")


def parse_loss(text: str) -> dict[str, Any]:
    iters, train, val = [], [], []
    for m in _TRAIN.finditer(text):
        iters.append(int(m.group(1)))
        train.append(float(m.group(2)))
    for m in _VAL.finditer(text):
        val.append({"iter": int(m.group(1)), "loss": float(m.group(2))})
    return {"iters": iters, "train_loss": train, "val": val}


def parse_trainable(text: str) -> dict[str, Any]:
    out: dict[str, Any] = {
        "trainable_pct": None,
        "trainable_m": None,
        "total_m": None,
        "peak_mem_gb": None,
    }
    if (m := _TRAINABLE.search(text)) is not None:
        out["trainable_pct"] = float(m.group(1))
        out["trainable_m"] = float(m.group(2))
        out["total_m"] = float(m.group(3))
    peaks = _PEAK.findall(text)
    if peaks:
        out["peak_mem_gb"] = max(float(p) for p in peaks)
    return out
