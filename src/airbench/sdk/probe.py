"""Hardware probe → the machine spec for results/hardware.json (H1).

Merges live readings (platform + psutil) with the curated peaks in
config/hardware.json (GPU GFLOP/s, memory bandwidth, measured SSD throughput).
"""

from __future__ import annotations

import platform
from typing import Any

from airbench.shared import config


def hardware_report(vm=None, cpu_logical=None) -> dict[str, Any]:
    """Build the hardware report. ``vm``/``cpu_logical`` are injectable for tests."""
    if vm is None or cpu_logical is None:
        import psutil

        vm = vm or psutil.virtual_memory()
        cpu_logical = cpu_logical or psutil.cpu_count(logical=True)
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "cpu_logical": cpu_logical,
        "ram_total_gb": round(vm.total / 1e9, 2),
        "ram_available_gb": round(vm.available / 1e9, 2),
        "curated": config.get_hardware(),
    }
