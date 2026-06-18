"""5.3 equivalent — layer-by-layer streaming demo (AirLLM's mechanism, MPS/CPU).

Each transformer block is materialized from the SSD on demand, forwarded, then
freed — so peak memory stays at one block, not the whole model. We instrument IO
time (load) vs compute time (forward) per layer, and optionally the *predicted*
IO time from SSD bandwidth, to quantify the paging trade-off (H3/H8).
"""

from __future__ import annotations

from collections.abc import Callable

from airbench.runners.run_types import LayeredMetrics


def run_layered(
    n_layers: int,
    load_block: Callable[[int], object],
    forward: Callable[[object], object],
    free_block: Callable[[object], None],
    clock: Callable[[], float],
    ssd_bytes_per_s: float | None = None,
) -> LayeredMetrics:
    """Stream ``n_layers`` blocks one at a time, timing IO vs compute per layer."""
    io_s: list[float] = []
    compute_s: list[float] = []
    predicted_io_s: list[float] = []
    cumulative_bytes = 0
    for i in range(n_layers):
        t0 = clock()
        block = load_block(i)
        t1 = clock()
        forward(block)
        t2 = clock()
        nbytes = int(getattr(block, "nbytes", 0))
        cumulative_bytes += nbytes
        free_block(block)
        io_s.append(t1 - t0)
        compute_s.append(t2 - t1)
        if ssd_bytes_per_s:
            predicted_io_s.append(nbytes / ssd_bytes_per_s)
    return LayeredMetrics(n_layers, io_s, compute_s, cumulative_bytes, predicted_io_s)
