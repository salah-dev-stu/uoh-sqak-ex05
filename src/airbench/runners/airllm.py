"""5.3 — the REAL AirLLM attempt (time-boxed), with an honest fallback.

We try to load the model via the ``airllm`` library through the Gatekeeper. AirLLM
is CUDA-centric, so on Apple-Silicon/MPS this may raise (ImportError / no CUDA);
that failure is captured as a ConstraintReport and the SDK falls back to the
``layered`` llama.cpp-style streaming demo. Either outcome is gradeable (H3).
"""

from __future__ import annotations

from collections.abc import Callable

from airbench.runners.run_types import ConstraintReport, RunMetrics


def run_airllm(
    gatekeeper,
    loader: Callable[[], object],
    generate: Callable[[object], RunMetrics],
    timebox_min: int = 60,
    label: str = "airllm",
) -> RunMetrics | ConstraintReport:
    """Attempt the real AirLLM run; on any load failure return a ConstraintReport."""
    try:
        model = gatekeeper.load_model(loader, label)
    except Exception as exc:
        return ConstraintReport(
            reason=f"{type(exc).__name__}: {exc}",
            fallback="layered",
            detail=f"time-box was {timebox_min} min; falling back to layer-streaming demo",
        )
    metrics = generate(model)
    metrics.extra.setdefault("timebox_min", timebox_min)
    metrics.extra.setdefault("airllm", True)
    return metrics
