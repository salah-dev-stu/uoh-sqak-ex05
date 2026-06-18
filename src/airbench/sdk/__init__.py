"""Public SDK surface (R1): all business logic is reached via ``BenchSDK``.

Result types are re-exported here so callers (CLI, experiments) never reach into
``airbench.runners`` directly.
"""

from airbench.runners.run_types import (
    ConstraintReport,
    FailureReport,
    LayeredMetrics,
    RunMetrics,
)
from airbench.sdk.facade import BenchSDK

__all__ = [
    "BenchSDK",
    "ConstraintReport",
    "FailureReport",
    "LayeredMetrics",
    "RunMetrics",
]
