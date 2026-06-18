"""Phase 7 — experiment wrappers dispatch via the SDK, and stay SDK-only (§8)."""

from __future__ import annotations

from pathlib import Path

import pytest

from airbench.sdk import ConstraintReport
from experiments import (
    airllm_run,
    baseline_oom,
    economics_run,
    extreme_airllm,
    probe,
    quant_sweep,
)


class RecordingSDK:
    def __init__(self, airllm_result=None):
        self.results_dir = Path("results/test")
        self.calls = []
        self._airllm_result = airllm_result
        self.gatekeeper = type(
            "GK", (), {"download": lambda *a, **k: None, "run_subprocess": lambda *a, **k: None}
        )()

    def __getattr__(self, name):
        def _m(*a, **k):
            self.calls.append(name)
            if name == "run_airllm":
                return self._airllm_result
            return {"ok": name}

        return _m


@pytest.mark.parametrize(
    "module,method",
    [
        (probe, "probe_hardware"),
        (baseline_oom, "run_baseline_oom"),
        (quant_sweep, "run_quant_sweep"),
        (economics_run, "compute_economics"),
        (extreme_airllm, "run_extreme"),
    ],
)
def test_experiment_dispatches(module, method):
    sdk = RecordingSDK()
    module.main(sdk=sdk)
    assert method in sdk.calls


def test_airllm_falls_back_to_layered_on_constraint():
    sdk = RecordingSDK(airllm_result=ConstraintReport(reason="ImportError: cuda"))
    airllm_run.main(sdk=sdk)
    assert "run_airllm" in sdk.calls and "run_layered_demo" in sdk.calls


def test_experiments_are_sdk_only():
    forbidden = ("airbench.runners", "airbench.metrics", "airbench.economics", "airbench.figures")
    exp_dir = Path(__file__).resolve().parents[1] / "experiments"
    offenders = []
    for py in exp_dir.glob("*.py"):
        text = py.read_text(encoding="utf-8")
        for f in forbidden:
            if f in text:
                offenders.append(f"{py.name}: {f}")
    assert not offenders, f"experiments must reach runners/metrics via the SDK: {offenders}"
