"""Phase 2 — SDK façade: wiring + delegation (runners mocked)."""

from __future__ import annotations

from pathlib import Path

import pytest

from airbench.sdk import facade
from airbench.sdk.facade import BenchSDK


@pytest.fixture
def sdk(tmp_path):
    return BenchSDK(run_id="t", results_root=tmp_path)


def test_init_builds_gatekeeper_and_run_dir(sdk, tmp_path):
    assert sdk.gatekeeper is not None
    assert sdk.results_dir == tmp_path / "t"


def test_run_ids_unique():
    a, b = BenchSDK(results_root="results"), BenchSDK(results_root="results")
    assert a.run_id != b.run_id


def test_probe_writes_hardware_json(sdk):
    report = sdk.probe_hardware()
    assert "curated" in report and report["ram_total_gb"] > 0
    assert (sdk.results_dir / "hardware.json").exists()


def test_run_baseline_oom_delegates(sdk, monkeypatch):
    from airbench.runners.run_types import FailureReport

    sentinel = FailureReport("RuntimeError", "oom", 7800.0, "memory-bound")
    monkeypatch.setattr(facade.baseline_oom, "run_oom", lambda gk, loader, peak_mem_fn: sentinel)
    out = sdk.run_baseline_oom(loader=lambda: None)
    assert out is sentinel
    assert (sdk.results_dir / "baseline" / "oom.json").exists()


def test_run_baseline_llamacpp_delegates(sdk, monkeypatch):
    from airbench.runners.run_types import RunMetrics

    rm = RunMetrics("llama.cpp Q4_K_M", ttft_s=1.2, tpot_s=0.09)
    monkeypatch.setattr(facade.baseline_llamacpp, "run", lambda gk, p, prompt, rt, ctx, label: rm)
    out = sdk.run_baseline_llamacpp("/Volumes/Backup/m.gguf")
    assert out is rm
    assert (sdk.results_dir / "baseline" / "Q4_K_M.json").exists()


def test_run_airllm_constraint_path(sdk, monkeypatch):
    from airbench.runners.run_types import ConstraintReport

    cr = ConstraintReport(reason="ImportError: cuda")
    monkeypatch.setattr(facade, "run_airllm", lambda *a, **k: cr)
    out = sdk.run_airllm(generate=lambda m: None, loader=lambda: None)
    assert out is cr
    assert (sdk.results_dir / "airllm" / "airllm.json").exists()


def test_run_quant_sweep_delegates(sdk, monkeypatch):
    monkeypatch.setattr(
        facade.quant_sweep, "run_sweep", lambda *a, **k: {"levels": [], "red_line": "Q2_K"}
    )
    out = sdk.run_quant_sweep(
        lambda lvl: "p", lambda lvl, p: None, lambda lvl, p: 7.0, lambda p: None
    )
    assert out["red_line"] == "Q2_K"
    assert (sdk.results_dir / "quant" / "sweep.json").exists()


def test_run_lora_delegates(sdk, monkeypatch):
    from airbench.runners.run_types import LoraResult

    res = LoraResult(base_model="m", iters=[10], train_loss=[2.0], val=[])
    monkeypatch.setattr(facade.lora, "run_lora", lambda gk, cfg, model: res)
    out = sdk.run_lora()
    assert out is res
    assert (sdk.results_dir / "lora" / "metrics.json").exists()


def test_compute_economics(sdk):
    report = sdk.compute_economics()
    assert report["onprem_annual"] > 0
    assert (sdk.results_dir / "economics.json").exists()


def test_make_figures_delegates(sdk, monkeypatch):
    monkeypatch.setattr(facade.figures, "build_all", lambda rd, od: [Path("x.png")])
    assert sdk.make_figures() == [Path("x.png")]
