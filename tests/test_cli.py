"""Phase 6 — CLI dispatch (SDK methods mocked)."""

from __future__ import annotations

import pytest

from airbench import cli


@pytest.fixture
def patched_sdk(monkeypatch, tmp_path):
    """Patch BenchSDK so commands dispatch without touching real runners."""
    calls = []

    class FakeSDK:
        def __init__(self, **kw):
            self.results_dir = tmp_path / "run"

        def __getattr__(self, name):
            def _method(*a, **k):
                calls.append(name)
                return {"ok": name}

            return _method

    monkeypatch.setattr(cli, "BenchSDK", FakeSDK, raising=False)
    # cli imports BenchSDK lazily inside main(); patch the source module too
    import airbench.sdk as sdkmod

    monkeypatch.setattr(sdkmod, "BenchSDK", FakeSDK)
    return calls


def test_probe_dispatch(patched_sdk):
    assert cli.main(["probe"]) == 0
    assert "probe_hardware" in patched_sdk


def test_baseline_oom_dispatch(patched_sdk):
    assert cli.main(["baseline-oom"]) == 0
    assert "run_baseline_oom" in patched_sdk


def test_baseline_requires_model_path(patched_sdk):
    with pytest.raises(SystemExit):
        cli.main(["baseline"])  # missing --model-path


def test_baseline_dispatch(patched_sdk):
    assert cli.main(["baseline", "--model-path", "/Volumes/Backup/m.gguf"]) == 0
    assert "run_baseline_llamacpp" in patched_sdk


def test_economics_dispatch(patched_sdk):
    assert cli.main(["economics"]) == 0
    assert "compute_economics" in patched_sdk


def test_figures_dispatch(patched_sdk):
    assert cli.main(["figures"]) == 0
    assert "make_figures" in patched_sdk


def test_all_dispatch(patched_sdk):
    assert cli.main(["all"]) == 0
    assert "probe_hardware" in patched_sdk and "compute_economics" in patched_sdk


def test_quant_sweep_dispatch(patched_sdk):
    assert cli.main(["quant-sweep"]) == 0
    assert "run_quant_sweep" in patched_sdk


def test_airllm_dispatch(patched_sdk):
    assert cli.main(["airllm"]) == 0
    assert "run_airllm" in patched_sdk


def test_layered_dispatch(patched_sdk):
    assert cli.main(["layered"]) == 0
    assert "run_layered_demo" in patched_sdk


def test_extreme_dispatch(patched_sdk):
    assert cli.main(["extreme"]) == 0
    assert "run_extreme" in patched_sdk


def test_version_flag(capsys):
    with pytest.raises(SystemExit) as e:
        cli.main(["--version"])
    assert e.value.code == 0
    assert "1.00" in capsys.readouterr().out


def test_unknown_command_exits():
    with pytest.raises(SystemExit):
        cli.main(["nope"])


def test_handler_error_returns_nonzero(monkeypatch, tmp_path):
    class BoomSDK:
        def __init__(self, **kw):
            self.results_dir = tmp_path

        def probe_hardware(self):
            raise RuntimeError("boom")

    import airbench.sdk as sdkmod

    monkeypatch.setattr(sdkmod, "BenchSDK", BoomSDK)
    assert cli.main(["probe"]) == 1
