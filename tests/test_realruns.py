"""Phase 7 — real-run closure builders (Gatekeeper mocked; no real models)."""

from __future__ import annotations

from pathlib import Path

from airbench.sdk import realruns


class FakeGK:
    def __init__(self, stdout="Final estimate: PPL = 7.5 +/- 0.1"):
        self.stdout = stdout
        self.downloads = []

    def download(self, repo, filename, dest, size_gb=None):
        self.downloads.append((repo, filename, size_gb))
        return Path(dest) / filename

    def run_subprocess(self, argv, timeout=None):
        return type("R", (), {"stdout_tail": self.stdout, "returncode": 0, "timed_out": False})()


class FakeSDK:
    def __init__(self):
        self.gatekeeper = FakeGK()


def test_quant_sweep_download_and_perplexity():
    sdk = FakeSDK()
    download, bench, perplexity, cleanup = realruns.quant_sweep_args(sdk)
    lvl = {"gguf_file": "Qwen2.5-7B-Instruct-Q4_K_M.gguf", "approx_gb": 4.7, "name": "Q4_K_M"}
    path = download(lvl)
    assert path.endswith("Q4_K_M.gguf")
    assert sdk.gatekeeper.downloads[0][2] == 4.7  # size_gb forwarded for budget gating
    assert perplexity(lvl, path) == 7.5
    rm = bench(lvl, path)
    assert rm.label == "Q4_K_M"


def test_quant_sweep_cleanup_removes_file(tmp_path):
    sdk = FakeSDK()
    _, _, _, cleanup = realruns.quant_sweep_args(sdk)
    f = tmp_path / "w.gguf"
    f.write_bytes(b"x")
    cleanup(str(f))
    assert not f.exists()
    cleanup(str(tmp_path / "missing.gguf"))  # no error on absent


def test_shard_layered_args_structure():
    sdk = FakeSDK()
    n, load_block, forward, free_block, clock, ssd_bps = realruns.shard_layered_args(sdk)
    assert isinstance(n, int)  # 0 shards if weights dir absent
    assert callable(load_block) and callable(forward) and callable(free_block)
    assert ssd_bps > 0


def test_airllm_generate_returns_callable():
    assert callable(realruns.airllm_generate(FakeSDK()))
