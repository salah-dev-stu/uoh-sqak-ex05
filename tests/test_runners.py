"""Phase 4 — runners (all externals mocked: no GPU/model/key)."""

from __future__ import annotations

import pytest

from airbench.runners import (
    airllm,
    baseline_llamacpp,
    baseline_oom,
    extreme,
    layered,
    llamacpp_parser,
    quant_sweep,
)
from airbench.runners.run_types import ConstraintReport, RunMetrics

LLAMA_OUT = """
llama_print_timings: prompt eval time =  1200.00 ms /    10 tokens (  120.00 ms per token,     8.33 tokens per second)
llama_print_timings:        eval time =  5670.00 ms /    63 runs   (   90.00 ms per token,    11.11 tokens per second)
"""

PPL_OUT = "[1]6.90,[2]7.05\nFinal estimate: PPL = 7.1234 +/- 0.05\n"


class FakeGK:
    """Minimal gatekeeper: load_model runs the loader; run_subprocess returns canned stdout."""

    def __init__(self, stdout="", rc=0, raise_exc=None):
        self.stdout, self.rc, self.raise_exc = stdout, rc, raise_exc

    def load_model(self, loader, label):
        if self.raise_exc:
            raise self.raise_exc
        return loader()

    def run_subprocess(self, argv, timeout=None):
        return type(
            "R", (), {"stdout_tail": self.stdout, "returncode": self.rc, "timed_out": False}
        )()


# ---------------------- parser ----------------------


def test_parse_metrics():
    m = llamacpp_parser.parse_metrics(LLAMA_OUT)
    assert m["prefill_ms"] == 1200.0 and m["prefill_tokens"] == 10
    assert m["tpot_ms"] == 90.0 and m["decode_tps"] == 11.11


def test_parse_metrics_missing():
    m = llamacpp_parser.parse_metrics("nothing here")
    assert m["prefill_ms"] is None and m["tpot_ms"] is None


def test_parse_perplexity():
    assert llamacpp_parser.parse_perplexity(PPL_OUT) == 7.1234


def test_parse_perplexity_absent():
    assert llamacpp_parser.parse_perplexity("no ppl") is None


# ---------------------- baseline_oom ----------------------


def test_oom_captures_failure():
    gk = FakeGK(raise_exc=RuntimeError("MPS backend out of memory"))
    rep = baseline_oom.run_oom(gk, lambda: None, peak_mem_fn=lambda: 7800.0)
    assert rep.error_type == "RuntimeError"
    assert "MPS" in rep.diagnosis and rep.peak_mem_mb == 7800.0


def test_oom_host_oom_diagnosis():
    assert "RAM" in baseline_oom.diagnose(MemoryError("cannot allocate memory"))


def test_oom_unexpected_success():
    rep = baseline_oom.run_oom(FakeGK(), lambda: "model")
    assert rep.error_type == "NoError"


# ---------------------- baseline_llamacpp ----------------------


def test_llamacpp_run_parses_timings():
    gk = FakeGK(stdout=LLAMA_OUT)
    runtime = {"max_tokens": 64, "seed": 42, "ctx_sizes": [512], "timeout_s": 60}
    rm = baseline_llamacpp.run(gk, "/Volumes/Backup/m.gguf", "hello", runtime)
    assert rm.ttft_s == pytest.approx(1.2)
    assert rm.tpot_s == pytest.approx(0.09)
    assert rm.throughput_tps == 11.11


def test_build_argv():
    argv = baseline_llamacpp.build_argv("llama-cli", "m.gguf", "hi", 64, 512, 42)
    assert argv[0] == "llama-cli" and "-m" in argv and "m.gguf" in argv


# ---------------------- layered ----------------------


def test_layered_streams_and_times():
    clock = iter([0, 1, 2, 2, 3, 4]).__next__  # 2 layers: io=1, compute=1 each

    class Blk:
        nbytes = 500_000_000

    freed = []
    m = layered.run_layered(
        2,
        load_block=lambda i: Blk(),
        forward=lambda b: None,
        free_block=lambda b: freed.append(b),
        clock=clock,
        ssd_bytes_per_s=500_000_000,
    )
    assert m.n_layers == 2
    assert m.io_s == [1, 1] and m.compute_s == [1, 1]
    assert m.cumulative_io_bytes == 1_000_000_000
    assert m.predicted_io_s == [1.0, 1.0]
    assert len(freed) == 2


# ---------------------- airllm ----------------------


def test_airllm_success_path():
    gk = FakeGK()
    out = airllm.run_airllm(
        gk, lambda: "model", lambda m: RunMetrics("airllm", ttft_s=1.0), timebox_min=45
    )
    assert isinstance(out, RunMetrics)
    assert out.extra["airllm"] is True and out.extra["timebox_min"] == 45


def test_airllm_constraint_on_load_failure():
    gk = FakeGK(raise_exc=ImportError("No module named 'cuda'"))
    out = airllm.run_airllm(gk, lambda: None, lambda m: RunMetrics("x"))
    assert isinstance(out, ConstraintReport)
    assert out.fallback == "layered" and "ImportError" in out.reason


# ---------------------- quant_sweep ----------------------


def test_quant_sweep_serial_delete_and_red_line():
    levels = [{"name": "Q8_0"}, {"name": "Q5_K_M"}, {"name": "Q4_K_M"}, {"name": "Q2_K"}]
    ppls = {"Q8_0": 7.0, "Q5_K_M": 7.1, "Q4_K_M": 7.3, "Q2_K": 9.5}
    events = []

    def download(lvl):
        events.append(("dl", lvl["name"]))
        return f"/Volumes/Backup/{lvl['name']}.gguf"

    def cleanup(path):
        events.append(("rm", path))

    out = quant_sweep.run_sweep(
        levels,
        download=download,
        bench=lambda lvl, p: RunMetrics(lvl["name"], tpot_s=0.1),
        perplexity=lambda lvl, p: ppls[lvl["name"]],
        cleanup=cleanup,
        threshold=0.5,
    )
    # red line: first level with ppl - min(7.0) > 0.5 → Q2_K (9.5)
    assert out["red_line"] == "Q2_K"
    # every download is followed by a cleanup before the next download (≤1 weight at a time)
    kinds = [e[0] for e in events]
    assert kinds == ["dl", "rm", "dl", "rm", "dl", "rm", "dl", "rm"]


def test_quant_sweep_continues_after_failure():
    levels = [{"name": "Q8_0"}, {"name": "Q4_K_M"}]
    cleaned = []

    def bench(lvl, p):
        if lvl["name"] == "Q8_0":
            raise RuntimeError("llama.cpp crashed")
        return RunMetrics("Q4_K_M")

    out = quant_sweep.run_sweep(
        levels,
        download=lambda lvl: "p",
        bench=bench,
        perplexity=lambda lvl, p: 7.0,
        cleanup=lambda p: cleaned.append(p),
        threshold=0.5,
    )
    assert out["levels"][0]["error"] is not None
    assert out["levels"][1]["metrics"]["label"] == "Q4_K_M"
    assert len(cleaned) == 2  # cleanup ran even for the failed level


# ---------------------- extreme ----------------------


def test_extreme_tags_metrics():
    gk = FakeGK()
    out = extreme.run_extreme(gk, lambda: "m", lambda m: RunMetrics("70b"))
    assert out.extra["extreme"] is True


def test_extreme_constraint_passthrough():
    gk = FakeGK(raise_exc=RuntimeError("CUDA required"))
    out = extreme.run_extreme(gk, lambda: None, lambda m: RunMetrics("x"))
    assert isinstance(out, ConstraintReport)
