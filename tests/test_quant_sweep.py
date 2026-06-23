"""Phase 4 — quant sweep + extreme runners (split from test_runners.py; all mocked)."""

from __future__ import annotations

from airbench.runners import extreme, quant_sweep
from airbench.runners.run_types import ConstraintReport, RunMetrics


class FakeGK:
    def __init__(self, raise_exc=None):
        self.raise_exc = raise_exc

    def load_model(self, loader, label):
        if self.raise_exc:
            raise self.raise_exc
        return loader()


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
    assert out["red_line"] == "Q2_K"  # first level with ppl - min(7.0) > 0.5
    # every download is followed by a cleanup before the next (≤1 weight at a time)
    assert [e[0] for e in events] == ["dl", "rm", "dl", "rm", "dl", "rm", "dl", "rm"]


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


def test_extreme_tags_metrics():
    out = extreme.run_extreme(FakeGK(), lambda: "m", lambda m: RunMetrics("70b"))
    assert out.extra["extreme"] is True


def test_extreme_constraint_passthrough():
    gk = FakeGK(raise_exc=RuntimeError("CUDA required"))
    out = extreme.run_extreme(gk, lambda: None, lambda m: RunMetrics("x"))
    assert isinstance(out, ConstraintReport)
