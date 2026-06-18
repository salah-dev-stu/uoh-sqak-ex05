"""Phase 3 — metrics: timing (TTFT/TPOT), memory, paging, roofline, aggregate."""

from __future__ import annotations

import math

import pytest

from airbench.metrics import aggregate, memory, paging, roofline, timing

# ----------------------------- timing -----------------------------


def test_ttft_and_tpot_split():
    # submit at t=0; first token at 2.0 (TTFT); then +0.5 each (TPOT)
    rt = timing.measure_once(0.0, [2.0, 2.5, 3.0, 3.5])
    assert rt.ttft_s == 2.0
    assert rt.tpot_s == pytest.approx(0.5)
    assert rt.n_tokens == 4
    assert rt.throughput_tps == pytest.approx(4 / 3.5)


def test_single_token_tpot_none():
    rt = timing.measure_once(0.0, [1.0])
    assert rt.tpot_s is None


def test_empty_tokens_raises():
    with pytest.raises(ValueError):
        timing.measure_once(0.0, [])


def test_measure_generation_discards_warmup():
    seq = iter([(0.0, [9.0, 9.5]), (0.0, [1.0, 1.5]), (0.0, [2.0, 2.5])])
    stats = timing.measure_generation(lambda: next(seq), n_runs=2, warmup=1)
    assert len(stats.runs) == 2  # first (warmup) dropped
    s = stats.summary()
    assert s["ttft_s"]["mean"] == pytest.approx(1.5)  # (1.0 + 2.0)/2
    assert s["ttft_s"]["std"] >= 0


def test_summary_handles_all_none_tpot():
    stats = timing.TimingStats([timing.RunTiming(1.0, None, 5.0, 1)])
    assert stats.summary()["tpot_s"]["n"] == 0
    assert math.isnan(stats.summary()["tpot_s"]["mean"])


# ----------------------------- memory -----------------------------

VM_STAT = """Mach Virtual Memory Statistics: (page size of 16384 bytes)
Pages free:                  10000.
Pages active:                50000.
Pages wired down:            20000.
"""


def test_peak_rss_injected_proc():
    class P:
        def memory_info(self):
            return type("M", (), {"rss": 1024 * 1024 * 512})()

    assert memory.peak_rss_mb(P()) == pytest.approx(512)


def test_parse_vm_stat_to_mb():
    cats = memory.parse_vm_stat(VM_STAT)
    assert cats["free"] == pytest.approx(10000 * 16384 / 1024 / 1024)
    assert "active" in cats


def test_sample_unified_memory(monkeypatch):
    class FakeGK:
        def run_subprocess(self, argv):
            return type("R", (), {"stdout_tail": VM_STAT})()

    class P:
        def memory_info(self):
            return type("M", (), {"rss": 1024 * 1024 * 100})()

    s = memory.sample_unified_memory(FakeGK(), rss_proc=P())
    assert s.rss_mb == pytest.approx(100)
    assert s.free_mb > 0


# ----------------------------- paging -----------------------------

BEFORE = "Pageins: 100.\nSwapins: 5.\nSwapouts: 2.\nPageouts: 10.\n"
AFTER = "Pageins: 250.\nSwapins: 30.\nSwapouts: 2.\nPageouts: 40.\n"


def test_paging_delta():
    d = paging.delta(BEFORE, AFTER)
    assert d["pageins"] == 150
    assert d["swapins"] == 25
    assert d["swapouts"] == 0
    assert d["pageouts"] == 30


def test_paging_tolerates_missing():
    d = paging.delta("garbage", AFTER)
    assert d["swapins"] == 30  # before missing → treated as 0


# ----------------------------- roofline -----------------------------


def test_intensity_and_classify_memory_bound():
    # decode FP16: intensity ~1 FLOP/byte → memory-bound on M2 (ridge = 2840/100 = 28.4)
    flops = roofline.flops_per_token(7.62)
    byts = roofline.bytes_per_token(7.62, 2)
    pt = roofline.evaluate(flops, byts, achieved_gflops=80, peak_gflops=2840, bw_gbps=100)
    assert pt.intensity == pytest.approx(1.0)
    assert pt.bound == "memory-bound"
    assert pt.attainable_gflops == pytest.approx(100.0)  # bw * intensity


def test_classify_compute_bound():
    assert roofline.classify(50.0, 2840, 100) == "compute-bound"  # 50 > ridge 28.4


def test_ridge_point_boundary():
    ridge = roofline.ridge_intensity(2840, 100)
    assert roofline.classify(ridge, 2840, 100) == "compute-bound"  # >= ridge


def test_intensity_zero_bytes_raises():
    with pytest.raises(ValueError):
        roofline.arithmetic_intensity(10, 0)


# ----------------------------- aggregate -----------------------------


def test_json_round_trip(tmp_path):
    p = aggregate.write_json(tmp_path / "a" / "m.json", {"b": 1, "a": 2})
    assert p.exists()
    assert aggregate.read_json(p) == {"b": 1, "a": 2}


def test_csv_round_trip(tmp_path):
    rows = [{"level": "Q4", "ppl": 7.1}, {"level": "Q2", "ppl": 9.9}]
    p = aggregate.write_csv(tmp_path / "m.csv", rows)
    text = p.read_text()
    assert "level,ppl" in text
    assert "Q4,7.1" in text


def test_csv_empty(tmp_path):
    p = aggregate.write_csv(tmp_path / "e.csv", [])
    assert p.read_text() == ""
