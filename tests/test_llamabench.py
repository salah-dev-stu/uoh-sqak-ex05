"""llama-bench JSON parsing (the robust real-run engine)."""

from __future__ import annotations

import pytest

from airbench.runners import llamabench

# stderr Metal logs get mixed with the stdout JSON array by the gatekeeper runner:
SAMPLE = """ggml_metal_device_init: use shared buffers = true
load_backend: loaded MTL backend
[
  {"n_prompt": 64, "n_gen": 0, "avg_ts": 130.85, "stddev_ts": 12.36},
  {"n_prompt": 0, "n_gen": 64, "avg_ts": 17.61, "stddev_ts": 0.27}
]
"""


def test_build_argv_has_no_mmap_and_json():
    argv = llamabench.build_argv("llama-bench", "m.gguf", 64, 64, 3)
    assert "-mmp" in argv and argv[argv.index("-mmp") + 1] == "0"
    assert "-o" in argv and "json" in argv
    assert "-ngl" in argv


def test_parse_splits_prefill_and_decode():
    out = llamabench.parse_json(SAMPLE)
    assert out["prefill_tps"] == {"mean": 130.85, "std": 12.36}
    assert out["decode_tps"] == {"mean": 17.61, "std": 0.27}
    assert out["ttft_s"] == pytest.approx(64 / 130.85, rel=1e-3)
    assert out["tpot_ms"] == pytest.approx(1000 / 17.61, rel=1e-3)
    assert out["throughput_tps"] == 17.61


def test_parse_no_array_raises():
    with pytest.raises(ValueError):
        llamabench.parse_json("no json here")
