"""Builders that assemble the REAL-run closures from config + the Gatekeeper.

Everything external is routed through ``sdk.gatekeeper`` (download / run_subprocess);
the only direct OS call is ``os.remove`` to free a weight between quant levels.
The device-specific closures (perplexity, generate, shard streaming) are validated
on the Mac in Phase 9; unit tests mock the SDK methods that consume them.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path

from airbench.runners import baseline_llamacpp
from airbench.runners.llamacpp_parser import parse_perplexity
from airbench.runners.run_types import RunMetrics
from airbench.shared import config


def quant_sweep_args(sdk) -> tuple[Callable, Callable, Callable, Callable]:
    """Return (download, bench, perplexity, cleanup) for the serial GGUF sweep."""
    model, rt = config.get_model("primary"), config.get_runtime()
    repo, wdir = model["gguf_repo"], model["weights_dir"]

    def download(lvl: dict) -> str:
        return str(
            sdk.gatekeeper.download(repo, lvl["gguf_file"], wdir, size_gb=lvl.get("approx_gb"))
        )

    def bench(lvl: dict, path: str) -> RunMetrics:
        return baseline_llamacpp.run(sdk.gatekeeper, path, rt["prompts"][0], rt, label=lvl["name"])

    def perplexity(lvl: dict, path: str) -> float | None:
        argv = [
            rt.get("llamacpp_perplexity_bin", "llama-perplexity"),
            "-m",
            path,
            "-f",
            rt["perplexity_corpus"],
        ]
        res = sdk.gatekeeper.run_subprocess(argv, timeout=rt.get("timeout_s"))
        return parse_perplexity(res.stdout_tail)

    def cleanup(path: str) -> None:
        if path and os.path.exists(path):
            os.remove(path)

    return download, bench, perplexity, cleanup


def shard_layered_args(sdk, clock=None) -> tuple:
    """Layer-streaming demo over the model's on-disk safetensors shards (real SSD IO)."""
    import time

    hw = config.get_hardware()
    model = config.get_model("primary")
    shards = sorted(Path(model["weights_dir"]).glob("*.safetensors"))

    def load_block(i: int) -> object:
        from safetensors import safe_open

        data = {}
        with safe_open(str(shards[i]), framework="pt") as f:  # reads the shard from SSD
            for k in f.keys():  # noqa: SIM118
                data[k] = f.get_tensor(k)
        return _Block(data, shards[i].stat().st_size)

    def forward(block: object) -> object:
        return block  # the demo measures IO; a trivial touch stands in for compute

    def free_block(block: object) -> None:
        block.data.clear()

    ssd_bps = hw.get("ssd_read_mbps", 0) * 1_000_000
    return len(shards), load_block, forward, free_block, (clock or time.perf_counter), ssd_bps


class _Block:
    def __init__(self, data: dict, nbytes: int):
        self.data = data
        self.nbytes = nbytes


def airllm_generate(sdk) -> Callable[[object], RunMetrics]:
    """Return generate(model) -> RunMetrics measuring TTFT/TPOT (validated in Phase 9)."""
    rt = config.get_runtime()

    def generate(model: object) -> RunMetrics:
        from airbench.metrics.timing import measure_once

        def one() -> tuple[float, list[float]]:
            import time

            start = time.perf_counter()
            stamps: list[float] = []
            for _ in model.generate_iter(rt["prompts"][0], max_new_tokens=rt["max_tokens"]):
                stamps.append(time.perf_counter())
            return start, stamps

        start, stamps = one()
        t = measure_once(start, stamps)
        return RunMetrics("airllm", t.ttft_s, t.tpot_s, t.throughput_tps)

    return generate
