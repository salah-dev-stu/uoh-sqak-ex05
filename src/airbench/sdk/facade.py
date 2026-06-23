"""BenchSDK — the single public entry point (R1).

CLI and experiment scripts touch only this façade; it owns the shared Gatekeeper,
a per-run results directory, and delegates to runners / metrics / economics.
"""

from __future__ import annotations

import os
import time
import uuid
from pathlib import Path
from typing import Any

from airbench import figures
from airbench.economics import calculator
from airbench.metrics.aggregate import write_json
from airbench.metrics.memory import peak_rss_mb
from airbench.runners import baseline_llamacpp, baseline_oom, extreme, layered, lora, quant_sweep
from airbench.runners.airllm import run_airllm
from airbench.sdk import loaders, probe
from airbench.shared import config
from airbench.shared.gatekeeper import ApiGatekeeper
from airbench.shared.logging_config import get_logger


def _new_run_id() -> str:
    return f"run-{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"


class BenchSDK:
    def __init__(
        self,
        config_dir: str | Path | None = None,
        run_id: str | None = None,
        results_root: str | Path = "results",
    ):
        if config_dir:
            os.environ["AIRBENCH_CONFIG_DIR"] = str(config_dir)
        self.run_id = run_id or _new_run_id()
        self.results_dir = Path(results_root) / self.run_id
        self.gatekeeper = ApiGatekeeper(config.get_budgets(), self.results_dir / "gate_ledger.json")
        self.log = get_logger("sdk")

    def _save(self, name: str, data: Any) -> Path:
        return write_json(self.results_dir / name, data)

    def probe_hardware(self) -> dict[str, Any]:
        report = probe.hardware_report()
        self._save("hardware.json", report)
        return report

    def run_baseline_oom(self, loader=None) -> Any:
        loader = loader or loaders.hf_fp16_loader(config.get_model("primary"))
        rep = baseline_oom.run_oom(self.gatekeeper, loader, peak_mem_fn=peak_rss_mb)
        self._save("baseline/oom.json", rep.to_dict())
        return rep

    def run_baseline_llamacpp(self, model_path: str, quant: str = "Q4_K_M", ctx=None) -> Any:
        rt = config.get_runtime()
        rm = baseline_llamacpp.run(
            self.gatekeeper, model_path, rt["prompts"][0], rt, ctx=ctx, label=f"llama.cpp {quant}"
        )
        self._save(f"baseline/{quant}.json", rm.to_dict())
        return rm

    def run_layered_demo(
        self,
        n_layers,
        load_block,
        forward,
        free_block,
        clock=time.perf_counter,
        ssd_bytes_per_s=None,
    ) -> Any:
        m = layered.run_layered(n_layers, load_block, forward, free_block, clock, ssd_bytes_per_s)
        self._save("airllm/layered.json", m.to_dict())
        return m

    def run_airllm(self, generate, loader=None, label="airllm") -> Any:
        loader = loader or loaders.airllm_loader(config.get_model("primary"))
        rt = config.get_runtime()
        out = run_airllm(self.gatekeeper, loader, generate, rt.get("airllm_timebox_min", 60), label)
        self._save("airllm/airllm.json", out.to_dict())
        return out

    def run_quant_sweep(self, download, bench, perplexity, cleanup) -> dict[str, Any]:
        rt = config.get_runtime()
        out = quant_sweep.run_sweep(
            config.get_quant_levels(),
            download,
            bench,
            perplexity,
            cleanup,
            rt.get("red_line_ppl_delta", 0.5),
        )
        self._save("quant/sweep.json", out)
        return out

    def run_extreme(self, generate, loader=None) -> Any:
        loader = loader or loaders.airllm_loader(config.get_model("extreme"))
        rt = config.get_runtime()
        out = extreme.run_extreme(
            self.gatekeeper, loader, generate, rt.get("airllm_timebox_min", 60)
        )
        self._save("extreme/extreme.json", out.to_dict())
        return out

    def run_lora(self) -> Any:
        cfg = config.get_lora()
        out = lora.run_lora(self.gatekeeper, cfg, cfg["base"])
        self._save("lora/metrics.json", out.to_dict())
        return out

    def compute_economics(self) -> dict[str, Any]:
        report = calculator.compute(config.get_economics())
        self._save("economics.json", report)
        return report

    def make_figures(self, out_dir="reports/figures") -> list[Path]:
        return figures.build_all(self.results_dir, out_dir)
