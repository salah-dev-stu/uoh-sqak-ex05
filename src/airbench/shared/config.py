"""Config loading + typed accessors (R4/R10).

All run parameters — model ids, quant levels, prompts, budgets, prices, hardware
peaks — live in ``config/*.json``. Nothing is hardcoded at call sites. Each config
is validated against a small schema (required keys must be present; unknown
top-level keys are rejected) so a typo fails loudly instead of silently.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

DEFAULT_DIR = Path(__file__).resolve().parents[3] / "config"

# name -> (required keys, optional keys). Unknown top-level keys are rejected.
_SCHEMA: dict[str, tuple[set[str], set[str]]] = {
    "models": ({"primary"}, {"alternative", "extreme"}),
    "quant_levels": ({"levels"}, set()),
    "runtime": (
        {"n_runs", "warmup", "max_tokens", "seed", "prompts"},
        {
            "ctx_sizes",
            "timeout_s",
            "llamacpp_bin",
            "llamacpp_perplexity_bin",
            "perplexity_corpus",
            "red_line_ppl_delta",
            "airllm_timebox_min",
        },
    ),
    "economics": (
        {"api_providers", "hardware_capex_usd", "amortize_years"},
        {
            "ssd_capex_usd",
            "idle_w",
            "active_w",
            "kwh_usd",
            "hours_active_per_day",
            "avg_req_tokens_in",
            "avg_req_tokens_out",
        },
    ),
    "budgets": ({"max_download_gb", "max_subprocess_s", "max_api_calls", "allow_network"}, set()),
    "hardware": (
        {"ram_gb", "gpu_peak_gflops_fp16", "unified_mem_bandwidth_gbps", "ssd_read_mbps"},
        {
            "chip",
            "model_identifier",
            "cpu_cores",
            "perf_cores",
            "eff_cores",
            "gpu_cores",
            "ssd_write_mbps",
            "weights_volume",
            "notes",
        },
    ),
    "lora": (
        {"base", "data_dir", "adapter_dir", "iters", "batch_size", "num_layers"},
        {
            "fallback",
            "lora_rank",
            "max_seq_len",
            "learning_rate",
            "steps_per_eval",
            "timeout_s",
            "eval_prompt",
        },
    ),
}


def config_dir() -> Path:
    """Config directory, overridable via ``AIRBENCH_CONFIG_DIR`` (used by tests)."""
    return Path(os.environ.get("AIRBENCH_CONFIG_DIR", str(DEFAULT_DIR)))


def load(name: str) -> dict[str, Any]:
    """Load + validate ``config/<name>.json``."""
    path = config_dir() / f"{name}.json"
    if not path.exists():
        raise FileNotFoundError(f"config not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    _validate(name, data)
    return data


def _validate(name: str, data: dict[str, Any]) -> None:
    if name not in _SCHEMA:
        return
    required, optional = _SCHEMA[name]
    missing = required - data.keys()
    if missing:
        raise KeyError(f"config '{name}' missing required key(s): {sorted(missing)}")
    unknown = data.keys() - required - optional
    if unknown:
        raise KeyError(f"config '{name}' has unknown key(s): {sorted(unknown)}")


def get_model(role: str) -> dict[str, Any]:
    models = load("models")
    if role not in models:
        raise KeyError(f"model role not found: {role}")
    return models[role]


def get_runtime() -> dict[str, Any]:
    return load("runtime")


def get_economics() -> dict[str, Any]:
    return load("economics")


def get_budgets() -> dict[str, Any]:
    return load("budgets")


def get_hardware() -> dict[str, Any]:
    return load("hardware")


def get_quant_levels() -> list[dict[str, Any]]:
    return load("quant_levels")["levels"]


def get_lora() -> dict[str, Any]:
    return load("lora")
