"""CLI subcommand handlers — each takes (sdk, args) and calls one SDK method (R1)."""

from __future__ import annotations

from typing import Any


def _probe(sdk, args) -> Any:
    return sdk.probe_hardware()


def _baseline_oom(sdk, args) -> Any:
    return sdk.run_baseline_oom()


def _baseline(sdk, args) -> Any:
    return sdk.run_baseline_llamacpp(args.model_path, args.quant)


def _quant_sweep(sdk, args) -> Any:
    from airbench.sdk.realruns import quant_sweep_args

    return sdk.run_quant_sweep(*quant_sweep_args(sdk))


def _airllm(sdk, args) -> Any:
    from airbench.sdk.realruns import airllm_generate

    return sdk.run_airllm(generate=airllm_generate(sdk))


def _layered(sdk, args) -> Any:
    from airbench.sdk.realruns import shard_layered_args

    return sdk.run_layered_demo(*shard_layered_args(sdk))


def _extreme(sdk, args) -> Any:
    from airbench.sdk.realruns import airllm_generate

    return sdk.run_extreme(generate=airllm_generate(sdk))


def _lora(sdk, args) -> Any:
    return sdk.run_lora()


def _economics(sdk, args) -> Any:
    return sdk.compute_economics()


def _figures(sdk, args) -> Any:
    return sdk.make_figures()


def _all(sdk, args) -> Any:
    """The no-extra-arg safe subset, for a quick end-to-end smoke (real runs use the per-command forms)."""
    return {
        "hardware": sdk.probe_hardware(),
        "economics": sdk.compute_economics(),
        "figures": [str(p) for p in sdk.make_figures()],
    }


HANDLERS = {
    "probe": _probe,
    "baseline-oom": _baseline_oom,
    "baseline": _baseline,
    "quant-sweep": _quant_sweep,
    "airllm": _airllm,
    "layered": _layered,
    "extreme": _extreme,
    "lora": _lora,
    "economics": _economics,
    "figures": _figures,
    "all": _all,
}

# subcommands that take extra CLI arguments
NEEDS_MODEL_PATH = {"baseline"}
