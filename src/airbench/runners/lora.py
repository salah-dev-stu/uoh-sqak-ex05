"""QLoRA fine-tune runner — drives mlx_lm.lora through the Gatekeeper (H9).

The 8 GB M2 cannot run a 7B for inference, but it CAN train LoRA adapters on a 4-bit base
(QLoRA) via Apple's mlx-lm. Every external call goes through the Gatekeeper; the heavy mlx
deps live in the `heavy` extra so tests/CI never need them.
"""

from __future__ import annotations

from airbench.runners.mlx_lora_parser import parse_loss, parse_trainable
from airbench.runners.run_types import ConstraintReport, LoraResult


def build_train_argv(cfg: dict, model: str, adapter_dir: str) -> list[str]:
    return [
        "python",
        "-m",
        "mlx_lm",
        "lora",
        "--model",
        model,
        "--train",
        "--data",
        cfg["data_dir"],
        "--iters",
        str(cfg["iters"]),
        "--batch-size",
        str(cfg["batch_size"]),
        "--num-layers",
        str(cfg["num_layers"]),
        "--adapter-path",
        adapter_dir,
        "--max-seq-length",
        str(cfg["max_seq_len"]),
        "--learning-rate",
        str(cfg["learning_rate"]),
        "--steps-per-eval",
        str(cfg["steps_per_eval"]),
    ]


def build_generate_argv(
    model: str, prompt: str, adapter_path: str | None = None, max_tokens: int = 64
) -> list[str]:
    argv = [
        "python",
        "-m",
        "mlx_lm",
        "generate",
        "--model",
        model,
        "--prompt",
        prompt,
        "--max-tokens",
        str(max_tokens),
    ]
    if adapter_path:
        argv += ["--adapter-path", adapter_path]
    return argv


def run_lora(gatekeeper, cfg: dict, model: str) -> LoraResult | ConstraintReport:
    adapter_dir = cfg["adapter_dir"]
    argv = build_train_argv(cfg, model, adapter_dir)
    try:
        res = gatekeeper.run_subprocess(argv, timeout=cfg.get("timeout_s", 3600))
    except Exception as exc:
        return ConstraintReport(reason=f"{type(exc).__name__}: {exc}", fallback="0.5B base")
    loss = parse_loss(res.stdout_tail)
    tr = parse_trainable(res.stdout_tail)
    if res.returncode != 0 and not loss["iters"]:
        return ConstraintReport(reason=f"mlx_lm lora rc={res.returncode}", fallback="0.5B base")
    return LoraResult(
        base_model=model,
        iters=loss["iters"],
        train_loss=loss["train_loss"],
        val=loss["val"],
        trainable_pct=tr["trainable_pct"],
        trainable_m=tr["trainable_m"],
        total_m=tr["total_m"],
        peak_mem_gb=tr["peak_mem_gb"],
        duration_s=getattr(res, "duration_s", None),
        adapter_path=adapter_dir,
    )
