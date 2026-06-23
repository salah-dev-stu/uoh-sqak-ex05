#!/usr/bin/env python3
"""Run the REAL QLoRA fine-tune on the Mac and capture artifacts (H9).

    uv sync --extra heavy
    uv run python scripts/run_lora.py
Memory-monitored; if 1.5B is too heavy, set config/lora.json "base" to the 0.5B fallback and re-run.
"""

from __future__ import annotations

import sys
from pathlib import Path

from airbench import figures
from airbench.metrics.aggregate import write_json
from airbench.runners import lora
from airbench.sdk import BenchSDK
from airbench.shared import config


def _gen(sdk, model, prompt, adapter=None):
    argv = lora.build_generate_argv(model, prompt, adapter_path=adapter, max_tokens=80)
    return sdk.gatekeeper.run_subprocess(argv, timeout=600).stdout_tail


def main() -> int:
    sdk = BenchSDK(run_id="real")
    cfg = config.get_lora()
    model = cfg["base"]
    out = Path("results/real/lora")
    out.mkdir(parents=True, exist_ok=True)

    print(f"[1/3] before-generation ({model})")
    before = _gen(sdk, model, cfg["eval_prompt"])
    (out / "before.txt").write_text(before, encoding="utf-8")

    print("[2/3] QLoRA training (mlx_lm.lora) ...")
    result = sdk.run_lora()
    if not hasattr(result, "train_loss"):  # ConstraintReport
        print(f"constraint: {result}")
        return 1
    (out / "loss.log").write_text(str(result.to_dict()), encoding="utf-8")

    print("[3/3] after-generation (with adapter)")
    after = _gen(sdk, model, cfg["eval_prompt"], adapter=result.adapter_path)
    (out / "after.txt").write_text(after, encoding="utf-8")

    write_json(out / "metrics.json", result.to_dict())
    figures.lora_loss("reports/figures/lora_loss.png", result.iters, result.train_loss, result.val)
    print(f"done -> {out}  (trainable {result.trainable_pct}% , peak {result.peak_mem_gb} GB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
