"""Heavy model-load CLOSURES (lazy imports).

These build callables that perform the real ``transformers`` / ``airllm`` loads.
They are NEVER called directly — the SDK passes each closure to
``ApiGatekeeper.load_model``, which times, records, and (for the OOM baseline)
captures the failure. So the external load is gated even though the import lives
here. The lazy imports keep torch/transformers/airllm out of tests + CI (Path D).

This module is on the R3 meta-test allow-list precisely because its closures only
ever execute through the Gatekeeper (see docs/adr/ADR-003).
"""

from __future__ import annotations

from collections.abc import Callable


def hf_fp16_loader(model_cfg: dict) -> Callable[[], object]:
    """Closure that loads the model in FP16 onto MPS (expected to OOM 8 GB)."""

    def _load() -> object:
        import torch
        from transformers import AutoModelForCausalLM

        return AutoModelForCausalLM.from_pretrained(
            model_cfg["hf_id"], torch_dtype=torch.float16, device_map="mps"
        )

    return _load


def airllm_loader(model_cfg: dict) -> Callable[[], object]:
    """Closure that loads the model via AirLLM (CUDA-centric; may raise on MPS)."""

    def _load() -> object:
        from airllm import AutoModel

        return AutoModel.from_pretrained(model_cfg["hf_id"])

    return _load
