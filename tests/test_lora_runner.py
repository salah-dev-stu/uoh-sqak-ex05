"""QLoRA runner — argv shapes + run_lora (gatekeeper mocked; no MLX)."""

from __future__ import annotations

from airbench.runners import lora
from airbench.runners.run_types import ConstraintReport, LoraResult

CFG = {
    "data_dir": "data/lora",
    "adapter_dir": "out/ad",
    "iters": 150,
    "batch_size": 1,
    "num_layers": 8,
    "max_seq_len": 512,
    "learning_rate": 0.0001,
    "steps_per_eval": 25,
    "timeout_s": 60,
}
OUT = "Trainable parameters: 0.05% (0.8M/1543.7M)\nIter 10: Train loss 2.0, Peak mem 2.3 GB\n"


class FakeGK:
    def __init__(self, stdout=OUT, rc=0, raise_exc=None):
        self.stdout, self.rc, self.raise_exc = stdout, rc, raise_exc

    def run_subprocess(self, argv, timeout=None):
        if self.raise_exc:
            raise self.raise_exc
        return type(
            "R",
            (),
            {
                "stdout_tail": self.stdout,
                "returncode": self.rc,
                "timed_out": False,
                "duration_s": 12.3,
            },
        )()


def test_build_train_argv():
    argv = lora.build_train_argv(CFG, "mlx-community/Qwen2.5-1.5B-Instruct-4bit", "out/ad")
    assert "lora" in argv and "--train" in argv
    assert "--data" in argv and "data/lora" in argv
    assert "--adapter-path" in argv and "out/ad" in argv
    assert "mlx-community/Qwen2.5-1.5B-Instruct-4bit" in argv


def test_build_generate_argv_with_adapter():
    argv = lora.build_generate_argv("m", "hello", adapter_path="out/ad")
    assert "generate" in argv and "--adapter-path" in argv and "out/ad" in argv


def test_run_lora_success():
    out = lora.run_lora(FakeGK(), CFG, "m")
    assert isinstance(out, LoraResult)
    assert out.train_loss == [2.0] and out.trainable_pct == 0.05
    assert out.peak_mem_gb == 2.3 and out.duration_s == 12.3
    assert out.adapter_path == "out/ad"


def test_run_lora_failure_returns_constraint():
    out = lora.run_lora(FakeGK(raise_exc=RuntimeError("mlx missing")), CFG, "m")
    assert isinstance(out, ConstraintReport)
