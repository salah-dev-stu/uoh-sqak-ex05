"""mlx_lm.lora stdout parsing (loss curve + parameter efficiency)."""

from __future__ import annotations

from airbench.runners import mlx_lora_parser

SAMPLE = """Trainable parameters: 0.052% (0.812M/1543.714M)
Iter 1: Val loss 2.345, Val took 1.2s
Iter 10: Train loss 2.100, Learning Rate 1.000e-04, It/sec 0.5, Peak mem 2.30 GB
Iter 25: Val loss 1.800, Val took 1.1s
Iter 150: Train loss 0.500, Learning Rate 1.000e-04, Peak mem 2.45 GB
"""


def test_parse_loss():
    out = mlx_lora_parser.parse_loss(SAMPLE)
    assert out["iters"] == [10, 150]
    assert out["train_loss"] == [2.1, 0.5]
    assert out["val"] == [{"iter": 1, "loss": 2.345}, {"iter": 25, "loss": 1.8}]


def test_parse_trainable():
    t = mlx_lora_parser.parse_trainable(SAMPLE)
    assert t["trainable_pct"] == 0.052
    assert t["trainable_m"] == 0.812 and t["total_m"] == 1543.714
    assert t["peak_mem_gb"] == 2.45


def test_parse_empty():
    out = mlx_lora_parser.parse_loss("nothing")
    assert out["iters"] == [] and out["val"] == []
    assert mlx_lora_parser.parse_trainable("nothing")["trainable_pct"] is None
