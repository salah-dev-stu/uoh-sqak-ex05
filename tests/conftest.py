"""Shared test fixtures. No GPU/model/key ever required — everything is mocked."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import pytest


@pytest.fixture
def write_config(tmp_path: Path, monkeypatch) -> Callable[[dict[str, dict]], Path]:
    """Write a {name: data} map of configs to a tmp dir and point AIRBENCH_CONFIG_DIR at it."""

    def _write(configs: dict[str, dict]) -> Path:
        cfg = tmp_path / "config"
        cfg.mkdir(exist_ok=True)
        for name, data in configs.items():
            (cfg / f"{name}.json").write_text(json.dumps(data), encoding="utf-8")
        monkeypatch.setenv("AIRBENCH_CONFIG_DIR", str(cfg))
        return cfg

    return _write


class FakeClock:
    """Deterministic clock: each call advances by 1.0s (so one gated op measures 1.0s)."""

    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        self.t += 1.0
        return self.t


@pytest.fixture
def fake_clock() -> FakeClock:
    return FakeClock()


@pytest.fixture
def budgets() -> dict:
    return {
        "max_download_gb": 60.0,
        "max_subprocess_s": 1800,
        "max_api_calls": 3,
        "allow_network": True,
    }
