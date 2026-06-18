"""R3 — the Gatekeeper gates AND records every external call."""

from __future__ import annotations

import json
import subprocess

import pytest

from airbench.shared.gatekeeper import ApiGatekeeper
from airbench.shared.gatekeeper_types import BudgetExceededError


def _gk(tmp_path, budgets, fake_clock):
    return ApiGatekeeper(budgets, tmp_path / "run" / "gate_ledger.json", clock=fake_clock)


def test_download_happy_records_event(tmp_path, budgets, fake_clock):
    gk = _gk(tmp_path, budgets, fake_clock)

    def fetcher(repo, filename, dest):
        p = tmp_path / filename
        p.write_bytes(b"x" * 100)
        return p

    out = gk.download("repo/x", "w.gguf", tmp_path, size_gb=1.0, fetcher=fetcher)
    assert out.exists()
    assert len(gk.ledger) == 1
    ev = gk.ledger[0]
    assert ev.kind == "download" and ev.ok and ev.bytes_in == 100
    assert ev.duration_s == 1.0


def test_download_denies_over_budget(tmp_path, budgets, fake_clock):
    gk = _gk(tmp_path, budgets, fake_clock)
    with pytest.raises(BudgetExceededError, match="max_download_gb"):
        gk.download("r", "f", tmp_path, size_gb=999.0, fetcher=lambda *a: tmp_path)


def test_subprocess_records_rc_and_tail(tmp_path, budgets, fake_clock):
    gk = _gk(tmp_path, budgets, fake_clock)
    res = gk.run_subprocess(["echo", "hi"], runner=lambda argv, t: (0, "hello world"))
    assert res.returncode == 0
    assert res.stdout_tail == "hello world"
    assert gk.ledger[0].ok is True


def test_subprocess_timeout(tmp_path, budgets, fake_clock):
    gk = _gk(tmp_path, budgets, fake_clock)

    def runner(argv, t):
        raise subprocess.TimeoutExpired(argv, t)

    res = gk.run_subprocess(["sleep", "9"], timeout=1, runner=runner)
    assert res.timed_out is True
    assert gk.ledger[0].ok is False and gk.ledger[0].detail == "TIMEOUT"


def test_subprocess_timeout_capped_by_budget(tmp_path, fake_clock):
    captured = {}

    def runner(argv, t):
        captured["t"] = t
        return 0, ""

    gk = _gk(tmp_path, {"max_subprocess_s": 5, "allow_network": True}, fake_clock)
    gk.run_subprocess(["x"], timeout=999, runner=runner)
    assert captured["t"] == 5  # clamped to budget


def test_load_model_success(tmp_path, budgets, fake_clock):
    gk = _gk(tmp_path, budgets, fake_clock)
    obj = gk.load_model(lambda: "MODEL", "load qwen")
    assert obj == "MODEL"
    assert gk.ledger[0].kind == "load_model" and gk.ledger[0].ok


def test_load_model_failure_records_then_raises(tmp_path, budgets, fake_clock):
    gk = _gk(tmp_path, budgets, fake_clock)

    def loader():
        raise RuntimeError("CUDA not available")

    with pytest.raises(RuntimeError):
        gk.load_model(loader, "load airllm")
    assert gk.ledger[0].ok is False
    assert "CUDA" in gk.ledger[0].detail


def test_api_call_counts_and_caps(tmp_path, fake_clock):
    gk = _gk(tmp_path, {"max_api_calls": 2, "allow_network": True}, fake_clock)
    gk.api_call(lambda: "a", "price1")
    gk.api_call(lambda: "b", "price2")
    with pytest.raises(BudgetExceededError, match="budget"):
        gk.api_call(lambda: "c", "price3")
    assert len(gk.ledger) == 2


def test_network_disabled_blocks(tmp_path, fake_clock):
    gk = _gk(tmp_path, {"allow_network": False, "max_api_calls": 9}, fake_clock)
    with pytest.raises(BudgetExceededError, match="network disabled"):
        gk.download("r", "f", tmp_path, fetcher=lambda *a: tmp_path)
    with pytest.raises(BudgetExceededError, match="network disabled"):
        gk.api_call(lambda: 1, "p")


def test_ledger_persisted_and_append(tmp_path, budgets, fake_clock):
    gk = _gk(tmp_path, budgets, fake_clock)
    gk.load_model(lambda: 1, "one")
    gk.load_model(lambda: 2, "two")
    saved = json.loads((tmp_path / "run" / "gate_ledger.json").read_text())
    assert len(saved) == 2
    assert [e["label"] for e in saved] == ["one", "two"]


def test_secret_redaction(tmp_path, budgets, fake_clock, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_SECRET123")
    gk = _gk(tmp_path, budgets, fake_clock)
    gk.load_model(lambda: 1, "load with hf_SECRET123 in label")
    assert "hf_SECRET123" not in gk.ledger[0].label
    assert "***" in gk.ledger[0].label


def _counter_clock():
    t = [0.0]

    def c():
        t[0] += 1.0
        return t[0]

    return c


def test_separate_gatekeepers_separate_ledgers(tmp_path, budgets):
    a = ApiGatekeeper(budgets, tmp_path / "a.json", clock=_counter_clock())
    b = ApiGatekeeper(budgets, tmp_path / "b.json", clock=_counter_clock())
    a.load_model(lambda: 1, "a-evt")
    assert len(a.ledger) == 1 and len(b.ledger) == 0
