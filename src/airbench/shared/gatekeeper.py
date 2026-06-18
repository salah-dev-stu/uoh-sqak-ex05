"""ApiGatekeeper — the single chokepoint for EVERY external interaction (R3).

Model downloads, llama.cpp subprocesses, in-process model loads, and API price
lookups all route through here. It GATES (budget + network enforcement) and
RECORDS (an append-only JSON ledger), so the committed audit trail proves the
Gatekeeper is wired, not decorative. No other module may call subprocess /
requests / hf_hub_download directly — a meta-test enforces this.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from airbench.shared.gatekeeper_types import BudgetExceededError, GateEvent, RunResult
from airbench.shared.logging_config import get_logger

_SECRET_ENV = ("HF_TOKEN", "OPENAI_API_KEY")


class ApiGatekeeper:
    def __init__(self, budgets: dict[str, Any], ledger_path: str | Path, clock=time.perf_counter):
        self.budgets = budgets
        self.ledger_path = Path(ledger_path)
        self._events: list[GateEvent] = []
        self._api_calls = 0
        self._clock = clock
        self._log = get_logger("gatekeeper")

    def _require_network(self, label: str) -> None:
        if not self.budgets.get("allow_network", True):
            raise BudgetExceededError(f"network disabled by budget; blocked: {label}")

    def _redact(self, text: str) -> str:
        for env in _SECRET_ENV:
            val = os.environ.get(env)
            if val:
                text = text.replace(val, "***")
        return text

    def record(self, event: GateEvent) -> None:
        event.label = self._redact(event.label)
        event.detail = self._redact(event.detail)
        self._events.append(event)
        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.ledger_path.with_suffix(".tmp")
        tmp.write_text(json.dumps([e.to_dict() for e in self._events], indent=2), encoding="utf-8")
        os.replace(tmp, self.ledger_path)

    @property
    def ledger(self) -> list[GateEvent]:
        return list(self._events)

    def download(self, repo: str, filename: str, dest, *, size_gb=None, fetcher=None) -> Path:
        self._require_network(f"download {repo}/{filename}")
        cap = self.budgets.get("max_download_gb")
        if size_gb is not None and cap is not None and size_gb > cap:
            raise BudgetExceededError(f"download {size_gb}GB exceeds max_download_gb={cap}")
        fetcher = fetcher or _default_fetcher
        start = self._clock()
        path = Path(fetcher(repo, filename, dest))
        dur = self._clock() - start
        nbytes = path.stat().st_size if path.exists() else 0
        self.record(
            GateEvent(
                f"download {repo}/{filename}",
                "download",
                dur,
                True,
                bytes_in=nbytes,
                detail=str(dest),
            )
        )
        return path

    def run_subprocess(self, argv: list[str], *, timeout=None, runner=None) -> RunResult:
        cap = self.budgets.get("max_subprocess_s")
        if cap:
            timeout = min(timeout or cap, cap)
        runner = runner or _default_runner
        start = self._clock()
        try:
            rc, out = runner(argv, timeout)
            timed_out = False
        except subprocess.TimeoutExpired:
            rc, out, timed_out = -1, "", True
        dur = self._clock() - start
        ok = rc == 0 and not timed_out
        self.record(
            GateEvent(
                " ".join(argv[:3]),
                "subprocess",
                dur,
                ok,
                detail=("TIMEOUT" if timed_out else f"rc={rc}"),
            )
        )
        return RunResult(rc, (out or "")[-2000:], dur, timed_out, list(argv))

    def load_model(self, loader: Callable[[], Any], label: str) -> Any:
        start = self._clock()
        try:
            obj = loader()
        except Exception as exc:
            dur = self._clock() - start
            self.record(
                GateEvent(label, "load_model", dur, False, detail=f"{type(exc).__name__}: {exc}")
            )
            raise
        self.record(GateEvent(label, "load_model", self._clock() - start, True))
        return obj

    def api_call(self, fn: Callable[[], Any], label: str) -> Any:
        cap = self.budgets.get("max_api_calls")
        if cap is not None and self._api_calls >= cap:
            raise BudgetExceededError(f"api_call budget {cap} exhausted; blocked: {label}")
        self._require_network(label)
        self._api_calls += 1
        start = self._clock()
        result = fn()
        self.record(
            GateEvent(label, "api_call", self._clock() - start, True, bytes_out=len(str(result)))
        )
        return result


def _default_fetcher(repo: str, filename: str, dest) -> str:
    from huggingface_hub import hf_hub_download

    return hf_hub_download(repo_id=repo, filename=filename, local_dir=str(dest))


def _default_runner(argv: list[str], timeout):
    proc = subprocess.run(argv, capture_output=True, text=True, timeout=timeout, check=False)
    return proc.returncode, (proc.stdout or "") + (proc.stderr or "")
