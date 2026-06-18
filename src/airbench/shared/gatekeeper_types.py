"""Data types for the Gatekeeper ledger (kept separate so gatekeeper.py stays ≤150 lines)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


class BudgetExceededError(RuntimeError):
    """Raised when a call would exceed a configured budget (download size, api calls, network off)."""


@dataclass
class GateEvent:
    """One recorded external interaction. Forms the audit ledger proving the Gatekeeper is wired."""

    label: str
    kind: str  # "download" | "subprocess" | "load_model" | "api_call"
    duration_s: float
    ok: bool
    bytes_in: int = 0
    bytes_out: int = 0
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "kind": self.kind,
            "duration_s": round(self.duration_s, 6),
            "ok": self.ok,
            "bytes_in": self.bytes_in,
            "bytes_out": self.bytes_out,
            "detail": self.detail,
        }


@dataclass
class RunResult:
    """Result of a gated subprocess call."""

    returncode: int
    stdout_tail: str
    duration_s: float
    timed_out: bool = False
    argv: list[str] = field(default_factory=list)
