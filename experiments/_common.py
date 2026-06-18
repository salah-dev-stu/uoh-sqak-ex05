"""Shared helper so each experiment stays a thin, DRY entrypoint over the SDK."""

from __future__ import annotations

from types import SimpleNamespace

from airbench.cli_handlers import HANDLERS
from airbench.sdk import BenchSDK


def run(command: str, sdk=None, **arg_kwargs):
    """Build (or accept) a BenchSDK and dispatch the named command via its handler."""
    sdk = sdk or BenchSDK()
    args = SimpleNamespace(**arg_kwargs)
    result = HANDLERS[command](sdk, args)
    print(f"[{command}] -> {sdk.results_dir}")
    return result
