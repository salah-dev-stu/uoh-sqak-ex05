"""airbench CLI — operator entry for the experiments. Dispatches to BenchSDK (R1)."""

from __future__ import annotations

import argparse
import sys

from airbench.cli_handlers import HANDLERS, NEEDS_MODEL_PATH
from airbench.shared.version import VERSION


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="airbench", description="AirLLM/quantization benchmarking (EX05)"
    )
    p.add_argument("--version", action="version", version=VERSION)
    p.add_argument("--config-dir", default=None)
    p.add_argument("--run-id", default=None)
    p.add_argument("--results-root", default="results")
    sub = p.add_subparsers(dest="command", required=True)
    for name in HANDLERS:
        sp = sub.add_parser(name)
        if name in NEEDS_MODEL_PATH:
            sp.add_argument("--model-path", required=True)
            sp.add_argument("--quant", default="Q4_K_M")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    from airbench.sdk import BenchSDK

    sdk = BenchSDK(config_dir=args.config_dir, run_id=args.run_id, results_root=args.results_root)
    try:
        result = HANDLERS[args.command](sdk, args)
    except Exception as exc:  # operator-facing: report and exit nonzero
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(f"[airbench] {args.command} → {sdk.results_dir}")
    if result is not None:
        print(_summarize(result))
    return 0


def _summarize(result: object) -> str:
    if hasattr(result, "to_dict"):
        result = result.to_dict()
    text = str(result)
    return text if len(text) <= 400 else text[:400] + "…"


if __name__ == "__main__":
    sys.exit(main())
