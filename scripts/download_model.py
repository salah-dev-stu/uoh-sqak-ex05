#!/usr/bin/env python3
"""Download model weights to the EXTERNAL SSD via the Gatekeeper.

Weights are NEVER committed (H10) and NEVER written to the internal disk (the
SSD-path guard refuses that). Usage:
    uv run python scripts/download_model.py --role primary --quant Q4_K_M
    uv run python scripts/download_model.py --role extreme           # 70B GGUF file
"""

from __future__ import annotations

import argparse
import sys

from airbench.shared import config
from airbench.shared.gatekeeper import ApiGatekeeper
from airbench.shared.paths import assert_external


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Download GGUF weights to the external SSD")
    ap.add_argument("--role", default="primary")
    ap.add_argument(
        "--quant",
        default=None,
        help="GGUF level name (e.g. Q4_K_M); omit for a model's own gguf_file",
    )
    args = ap.parse_args(argv)

    model = config.get_model(args.role)
    weights_dir = assert_external(model["weights_dir"])  # refuses internal disk (T103a)
    weights_dir.mkdir(parents=True, exist_ok=True)
    gk = ApiGatekeeper(config.get_budgets(), weights_dir / "download_ledger.json")

    if args.quant:
        lvl = next(x for x in config.get_quant_levels() if x["name"] == args.quant)
        repo, filename, size = model["gguf_repo"], lvl["gguf_file"], lvl.get("approx_gb")
    else:
        repo, filename, size = model["gguf_repo"], model["gguf_file"], model.get("size_gb")

    path = gk.download(repo, filename, weights_dir, size_gb=size)
    print(f"downloaded {filename} -> {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
