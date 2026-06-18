#!/usr/bin/env python3
"""Render comparative figures from a results dir into reports/figures/ (H6).

uv run python scripts/make_figures.py --results results/<run-id>
"""

from __future__ import annotations

import argparse
import sys

from airbench import figures


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Build figures from committed results")
    ap.add_argument("--results", required=True, help="results/<run-id> directory")
    ap.add_argument("--out", default="reports/figures")
    args = ap.parse_args(argv)
    written = figures.build_all(args.results, args.out)
    for p in written:
        print(f"wrote {p}")
    print(f"{len(written)} figure(s) written to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
