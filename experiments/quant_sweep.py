#!/usr/bin/env python3
"""5.4 — serial GGUF quant sweep (Q8/Q5/Q4/Q2) with the perplexity red-line."""

from __future__ import annotations

from experiments._common import run


def main(sdk=None):
    return run("quant-sweep", sdk=sdk)


if __name__ == "__main__":
    main()
