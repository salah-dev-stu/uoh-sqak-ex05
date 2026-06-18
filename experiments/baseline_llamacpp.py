#!/usr/bin/env python3
"""5.2b — runnable quantized baseline via llama.cpp (the comparison anchor)."""

from __future__ import annotations

import sys

from experiments._common import run


def main(model_path=None, quant="Q4_K_M", sdk=None):
    model_path = model_path or (sys.argv[1] if len(sys.argv) > 1 else None)
    return run("baseline", sdk=sdk, model_path=model_path, quant=quant)


if __name__ == "__main__":
    main()
