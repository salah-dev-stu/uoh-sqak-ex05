#!/usr/bin/env python3
"""5.3 — real AirLLM attempt; on a ConstraintReport, fall back to the layered demo."""

from __future__ import annotations

from airbench.sdk import ConstraintReport
from experiments._common import run


def main(sdk=None):
    result = run("airllm", sdk=sdk)
    if isinstance(result, ConstraintReport):
        print(f"AirLLM constraint ({result.reason}); running layered fallback")
        return run("layered", sdk=sdk)
    return result


if __name__ == "__main__":
    main()
