#!/usr/bin/env python3
"""5.2a — direct HF FP16 load that OOMs the 8 GB Mac (the baseline failure)."""

from __future__ import annotations

from experiments._common import run


def main(sdk=None):
    return run("baseline-oom", sdk=sdk)


if __name__ == "__main__":
    main()
