#!/usr/bin/env python3
"""5.7 — the extreme 70B-class AirLLM streaming experiment."""

from __future__ import annotations

from experiments._common import run


def main(sdk=None):
    return run("extreme", sdk=sdk)


if __name__ == "__main__":
    main()
