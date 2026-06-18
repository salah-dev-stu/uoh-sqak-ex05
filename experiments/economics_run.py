#!/usr/bin/env python3
"""5.5 — On-Prem vs API economics + break-even → results/economics.json."""

from __future__ import annotations

from experiments._common import run


def main(sdk=None):
    return run("economics", sdk=sdk)


if __name__ == "__main__":
    main()
