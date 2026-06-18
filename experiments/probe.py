#!/usr/bin/env python3
"""5.1 — probe + document the machine spec → results/hardware.json."""

from __future__ import annotations

from experiments._common import run


def main(sdk=None):
    return run("probe", sdk=sdk)


if __name__ == "__main__":
    main()
