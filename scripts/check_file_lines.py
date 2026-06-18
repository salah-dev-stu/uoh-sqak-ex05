#!/usr/bin/env python3
"""Fail if any package .py file exceeds the logical-line limit (R7: <=150).

Logical lines = non-blank lines that are not pure comments. Docstrings count
(they are real source). Run from the repo root: ``python scripts/check_file_lines.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path

LIMIT = 150
TARGET = Path("src/airbench")


def logical_lines(path: Path) -> int:
    count = 0
    for raw in path.read_text(encoding="utf-8").splitlines():
        stripped = raw.strip()
        if not stripped or stripped.startswith("#"):
            continue
        count += 1
    return count


def main() -> int:
    violations: list[tuple[Path, int]] = []
    for path in sorted(TARGET.rglob("*.py")):
        n = logical_lines(path)
        if n > LIMIT:
            violations.append((path, n))
    if violations:
        print(f"FAIL: {len(violations)} file(s) exceed {LIMIT} logical lines:")
        for path, n in violations:
            print(f"  {path}: {n}")
        return 1
    print(f"OK: all .py files under {TARGET} within {LIMIT} logical lines.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
