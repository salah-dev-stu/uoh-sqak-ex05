#!/usr/bin/env python3
"""Fail if the single-source version desyncs (R5).

``src/airbench/shared/version.py`` is the one literal. This asserts
``airbench.__version__`` agrees with it. hatchling reads the same file for the
build version, so all three stay in lock-step.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

VERSION_FILE = Path("src/airbench/shared/version.py")


def literal_version() -> str:
    text = VERSION_FILE.read_text(encoding="utf-8")
    match = re.search(r'__version__\s*=\s*"([^"]+)"', text)
    if not match:
        raise SystemExit("FAIL: __version__ literal not found in version.py")
    return match.group(1)


def main() -> int:
    src = literal_version()
    sys.path.insert(0, "src")
    import airbench  # noqa: E402

    if airbench.__version__ != src:
        print(f"FAIL: airbench.__version__={airbench.__version__} != version.py {src}")
        return 1
    print(f"OK: version single-source in sync ({src}).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
