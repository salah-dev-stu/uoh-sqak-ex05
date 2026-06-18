"""R3 meta-test — NO module may make external calls outside the Gatekeeper.

Greps the package source for raw external-call markers. The only file allowed to
contain them is the gatekeeper itself (the single chokepoint).
"""

from __future__ import annotations

from pathlib import Path

import airbench

# Precise usage markers (avoid false positives like the config key "max_subprocess_s").
FORBIDDEN = [
    "import subprocess",
    "subprocess.",
    "import requests",
    "requests.",
    "import urllib",
    "urllib.",
    "hf_hub_download(",
    "snapshot_download(",
    ".from_pretrained(",
]

ALLOWED_FILES = {"gatekeeper.py", "gatekeeper_types.py"}

SRC = Path(airbench.__file__).parent


def test_no_raw_external_calls_outside_gatekeeper():
    offenders: list[str] = []
    for py in SRC.rglob("*.py"):
        if py.name in ALLOWED_FILES:
            continue
        text = py.read_text(encoding="utf-8")
        for marker in FORBIDDEN:
            if marker in text:
                offenders.append(f"{py.relative_to(SRC)}: {marker}")
    assert not offenders, f"raw external calls found outside gatekeeper: {offenders}"
