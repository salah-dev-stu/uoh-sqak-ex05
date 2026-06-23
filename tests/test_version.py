"""R5 — version single-source of truth."""

from __future__ import annotations

import importlib.metadata

from packaging.version import Version

import airbench
from airbench.shared.version import VERSION


def test_version_literal():
    assert airbench.__version__ == "1.31"
    assert VERSION == "1.31"


def test_version_matches_installed_metadata():
    # metadata normalizes "1.00" -> "1.0"; compare canonically.
    assert Version(importlib.metadata.version("airbench")) == Version(airbench.__version__)
