"""SSD-path guard: weights must live on an external volume (T103a)."""

from __future__ import annotations

import pytest

from airbench.shared.paths import assert_external


def test_accepts_external():
    assert str(assert_external("/Volumes/Backup/hw5-weights")).startswith("/Volumes/")


def test_rejects_internal_disk():
    with pytest.raises(ValueError, match="external volume"):
        assert_external("/Users/salah/weights")


def test_rejects_relative():
    with pytest.raises(ValueError):
        assert_external("weights")
