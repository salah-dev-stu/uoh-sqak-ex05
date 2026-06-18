"""Phase 5 — economics (hand-checked math)."""

from __future__ import annotations

import pytest

from airbench.economics import calculator as calc
from airbench.shared import config


def test_api_cost_per_request():
    # 500 in @ 0.15/1M + 200 out @ 0.60/1M = 0.000075 + 0.00012
    assert calc.api_cost_per_request(500, 200, 0.15, 0.60) == pytest.approx(0.000195)


def test_onprem_annual_cost():
    # (1199+90)/3 = 429.666...; kWh = 25/1000*8*365 = 73 kWh; *0.17 = 12.41
    got = calc.onprem_annual_cost(1199, 90, 3, 25, 0.17, 8)
    assert got == pytest.approx(1289 / 3 + 73 * 0.17)


def test_break_even():
    be = calc.break_even_requests_per_day(0.000195, 442.07)
    assert be == pytest.approx(442.07 / (0.000195 * 365))


def test_break_even_zero_api_is_inf():
    assert calc.break_even_requests_per_day(0, 100) == float("inf")


def test_tco_curve_shape():
    curve = calc.tco_curve(0.0002, 442.0, max_req_per_day=1000, points=5)
    assert curve[0]["api_annual"] == 0
    assert curve[-1]["api_annual"] > 0
    assert all(p["onprem_annual"] == 442.0 for p in curve)


def test_compute_from_real_config():
    report = calc.compute(config.get_economics())
    assert report["onprem_annual"] > 0
    assert len(report["providers"]) == 2
    assert all(p["break_even_req_per_day"] > 0 for p in report["providers"])
