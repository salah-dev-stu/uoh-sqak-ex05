"""On-Prem vs API economics: per-request API cost, amortized local cost, break-even (H7).

Honest model: API charges $/token; On-Prem is amortized hardware CAPEX + electricity
(active power × hours/day × kWh price). Break-even is the requests/day at which the
two annual costs cross. Below it, the API is cheaper; above it, local wins.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

DAYS_PER_YEAR = 365


def api_cost_per_request(
    tokens_in: int, tokens_out: int, usd_per_1m_in: float, usd_per_1m_out: float
) -> float:
    return tokens_in / 1_000_000 * usd_per_1m_in + tokens_out / 1_000_000 * usd_per_1m_out


def onprem_annual_cost(
    capex_usd: float,
    ssd_capex_usd: float,
    amortize_years: float,
    active_w: float,
    kwh_usd: float,
    hours_active_per_day: float,
) -> float:
    amortized = (capex_usd + ssd_capex_usd) / amortize_years
    annual_kwh = active_w / 1000 * hours_active_per_day * DAYS_PER_YEAR
    return amortized + annual_kwh * kwh_usd


def break_even_requests_per_day(api_per_request: float, onprem_annual: float) -> float:
    """Requests/day where API annual cost == On-Prem annual cost."""
    if api_per_request <= 0:
        return float("inf")
    return onprem_annual / (api_per_request * DAYS_PER_YEAR)


def tco_curve(
    api_per_request: float, onprem_annual: float, max_req_per_day: int, points: int = 10
) -> list[dict[str, float]]:
    step = max(1, max_req_per_day // points)
    curve = []
    for req in range(0, max_req_per_day + 1, step):
        curve.append(
            {
                "req_per_day": req,
                "api_annual": api_per_request * req * DAYS_PER_YEAR,
                "onprem_annual": onprem_annual,
            }
        )
    return curve


@dataclass
class ProviderEconomics:
    provider: str
    api_per_request: float
    onprem_annual: float
    break_even_req_per_day: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def compute(econ: dict) -> dict[str, Any]:
    """Build the full economics report from ``config/economics.json``."""
    onprem = onprem_annual_cost(
        econ["hardware_capex_usd"],
        econ.get("ssd_capex_usd", 0.0),
        econ["amortize_years"],
        econ.get("active_w", 0.0),
        econ.get("kwh_usd", 0.0),
        econ.get("hours_active_per_day", 0.0),
    )
    tin, tout = econ.get("avg_req_tokens_in", 0), econ.get("avg_req_tokens_out", 0)
    providers = []
    for p in econ["api_providers"]:
        per_req = api_cost_per_request(tin, tout, p["usd_per_1m_in"], p["usd_per_1m_out"])
        providers.append(
            ProviderEconomics(
                p["name"], per_req, onprem, break_even_requests_per_day(per_req, onprem)
            ).to_dict()
        )
    return {"onprem_annual": onprem, "providers": providers}
