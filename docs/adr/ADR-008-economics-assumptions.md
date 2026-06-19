# ADR-008 — Economics assumptions (CAPEX/OPEX, break-even)

**Status:** accepted

**Context.** H7 requires real On-Prem-vs-API math with a break-even, not hand-waving.

**Decision.** All inputs live in `config/economics.json`: API `$/1M` in/out for two providers
(gpt-4o-mini, claude-haiku), hardware + SSD CAPEX amortized over 3 years, active power (W) ×
hours/day × `$/kWh`, and an average token mix. `economics/calculator.py` computes per-request
API cost, On-Prem annual cost (amortized CAPEX + electricity), the **break-even requests/day**
(API_annual = OnPrem_annual), and a TCO curve. Assumptions are documented and editable.

**Consequences.** Reproducible, sensitivity-friendly economics (H7). The honest conclusion for
an 8 GB Mac (few req/day throughput) is expected to favour the API at low volume — the analysis
states exactly where it flips.
