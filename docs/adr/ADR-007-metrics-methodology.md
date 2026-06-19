# ADR-007 — Metrics methodology: TTFT/TPOT split, N≥5, mean±std

**Status:** accepted

**Context.** H5 requires TTFT and TPOT measured **separately** (Prefill vs Decode), over
multiple runs, as mean±std — not a single sample.

**Decision.** `metrics/timing.py` takes a `generate_fn` returning `(start, [token_timestamps])`.
TTFT = first_token − start (Prefill/compute); TPOT = (last − first)/(n−1) (Decode/memory);
throughput = n/total. `measure_generation` runs `warmup + n_runs` (config: warmup=1, n_runs=5),
discards warmups, and reports mean±std. Peak memory via psutil RSS + `vm_stat`; paging via
`vm_stat` swap deltas. Roofline classifies compute- vs memory-bound from `config/hardware.json`.

**Consequences.** Comparable, statistically-meaningful numbers across baseline / quant levels /
layered / extreme, all from the same harness (H5/H6/H8).
