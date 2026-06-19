# Mini-PRD — AirLLM + layer streaming (5.3)
**Goal:** make the oversized model run via layer-by-layer streaming; analyze paging.
**Inputs:** weights on SSD, `airllm_timebox_min`. **Outputs:** `results/airllm/airllm.json` or a
ConstraintReport + `results/airllm/layered.json`, plus `reports/airllm_constraint.md`.
**Done when:** real AirLLM attempted (time-boxed) and the layered demo's per-layer I/O-vs-compute
+ vm_stat paging deltas are captured (H3/H8). See [ADR-002](../adr/ADR-002-airllm-honest-path.md).
