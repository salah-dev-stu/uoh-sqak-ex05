# Mini-PRD — Quantization sweep (5.4)
**Goal:** measure memory/speed/quality across Q8/Q5/Q4/Q2 and find the accuracy red line.
**Inputs:** `config/quant_levels.json`, `red_line_ppl_delta`. **Outputs:** `results/quant/sweep.json`.
**Done when:** each level benchmarked serially (download→bench→ppl→delete) and the red line
(first ΔPPL > threshold) reported (H4/H9). See [ADR-004](../adr/ADR-004-serial-quant-sweep.md).
