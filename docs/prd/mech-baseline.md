# Mini-PRD — Baseline (5.2)
**Goal:** prove the model fails/crawls on a direct run and diagnose the bottleneck.
**Inputs:** primary model (Qwen2.5-7B), `config/runtime.json`. **Outputs:** `results/baseline/oom.json`
(+ log/screenshot), `results/baseline/<quant>.json`.
**Done when:** FP16 load OOMs with a captured trace + memory-bound diagnosis (H2); a llama.cpp
quantized run yields TTFT/TPOT as the comparison anchor (H5).
