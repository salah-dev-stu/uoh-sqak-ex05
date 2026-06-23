# Research Questions (spec §4 / H11)

Each question → the answer → the evidence file that backs it. All numbers are filled from the committed
`results/` artifacts (real on-device runs); where a measurement is infeasible on 8 GB it is stated explicitly.

### Q1 — What blocked the direct run: memory or compute, and how did you identify it?
**Memory-bound.** Qwen2.5-7B FP16 ≈ 15.2 GB > 8 GB unified memory, so the load OOMs before sustained
compute begins. Identified by: the load-time failure (no tokens produced), peak memory/pressure at
failure, and Roofline placement (Decode intensity ≈ 1 FLOP/byte ≪ ridge 28.4) — see report §3, §6.
**Evidence:** `results/baseline/oom.json`, `reports/figures/roofline.png`.

### Q2 — How does AirLLM change resource allocation, and how does it map to OS virtual memory/paging?
It trades **memory for I/O**: instead of holding all weights resident, it streams one layer at a time
from the SSD, capping peak memory at a single block. This is manual **demand paging** — a non-resident
layer is "faulted in" from the SSD backing store (~0.5 GB/s), used, evicted. Measured via per-layer
I/O-vs-compute and `vm_stat` swap/pagein deltas. See report §4.
**Evidence:** `results/airllm/layered.json`, `reports/airllm_constraint.md`.

### Q3 — Quantization's effect on memory/speed/quality — where's the accuracy red line?
Lower bits ⇒ smaller footprint and (usually) faster memory-bound decode, at rising perplexity. The red
line is the first GGUF level whose ΔPPL vs Q8 exceeds the configured threshold (objective, via
`llama-perplexity`). See report §5. **Beyond inference, quantization also enables *training*:** QLoRA
(LoRA adapters on a 4-bit base) fine-tuned a 1.5B model on the same 8 GB Mac at 1.36 GB peak in ~95 s —
quantization is what makes both running *and* adapting feasible on modest hardware (report §8b).
**Evidence:** `results/quant/sweep.json`, `results/real/lora/metrics.json`, `reports/figures/lora_loss.png`.

### Q4 — How do Prefill/Decode show up in your TTFT vs TPOT split?
**TTFT** is dominated by **Prefill** (parallel, compute-bound, matrix–matrix). **TPOT** is the
**Decode** phase (autoregressive, memory-bound, matrix–vector re-reading weights each step). Measured
separately by the timing harness, mean±std over N≥5. See report §6, ADR-007.
**Evidence:** `results/baseline/*.json`, `results/airllm/*.json`, `reports/figures/{ttft,tpot}.png`.

### Q5 — What's the throughput/latency price of running big on modest hardware?
Streaming makes the impossible possible but is **I/O-bound**: effective bandwidth falls from ~100 GB/s
(RAM) to ~0.5 GB/s (SSD), so TPOT balloons (7B FP16 ≈ 32 s/token of pure I/O; 70B ≈ 80 s+). Quantized
llama.cpp is far faster but is the only thing that *fits*. See report §4, §6, §8.
**Evidence:** comparative figures + `results/extreme/*`.

### Q6 — When is local worth it economically vs an external API?
Below the break-even (requests/day), the API is cheaper; above it, amortized On-Prem wins. For an 8 GB
Mac the sustained throughput is low, so the API is expected to win until fairly high volume; the exact
crossover + TCO curve are computed. See report §7, ADR-008.
**Evidence:** `results/economics.json`, `reports/figures/breakeven.png`.
