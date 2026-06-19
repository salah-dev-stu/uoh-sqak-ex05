# EX05 — Deep-Dive Technical Report

> Running a massive LLM locally on an 8 GB Apple M2 via layer-streaming + GGUF quantization.
> Course 203.3763 · Dr. Yoram Segal · Salah Qadah + Andalus Kalash · group `uoh-sqak`.
>
> **Note on status:** the engineering, methodology, and conceptual analysis below are complete.
> Cells/blocks marked _⟢ measured in run_ are filled from the committed `results/` artifacts once
> the on-device experiments (Phase 9) execute; the harness that produces them is done and tested.

---

## 1. Hardware (5.1 / H1)

| Component | Spec | Why it matters |
|---|---|---|
| Chip | Apple **M2** (Mac14,2) | Metal/MPS GPU; CUDA-only tooling (AirLLM, bitsandbytes) off the happy path |
| CPU | 8-core (4 performance + 4 efficiency) | modest parallelism for Prefill |
| **Unified memory** | **8 GB** (CPU+GPU shared) | **the hard wall** — there is no separate VRAM to spill to |
| Internal SSD | ~9 GB free (96% full) | too small for large weights |
| **External USB SSD** | `/Volumes/Backup`, 489 GB free, **~498 MB/s read / ~358 MB/s write** | weight store + layer-streaming source; its ~0.5 GB/s read is the predicted streaming ceiling |
| OS / Python | macOS 26.2 / Python 3.13 (uv) | — |

Live values are captured to `results/<run>/hardware.json` by `airbench probe`.

## 2. Model justification (5.1 / H1)

**Qwen2.5-7B-Instruct** (ungated). 7.62 B params, 28 layers, hidden 3584. **FP16 ≈ 15.2 GB ≈ 2× the
8 GB RAM** → a direct full-precision load cannot fit and OOMs. GGUF quants (Q8 ~8.1 GB … Q2 ~3.0 GB)
are storable on the SSD and runnable. Using one model across baseline → quantization → streaming keeps
the whole narrative coherent. (Llama-3.1-8B is the documented gated alternative; see
[ADR-001](../docs/adr/ADR-001-model-choice.md).)

## 3. Baseline: it fails (5.2 / H2)

**3a. Direct FP16 load (the failure).** `transformers` FP16 load on MPS through the Gatekeeper.
Expected: an MPS/host out-of-memory error — 15.2 GB of weights cannot fit 8 GB. The exception, the
peak-memory trace at failure, and a diagnosis are captured to `results/baseline/oom.json`.

> ⟢ measured in run: error type, peak RSS / memory-pressure at failure, screenshot.

**Bottleneck diagnosis.** This is **memory-bound**, not compute-bound: the process never gets to
sustained compute — it dies allocating weights. Evidence: weight bytes (15.2 GB) > capacity (8 GB);
memory pressure goes red before tokens are produced. (Roofline placement in §6.)

**3b. Quantized runnable baseline (the anchor).** llama.cpp with a Q4_K_M GGUF runs and gives the
first real TTFT/TPOT numbers we compare everything against.

> ⟢ measured in run: TTFT, TPOT, throughput (mean±std, N≥5) → `results/baseline/Q4_K_M.json`.

## 4. AirLLM + layer streaming (5.3 / H3)

We attempt the **real AirLLM** library (streaming FP16 from the SSD), **time-boxed** to 45–60 min. If
it cannot initialize on MPS (CUDA-centric), we record a `ConstraintReport` and run our **equivalent
layer-streaming demo**: each on-disk safetensors shard is materialized, forwarded, then freed, so peak
memory stays at one block instead of the whole model — exactly AirLLM's mechanism (see
[ADR-002](../docs/adr/ADR-002-airllm-honest-path.md)).

**Virtual-memory / paging mapping (H8).** Layer streaming *is* manual paging: a layer not resident is
"page-faulted" in from the SSD (~0.5 GB/s), used, then evicted — the OS analogue is demand paging with
the SSD as backing store. We measure the real signature via `vm_stat` swapins/pageins deltas around the
run, and per-layer I/O vs compute time.

> ⟢ measured in run: per-layer I/O vs compute, predicted-vs-actual I/O, vm_stat swap deltas,
> AirLLM outcome (`reports/airllm_constraint.md`).

## 5. Quantization sweep + the accuracy red line (5.4 / H4)

Serial sweep Q8 → Q5 → Q4 → Q2 (download → benchmark → **perplexity** via `llama-perplexity` → delete).
Lower precision ⇒ less memory + (often) faster decode, but rising perplexity. The **red line** is the
first level whose ΔPPL vs the Q8 baseline exceeds `red_line_ppl_delta` (config). See
[ADR-004](../docs/adr/ADR-004-serial-quant-sweep.md).

| Level | ~size | peak mem | TTFT | TPOT | throughput | PPL | ΔPPL |
|---|---|---|---|---|---|---|---|
| Q8_0 | 8.1 GB | ⟢ | ⟢ | ⟢ | ⟢ | ⟢ | 0 (ref) |
| Q5_K_M | 5.4 GB | ⟢ | ⟢ | ⟢ | ⟢ | ⟢ | ⟢ |
| Q4_K_M | 4.7 GB | ⟢ | ⟢ | ⟢ | ⟢ | ⟢ | ⟢ |
| Q2_K | 3.0 GB | ⟢ | ⟢ | ⟢ | ⟢ | ⟢ | ⟢ |

> ⟢ red line: _(level where ΔPPL crosses the threshold)_ → `results/quant/sweep.json`.

## 6. Prefill/Decode, compute- vs memory-bound, Roofline (5.6 / H8)

- **Prefill** processes the whole prompt in parallel (matrix–matrix): high arithmetic intensity →
  **compute-bound**, and it dominates **TTFT**.
- **Decode** generates one token at a time (matrix–vector), re-reading all weights each step → low
  arithmetic intensity → **memory-bound**, and it sets **TPOT**.
- **Roofline (M2).** Compute ceiling ≈ 2.84 TFLOP/s (config); memory bandwidth = 100 GB/s ⇒ ridge
  intensity ≈ 28.4 FLOP/byte. Decode intensity ≈ 2 FLOP / bytes_per_param ≈ **1 FLOP/byte at FP16** —
  far left of the ridge, i.e. firmly **memory-bound**. With layer streaming the effective bandwidth
  drops from 100 GB/s (RAM) to ~0.5 GB/s (SSD), pushing the attainable performance two orders of
  magnitude lower — which is *why* streaming runs but crawls.

> ⟢ measured in run: achieved GFLOP/s points + `reports/figures/roofline.png`.

## 7. Economics: On-Prem vs API (5.5 / H7)

Inputs in `config/economics.json` (see [ADR-008](../docs/adr/ADR-008-economics-assumptions.md)).
Per-request API cost vs amortized On-Prem (CAPEX/3yr + electricity); **break-even = requests/day where
the two annual costs cross**. For an 8 GB Mac whose sustained throughput is low, the API is expected to
win at low volume; the analysis reports the exact crossover and a TCO curve.

> ⟢ measured/derived: per-provider break-even req/day, TCO curve → `results/economics.json`,
> `reports/figures/breakeven.png`.

## 8. Original extensions (5.7 / H9)

1. **Quant Pareto** — quality (PPL) vs memory/speed across the sweep → `quant_pareto.png`.
2. **Extreme 70B AirLLM** — stream a 70B Q4 (~47 GB) from the SSD; expect ~80 s+/token of pure I/O,
   the textbook I/O-bound result.
3. **Context-length sweep** — vary ctx (512/2048/8192) and watch the compute→memory bound transition.

> ⟢ measured in run: `results/extreme/*`, context-sweep figure.

## 9. Honest negative results

If AirLLM won't run on MPS, that is reported plainly with the equivalent demo standing in (the spec
weights a well-analyzed negative result equally). If layer streaming is *slower* than the quantized
baseline (it likely is, by orders of magnitude, due to SSD I/O), that is the **point** — it makes an
otherwise-impossible run possible, and the cost is exactly the paging overhead we quantify.

## 10. References

Hugging Face (Qwen2.5-7B-Instruct, bartowski GGUF) · AirLLM · llama.cpp · Ollama · Lecture 08
(On-Premises LLM Deployment). Engineering standards + per-mechanism PRDs in `docs/`.
