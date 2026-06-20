# EX05 — Running a Massive LLM Locally: AirLLM, Quantization & Performance Benchmarking

[![CI](https://github.com/salah-dev-stu/uoh-sqak-ex05/actions/workflows/ci.yml/badge.svg)](https://github.com/salah-dev-stu/uoh-sqak-ex05/actions/workflows/ci.yml)

> **Course** 203.3763 Orchestration of AI Agents · University of Haifa · Spring 2026 · Dr. Yoram Segal
> **Authors** Salah Qadah · Andalus Kalash · **Group** `uoh-sqak`

Take an LLM **too big for an 8 GB Apple M2**, show it **fails on a direct run**, then make it run via
**layer-by-layer streaming + GGUF quantization**, and analyze the cost/benefit with measured metrics,
comparative graphs, a Roofline model, and an On-Prem-vs-API economic break-even. The goal is the
**engineering + economic analysis**, not output quality — a well-analyzed negative result counts equally.

📄 **Deep-dive report:** [`reports/technical_report.md`](reports/technical_report.md) ·
**Research-question answers:** [`reports/research_questions.md`](reports/research_questions.md) ·
**Design decisions:** [`docs/adr/`](docs/adr)

## Hardware (the premise)

| Component | Spec |
|---|---|
| Chip | Apple **M2** (Mac14,2), Metal/MPS GPU, 8-core CPU (4P+4E) |
| **Unified memory** | **8 GB** (CPU+GPU shared) — the hard wall |
| Internal SSD | ~9 GB free |
| **External USB SSD** | `/Volumes/Backup`, 489 GB free, ~498 MB/s read / ~358 MB/s write |

**Model:** `Qwen/Qwen2.5-7B-Instruct` (ungated). FP16 ≈ 15.2 GB ≈ **2× the RAM** → guaranteed OOM
baseline; GGUF quants (3–8 GB) are storable on the SSD and runnable.

## Architecture

CLI / experiments → **`BenchSDK`** (the one public entry, R1) → runners / metrics / economics, with
**every** external call (download, llama.cpp subprocess, model load, API lookup) funneled through one
wired **`ApiGatekeeper`** that gates (budgets) and records (an audit ledger). See
[`diagrams/block_diagram.mmd`](diagrams/block_diagram.mmd) and
[`diagrams/class_diagram.mmd`](diagrams/class_diagram.mmd) (R2/R3).

## Install

```bash
uv sync                 # runtime + dev deps; tests need NO GPU/model/key
uv run pytest           # green, fully mocked (grader Path D)
uv run airbench --help
```

Real experiments additionally need the heavy extra and llama.cpp:

```bash
uv sync --extra heavy   # torch / transformers / airllm / llama-cpp-python
brew install llama.cpp  # provides llama-cli + llama-perplexity
cp .env-example .env     # only needed for the gated Llama-3.1 alternative
```

## Reproduce the experiments

Weights download to the external SSD (never committed; an internal-disk path is refused):

```bash
uv run python scripts/download_model.py --role primary --quant Q4_K_M   # → /Volumes/Backup

uv run airbench probe                                   # 5.1 hardware → results/<run>/hardware.json
uv run airbench baseline-oom                            # 5.2a HF FP16 → OOM (the failure)
uv run airbench baseline --model-path /Volumes/Backup/hw5-weights/<q4>.gguf   # 5.2b anchor
uv run airbench quant-sweep                             # 5.4 Q8/Q5/Q4/Q2 + perplexity red line
uv run airbench airllm                                  # 5.3 real AirLLM attempt (→ layered fallback)
uv run airbench economics                               # 5.5 break-even → results/<run>/economics.json
uv run python scripts/make_figures.py --results results/<run>   # 5.4/5.6 graphs
```

(Global flags go before the subcommand, e.g. `uv run airbench --run-id demo economics`.)

## Results summary (real, measured on the 8 GB M2)

**The headline:** on 8 GB you either run **fast and crash** or **safe and crawl** — the memory wall, no
free lunch. Benchmarked with `llama-bench` (Metal) on Qwen2.5-7B Q4_K_M (`results/real/baseline_q4.json`):

| Regime | Prefill | Decode | Stable? |
|---|---|---|---|
| GPU + RAM (`-ngl 99 -mmp 0`) | **130.9 ± 12.4 tok/s** | **17.6 ± 0.27 tok/s** | ❌ **froze the Mac** (4.4 GB resident on 8 GB) |
| CPU + mmap (`-ngl 0 -mmp 1`) | 0.75 ± 0.03 tok/s | **< 0.03 tok/s** | ✅ stable but unusable (USB paging per token) |

Decode collapses **17.6 → <0.03 tok/s (>500×)** the moment weights must page from disk — the memory-bound /
paging result the lecture predicts. Both points sit far below the M2 compute ceiling (Roofline §6), i.e.
neither is compute-limited. **Economics:** On-Prem ≈ **$442/yr**, break-even **6,211 req/day** vs
gpt-4o-mini — but the memory wall caps this Mac at a few hundred req/day, so the API wins decisively
($0 was actually spent — the comparison uses published list prices). **AirLLM** is CUDA/MLX-centric and
does not run cleanly here ([`reports/airllm_constraint.md`](reports/airllm_constraint.md)); the layered
streaming demo stands in.

![decode](reports/figures/decode_throughput.png)
![prefill](reports/figures/prefill_throughput.png)
![roofline](reports/figures/roofline.png)
![break-even](reports/figures/breakeven.png)

Full analysis: [`reports/technical_report.md`](reports/technical_report.md).

## Repo structure

```
src/airbench/   shared/ (config, version, gatekeeper, paths) · sdk/ · runners/ · metrics/ · economics/ · figures.py
experiments/    thin SDK-only run scripts (§8)
config/         models · quant_levels · runtime · economics · budgets · hardware  (R4/R10)
scripts/        download_model · make_figures · check_file_lines · check_version_sync · fill_submission_pdf
results/        committed metrics/ledgers/logs (weights are NOT committed)
reports/        technical_report.md · research_questions.md · airllm_constraint.md · figures/
docs/           adr/ (8 decisions) · prd/ (per-mechanism)
diagrams/       class_diagram.mmd · block_diagram.mmd
tests/          110 tests, fully mocked, ≥85% coverage
```

## Engineering standards

SDK layer (R1) · OOP + class diagram (R2) · wired Gatekeeper on every external call (R3) · config-driven
(R4/R10) · version single-source (R5) · TDD, ≥85% coverage (R6/R9) · ≤150 logical lines/file (R7) ·
ruff clean (R8) · no secrets (R11) · uv only (R12) · continuous commits + green Python-3.13 CI (R13).

## References & acknowledgments

Qwen2.5-7B-Instruct (Hugging Face) · bartowski GGUF quants · AirLLM · llama.cpp · Ollama · **Lecture 08 —
On-Premises LLM Deployment** (Dr. Yoram Segal). Built for course 203.3763; co-authored by Salah Qadah and
Andalus Kalash.
