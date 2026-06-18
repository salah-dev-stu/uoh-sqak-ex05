# EX05 — Running a Massive LLM Locally: AirLLM, Quantization & Performance Benchmarking

[![CI](https://github.com/salah-dev-stu/uoh-sqak-ex05/actions/workflows/ci.yml/badge.svg)](https://github.com/salah-dev-stu/uoh-sqak-ex05/actions/workflows/ci.yml)

> **Course** 203.3763 Orchestration of AI Agents · University of Haifa · Spring 2026 · Dr. Yoram Segal
> **Authors** Salah Qadah · Andalus Kalash · **Group** `uoh-sqak`

🚧 **Work in progress** — this README becomes the deep-dive technical report. See `prd.md`, `Plan.md`,
and `Todo.md` for the build plan.

## What this is

Take an LLM **too big for an 8 GB Apple-Silicon Mac**, show it fails/crawls on a direct run, then make it run
via **AirLLM-style layer streaming + GGUF quantization**, and analyze the cost/benefit with measured metrics,
graphs, a Roofline model, and an On-Prem-vs-API economic break-even.

- **Primary model:** Qwen2.5-7B-Instruct (ungated). FP16 ≈ 15 GB ≈ 2× the 8 GB RAM → OOM baseline.
- **Hardware:** Apple M2, 8 GB unified memory, external USB SSD for weights + streaming.
- **Tooling:** `uv` · `llama.cpp` (GGUF) · `transformers` (OOM demo) · `airllm` (real attempt) · matplotlib.

## Quick start

```bash
uv sync                       # dev + runtime deps (tests need no GPU/model/key)
uv run pytest                 # green, fully mocked
uv run airbench --help        # experiment CLI
```

Full reproduction steps, results, and figures land here as the experiments run.
