# PRD — EX05: Running a Massive LLM Locally (AirLLM, Quantization & Performance Benchmarking)

> **Course** 203.3763 Orchestration of AI Agents · University of Haifa · Spring 2026 · Dr. Yoram Segal
> **Authors** Salah Qadah (323039974) + Andalus Kalash (211435797) · **Group** `uoh-sqak`
> **Repo** https://github.com/salah-dev-stu/uoh-sqak-ex05 (PUBLIC) · **Deadline** Fri 2026-06-26 23:59 (−5/24h late)
> **Status** DRAFT — awaiting **Gate 1** approval. Version 1.00.

---

## 1. Problem & Goal

Run a HuggingFace LLM that is **too big for an 8 GB Apple-Silicon Mac**. First prove it **fails / crawls** on a direct run (the baseline) and diagnose *why* (compute-bound vs memory-bound). Then make it run anyway via **AirLLM (layer-by-layer streaming)** + **GGUF quantization**, and deliver a **deep-dive technical report** with comparative metrics, graphs, an On-Prem-vs-API economic break-even, and a lecture-concept analysis (VRAM, Prefill/Decode, paging, Roofline).

**The grade is on engineering + economic insight, NOT output quality.** A well-analyzed *negative* result (e.g. "AirLLM runs it but is brutally I/O-bound, here's the math") scores as fully as a positive one. This is explicit in the spec and shapes every decision below.

---

## 2. The Description This PRD Is Built On (locked decisions)

These were confirmed interactively with the user and are the "bullets" this PRD realizes:

- **A. Hardware is the premise (measured, not assumed).**
  - Apple **M2** MacBook Air · **8 GB unified memory** (the single hard wall; CPU & GPU share it) · 8-core CPU (4 performance + 4 efficiency) · Metal/MPS GPU · macOS 26.2.
  - Internal SSD: **~9.3 GB free** (96% full) — too small for large weights.
  - **External USB SSD `/Volumes/Backup`: 489 GB free**, measured **~498 MB/s read / ~358 MB/s write**. Holds all weights and is the AirLLM layer-streaming source. Its ~0.5 GB/s read bandwidth is the predicted AirLLM bottleneck.
- **B. Model & experiment ladder.**
  - **Primary model: `Qwen/Qwen2.5-7B-Instruct` (ungated — no HF license click, no token).** FP16 ≈ **15.2 GB ≈ 2× RAM** → guaranteed OOM on a direct load (baseline failure). Quantized GGUF builds (3–8 GB) are storable and runnable, so it spans the full story on one model. (`meta-llama/Llama-3.1-8B-Instruct` is the documented gated alternative; chosen Qwen to avoid a mid-run license/token stall at zero grade cost.)
  - **Baseline (5.2):** two angles — (1) **raw HF `transformers` FP16 load** → live OOM/MPS-allocation failure (the bottleneck demo); (2) **llama.cpp** runnable quantized baseline to benchmark the optimized run against.
  - **Quantizer (5.4):** **GGUF sweep via llama.cpp — Q8 / Q5_K_M / Q4_K_M / Q2_K**, run **serially** (download → benchmark → delete) to respect disk. Locate the accuracy "red line".
  - **AirLLM (5.3):** **attempt the real AirLLM library** streaming the 8B FP16 weights from the SSD on MPS/CPU. If it cannot initialize on Apple Silicon (it is CUDA-centric), document the constraint and run **our equivalent layer-by-layer streaming demo** — spec-sanctioned, fully gradeable.
  - **Extension (5.7):** after the core 8B experiment is solid, add an **"extreme AirLLM" 70B-class Q4 (~40 GB)** streamed from the SSD (~80 s+/token I/O-bound) as an original experiment.
- **C. Engineering standards (carry-over from HW1–HW4, grader-enforced).**
  - SDK façade · **Gatekeeper wrapping every external call** (download, inference subprocess, API hit) — wired, not decorative · config-driven (no hardcoded values) · version single-source · uv only · ruff-clean · pytest ≥85% GREEN with heavy bits **mocked** (grader has no GPU/model/key) · ≤150 logical lines/file · GitHub CI green on **Python 3.13** · class diagram · **commits spread over time**.

---

## 3. Scope — Core Tasks 5.1–5.7 → Concrete Deliverables

| Spec | Deliverable in this repo | Evidence committed |
|---|---|---|
| **5.1** Hardware + model justification | `airbench probe` writes `results/hardware.json`; README §Hardware; model-size math | `results/hardware.json`, README table |
| **5.2** Baseline (fails/crawls) + bottleneck | `experiments/baseline_oom.py` (HF FP16 → OOM) + `experiments/baseline_llamacpp.py` (slow/runnable) | `results/baseline/*.log`, `*_metrics.json`, screenshots |
| **5.3** AirLLM + quantization | `airbench.airllm` (real attempt) + `airbench.layered` (equivalent streaming demo) | `results/airllm/*.json` + `reports/airllm_constraint.md` |
| **5.4** Measure TTFT/TPOT/throughput/mem | `airbench.metrics` harness; **TTFT & TPOT measured separately**, N≥5 runs, mean±std | `results/*/metrics.json`, CSVs |
| **5.5** Economics On-Prem vs API | `airbench.economics` + `experiments/economics_run.py`; break-even req/day | `results/economics.json`, `reports/figures/breakeven.png` |
| **5.6** Lecture-concept analysis | `reports/technical_report.md` ties every result to the *why* | report + Roofline plot |
| **5.7** Original extension(s) | quant Pareto sweep + 70B "extreme AirLLM" + context-length bound-transition | `reports/figures/pareto.png`, `results/extreme/*` |

---

## 4. Research Questions (spec §4) — each answered in the report

1. **What blocked the direct run — memory or compute, and how identified?** → 5.2 + Roofline.
2. **How does AirLLM change resource allocation, and how does it map to OS virtual-memory/paging?** → 5.3 + paging analysis.
3. **Quantization's effect on memory/speed/quality — where's the accuracy red line?** → 5.4 sweep + Pareto.
4. **How do Prefill/Decode show up in TTFT vs TPOT?** → 5.4 disaggregated metrics.
5. **What's the throughput/latency price of running big on modest hardware?** → 5.4 + economics.
6. **When is local worth it vs an external API?** → 5.5 break-even.

A `reports/research_questions.md` cross-references each answer to its evidence file.

---

## 5. Architecture

Mirrors the HW4 pattern that scored well. Single installable package `airbench`.

```
CLI (airbench.cli)  ──►  SDK façade (airbench.sdk.BenchSDK)   ◄── the only public entry (R1)
                                   │
        ┌──────────────────────────┼───────────────────────────────┐
        ▼                          ▼                                ▼
 runners/ (baseline,        metrics/ (TTFT, TPOT,            economics/ (API vs
 airllm, layered,          throughput, peak-mem,            On-Prem, break-even)
 quant_sweep)              roofline, paging)
        │                          │
        └──────────►  shared/gatekeeper.ApiGatekeeper  ◄──────────┘
                      (EVERY external call: HF download, llama.cpp
                       subprocess, AirLLM load, API pricing fetch —
                       gates + records: timing, bytes, exit codes)   (R3)

 shared/: config.py (loads config/*.json)  ·  version.py (single source, R5)
          token_meter.py (reused)          ·  logging_config.py
```

**Key principles**
- **R1 SDK layer** — all logic behind `BenchSDK`; CLI and experiments call only the SDK.
- **R3 Gatekeeper** — one `ApiGatekeeper` wraps *every* external interaction: model downloads (`huggingface_hub`), every `subprocess` to `llama.cpp`, the AirLLM model load, and the API-pricing lookup for §5.5. It **gates** (allow/deny by config budgets) **and records** (duration, bytes in/out, exit status) to a run ledger. Grep for raw `subprocess`/`requests`/`hf_hub_download` outside it must return nothing.
- **R4/R10 Config-driven** — `config/models.json` (ids, quant levels, file sizes), `config/runtime.json` (N runs, prompt set, max tokens, timeouts), `config/economics.json` (API prices, hardware CAPEX, kWh, amortization years), `config/budgets.json` (Gatekeeper limits). Zero magic constants in code.
- **R5 Version** — `airbench/shared/version.py` holds `VERSION = "1.00"`; `__init__.py` imports it; hatchling reads it; a test asserts the match. +0.01 per change.
- **R7 ≤150 logical lines/file** — enforced by `scripts/check_file_lines.py` (reused from HW4) in CI + pre-commit.

---

## 6. Measurement Methodology (5.4 — the heart of the grade)

- **TTFT (Prefill / compute)** and **TPOT (Decode / memory)** measured **separately**:
  - TTFT = wall time from submit → first token. TPOT = (total_gen_time − TTFT) / (n_tokens − 1).
  - Throughput = output_tokens / total_time. Peak memory via `psutil` (RSS + unified-memory pressure) + macOS `footprint`/`vm_stat` sampling; **paging** via `vm_stat` swapins/swapouts deltas captured around each run.
- **Multiple runs:** N ≥ 5 per configuration (configurable). Report **mean ± std**; discard a warmup run. Fixed prompt set + fixed seed + fixed `max_tokens` for comparability.
- **Configurations benchmarked:** baseline-llamacpp (per quant level), AirLLM/layered (8B FP16), and the 70B extreme run. Baseline-OOM contributes the failure trace, not timing.
- **Graphs (matplotlib, committed to `reports/figures/`):** TTFT bar (±std), TPOT bar (±std), throughput vs quant level, peak-memory bar, latency breakdown (prefill/decode/IO), **Roofline** (arithmetic intensity vs achieved GFLOP/s with hardware ceilings), **quant Pareto** (quality vs memory/speed), **break-even** curve.
- **Quality proxy** (to find the red line, not graded as output quality): perplexity on a fixed text and/or a small deterministic exact-match probe set per quant level.

---

## 7. Economic Analysis (5.5)

- **API side:** configurable per-provider $/1M input & output tokens (`config/economics.json`), cost per representative request.
- **On-Prem side:** hardware CAPEX (this Mac + SSD) amortized over N years + electricity (measured/estimated W × kWh price) + negligible idle.
- **Break-even:** requests/day where amortized local cost = API cost; sensitivity to utilization and token mix. Output `results/economics.json` + a break-even chart + a TCO curve. Honest treatment of throughput limits (this Mac serves few req/day → likely API-favored at low volume; analysis says exactly where it flips).

---

## 8. Repository Structure

```
uoh-sqak-ex05/
├── README.md                      # deep-dive technical report (the core deliverable) + run guide
├── pyproject.toml  uv.lock        # uv only (R12); hatchling dynamic version
├── .github/workflows/ci.yml       # Python 3.13 pinned (R13)
├── .pre-commit-config.yaml  .env-example  .gitignore
├── prd.md  Plan.md  Todo.md       # lifecycle docs
├── config/                        # models, runtime, economics, budgets (R4/R10)
├── src/airbench/
│   ├── __init__.py  cli.py
│   ├── shared/   {version,config,gatekeeper,token_meter,logging_config}.py
│   ├── sdk/      facade.py
│   ├── runners/  {baseline,airllm,layered,quant_sweep,extreme}.py
│   ├── metrics/  {timing,memory,paging,roofline}.py
│   └── economics/ calculator.py
├── experiments/  {probe,baseline_oom,baseline_llamacpp,airllm_run,quant_sweep,economics_run,extreme_airllm}.py
├── scripts/      {download_model.py,check_file_lines.py,fill_submission_pdf.py,make_figures.py}
├── results/      hardware.json  baseline/  airllm/  quant/  extreme/  economics.json   # committed artifacts (NO weights)
├── reports/      technical_report.md  airllm_constraint.md  research_questions.md  figures/*.png
├── diagrams/     class_diagram.mmd  block_diagram.mmd                                  # (R2)
└── tests/        unit tests, all heavy bits mocked (R6/R9)
```

Weights are **never committed** (R/H10): `scripts/download_model.py` (Gatekeeper-wrapped) fetches to `/Volumes/Backup`; `.gitignore` excludes weight dirs.

---

## 9. Testing & Quality Gates

- **R6 TDD / R9 ≥85% cov:** every module unit-tested; **all heavy operations mocked** — `subprocess`, downloads, AirLLM load, `psutil`/`vm_stat`, file sizes — so `uv run pytest` is green on the grader's hardware with no GPU/model/key (grader Path D). Metrics math, Gatekeeper gating/recording, economics break-even, config loading, version sync all tested against fixtures.
- **R8 ruff:** `select=[E,F,W,I,N,UP,B,C4,SIM]`, `ignore=[E501]`, line-length 100, py313 — zero failures.
- **R7 file lines:** `check_file_lines.py` ≤150 logical lines.
- **R13 CI:** `actions/setup-python@v5` pinned to 3.13; runs ruff + file-lines + pytest-cov (fail_under=85). Green badge in README.
- **Pre-commit:** ruff, file-lines, version-sync.

---

## 10. Grading-Gate Traceability (H1–H11)

| Gate | Where satisfied |
|---|---|
| H1 hardware+model | `results/hardware.json`, README §Hardware, §2A/§1 here |
| H2 baseline failure + bottleneck | `experiments/baseline_oom.py` logs + Roofline diagnosis |
| H3 AirLLM run | `runners/airllm.py` real attempt + `runners/layered.py` fallback + `reports/airllm_constraint.md` |
| H4 quantization effect | `runners/quant_sweep.py`, Pareto, red-line discussion |
| H5 TTFT+TPOT separate, mean±std | `metrics/timing.py`, N≥5, results JSON |
| H6 comparative graphs | `reports/figures/*.png` |
| H7 economics + break-even | `economics/calculator.py`, break-even chart |
| H8 lecture-concept analysis + Roofline | `reports/technical_report.md`, `figures/roofline.png` |
| H9 original extension | 70B extreme run + quant Pareto + context-length sweep |
| H10 reproducible repo + report | README, download script, committed artifacts |
| H11 §4 research questions | `reports/research_questions.md` |

---

## 11. Risks & Mitigations

| Risk | Mitigation |
|---|---|
| AirLLM won't init on MPS (CUDA-centric) | **Expected**; honest constraint doc + equivalent layered demo (spec allows; counts equally) |
| 70B download huge/slow; runs ~80s+/token | Extension only, after core; few tokens suffice; can be skipped without losing core grade |
| 8 GB RAM kills even quantized runs / system instability | Serial single-config runs; small `max_tokens`; capture failures as data |
| HF gated model | Primary is **ungated Qwen2.5-7B** (no token needed); `HF_TOKEN` support kept optional for the documented Llama-3.1 alternative via `os.environ.get` |
| Grader has no hardware | All tests mocked; real artifacts committed; never gate grade on a re-run |
| One-day commit history (HW4 nit) | Commit continuously across the build, logically grouped |

---

## 12. Milestones & Commit Cadence

1. Scaffolding (uv, pyproject, CI, shared/, config, version) → commit.
2. Gatekeeper + SDK façade + tests → commit.
3. Metrics harness + tests → commit.
4. Runners (baseline/layered/quant) + tests → commit.
5. Real experiments on the Mac; commit artifacts as produced (spread over time).
6. Economics + figures → commit.
7. Report + README + diagrams → commit.
8. 70B extension → commit.
9. Submission PDF, final polish, push public.

---

## 13. Out of Scope

- Output-quality optimization / prompt engineering (explicitly not graded).
- Training/fine-tuning (LoRA mentioned only as concept).
- Multi-GB weights in git. Hours-long runs (pick configs that *demonstrate* strain quickly).

---

## 14. Definition of Done

All H1–H11 + R1–R13 satisfied; CI green; README is a deep-dive report with ≥6 figures incl. Roofline; baseline failure + AirLLM/quantized runs + economics all backed by committed artifacts; §4 questions answered; repo public & shared with `rmisegal@gmail.com`; submission PDF generated.

---

### → GATE 1: Approve this PRD before I write `Plan.md` and `Todo.md` (300–800 tasks).
