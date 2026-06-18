# Plan.md — Implementation Plan for EX05 (`airbench`)

> Realizes `prd.md`. Read that first. This is the *how*: module designs, interfaces, config schemas, test
> strategy, experiment execution, and commit cadence. Version 1.00. Awaiting **Gate 2** (with `Todo.md`).

---

## 0. Conventions

- **Package:** `src/airbench/` (installable via uv + hatchling, dynamic version).
- **Language/level:** Python 3.13 only. ≤150 logical lines/file (split aggressively).
- **TDD:** write the test first (red) → implement (green) → refactor. Heavy ops mocked.
- **Public surface:** everything goes through `airbench.sdk.facade.BenchSDK`. CLI + experiments import only the SDK.
- **No magic values:** all ids/paths/levels/prices/limits in `config/*.json`, loaded via `shared/config.py`.
- **External calls:** ONLY inside `shared/gatekeeper.ApiGatekeeper`. Nothing else may call `subprocess`,
  `requests`, `huggingface_hub`, or load a model directly.

---

## 1. Phase Map (build order)

| Ph | Name | Output | Gate dep |
|----|------|--------|----------|
| 0 | Scaffolding | uv project, pyproject, CI, pre-commit, dirs, version, .env-example | — |
| 1 | shared/ | config, version, logging, **gatekeeper**, token_meter (+ tests) | R3/R4/R5 |
| 2 | SDK façade | `BenchSDK` thin orchestrator (+ tests) | R1 |
| 3 | metrics/ | timing (TTFT/TPOT), memory, paging, roofline (+ tests) | H5/H8 |
| 4 | runners/ | baseline, layered, airllm, quant_sweep, extreme (+ tests, mocked) | H2/H3/H4 |
| 5 | economics/ | calculator + break-even (+ tests) | H7 |
| 6 | CLI | `airbench probe/baseline/airllm/quant/economics/figures` | — |
| 7 | scripts | download_model, check_file_lines, make_figures, fill_submission_pdf | H10 |
| 8 | Test sweep | coverage ≥85%, ruff clean, file-lines pass | R6/R8/R9 |
| 9 | Real runs | execute experiments on the Mac; capture artifacts | H1/H2/H3/H4/H5 |
| 10 | Figures | matplotlib graphs incl. Roofline, Pareto, break-even | H6/H8 |
| 11 | Report+README | technical_report, README, diagrams, research_questions | H8/H10/H11 |
| 12 | Extension | 70B extreme AirLLM run + context-length sweep | H9 |
| 13 | Submission | PDF, public push, share w/ rmisegal | submit |

Commits happen *within and across* phases — never one big-bang day.

---

## 2. Configuration Schema (`config/`)

**`models.json`**
```json
{
  "primary": {
    "hf_id": "Qwen/Qwen2.5-7B-Instruct",
    "params_b": 7.62, "hidden": 3584, "layers": 28,
    "fp16_gb": 15.2, "gated": false,
    "gguf_repo": "bartowski/Qwen2.5-7B-Instruct-GGUF",
    "weights_dir": "/Volumes/Backup/hw5-weights"
  },
  "alternative": { "hf_id": "meta-llama/Llama-3.1-8B-Instruct", "gated": true, "needs": "HF_TOKEN + license", "...": "..." },
  "extreme":  { "hf_id": "...70B...", "gguf_file": "...Q4_K_M.gguf", "size_gb": 40 }
}
```
**`quant_levels.json`** — ordered list `[{name:"Q8_0",gguf_file,approx_gb}, {name:"Q5_K_M",...}, "Q4_K_M", "Q2_K"]`.
**`runtime.json`** — `{n_runs:5, warmup:1, max_tokens:64, seed:42, prompts:[...], llamacpp_bin, ctx_sizes:[512,2048,8192], timeout_s}`.
**`economics.json`** — `{api_providers:[{name,usd_per_1m_in,usd_per_1m_out}], hardware_capex_usd, ssd_capex_usd, amortize_years:3, idle_w, active_w, kwh_usd, avg_req_tokens_in, avg_req_tokens_out}`.
**`budgets.json`** — Gatekeeper limits `{max_download_gb, max_subprocess_s, max_api_calls, allow_network:true}`.

All schemas validated on load (lightweight dataclass/`TypedDict` + assertions; no hardcoding in modules).

---

## 3. Module Designs

### 3.1 `shared/version.py` (R5)
`VERSION = "1.00"`. `__init__.py`: `from airbench.shared.version import VERSION`. hatchling `[tool.hatch.version] path`. Test asserts `airbench.__version__ == VERSION` and matches pyproject.

### 3.2 `shared/config.py` (R4/R10)
- `load(name) -> dict` reads `config/<name>.json` (path from `AIRBENCH_CONFIG_DIR` env or default).
- `get_model(role)`, `get_runtime()`, `get_economics()`, `get_budgets()` typed accessors.
- Raises on missing keys. No defaults baked into call sites.

### 3.3 `shared/gatekeeper.py` (R3 — the linchpin)
```python
class ApiGatekeeper:
    def __init__(self, budgets: dict, ledger_path: Path): ...
    def download(self, repo, filename, dest) -> Path        # wraps hf_hub_download; checks max_download_gb
    def run_subprocess(self, argv, timeout) -> RunResult     # wraps subprocess.run; records time/exit/stdout tail
    def load_model(self, loader: Callable, label) -> Any     # wraps any in-proc heavy load (AirLLM/transformers)
    def api_call(self, fn: Callable, label) -> Any           # wraps pricing/HTTP; counts against max_api_calls
    def record(self, event: GateEvent)                       # append to JSON ledger: ts,label,kind,dur,bytes,ok
    @property
    def ledger(self) -> list[GateEvent]
```
- **Gates:** deny if budget exceeded (raise `BudgetExceeded`); honor `allow_network`.
- **Records:** every call → `results/<run>/gate_ledger.json`. This is the evidence it's wired, not decorative.
- Tests: mock the wrapped callables; assert gating denies over-budget, assert ledger entries written, assert
  no raw `subprocess`/`requests` exist elsewhere (a meta-test greps the source tree).

### 3.4 `shared/token_meter.py` — reuse HW4 (counts tokens for economics + prompt sizing).

### 3.5 `sdk/facade.py` (R1)
```python
class BenchSDK:
    def __init__(self, config_dir=None): build config + gatekeeper + logger
    def probe_hardware(self) -> HardwareReport
    def run_baseline_oom(self) -> FailureReport
    def run_baseline_llamacpp(self, quant) -> RunMetrics
    def run_airllm(self) -> RunMetrics | ConstraintReport
    def run_layered_demo(self) -> RunMetrics
    def run_quant_sweep(self) -> list[RunMetrics]
    def run_extreme(self) -> RunMetrics
    def compute_economics(self) -> EconomicsReport
    def make_figures(self) -> list[Path]
```
Thin: delegates to runners/metrics/economics; owns the shared gatekeeper instance.

### 3.6 `metrics/` (H5/H8)
- `timing.py`: `measure_generation(generate_fn, n_runs, warmup) -> TimingStats` returning **separate** TTFT &
  TPOT arrays → mean±std; throughput. `generate_fn` is a callback yielding tokens with timestamps so Prefill
  (first token) and Decode (subsequent) are split. Pure/mockable — no model inside.
- `memory.py`: peak RSS via `psutil`; macOS unified-memory pressure sampler (wraps `vm_stat`/`footprint`
  through gatekeeper subprocess). Returns `MemSample` series + peak.
- `paging.py`: capture `vm_stat` swapins/swapouts before/after a run → page-fault deltas (the paging story).
- `roofline.py`: given model FLOPs/token, bytes moved/token, and hardware peaks (config) → arithmetic
  intensity + achieved GFLOP/s + ceiling → classify compute- vs memory-bound. Pure math, fully tested.
- `aggregate.py`: dataclasses → JSON/CSV writers under `results/`.

### 3.7 `runners/` (H2/H3/H4/H9)
- `baseline_oom.py`: attempt HF `transformers` FP16 load via `gatekeeper.load_model`; expect
  RuntimeError/MPS OOM; capture the exception + memory trace → `FailureReport`. (In tests, the loader is
  mocked to raise; on the Mac it really fails.)
- `baseline_llamacpp.py`: download GGUF (gatekeeper) → run `llama-cli`/`llama.cpp` subprocess (gatekeeper) →
  parse tokens+timings → `RunMetrics`.
- `layered.py`: our **equivalent AirLLM demo** — load model config + iterate transformer blocks, materializing
  each block's weights from SSD on demand, running a forward pass, freeing it; instrument per-layer I/O vs
  compute. Demonstrates the paging principle on MPS/CPU regardless of AirLLM.
- `airllm.py`: real `airllm.AutoModel` attempt streaming from SSD via `gatekeeper.load_model`,
  **time-boxed to `runtime.airllm_timebox_min` (45–60 min)**; on ImportError/CUDA/MPS failure OR timeout →
  `ConstraintReport` (logged) and the SDK falls back to `layered` (llama.cpp disk-streaming). No open-ended debugging.
- `quant_sweep.py`: for each quant level (serial): download → bench (reuse baseline_llamacpp) → record →
  delete weight file → next. Adds a quality probe (perplexity/exact-match) per level.
- `extreme.py`: 70B Q4 streamed via airllm/layered; few tokens; capture the brutal I/O-bound numbers.

### 3.8 `economics/calculator.py` (H7)
- `api_cost(req, tokens_in, tokens_out, provider)`, `onprem_annual_cost(capex, years, watts, kwh)`,
  `break_even(api, onprem) -> requests_per_day`, plus a sweep producing the TCO curve points. Pure math.

### 3.9 `cli.py`
`argparse` subcommands: `probe | baseline-oom | baseline | airllm | quant-sweep | extreme | economics |
figures | all`. Each calls one `BenchSDK` method, writes artifacts, prints a summary. ≤150 lines (split helpers).

---

## 4. Test Strategy (R6/R9 — grader Path D)

- **Everything heavy is mocked:** `subprocess.run`, `hf_hub_download`, `airllm`/`transformers` loads,
  `psutil`, `vm_stat` output (fixture strings). No GPU/model/key ever needed.
- **Per-module unit tests:** config loading/validation, version sync, gatekeeper gating+ledger+source-grep
  meta-test, timing math (known token timestamps → known TTFT/TPOT/std), memory/paging parsing from fixture
  output, roofline classification (compute vs memory cases), economics break-even (hand-checked numbers),
  each runner's control flow with mocked externals, CLI dispatch.
- **Coverage:** `--cov=airbench --cov-report=term-missing`, `fail_under=85`.
- **Fixtures:** sample `vm_stat`/`llama.cpp` stdout, fake GGUF metadata, canned token-timing sequences.

---

## 5. CI & Pre-commit (R13/R8/R7)

- `.github/workflows/ci.yml`: `actions/setup-python@v5` python-version 3.13 → `uv sync` →
  `uv run ruff check` → `uv run python scripts/check_file_lines.py` → `uv run pytest`. Badge in README.
- `.pre-commit-config.yaml`: ruff (check+format), file-lines, version-sync hook.

---

## 6. Real Experiment Execution (Phase 9, on the Mac)

Order, each writing artifacts to `results/` and committed as produced:
1. `airbench probe` → `hardware.json`.
2. `airbench baseline-oom` → capture the FP16 OOM (log + screenshot). **The headline failure.**
3. `download primary Q4` → `airbench baseline` → runnable baseline metrics.
4. `airbench quant-sweep` (Q8→Q5→Q4→Q2, serial) → per-level metrics + quality + the red line.
5. `airbench airllm` (real attempt; record constraint) → `airbench` layered demo metrics.
6. `airbench economics` → economics.json.
7. `airbench figures` → all PNGs.
8. (Phase 12) `airbench extreme` → 70B numbers.

Safety: small `max_tokens`, serial single-config, timeouts via Gatekeeper; failures are captured as data.

---

## 7. Figures (Phase 10 — H6/H8)

`scripts/make_figures.py` reads `results/*` → emits to `reports/figures/`: `ttft.png`, `tpot.png`,
`throughput_vs_quant.png`, `peak_memory.png`, `latency_breakdown.png`, `roofline.png`, `quant_pareto.png`,
`breakeven.png`, `tco_curve.png`. Labeled axes, units, error bars (±std).

---

## 8. Report & README (Phase 11 — H8/H10/H11)

- `reports/technical_report.md`: hardware → baseline failure + bottleneck (Roofline) → AirLLM constraint +
  layered mechanism + paging → quantization sweep + red line → metrics + graphs → economics + break-even →
  every §4 research question answered → honest negative results.
- `README.md` **is** the deep-dive report front door: overview, hardware table, install (`uv sync`), how to
  reproduce each experiment, embedded figures, results summary, references, CI badge, acknowledgments.
- `diagrams/`: `class_diagram.mmd` (OOP, R2) + `block_diagram.mmd` (data flow).
- `reports/research_questions.md`: each §4 question → answer → evidence file.

---

## 9. Submission (Phase 13)

`scripts/fill_submission_pdf.py` (adapt HW4) → `uoh-sqak-ex05.pdf`. Repo public + shared with
`rmisegal@gmail.com`. Self-grade 85. Both members submit on Moodle id=278465.

---

## 10. Definition of Done

Maps to PRD §14: all H1–H11 + R1–R13 green; CI green; ≥6 figures incl. Roofline; baseline failure + AirLLM +
quant + economics backed by committed artifacts; §4 answered; PDF generated; repo public.
