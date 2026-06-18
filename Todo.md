# Todo.md — EX05 `airbench` (execute one-by-one, mark each `[x]` when done)

> Realizes `prd.md` + `Plan.md`. TDD order: test → impl → refactor. Heavy ops mocked so the grader needs no
> GPU/model/key. Commit continuously (`→commit`); never one big-bang day. Awaiting **Gate 2** before T001.
> Contiguous numbering; PRD→Todo coverage self-audit at the bottom. ≥500 atomic tasks.

---

## Phase 0 — Scaffolding & Repo Hygiene

- [x] T001 Create dir `src/airbench/`
- [x] T002 Create dir `src/airbench/shared/`
- [x] T003 Create dir `src/airbench/sdk/`
- [x] T004 Create dir `src/airbench/runners/`
- [x] T005 Create dir `src/airbench/metrics/`
- [x] T006 Create dir `src/airbench/economics/`
- [x] T007 Create dirs `tests/` and `tests/fixtures/`
- [x] T008 Create dir `config/`
- [x] T009 Create dir `experiments/`
- [x] T010 Create dir `scripts/`
- [x] T011 Create dirs `results/` and `reports/figures/`
- [x] T012 Create dir `diagrams/`
- [x] T013 `pyproject.toml`: `[project]` name=airbench + description
- [x] T014 pyproject: `requires-python = ">=3.13,<3.14"`
- [x] T015 pyproject: authors Salah Qadah + Andalus Kalash
- [x] T016 pyproject: `dynamic = ["version"]`
- [x] T017 pyproject: runtime deps huggingface_hub, psutil, matplotlib, numpy
- [x] T018 pyproject: optional extra `heavy = [transformers, torch, airllm, llama-cpp-python]`
- [x] T019 pyproject: dev group pytest, pytest-cov, ruff, pre-commit
- [x] T020 pyproject: `[project.scripts] airbench = "airbench.cli:main"`
- [x] T021 pyproject: `[build-system]` hatchling
- [x] T022 pyproject: `[tool.hatch.version] path = src/airbench/shared/version.py`
- [x] T023 pyproject: `[tool.hatch.build.targets.wheel] packages = ["src/airbench"]`
- [x] T024 pyproject: `[tool.ruff]` line-length 100, target-version py313
- [x] T025 pyproject: `[tool.ruff.lint]` select E/F/W/I/N/UP/B/C4/SIM, ignore E501
- [x] T026 pyproject: `[tool.pytest.ini_options]` testpaths + cov addopts
- [x] T027 pyproject: `[tool.coverage.run]` source=airbench, omit tests/fixtures
- [x] T028 pyproject: `[tool.coverage.report] fail_under = 85`
- [x] T029 Create `src/airbench/shared/version.py` with `VERSION = "1.00"`
- [x] T030 Create `src/airbench/shared/__init__.py`
- [x] T031 Create `src/airbench/__init__.py` importing VERSION as `__version__`
- [x] T032 Create `src/airbench/sdk/__init__.py`
- [x] T033 Create `src/airbench/runners/__init__.py`
- [x] T034 Create `src/airbench/metrics/__init__.py`
- [x] T035 Create `src/airbench/economics/__init__.py`
- [x] T036 `uv sync` → generate `uv.lock`
- [x] T037 Verify `uv.lock` tracked; no requirements.txt / venv / pip artifacts (R12)
- [x] T038 Verify `uv run python -c "import airbench; print(airbench.__version__)"` → 1.00
- [x] T039 `.gitignore`: `.venv`, `__pycache__`, `.pytest_cache`, `.ruff_cache`, `.coverage`
- [x] T040 `.gitignore`: `.env`
- [x] T041 `.gitignore`: weights (`*.gguf`, `*.safetensors`, `/Volumes/Backup` weights dir)
- [x] T042 `.env-example`: `HF_TOKEN=` placeholder (R11)
- [x] T043 `.env-example`: `OPENAI_API_KEY=` placeholder
- [x] T044 Verify `.env` not tracked by git
- [x] T045 Copy/adapt `scripts/check_file_lines.py` from HW4 (R7)
- [x] T046 Run check_file_lines on scaffolding → passes
- [x] T047 `.pre-commit-config.yaml`: ruff check hook
- [x] T048 pre-commit: ruff format hook
- [x] T049 pre-commit: file-lines hook
- [x] T050 pre-commit: version-sync hook
- [x] T051 `pre-commit install` + dry run
- [x] T052 `.github/workflows/ci.yml`: trigger on push/PR
- [x] T053 CI: `actions/setup-python@v5` python-version 3.13 (R13)
- [x] T054 CI: install uv
- [x] T055 CI: `uv sync`
- [x] T056 CI: `uv run ruff check`
- [x] T057 CI: `uv run python scripts/check_file_lines.py`
- [x] T058 CI: `uv run pytest`
- [x] T059 Create empty-valid `config/models.json`
- [x] T060 Create empty-valid `config/quant_levels.json`
- [x] T061 Create empty-valid `config/runtime.json`
- [x] T062 Create empty-valid `config/economics.json`
- [x] T063 Create empty-valid `config/budgets.json`
- [x] T064 Create empty-valid `config/hardware.json`
- [x] T065 Add LICENSE file
- [x] T066 `README.md` stub (title, authors, course, WIP)
- [x] T067 `git init`; set remote `uoh-sqak-ex05`
- [x] T068 `ruff check` clean on scaffolding
- [x] T069 Commit: scaffolding →commit

## Phase 1 — `shared/` (version, config, logging, gatekeeper, token_meter)

- [x] T080 Test: `test_version.py` `airbench.__version__ == "1.00"`
- [x] T081 Test: version equals importlib.metadata version
- [x] T082 Impl: confirm version single-source wiring green (R5)
- [x] T083 Test: `test_config.py` `load("models")` returns dict (fixture dir)
- [x] T084 Test: `load` honors `AIRBENCH_CONFIG_DIR` env override
- [x] T085 Test: `load` raises on missing file
- [x] T086 Test: `load` raises on malformed JSON
- [x] T087 Impl: `shared/config.py` `load(name)` + env override
- [x] T088 Test: `get_model("primary")` returns model dict
- [x] T089 Test: `get_model` raises on missing role
- [x] T090 Impl: `get_model(role)`
- [x] T091 Test+Impl: `get_runtime()`
- [x] T092 Test+Impl: `get_economics()`
- [x] T093 Test+Impl: `get_budgets()`
- [x] T094 Test+Impl: `get_quant_levels()` (ordered)
- [x] T095 Test+Impl: `get_hardware()`
- [x] T096 Test: required-key validation raises when a key is absent (no defaults) (R10)
- [x] T097 Impl: per-config required-key validators
- [x] T098 Fill `config/models.json`: primary hf_id **Qwen/Qwen2.5-7B-Instruct (ungated)**
- [x] T099 Fill primary params_b=7.62/hidden=3584/layers=28
- [x] T100 Fill primary fp16_gb=15.2, gated=false
- [x] T101 Fill primary gguf_repo (bartowski/Qwen2.5-7B-Instruct-GGUF) + weights_dir `/Volumes/Backup/hw5-weights`
- [x] T102 Fill alternative Llama-3.1-8B-Instruct (gated=true, needs HF_TOKEN+license) — documented, not used by default
- [x] T103 Fill extreme 70B (ungated GGUF repo + gguf_file Q4_K_M, size_gb=40)
- [x] T103a Test+Impl: `download_model` REFUSES a weights_dir on the internal volume (must be on SSD/`/Volumes`)
- [x] T104 Fill `config/quant_levels.json` Q8_0 (gguf_file, approx_gb)
- [x] T105 Fill Q5_K_M
- [x] T106 Fill Q4_K_M
- [x] T107 Fill Q2_K
- [x] T108 Fill `config/runtime.json` n_runs=5, warmup=1
- [x] T109 Fill runtime max_tokens=64, seed=42
- [x] T110 Fill runtime prompt set (deterministic)
- [x] T111 Fill runtime ctx_sizes [512,2048,8192] + timeout_s + llamacpp_bin
- [x] T111a Fill runtime `perplexity_corpus` path + `red_line_ppl_delta` threshold + `airllm_timebox_min` (45–60)
- [x] T112 Fill `config/economics.json` API providers (usd_per_1m_in/out)
- [x] T113 Fill economics CAPEX (Mac + SSD) + amortize_years=3
- [x] T114 Fill economics idle_w, active_w, kwh_usd
- [x] T115 Fill economics avg_req_tokens_in/out
- [x] T116 Fill `config/budgets.json` max_download_gb, max_subprocess_s
- [x] T117 Fill budgets max_api_calls, allow_network=true
- [x] T118 Fill `config/hardware.json` M2 GPU peak GFLOP/s (FP16)
- [x] T119 Fill hardware unified-memory bandwidth GB/s
- [x] T120 Fill hardware SSD read/write MB/s (measured ~498/358)
- [x] T121 Fill hardware cores (4P+4E), RAM_gb=8
- [x] T122 Test: `test_logging.py` level from env
- [x] T123 Test: logging no duplicate handlers
- [x] T124 Impl: `shared/logging_config.py` (adapt HW4)
- [x] T125 Impl: `GateEvent` dataclass (ts, label, kind, dur, bytes, ok)
- [x] T126 Impl: `RunResult` dataclass
- [x] T127 Impl: `BudgetExceeded` exception
- [x] T128 Test: `test_gatekeeper.py` `download` happy path records GateEvent (mock hf)
- [x] T129 Test: download writes to dest path
- [x] T130 Test: download denies size > max_download_gb → BudgetExceeded
- [x] T131 Test: `run_subprocess` records duration
- [x] T132 Test: run_subprocess records exit code + stdout tail
- [x] T133 Test: run_subprocess enforces timeout → records timeout event
- [x] T134 Test: `load_model` wraps callable, records label+duration
- [x] T135 Test: load_model propagates load error cleanly
- [x] T136 Test: `api_call` counts against max_api_calls
- [x] T137 Test: api_call denies over budget → BudgetExceeded
- [x] T138 Test: `allow_network=false` denies download + api_call
- [x] T139 Test: ledger persisted to `results/<run>/gate_ledger.json` (atomic write)
- [x] T140 Test: ledger property returns events in order
- [x] T141 Impl: `ApiGatekeeper.__init__(budgets, ledger_path)`
- [x] T142 Impl: `download`
- [x] T143 Impl: `run_subprocess`
- [x] T144 Impl: `load_model`
- [x] T145 Impl: `api_call`
- [x] T146 Impl: `record` + atomic ledger write
- [x] T147 Impl: `ledger` property
- [x] T148 Refactor: split IO helpers to `gatekeeper_io.py` if >150 lines (R7)
- [x] T149 **Meta-test** `test_no_raw_external_calls.py`: grep src for `subprocess`
- [x] T150 Meta-test: grep for `requests`/`urllib`
- [x] T151 Meta-test: grep for `hf_hub_download`/`snapshot_download`
- [x] T152 Meta-test: grep for raw `.from_pretrained`/`AutoModel(` outside gatekeeper (R3)
- [x] T153 Reuse `shared/token_meter.py` from HW4
- [x] T154 Test: `test_token_meter.py` counts tokens on fixture text
- [x] T158 Test: config rejects extra/unknown top-level key (strict schema)
- [x] T159 Test: `get_quant_levels` preserves declared order Q8→Q5→Q4→Q2
- [x] T160 Test: gatekeeper ledger entry JSON schema (keys + types)
- [x] T161 Test: gatekeeper run-id is unique per SDK session
- [x] T162 Test: download records bytes-in; api_call records bytes-out
- [x] T163 Test: BudgetExceeded message names the violated budget
- [x] T164 Test: ledger survives append across multiple calls (no overwrite)
- [x] T165 Test: gatekeeper redacts HF_TOKEN/API key from ledger + logs (R11)
- [x] T166 Impl: token/secret redaction in record()
- [x] T167 Test: token_meter handles empty + unicode text
- [x] T168 Test: config path resolution relative vs absolute
- [x] T169 Impl: close any gaps surfaced by T158–T168
- [x] T155 Confirm all Phase-1 files ≤150 lines
- [x] T156 ruff clean Phase 1
- [x] T157 Commit: shared layer →commit

## Phase 2 — SDK Façade

- [ ] T170 Test: `test_sdk.py` `BenchSDK(config_dir)` builds config
- [ ] T171 Test: SDK builds gatekeeper + logger
- [ ] T172 Test: SDK holds single shared gatekeeper instance
- [ ] T173 Test: `probe_hardware` delegates (mock)
- [ ] T174 Test: `run_baseline_oom` delegates
- [ ] T175 Test: `run_baseline_llamacpp` delegates
- [ ] T176 Test: `run_airllm` delegates
- [ ] T177 Test: `run_airllm` falls back to layered on ConstraintReport
- [ ] T178 Test: `run_layered_demo` delegates
- [ ] T179 Test: `run_quant_sweep` delegates
- [ ] T180 Test: `run_extreme` delegates
- [ ] T181 Test: `compute_economics` delegates
- [ ] T182 Test: `make_figures` delegates
- [ ] T183 Impl: `sdk/facade.py` `__init__`
- [ ] T184 Impl: `probe_hardware`
- [ ] T185 Impl: `run_baseline_oom`
- [ ] T186 Impl: `run_baseline_llamacpp(quant)`
- [ ] T187 Impl: `run_airllm` + layered fallback
- [ ] T188 Impl: `run_layered_demo`
- [ ] T189 Impl: `run_quant_sweep`
- [ ] T190 Impl: `run_extreme`
- [ ] T191 Impl: `compute_economics`
- [ ] T192 Impl: `make_figures`
- [ ] T196 Test: SDK generates a unique run-id + creates `results/<run-id>/` dir
- [ ] T197 Test: SDK passes run-id-scoped ledger path to gatekeeper
- [ ] T198 Test: SDK methods write artifacts under the run dir
- [ ] T199 Test: SDK surfaces ConstraintReport without raising to caller
- [ ] T200 Test: SDK config_dir=None falls back to repo `config/`
- [ ] T201 Test: SDK logs start/end of each run method
- [ ] T202 Impl: run-id + artifact-dir helper
- [ ] T203 Impl: result-path resolver used by all methods
- [ ] T204 Impl: error-to-report wrapping in run_airllm/run_extreme
- [ ] T205 Test: `run_quant_sweep` returns one RunMetrics per level
- [ ] T206 Impl: aggregation of sweep results into a summary dict
- [ ] T193 Refactor façade ≤150 lines (helpers/mixins)
- [ ] T194 Grep: only SDK imported by CLI/experiments (R1)
- [ ] T195 ruff clean + commit: SDK →commit

## Phase 3 — `metrics/`

- [x] T210 Test: `test_timing.py` first-token timestamp → exact TTFT
- [x] T211 Test: subsequent timestamps → exact TPOT mean
- [x] T212 Test: TPOT formula `(total-TTFT)/(n-1)`
- [x] T213 Test: std across n_runs
- [x] T214 Test: warmup run discarded
- [x] T215 Test: throughput = output_tokens/total_time
- [x] T216 Test: TPOT n=1 edge → None/NaN handled
- [x] T217 Impl: `metrics/timing.py` `measure_generation(generate_fn, n_runs, warmup)`
- [x] T218 Impl: split Prefill (first) vs Decode (rest) from token-timestamp callback
- [x] T219 Impl: `TimingStats` dataclass + mean±std
- [x] T220 Impl: `TimingStats.to_dict`
- [x] T221 Test: `test_memory.py` parse psutil peak RSS (mock) → MemSample
- [x] T222 Test: unified-mem sampler parses fixture `vm_stat`
- [x] T223 Test: unified-mem sampler parses fixture `footprint`
- [x] T224 Test: memory series peak extraction
- [x] T225 Impl: `metrics/memory.py` peak RSS sampler
- [x] T226 Impl: unified-memory pressure sampler (subprocess via gatekeeper)
- [x] T227 Impl: `MemSample` series + peak
- [x] T228 Test: `test_paging.py` swapins delta from two fixture snapshots
- [x] T229 Test: swapouts delta
- [x] T230 Test: page-fault delta + parse robustness
- [x] T231 Impl: `metrics/paging.py` before/after `vm_stat` → deltas
- [x] T232 Test: `test_roofline.py` compute-bound case classified
- [x] T233 Test: memory-bound case (Decode) classified
- [x] T234 Test: arithmetic intensity = FLOPs/Bytes hand-checked
- [x] T235 Test: achieved GFLOP/s vs ceiling min(compute, bw·intensity)
- [x] T236 Test: ceilings read from `config/hardware.json` (not hardcoded)
- [x] T237 Impl: `metrics/roofline.py` classification + ceilings
- [x] T238 Impl: model-FLOPs/token estimator (params, layers)
- [x] T239 Impl: bytes/token estimator (dtype, KV cache)
- [x] T240 Test: `test_aggregate.py` JSON round-trip
- [x] T241 Test: CSV round-trip
- [x] T242 Impl: `metrics/aggregate.py` JSON writer → `results/`
- [x] T243 Impl: CSV writer
- [x] T246 Test: timing rejects empty token stream
- [x] T247 Test: timing handles single-run (no std → 0/NaN documented)
- [x] T248 Test: throughput excludes TTFT when configured
- [x] T249 Test: memory sampler returns 0-safe when psutil missing (graceful)
- [x] T250 Test: paging parser tolerates locale/format variance in vm_stat
- [x] T251 Test: roofline boundary case (intensity exactly at ridge point)
- [x] T252 Test: FLOPs/token estimator matches hand calc for 8B dims
- [x] T253 Test: bytes/token includes KV-cache growth with context length
- [x] T254 Test: aggregate writer creates parent dirs
- [x] T255 Test: aggregate is deterministic (stable key order)
- [x] T256 Impl: close gaps from T246–T255
- [x] T244 Confirm metrics files ≤150 lines
- [x] T245 ruff clean + commit: metrics →commit

## Phase 4 — `runners/`

- [x] T260 Test: `test_baseline_oom.py` mocked loader raises → FailureReport captures exc type
- [x] T261 Test: FailureReport captures peak-mem trace
- [x] T262 Test: FailureReport captures timestamp + diagnosis hint
- [x] T263 Impl: `runners/baseline_oom.py` HF FP16 load via gatekeeper.load_model
- [x] T264 Impl: capture exception + memory trace
- [x] T265 Impl: `FailureReport` dataclass
- [x] T266 Test: `test_baseline_llamacpp.py` mocked subprocess stdout → RunMetrics
- [x] T267 Test: parser extracts prefill timing
- [x] T268 Test: parser extracts eval/decode timing + tokens/sec
- [x] T269 Test: parser handles malformed output
- [x] T270 Impl: `runners/baseline_llamacpp.py` download GGUF (gatekeeper)
- [x] T271 Impl: run llama.cpp subprocess (gatekeeper) + capture stdout
- [x] T272 Impl: `llamacpp_parser.py` stdout parser
- [x] T273 Impl: `RunMetrics` dataclass
- [x] T274 Test: `test_layered.py` iterate N fake blocks materialize→forward→free
- [x] T275 Test: per-layer IO time recorded
- [x] T276 Test: per-layer compute time recorded
- [x] T277 Test: peak mem bounded (frees each block before next)
- [x] T278 Impl: `runners/layered.py` per-block streaming from SSD
- [x] T279 Impl: per-layer instrumentation (IO vs compute)
- [x] T280 Impl: `LayeredMetrics` dataclass
- [x] T281 Test: `test_airllm.py` mocked airllm success → RunMetrics
- [x] T282 Test: ImportError → ConstraintReport (no crash)
- [x] T283 Test: CUDA-unavailable → ConstraintReport
- [x] T284 Impl: `runners/airllm.py` real attempt via gatekeeper.load_model
- [x] T285 Impl: `ConstraintReport` dataclass (reason, platform, fallback)
- [x] T286 Test: `test_quant_sweep.py` serial loop download→bench→quality→delete (mock)
- [x] T287 Test: never holds >1 weight file (delete before next download)
- [x] T288 Test: red-line rule = first level where ΔPPL vs Q8 baseline exceeds config threshold
- [ ] T288a Test: `llama-perplexity` fixture output parsed → final PPL float
- [x] T289 Impl: `runners/quant_sweep.py` serial sweep reusing baseline_llamacpp
- [x] T290 Impl: quality probe = **perplexity via llama.cpp `llama-perplexity`** (gatekeeper subprocess) on a fixed corpus; deterministic exact-match as secondary signal
- [ ] T290a Impl: perplexity parser + config-driven red-line threshold (ΔPPL)
- [x] T291 Impl: weight cleanup between levels
- [x] T292 Test: `test_extreme.py` 70B path mocked → RunMetrics high per-token IO
- [x] T293 Impl: `runners/extreme.py` 70B via airllm/layered, few tokens
- [x] T296 Test: baseline_oom maps MPS-OOM vs generic OOM to diagnosis hint
- [x] T297 Test: baseline_oom records partial memory growth before failure
- [x] T298 Test: llamacpp runner honors timeout → records failure not crash
- [x] T299 Test: llamacpp parser variant (Metal log lines) parsed
- [x] T300 Test: llamacpp parser extracts ctx size + n_threads echoed
- [x] T301 Test: layered runner records cumulative IO bytes
- [x] T302 Test: layered runner reports predicted vs actual per-layer IO
- [x] T303 Test: airllm runner passes weights_dir from config
- [x] T304 Test: quant_sweep continues after one level fails (records + moves on)
- [x] T305 Test: quality probe deterministic under fixed seed
- [x] T306 Test: extreme runner caps tokens from config
- [x] T307 Test: extreme runner ConstraintReport path
- [x] T308 Impl: close gaps from T296–T307
- [x] T309 Verify runners route ALL externals via gatekeeper (grep, R3)
- [x] T294 Confirm runner files ≤150 lines (split parsers)
- [x] T295 ruff clean + commit: runners →commit

## Phase 5 — `economics/`

- [x] T310 Test: `test_economics.py` api_cost provider A token mix hand-checked
- [x] T311 Test: api_cost provider B
- [x] T312 Test: onprem_annual_cost = capex/years + watts·hours·kwh
- [x] T313 Test: break_even requests/day hand-checked
- [x] T314 Test: tco_curve monotonic
- [x] T315 Test: utilization sensitivity
- [x] T316 Impl: `economics/calculator.py` api_cost
- [x] T317 Impl: onprem_annual_cost
- [x] T318 Impl: break_even
- [x] T319 Impl: tco_curve
- [x] T320 Impl: `EconomicsReport` dataclass
- [x] T321 Impl: JSON writer → `results/economics.json`
- [x] T322 Confirm ≤150 lines + ruff clean + commit →commit

## Phase 6 — CLI

- [ ] T335 Test: `test_cli.py` `probe` dispatches to SDK
- [ ] T336 Test: `baseline-oom` dispatches
- [ ] T337 Test: `baseline` dispatches (quant arg)
- [ ] T338 Test: `airllm` dispatches
- [ ] T339 Test: `layered` dispatches
- [ ] T340 Test: `quant-sweep` dispatches
- [ ] T341 Test: `extreme` dispatches
- [ ] T342 Test: `economics` dispatches
- [ ] T343 Test: `figures` dispatches
- [ ] T344 Test: `all` runs full pipeline order
- [ ] T345 Test: bad args → nonzero exit + usage
- [ ] T346 Impl: `cli.py` argparse + subcommand registry
- [ ] T347 Impl: `cli_handlers.py` probe/baseline handlers
- [ ] T348 Impl: airllm/layered/quant/extreme handlers
- [ ] T349 Impl: economics/figures/all handlers
- [ ] T350 Impl: each handler writes artifacts + prints summary table
- [ ] T351 Impl: `main()` entry
- [ ] T355 Test: CLI `--version` prints VERSION (single-source)
- [ ] T356 Test: CLI `--config-dir` overrides config location
- [ ] T357 Test: CLI `--run-id` / auto run-id behavior
- [ ] T358 Test: CLI `baseline --quant Q4_K_M` passes the level through
- [ ] T359 Test: CLI prints artifact paths on completion
- [ ] T360 Test: CLI nonzero exit when an SDK method raises
- [ ] T361 Impl: `--version`/`--config-dir`/`--run-id` flags
- [ ] T362 Impl: consistent summary-table formatter
- [ ] T363 Impl: exit-code mapping (success/failure/constraint)
- [ ] T364 Verify `airbench all` dry-run prints planned steps
- [ ] T352 Verify `uv run airbench --help`
- [ ] T353 Verify each subcommand `--help`
- [ ] T354 CLI ≤150 lines + ruff clean + commit →commit

## Phase 7 — scripts/ & experiments/

- [ ] T370 Test: `test_download_model.py` routes through gatekeeper (mock)
- [ ] T371 Test: download writes to weights_dir on SSD
- [ ] T372 Test: download uses HF_TOKEN for gated model
- [ ] T373 Test: download fallback model selection from config
- [ ] T374 Impl: `scripts/download_model.py` gatekeeper-wrapped
- [ ] T375 Impl: download quant-file selection + gated auth
- [ ] T376 Test: `test_make_figures.py` fixture results → expected PNG filenames (Agg)
- [ ] T377 Impl: `scripts/make_figures.py` reads `results/*`
- [ ] T378 Adapt `scripts/fill_submission_pdf.py` from HW4 for `uoh-sqak-ex05.pdf`
- [ ] T379 Confirm scripts ≤150 lines
- [ ] T380 Test: `test_experiments.py` probe wrapper calls SDK (mock)
- [ ] T381 Test: each experiment wrapper calls matching SDK method
- [ ] T382 Impl: `experiments/probe.py`
- [ ] T383 Impl: `experiments/baseline_oom.py`
- [ ] T384 Impl: `experiments/baseline_llamacpp.py`
- [ ] T385 Impl: `experiments/airllm_run.py` (+ layered fallback)
- [ ] T386 Impl: `experiments/quant_sweep.py`
- [ ] T387 Impl: `experiments/economics_run.py`
- [ ] T388 Impl: `experiments/extreme_airllm.py`
- [ ] T389 Grep: experiments import only the SDK
- [ ] T390 experiments ≤150 lines + ruff clean + commit →commit

## Phase 8 — Test Sweep & Quality Gates

- [ ] T400 `uv run pytest --cov` → measure coverage
- [ ] T401 List uncovered lines; add tests
- [ ] T402 Reach ≥85% coverage (R9)
- [ ] T403 `ruff check` → zero failures (R8)
- [ ] T404 `check_file_lines.py` → all ≤150 (R7)
- [ ] T405 Run meta-tests (no raw external calls) → green (R3)
- [ ] T406 Run version-sync tests → green (R5)
- [ ] T407 Run full suite with network disabled → green
- [ ] T408 Confirm no GPU/model/key needed (grader Path D)
- [ ] T411 Per-module coverage check: gatekeeper ≥90%
- [ ] T412 Per-module coverage check: metrics ≥90%
- [ ] T413 Per-module coverage check: economics ≥90%
- [ ] T414 Lint the tests themselves (ruff over tests/)
- [ ] T415 Fresh-clone smoke: `uv sync` + `uv run pytest` in a clean checkout
- [ ] T416 Confirm no network access in test run (monkeypatch socket)
- [ ] T417 Confirm deterministic test run (no flakiness over 3 runs)
- [ ] T409 Push branch → **GitHub CI green on Python 3.13** (R13)
- [ ] T410 Add CI badge to README →commit

## Phase 9 — Real Experiments on the Mac (commit artifacts as produced)

- [ ] T420 `airbench probe` → `results/hardware.json` (H1)
- [ ] T421 Verify hardware.json matches measured spec (8GB/M2)
- [ ] T422 Verify SSD entry (489GB @ ~498/358) in hardware.json
- [ ] T423 Commit hardware.json →commit
- [ ] T424 Primary Qwen2.5-7B is ungated — confirm download needs NO HF_TOKEN (no manual web step)
- [ ] T425 (Only if using the Llama alternative) put HF_TOKEN in `.env` + accept license — otherwise skip
- [ ] T426 `airbench baseline-oom` → HF FP16 load on the Mac
- [ ] T427 Capture OOM/MPS allocation failure → `results/baseline/oom.log` (H2)
- [ ] T428 Screenshot failure → `reports/figures/oom_screenshot.png`
- [ ] T429 Record peak RSS at failure
- [ ] T430 Record unified-mem pressure at failure (H2 evidence)
- [ ] T431 Write bottleneck diagnosis note (memory-bound: 16GB > 8GB)
- [ ] T432 Commit baseline OOM artifacts →commit
- [ ] T433 `download_model.py` primary Q4_K_M to SSD
- [ ] T434 Verify weight file + gate_ledger entry
- [ ] T435 `airbench baseline` (llama.cpp Q4) → N≥5 runs
- [ ] T436 Capture TTFT separately (Prefill) (H5)
- [ ] T437 Capture TPOT separately (Decode) (H5)
- [ ] T438 Save `results/baseline/q4_metrics.json` →commit
- [ ] T439 quant-sweep Q8_0 → bench N≥5 → quality → delete
- [ ] T440 quant-sweep Q5_K_M → bench → quality → delete
- [ ] T441 quant-sweep Q4_K_M → bench → quality → delete
- [ ] T442 quant-sweep Q2_K → bench → quality → delete
- [ ] T443 Identify accuracy **red line** from sweep using **perplexity (ΔPPL vs Q8)** crossing the configured threshold (H4/H9)
- [ ] T444 Save `results/quant/*.json` per level →commit
- [ ] T445 `airbench airllm` real attempt streaming 7B FP16 from SSD — **TIME-BOXED to `airllm_timebox_min` (45–60 min)**; on timeout/error → ConstraintReport + proceed (H3)
- [ ] T446 Capture success metrics OR ConstraintReport honestly (no open-ended CUDA/MPS debugging)
- [ ] T447 Write `reports/airllm_constraint.md`
- [ ] T448 `airbench layered` equivalent demo → per-layer IO vs compute
- [ ] T449 Capture `vm_stat` swap deltas during run (paging, H8)
- [ ] T450 Save `results/airllm/*.json` (TTFT/TPOT/throughput/peak-mem/IO), N≥5 (H5)
- [ ] T452 Capture `gate_ledger.json` for each real run (proves Gatekeeper wired) →commit
- [ ] T453 Record CPU/GPU utilization sample during baseline run (bottleneck evidence)
- [ ] T454 Record CPU/GPU utilization during layered/airllm run
- [ ] T455 Capture macOS memory-pressure (green/yellow/red) during runs
- [ ] T456 Save quality-probe outputs per quant level to `results/quant/quality/`
- [ ] T457 Record SSD read throughput observed during streaming (vs measured 498 MB/s)
- [ ] T458 Log wall-clock + tokens for each run into a master `results/index.json`
- [ ] T459 Verify every `results/*` artifact is committed (no gitignored data)
- [ ] T460 Re-run one config to confirm reproducibility of mean±std
- [ ] T451 Compare baseline vs layered/airllm; note honest result →commit

## Phase 10 — Economics & Figures

- [ ] T465 Measure/estimate active power draw → update economics.json
- [ ] T466 `airbench economics` → `results/economics.json` + break-even (H7)
- [ ] T467 Commit economics.json →commit
- [ ] T468 Figure `ttft.png` (±std bars) (H6)
- [ ] T469 Figure `tpot.png` (±std bars) (H6)
- [ ] T470 Figure `throughput_vs_quant.png` (H6)
- [ ] T471 Figure `peak_memory.png` baseline vs quant vs airllm (H6)
- [ ] T472 Figure `latency_breakdown.png` (prefill/decode/IO) (H8)
- [ ] T473 Figure `roofline.png` ceilings + run points (H8)
- [ ] T474 Figure `quant_pareto.png` quality vs memory/speed (H9)
- [ ] T475 Figure `breakeven.png` (H7)
- [ ] T476 Figure `tco_curve.png` (H7)
- [ ] T477 Verify each figure: labeled axes, units, title
- [ ] T478 Verify error bars (±std) where applicable
- [ ] T479 Commit figures →commit

## Phase 11 — Report, README, Diagrams

- [ ] T490 `diagrams/class_diagram.mmd` (SDK, Gatekeeper, runners, metrics, economics) (R2)
- [ ] T491 `diagrams/block_diagram.mmd` data-flow
- [ ] T492 Render diagrams to PNG/SVG
- [ ] T493 Report §Hardware (from hardware.json) (H1)
- [ ] T494 Report §Model justification: params/format/FP16-size vs 8GB (H1/5.1)
- [ ] T495 Report §Baseline failure narrative + logs (H2)
- [ ] T496 Report §Bottleneck diagnosis: memory-bound evidence (H2)
- [ ] T497 Report §Roofline analysis (compute vs memory bound) (H8)
- [ ] T498 Report §AirLLM constraint (Apple-Silicon/CUDA) (H3)
- [ ] T499 Report §Layered streaming mechanism (H3)
- [ ] T500 Report §Virtual-memory/paging mapping (H8)
- [ ] T501 Report §Quantization sweep + memory/speed/quality (H4)
- [ ] T502 Report §Accuracy red line discussion (H4)
- [ ] T503 Report §TTFT vs TPOT (Prefill/Decode), mean±std (H5)
- [ ] T504 Report §Throughput/latency price on modest hardware (§4)
- [ ] T505 Report §Economics API vs On-Prem + break-even + TCO (H7)
- [ ] T506 Report §Economics sensitivity analysis (H7)
- [ ] T507 Report §Original extensions (70B, Pareto, context-length) (H9)
- [ ] T508 Report §Honest negative results
- [ ] T509 `reports/research_questions.md` §4 Q1 (bottleneck: memory or compute, how identified)
- [ ] T510 research_questions §4 Q2 (AirLLM resource allocation ↔ paging)
- [ ] T511 research_questions §4 Q3 (quantization mem/speed/quality, red line)
- [ ] T512 research_questions §4 Q4 (Prefill/Decode ↔ TTFT/TPOT)
- [ ] T513 research_questions §4 Q5 (throughput/latency price)
- [ ] T514 research_questions §4 Q6 (when is local worth it vs API)
- [ ] T515 Link each research answer to its evidence file (H11)
- [ ] T516 README §Overview + course/instructor/authors
- [ ] T517 README §Hardware table
- [ ] T518 README §Install (`uv sync`) + heavy extra + HF token note
- [ ] T519 README §Reproduce each experiment (every `airbench` command)
- [ ] T520 README embed figures (ttft/tpot/throughput/memory)
- [ ] T521 README embed roofline + pareto + breakeven figures
- [ ] T522 README §Results summary + key findings
- [ ] T523 README §References (HF model, AirLLM, llama.cpp, Ollama, Lec 08)
- [ ] T524 README §Acknowledgments (co-authors, course staff)
- [ ] T525 README CI badge + repo structure
- [ ] T526 README class-diagram embed (R2)
- [ ] T527 Cross-check README vs spec §8 checklist (all items present)
- [ ] T529 `docs/adr/ADR-001-model-choice.md` (Qwen2.5-7B primary + why too big for 8GB; Llama-3.1 gated alternative)
- [ ] T530 `docs/adr/ADR-002-airllm-honest-path.md` (CUDA constraint + layered equivalent)
- [ ] T531 `docs/adr/ADR-003-gatekeeper-design.md` (wired-for-real, gate+record)
- [ ] T532 `docs/adr/ADR-004-serial-quant-sweep.md` (download→bench→delete rationale)
- [ ] T533 `docs/adr/ADR-005-ssd-streaming.md` (external SSD as weight+stream source)
- [ ] T534 `docs/adr/ADR-006-mock-strategy.md` (grader Path D, no GPU/model/key)
- [ ] T535 `docs/adr/ADR-007-metrics-methodology.md` (TTFT/TPOT split, N≥5, mean±std)
- [ ] T536 `docs/adr/ADR-008-economics-assumptions.md` (CAPEX/OPEX/break-even inputs)
- [ ] T537 Link ADRs from README + technical_report
- [ ] T538 `docs/prd/` per-mechanism mini-PRDs (baseline, airllm, quant, economics)
- [ ] T528 Commit report+README+diagrams+ADRs →commit

## Phase 12 — Extension: Extreme 70B AirLLM + Context Sweep

- [ ] T540 `download_model.py` 70B Q4 (~40GB) to SSD (gatekeeper; slow)
- [ ] T541 Verify 70B file + gate_ledger entry
- [ ] T542 `airbench extreme` → stream 70B, few tokens (H9)
- [ ] T543 Capture IO-bound per-token numbers
- [ ] T544 Compute per-token IO vs SSD-bandwidth prediction (~0.5GB/s)
- [ ] T545 Save `results/extreme/*.json`
- [ ] T546 Add extreme points to figures (throughput/memory/roofline)
- [ ] T547 Context-length sweep ctx 512 → metrics
- [ ] T548 Context sweep ctx 2048 → metrics
- [ ] T549 Context sweep ctx 8192 → metrics
- [ ] T550 Plot compute→memory bound transition vs ctx (H9)
- [ ] T551 Save context-sweep results + figure
- [ ] T552 Write/expand report §5.7 extensions section with 70B + context-sweep findings
- [ ] T552a **Fold extension results back into README**: update §Results summary, embed 70B/context-sweep figures, refresh research-question Q5 (throughput/latency price) with extreme data
- [ ] T552b Re-run README↔spec-§8 cross-check after the fold-back (T527 redo)
- [ ] T553 (If 70B infeasible) document attempt + why honestly (negative result counts) →commit

## Phase 13 — Submission & Push

- [ ] T565 Final `uv run pytest --cov` green ≥85%
- [ ] T566 Final `ruff check` clean
- [ ] T567 Final file-lines pass
- [ ] T568 Confirm GitHub CI green on main (Python 3.13)
- [ ] T569 Confirm NO weights committed
- [ ] T570 Confirm `results/` artifacts present + `.env` absent
- [ ] T571 Bump VERSION across the build; single-source test green
- [ ] T572 Verify commit history spread over time, logically grouped (R13)
- [ ] T573 Make repo **public**
- [ ] T574 Share repo with `rmisegal@gmail.com`
- [ ] T575 Verify repo accessible logged-out
- [ ] T576 `fill_submission_pdf.py` → `uoh-sqak-ex05.pdf` (self-grade 85, group uoh-sqak)
- [ ] T577 Verify PDF fields unaltered + filename exact + nothing extra
- [ ] T578 Remind user: both members submit on Moodle id=278465 (separate timestamps)
- [ ] T579 Final README pass; tag release
- [ ] T580 Push to GitHub; confirm deadline margin (Fri 2026-06-26 23:59) →commit/push

---

## Critical PRD→Todo Coverage Map (must be 100%)

| PRD demand | Tasks |
|---|---|
| **A** hardware premise (measured) | T118–T121, T420–T423, T493 |
| **A** SSD weights + streaming | T041, T101, T374, T433, T445–T450 |
| **5.1** model justification (params/format/size) (H1) | T494 |
| **B** primary Qwen2.5-7B (ungated); Llama-3.1 alt | T098–T103a, T424–T425 |
| red-line via perplexity (llama-perplexity) | T111a, T214, T288–T290a, T443 |
| AirLLM time-boxed + fallback | T111a, T445–T446 |
| weights download to SSD (not internal) | T101, T103a |
| 70B/context-sweep folded into report+README | T552–T552b |
| **B** baseline OOM (HF FP16) | T260–T265, T426–T432 |
| **B** baseline llama.cpp runnable | T266–T273, T433–T438 |
| **B** GGUF quant sweep serial + red line | T104–T107, T286–T291, T439–T444, T501–T502 |
| **B** real AirLLM + honest fallback | T281–T285, T445–T447, T498–T499 |
| **B** 70B extreme extension | T292–T293, T540–T546, T553 |
| **C** SDK layer (R1) | T170–T195 |
| **C** Gatekeeper wired + meta-test (R3) | T125–T152, T309, T405, T452 |
| **C** ADRs + per-mechanism PRDs (HW4 docs shape) | T529–T538 |
| **C** secret redaction in logs/ledger (R11) | T165–T166 |
| **C** config-driven (R4/R10) | T059–T064, T083–T121 |
| **C** version single-source (R5) | T029–T031, T080–T082, T406, T571 |
| **C** uv only (R12) | T036–T037 |
| **C** ruff clean (R8) | T024–T025, T403 |
| **C** pytest ≥85% mocked (R6/R9) | T400–T402, T407–T408, all test tasks |
| **C** ≤150 lines/file (R7) | T045, T148, T193, T244, T294, T322, T354, T379, T404 |
| **C** CI green py3.13 (R13) | T052–T058, T409, T568 |
| **C** no secrets (.env) (R11) | T040, T042–T044, T424, T570 |
| **C** class diagram (R2) | T490–T492, T526 |
| **C** spread commits (R13) | all →commit markers, T572 |
| **§8** experiments/*.py scripts | T380–T390 |
| **5.4** TTFT/TPOT separate, mean±std, N≥5 (H5) | T210–T220, T436–T437, T450, T503 |
| **5.4** memory + paging (H8) | T221–T231, T429–T430, T449 |
| roofline hardware peaks + SSD IO config | T118–T121, T237–T239, T544 |
| **5.6** Roofline + bound analysis (H8) | T232–T239, T473, T497 |
| **5.4/5.7** comparative graphs (H6) | T376–T377, T468–T479 |
| **5.5** economics + break-even (H7) | T310–T321, T465–T467, T475–T476, T505–T506 |
| **5.6** lecture-concept analysis (H8) | T495–T504 |
| **5.7** original extensions (H9) | T474, T540–T551 |
| **§4** research questions (H11) | T509–T515 |
| **§7/§8** reproducible repo + report (H10) | T370–T390, T493–T528 |
| **submission** PDF + public + share | T573–T578 |
| **risks** Qwen fallback / HF gate | T102, T373, T425, T447 |

**Audit result:** every PRD demand maps to ≥1 concrete task; no demand unmapped.

### → GATE 2: Approve Plan.md + Todo.md before any application code (T001 onward).
