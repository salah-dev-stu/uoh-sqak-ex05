# ADR-006 — Mock the heavy bits so the grader needs no GPU/model/key (Path D)

**Status:** accepted

**Context.** The grader runs `uv run pytest` with no GPU, no downloaded model, and no API key.

**Decision.** Heavy deps (`torch`, `transformers`, `airllm`, `llama-cpp-python`) live in an
optional `heavy` extra that **CI and tests never install**. Every external op is injected/mocked:
runners take callables; the gatekeeper takes `fetcher`/`runner`; `vm_stat`/`psutil`/`llama.cpp`
output is fed from fixtures. The **real** experiment artifacts (metrics JSON, ledgers, logs,
figures) are committed so the analysis stands alone.

**Consequences.** `pytest` is green with no hardware (87% coverage). The grade never depends on
re-running a multi-GB model.
