# ADR-003 — Gatekeeper: the single, wired chokepoint for all external calls (R3)

**Status:** accepted

**Context.** HW3/HW4 lost marks for a *decorative* gatekeeper. R3 requires every external
interaction (model download, inference subprocess, in-process model load, API price lookup) to
route through one component that genuinely gates and records.

**Decision.** `ApiGatekeeper` exposes `download / run_subprocess / load_model / api_call`. It
**gates** (download-size + subprocess-timeout + api-call + network-off budgets from
`config/budgets.json`) and **records** every call to `results/<run>/gate_ledger.json` (an
append-only, atomically-written audit trail with secret redaction). A **meta-test**
(`test_no_raw_external_calls`) greps the source tree and fails if any module outside the
gatekeeper makes a raw `subprocess`/`requests`/`hf_hub_download`/`from_pretrained` call. The
only allow-listed exception is `sdk/loaders.py`, whose heavy-load **closures execute solely via
`gatekeeper.load_model`** (gated + recorded).

**Consequences.** The committed ledgers from the real runs are concrete proof the Gatekeeper is
wired, not decorative. Adding any new external call without going through it breaks the suite.
