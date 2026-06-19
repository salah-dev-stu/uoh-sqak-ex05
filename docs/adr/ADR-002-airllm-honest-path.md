# ADR-002 — AirLLM on Apple Silicon: real attempt, time-boxed, with an equivalent fallback

**Status:** accepted

**Context.** AirLLM (layer-by-layer streaming) is **CUDA-centric**; MPS/Apple-Silicon support
is unreliable. The spec explicitly states a *well-analyzed negative result counts equally*.

**Decision.** Attempt the **real** AirLLM load (streaming FP16 from the SSD) through the
Gatekeeper, **time-boxed** to `runtime.airllm_timebox_min` (45–60 min). On ImportError /
no-CUDA / timeout, emit a `ConstraintReport` and fall back to our **equivalent layer-streaming
demo** (`runners/layered.py`) that materializes each on-disk safetensors shard, forwards, and
frees it — mirroring AirLLM's mechanism on MPS/CPU and measuring real SSD I/O vs compute.

**Consequences.** Either outcome is gradeable (H3). No open-ended CUDA debugging. The paging /
virtual-memory analysis (H8) is driven by the layered demo's measured per-layer I/O regardless
of whether the AirLLM library itself initializes.
