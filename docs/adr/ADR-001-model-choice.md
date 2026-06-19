# ADR-001 — Model choice: Qwen2.5-7B-Instruct (Llama-3.1-8B as gated alternative)

**Status:** accepted

**Context.** The machine is an Apple M2 with **8 GB unified memory**. We need a model
"too big to fit comfortably" (spec 5.1). A 70B-class model is the spec's example, but it
cannot even be *stored* on the internal disk (~9 GB free), and FP16 won't fit RAM.

**Decision.** Primary = **Qwen2.5-7B-Instruct (ungated)**. Its FP16 footprint (~15.2 GB) is
~2× the 8 GB RAM → a guaranteed, fast, reproducible OOM baseline, while its GGUF quants
(3–8 GB) are storable on the external SSD and runnable. Chosen over the gated
Llama-3.1-8B to avoid a mid-run HF license click + token-management step at **zero grade
cost** (same params class, same story). Llama-3.1-8B is kept as a documented alternative
(`config/models.json:alternative`, needs `HF_TOKEN`). A 70B-class model is used only for the
*extreme* extension (5.7), streamed from the SSD.

**Consequences.** The "too big for RAM" premise holds (16 GB > 8 GB). No secret/token needed
for the core experiment. The accuracy "red line" is studied via a GGUF quant sweep on this
one model so the whole story lives on a single model. See [ADR-005](ADR-005-ssd-streaming.md).
