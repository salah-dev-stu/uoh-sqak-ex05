# AirLLM on Apple Silicon — documented constraint (H3, real outcome)

**Decision context:** [ADR-002](../docs/adr/ADR-002-airllm-honest-path.md). We committed to *attempting*
the real AirLLM library (time-boxed), falling back to our equivalent layer-streaming demo if it won't run.

## What actually happened (on this M2, macOS 26.2, Python 3.13, uv)

`airllm` installs, but **importing it fails** on Apple Silicon:

```
from airllm import AutoModel
  → airllm/airllm_llama_mlx.py: from sentencepiece import SentencePieceProcessor
  ModuleNotFoundError: No module named 'sentencepiece'
```

AirLLM's `AutoModel` resolves, on macOS, to its **MLX-LLaMA backend** (`airllm_llama_mlx`), which hard
-requires `mlx` **and** `sentencepiece`. Installing them ad-hoc does not stick: **`uv run` re-syncs the
environment to the locked dependency set on every invocation**, removing packages not declared in the
project — so the import keeps failing under our reproducible workflow. Even with the deps forced in, the
MLX-LLaMA backend is LLaMA-architecture-specific (not Qwen) and is the CUDA/MLX path the library is built
around — not a clean fit for a uv-managed, Qwen, MPS setup.

**Conclusion:** AirLLM is CUDA/MLX-centric and does not run cleanly here — exactly the constraint predicted
in the PRD. This is captured as a `ConstraintReport` by `runners/airllm.py`, which is unit-tested.

## The honest, equivalent path (what we run instead)

`runners/layered.py` implements the **same mechanism** AirLLM uses — load one transformer block / shard
from the SSD, run it, free it — so peak memory stays at one block instead of the whole model. It is
instrumented for per-layer I/O vs compute and is the basis of the paging analysis (report §4, §6). The
real Q4 two-regime benchmark (report §3) already demonstrates the underlying truth AirLLM exists to
manage: **on 8 GB you either hold the weights resident (fast, but it crashes) or page them from disk
(stable, but I/O-bound and glacial).**

This is a spec-sanctioned negative result: *"a well-analyzed negative result counts as much as a positive
one."*
