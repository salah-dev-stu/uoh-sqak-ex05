# ADR-005 — External USB SSD as the weight store and streaming source

**Status:** accepted

**Context.** Internal disk has ~9 GB free. An external USB SSD (`/Volumes/Backup`, 489 GB free)
was measured at **~498 MB/s read / ~358 MB/s write**.

**Decision.** All weights live under `model.weights_dir` on the SSD. A guard
(`shared/paths.assert_external`) **refuses** any weights path not under `/Volumes/`. The layered
demo streams shards from the SSD; its ~0.5 GB/s read bandwidth is the predicted I/O ceiling
(7B FP16 ≈ 16 GB ⇒ ~32 s/token of pure I/O), which we compare against measured per-layer I/O.

**Consequences.** Weights are never written to the internal disk and never committed (gitignored,
H10). SSD bandwidth becomes the explicit AirLLM/paging bottleneck in the analysis (H8).
