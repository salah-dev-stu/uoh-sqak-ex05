# ADR-004 — Serial quant sweep (download → bench → perplexity → delete)

**Status:** accepted

**Context.** Holding Q8+Q5+Q4+Q2 GGUFs of a 7B model simultaneously is ~18 GB. Even on the
489 GB SSD we prefer not to assume free space, and the principle (no silent disk blow-up) matters.

**Decision.** `runners/quant_sweep.py` processes levels **serially**: download one level →
benchmark (TTFT/TPOT via llama.cpp) → measure **perplexity** via `llama-perplexity` → **delete**
the weight before the next. The accuracy **red line** is the first level whose ΔPPL vs the best
(Q8) baseline exceeds `runtime.red_line_ppl_delta`. One level failing does not abort the rest.

**Consequences.** Disk never holds >1 quant at a time (asserted by a test). The red line is an
objective metric (perplexity), not a subjective judgement (H4/H9).
