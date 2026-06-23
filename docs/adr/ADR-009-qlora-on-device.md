# ADR-009 — On-device QLoRA fine-tuning (mlx-lm) as the standout extension

**Status:** accepted

**Context.** Lecture 08 spends real time on **LoRA / QLoRA** and on-device fine-tuning ("with LoRA,
individual students can now fine-tune foundation models… on a laptop in hours"). Our core result shows the
8 GB M2 **cannot run a 7B for inference**. An ambitious, on-theme extension is to show the same machine
**can train an LLM** — and to do it with real measurements, not a toy.

Earlier we found `mlx` would not load for **AirLLM** on this machine ([ADR-002](ADR-002-airllm-honest-path.md)).
But `mlx-lm` is the *correct, native* tool for LoRA on Apple Silicon — so this turns that dead-end into a win.

**Decision.** Fine-tune **LoRA adapters on a 4-bit quantized base** (= QLoRA) via `mlx_lm.lora`:
`mlx-community/Qwen2.5-1.5B-Instruct-4bit`, ~150 iters, batch 1, 8 layers, seq 512. A small self-referential
dataset teaches the model **airbench's own findings** (clean, verifiable before/after; quality is not graded).
Everything is integrated into `airbench` (config, runner, SDK, CLI, tests) with all external calls through the
Gatekeeper; `mlx`/`mlx-lm` live in the `heavy` extra so CI/tests need no MLX. A 0.5B fallback is config-ready.

**Consequences.** Real, measured outcome on the 8 GB M2: **trainable 0.171 % (2.638 M / 1543.7 M params),
peak 1.36 GB, ~95 s, train loss 2.56 → 0.04, val 4.70 → 2.54**, and a clean behaviour change (base model
knows nothing about "airbench"; the tuned model recites the finding). The machine that can't *run* a 7B can
*fine-tune* a 1.5B. One real debugging loop (the Gatekeeper truncated the training log to 2000 chars, hiding
the parameter header → fixed to keep the full log) is itself the lecturer's "I tried, it didn't work, then I
debugged" arc, captured honestly.
