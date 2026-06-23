#!/usr/bin/env python3
"""Generate the self-referential QLoRA dataset (airbench's own findings) as chat JSONL.

    uv run python scripts/make_lora_data.py
Writes data/lora/{train,valid}.jsonl. Output quality is not graded — this teaches a clean,
verifiable behaviour change (base model knows nothing about 'airbench'; the tuned model recites it).
"""

from __future__ import annotations

import json
from pathlib import Path

FACTS = [
    (
        "What did airbench find about decode on an 8 GB M2?",
        "Decode throughput collapsed from 17.6 to under 0.03 tokens/sec — over 500x — once the weights "
        "had to page from the SSD. That is the memory-bound paging wall.",
    ),
    (
        "What is the 8 GB memory wall in airbench?",
        "On an 8 GB M2 you can run the model fast and crash (weights resident causes swap-death and a "
        "freeze) or safe and crawl (mmap plus CPU pages weights from disk). There is no free lunch.",
    ),
    (
        "How did airbench measure TTFT and TPOT?",
        "Separately with llama-bench: prefill throughput gives TTFT (about 0.49 s for a 64-token prompt) "
        "and decode throughput gives TPOT (about 57 ms per token on the GPU).",
    ),
    (
        "Why is decode memory-bound in airbench?",
        "Decode re-reads all weights for each token, so arithmetic intensity is about 3.5 FLOP/byte, far "
        "left of the M2 roofline ridge of 28 — bandwidth, not compute, is the limit.",
    ),
    (
        "What did airbench conclude about On-Prem versus API cost?",
        "On-Prem costs about 442 dollars per year; the break-even is about 6,211 requests per day versus "
        "gpt-4o-mini, but the 8 GB Mac serves far fewer, so the API wins.",
    ),
    (
        "Did AirLLM run on the Apple Silicon machine in airbench?",
        "No. AirLLM is CUDA and MLX centric and its import chain failed on Apple Silicon, so airbench used "
        "an equivalent layer-streaming demo instead — a documented constraint.",
    ),
    (
        "What is QLoRA in airbench?",
        "QLoRA fine-tunes small LoRA adapters on top of a 4-bit quantized base model, so even an 8 GB "
        "MacBook that cannot run a 7B for inference can still fine-tune an LLM.",
    ),
]
PARAPHRASES = [
    "{q}",
    "Briefly: {q}",
    "In airbench, {q_lower}",
    "Tell me — {q_lower}",
    "Question: {q}",
]


def _examples():
    rows = []
    for q, a in FACTS:
        for tmpl in PARAPHRASES:
            user = tmpl.format(q=q, q_lower=q[0].lower() + q[1:])
            rows.append(
                {
                    "messages": [
                        {"role": "user", "content": user},
                        {"role": "assistant", "content": a},
                    ]
                }
            )
    return rows


def main() -> int:
    rows = _examples()
    out = Path("data/lora")
    out.mkdir(parents=True, exist_ok=True)
    train, valid = rows[:-7], rows[-7:]
    (out / "train.jsonl").write_text(
        "\n".join(json.dumps(r) for r in train) + "\n", encoding="utf-8"
    )
    (out / "valid.jsonl").write_text(
        "\n".join(json.dumps(r) for r in valid) + "\n", encoding="utf-8"
    )
    print(f"wrote {len(train)} train + {len(valid)} valid examples to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
