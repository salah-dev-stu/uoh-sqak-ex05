"""Lightweight, provider-agnostic token estimator for cost sizing (§5.5).

Avoids a heavy tokenizer dependency: uses the well-known ~4-chars-per-token
heuristic. Exact token counts are not needed — the economics analysis drives
volume from ``config/economics.json``; this only sizes ad-hoc strings.
"""

from __future__ import annotations

_CHARS_PER_TOKEN = 4


def count_tokens(text: str) -> int:
    """Estimate token count for ``text`` (0 for empty)."""
    if not text:
        return 0
    return max(1, round(len(text) / _CHARS_PER_TOKEN))


def estimate_cost_usd(
    tokens_in: int, tokens_out: int, usd_per_1m_in: float, usd_per_1m_out: float
) -> float:
    """Cost in USD for a request with the given token counts at the given prices."""
    return tokens_in / 1_000_000 * usd_per_1m_in + tokens_out / 1_000_000 * usd_per_1m_out
