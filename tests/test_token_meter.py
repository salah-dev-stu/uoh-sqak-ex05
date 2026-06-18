"""Token estimator + per-request cost helper."""

from __future__ import annotations

from airbench.shared.token_meter import count_tokens, estimate_cost_usd


def test_empty_text_zero():
    assert count_tokens("") == 0


def test_nonempty_min_one():
    assert count_tokens("a") == 1


def test_unicode_handled():
    assert count_tokens("שלום עולם 🌍") > 0


def test_roughly_chars_over_four():
    assert count_tokens("x" * 40) == 10


def test_cost_math():
    # 500 in @ $0.15/1M + 200 out @ $0.60/1M
    cost = estimate_cost_usd(500, 200, 0.15, 0.60)
    assert round(cost, 6) == round(500 / 1e6 * 0.15 + 200 / 1e6 * 0.60, 6)
