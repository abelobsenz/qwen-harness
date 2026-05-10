#!/usr/bin/env python3
"""runtime_patch — token-level repetition penalty for dflash_mlx greedy decode.

dflash_mlx uses pure greedy argmax sampling (see runtime.greedy_tokens_with_mask).
That's mathematically guaranteed to loop in low-loss n-gram cycles.

The proxy-side `loop_guard` catches loops AFTER they happen, but the model
still wastes ~600 chars of decoded tokens before the abort fires. A
runtime-level repetition penalty that modifies logits BEFORE the argmax
prevents the loop from forming in the first place.

This module is loaded by `dflash_mlx.runtime` when `DFLASH_REP_PENALTY`
is non-zero. It provides:

  - `apply_repetition_penalty(logits, recent_token_ids, penalty)` — pure
    function. Subtracts `penalty` from the logit at each recent token ID.
    Trivial to call from any inference loop that already tracks recent
    tokens (which both `generate_baseline_once` and `generate_dflash_once`
    do — they have a `generated_tokens` list in scope).

  - `RepetitionContext(history_size, penalty)` — small stateful wrapper
    suitable for thread-local use during one generation. Tracks a deque
    of recent token IDs and applies the penalty when `.modify_logits()`
    is called.

Runtime integration:

  1. In `dflash_mlx/runtime.py:generate_baseline_once`, after each greedy
     argmax call, push the new token onto a RepetitionContext:

         ctx = RepetitionContext()  # at function start
         ...
         logits = ctx.modify_logits(logits)
         next_token = greedy_tokens_with_mask(logits, mask)
         ctx.observe(int(next_token))

  2. The same modification goes into `generate_dflash_once`'s verify
     loop, AND it must be applied identically to BOTH draft and target
     argmax calls so speculative-decoding acceptance still holds.

The library functions themselves are tested in
`scripts/test_runtime_patch.py` against synthetic logits.

Disable globally with `DFLASH_REP_PENALTY=0` (the default — this whole
module is a no-op on import; nothing happens until you call its
functions).
"""

from __future__ import annotations

import os
from collections import deque
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    import mlx.core as mx  # noqa: F401


# Env var contract:
#   DFLASH_REP_PENALTY: float, default 0.0 (off). Reasonable values:
#     0.05 — gentle nudge; rarely changes argmax decisions, but enough
#            to break short literal cycles.
#     0.10 — moderate; may slightly degrade quality on legitimately
#            repetitive outputs (lists, structured data).
#     >0.20 — aggressive; not recommended without quality validation.
DEFAULT_PENALTY = float(os.environ.get("DFLASH_REP_PENALTY", "0.0"))

# How many recent tokens count as "recent". 64 covers the typical
# n-gram-cycle length seen in greedy collapses (5-15 tokens), with
# headroom for longer phrase-level loops.
DEFAULT_HISTORY_SIZE = int(os.environ.get("DFLASH_REP_HISTORY", "64"))


def apply_repetition_penalty(logits, recent_token_ids, penalty: float = DEFAULT_PENALTY):
    """Subtract `penalty` from the logit at each recent token id.

    Pure function: returns a NEW array (the original is untouched).
    No-op when penalty == 0.0 (returns the input unchanged).

    Args:
        logits: mx.array of shape (V,) or (B, V). Last axis is vocab.
        recent_token_ids: iterable of int — recently emitted token IDs.
                          Duplicates are folded automatically (each
                          unique id is penalized once).
        penalty: float — additive subtraction in logit space. Set 0
                 to short-circuit.

    Returns:
        mx.array of the same shape as `logits`, with the recent-token
        positions reduced by `penalty`.

    Note: subtracts in logit space, NOT softmax space. A 0.05 penalty
    on a 0.4 logit yields 0.35; this is mathematically equivalent to
    multiplying the post-softmax probability by exp(-penalty) ≈ 0.95.
    """
    if penalty == 0.0:
        return logits
    if not recent_token_ids:
        return logits
    # Lazy import so this module can be imported (e.g. by tests) even
    # without mlx in scope. The actual application requires mlx.
    import mlx.core as mx  # noqa: F401

    # Dedupe + sort for predictable ordering in the kernel.
    unique_ids = sorted(set(int(t) for t in recent_token_ids))
    if not unique_ids:
        return logits

    # Build a (V,)-shaped delta array with -penalty at the recent
    # positions. This works for both 1D (V,) and 2D (B, V) logits via
    # broadcasting.
    vocab = int(logits.shape[-1])
    # Filter out-of-range ids (shouldn't happen in practice, but defensive).
    in_range = [i for i in unique_ids if 0 <= i < vocab]
    if not in_range:
        return logits

    # Constructing a sparse delta via mx.zeros + index assignment is the
    # cheapest approach mlx supports for this size; the array is float32
    # and the assignment vectorizes.
    delta = mx.zeros((vocab,), dtype=logits.dtype)
    idx = mx.array(in_range, dtype=mx.int32)
    pen = mx.full((len(in_range),), -float(penalty), dtype=logits.dtype)
    # mlx scatter via index assignment.
    delta = delta.at[idx].add(pen)
    return logits + delta


class RepetitionContext:
    """Sliding-window of recent token IDs + apply-on-demand penalty.

    Designed for one-generation-at-a-time use. Each generation should
    instantiate its own RepetitionContext (the runtime's `generate_*_once`
    functions are stateless except for their KV cache, so this fits
    naturally — the context lives only as long as one generation).

    Usage:

        ctx = RepetitionContext()
        ...
        for step in range(max_new_tokens):
            logits = ctx.modify_logits(logits)
            next_tok = int(greedy_tokens_with_mask(logits, mask).item())
            ctx.observe(next_tok)
    """

    __slots__ = ("_recent", "_penalty", "_history_size")

    def __init__(
        self,
        history_size: int = DEFAULT_HISTORY_SIZE,
        penalty: float = DEFAULT_PENALTY,
    ) -> None:
        self._history_size = max(0, int(history_size))
        self._penalty = float(penalty)
        self._recent: "deque[int]" = deque(maxlen=self._history_size or 1)

    @property
    def penalty(self) -> float:
        return self._penalty

    @property
    def is_active(self) -> bool:
        """Whether the context will actually modify logits when called.
        False when penalty == 0 OR history_size == 0."""
        return self._penalty != 0.0 and self._history_size > 0

    def observe(self, token_id: int) -> None:
        if not self.is_active:
            return
        self._recent.append(int(token_id))

    def modify_logits(self, logits):
        """Return logits with penalty applied to recent tokens. No-op
        when inactive."""
        if not self.is_active:
            return logits
        return apply_repetition_penalty(logits, self._recent, self._penalty)

    def clear(self) -> None:
        self._recent.clear()


def is_globally_enabled() -> bool:
    """True if the env-var-driven default would activate the penalty.
    Useful for guarding wiring code: integrations should consult this
    before installing a RepetitionContext into the inference loop."""
    return DEFAULT_PENALTY != 0.0 and DEFAULT_HISTORY_SIZE > 0


__all__ = [
    "apply_repetition_penalty",
    "RepetitionContext",
    "DEFAULT_PENALTY",
    "DEFAULT_HISTORY_SIZE",
    "is_globally_enabled",
]
