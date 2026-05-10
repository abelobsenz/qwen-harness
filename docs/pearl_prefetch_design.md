# PEARL prefetch — design notes (staged, not shipped)

## Goal
Overlap cycle N+1's draft forward with cycle N's target verify, hiding draft latency behind verify latency. On long-context prompts where verify dominates (e.g. 500ms verify vs 50ms draft at 5K), this could yield ~10% TPS.

## Why iter2 (2026-05-04) failed
Two separate issues compounded:
1. **Per-cycle buffer alloc** — every prefetch attempt allocated fresh `mx.array` for draft inputs/outputs, costing +0.9 GB RSS with zero gain.
2. **MLX scheduler** — even with overlap intent, the draft and verify forwards landed on the same Metal stream, so they ran sequentially despite async_eval.

The audit's proposed fix only addresses (1). Fixing (2) requires actual MLX stream isolation, which hasn't been validated for this codebase.

## Sequential dependency (the actual blocker)
```
cycle N:                           cycle N+1:
  draft → verify → posterior         draft (← needs staged_first
                  ↓                          ← needs posterior[accept_len:])
              staged_first ─────────────────┘
```
Cycle N+1's draft input depends on cycle N's verify output. Pure parallelism is impossible without speculation.

## PEARL approach (speculative)
Speculate that `staged_first` will equal the draft's highest-probability token at position `[acceptance_len]` in cycle N. Start drafting N+1 with that guess. If guess matches actual `staged_first`, prefetch is valid — save the draft cycle. If not, discard.

Hit rate matters: high-acceptance regimes (math, structured text) → guess almost always right → near-100% win. Low-acceptance regimes (code, JSON) → guess often wrong → potential loss.

## Implementation skeleton (when revisited)

```python
# Pre-allocate ONCE at function start (fix iter2 issue 1):
prefetch_block_buffer = mx.full((effective_block_tokens,), draft_model.mask_token_id, dtype=mx.uint32)
prefetched_drafted: mx.array | None = None
prefetch_assumed_first: int | None = None

# Inside the spec loop, AFTER staged_first is computed for cycle N:

# Validate prefetch from previous cycle
if prefetched_drafted is not None and prefetch_assumed_first == int(staged_first.item()):
    # Hit: reuse prefetched draft tokens
    drafted = prefetched_drafted
else:
    # Miss or first cycle: do draft normally
    drafted = greedy_tokens_with_mask(...)

# After committing cycle N, speculate cycle N+1:
# Use the highest-prob non-mask token as the assumed first
# (this requires a second argmax on the verify_logits, cheap)
assumed_first = ... # tensor[1]
with mx.stream(...):  # if MLX supports it
    prefetched_drafted = run_draft_forward(assumed_first)
prefetch_assumed_first = int(assumed_first.item())
```

## Validation plan (when shipped)
1. **Token equivalence check**: sha1 of generated_token_ids matches non-prefetch path on deterministic prompts (HumanEval, GSM at temp=0).
2. **Hit rate metric**: log prefetch hit/miss ratio per request. Should be >70% on math, >40% on code.
3. **Wall-time bench**: full perf_bench at 4 context sizes; need ≥+5% TPS on long context to justify.
4. **Memory check**: peak RSS shouldn't exceed pre-prefetch baseline by more than 100 MB (pre-allocated buffers only).

## Risk
- If MLX's `mx.stream` doesn't actually overlap on Metal (currently uncertain), this becomes pure overhead.
- Adaptive block_tokens interacts: when block shrinks dynamically, prefetched draft may have wrong block_len, requiring re-draft anyway.

## Decision
Defer to a focused session. Attempting this in a multi-optimization sprint risks regressing the working stack. Mark as `DFLASH_DRAFT_PREFETCH=1` env-gated when implemented — default OFF.

## EMPIRICAL UPDATE (2026-05-08)
Measured MLX stream overlap on Metal directly via `/tmp/mlx_stream_overlap_test.py`:
- Two 4096² matmuls sequentially: 44.17 ms each
- Two 4096² matmuls on `mx.new_stream` × 2: 45.89 ms each
- **Speedup: 0.96× (negative)**

MLX's GPU streams do NOT actually overlap on Metal on this hardware. Both streams compete for the same Metal command queue. **This kills PEARL prefetch as a TPS optimization** — even with perfect speculation hit rate and zero buffer allocation overhead, parallel draft+verify would not run in parallel.

The +0.9 GB RSS in iter2 was a red herring. The fundamental blocker is Metal scheduler / MLX integration. Without low-level Metal API access (multiple `MTLCommandQueue` with manual sync), there is no way to implement the overlap.

**Status: DEAD.** Re-evaluate only if MLX adds proper concurrent stream support, or if the project moves to CUDA/ROCm.
