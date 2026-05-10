#!/usr/bin/env python3
"""Probe: how much work is needed to actually plug PagedKVCache into dflash?

The standalone PagedKVCache (scripts/paged_attn.py) was already validated:
byte-perfect round-trip, attend() matches reference SDPA. The remaining
question is whether it can REPLACE the per-layer caches dflash constructs
in `dflash_mlx.runtime._construct_caches()`.

This probe doesn't try to actually run inference — too risky given dflash's
adaptive logic + cross-layer dependencies. Instead it surfaces the exact
interface mismatches we'd have to fix.
"""

from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

import mlx.core as mx
from mlx_lm.models.cache import QuantizedKVCache, KVCache
from paged_attn import PagedKVCache


def _summarize_cache(name: str, cache) -> None:
    print(f"\n--- {name} ---")
    print(f"  type: {type(cache).__name__}")
    print(f"  has update_and_fetch: {hasattr(cache, 'update_and_fetch')}")
    print(f"  has state property:   {hasattr(type(cache), 'state')}")
    if hasattr(cache, "offset"):
        print(f"  offset: {cache.offset}")


# ---- 1. What does dflash's cache list look like? ----
print("=" * 70)
print("PROBE 1: What dflash currently constructs")
print("=" * 70)
print("Per `runtime._construct_caches`:")
print("  - Linear-attention layers → RecurrentRollbackCache or ArraysCache")
print("  - Full-attention layers   → QuantizedKVCache (group=64, 4 or 8 bits)")
print()

# Show the QuantizedKVCache shape
qkv = QuantizedKVCache(group_size=64, bits=8)
B, H, T, D = 1, 8, 4, 128
keys = mx.random.normal((B, H, T, D)).astype(mx.bfloat16)
values = mx.random.normal((B, H, T, D)).astype(mx.bfloat16)
out_k, out_v = qkv.update_and_fetch(keys, values)
print(f"QuantizedKVCache.update_and_fetch returns:")
print(f"  K: tuple of {len(out_k)} arrays, shapes/dtypes:")
for i, x in enumerate(out_k):
    print(f"     [{i}] {tuple(x.shape)} {x.dtype}")
print(f"  V: tuple of {len(out_v)} arrays, shapes/dtypes:")
for i, x in enumerate(out_v):
    print(f"     [{i}] {tuple(x.shape)} {x.dtype}")
print(f"  → 3-tuple = (quantized values uint32, scales bf16, biases bf16)")
print(f"  → consumed by mx.fast.quantized_dot_product_attention")


# ---- 2. What does PagedKVCache currently provide? ----
print()
print("=" * 70)
print("PROBE 2: What PagedKVCache provides today")
print("=" * 70)
pkv = PagedKVCache(num_blocks=4, num_kv_heads=H, head_size=D,
                   block_size=8, x_width=8)
print(f"  storage: 5D K cache {tuple(pkv.k_cache.shape)} {pkv.k_cache.dtype}")
print(f"           4D V cache {tuple(pkv.v_cache.shape)} {pkv.v_cache.dtype}")
print(f"  write API:  cache.write(seq_id, K[N,h,d], V[N,h,d])")
print(f"  read API:   cache.gather([seq_id]) → contiguous [Total,h,d]")
print(f"  → plain bf16 throughout, NOT quantized.")


# ---- 3. The gap ----
print()
print("=" * 70)
print("THE GAP: What full integration needs")
print("=" * 70)
gap = """
Three additions to PagedKVCache, in priority order:

(A) Quantized paged storage.
    Each of K and V becomes 3 paged tensors:
      values_q: [n_blocks, n_kv_heads, head_size_q/x, block_size, x] uint32
      scales:   [n_blocks, n_kv_heads, head_size/group, block_size, x] bf16
      biases:   [n_blocks, n_kv_heads, head_size/group, block_size, x] bf16
    where head_size_q = head_size * bits / 32 (uint32-packed).
    The reshape_and_cache kernel needs to be split into 3 parallel
    scatters; mx.quantize() runs on the host side before the kernel.

(B) update_and_fetch shim.
    Wrap PagedKVCache in an adapter that:
      - on first update, allocs blocks for the (implicit) seq_id
      - on each update, scatters the new step's K/V into the next slots
      - returns either:
          (a) the full gathered slice as a quantized 3-tuple (slow but
              keeps mx.fast.quantized_dot_product_attention working), or
          (b) registers a custom attention forward that consumes the
              paged blocks directly (the actual win — needs the fused
              paged_attention V1 kernel that's NOT yet ported)

(C) Layer-wise mixed-precision support.
    The current dflash setup uses 8-bit KV on layers 0-19 and 4-bit on
    layers 20+. PagedKVCache currently has one uniform dtype. Need
    per-layer instances OR a single cache with a per-layer bits field
    threaded through reshape_and_cache + gather_kv_cache.

Engineering estimate: 3-5 days of focused work to get (A) and (B-a)
shipping, validating that token output matches the non-paged path on
fixed prompts. Another 2-3 days for (B-b) (the real perf win) once the
fused V1 kernel is ported.
"""
print(gap)


# ---- 4. Decision criteria ----
print("=" * 70)
print("DECISION GIVEN CURRENT STATE")
print("=" * 70)
print("""
Cumulative wins from this session (single-stream chat workload):
  short  68.6 → ~84   (+22%)
  p_3k   19.7 → ~23.5 (+19%)
  p_5k   12.6 → ~14.8 (+17%)
  p_9k    7.4 →  ~8.3 (+12%)

Paged attention's primary value is multi-tenant concurrency — N
sequences sharing a block pool without head-of-line blocking. For the
typical user's workload (one user + occasional supervisor-serialized
agents), that scenario is rare.

Recommendation: defer the 5-7 day full integration. Keep the validated
paged-storage primitives in scripts/paged_attn.py as a starting point.
Re-evaluate when (a) multi-agent concurrency becomes a real bottleneck,
or (b) the upstream MLX paged-attention PR (ml-explore/mlx#2228) lands,
making a swappable attention implementation available without a full
custom-kernel port.
""")
