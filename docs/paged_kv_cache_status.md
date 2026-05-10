# Paged KV cache on MLX — status and next steps

What works today, what's still to build, and the path to actual paged
attention on top of MLX. Source for the kernels is
[scripts/paged_attn.py](../scripts/paged_attn.py); reference Metal
kernels (downloaded for adaptation) are in
[scripts/paged_attn_kernels/](../scripts/paged_attn_kernels/).

## What ships in this turn

`scripts/paged_attn.py` — pure Python module exposing:

- `reshape_and_cache(key, value, k_cache, v_cache, slot_mapping, ...)` —
  scatter K/V from contiguous shape `[num_tokens, num_kv_heads, head_size]`
  into paged storage `[num_blocks, num_kv_heads, head_size/x, block_size, x]`
  (K) / `[num_blocks, num_kv_heads, head_size, block_size]` (V). Uses an
  inverse-slot lookup table (built CPU-side) so each GPU thread does an
  O(1) check instead of looping `num_tokens`.
- `gather_kv_cache(k_cache, v_cache, block_table, cu_seq_lens, ...)` —
  inverse: read paged blocks back into contiguous K/V with full byte
  fidelity. Used as the input path to `mx.fast.scaled_dot_product_attention`.
- `copy_blocks(k_cache, v_cache, block_mapping, ...)` — page-level move
  for fork/clone semantics (beam search, prefix-cache splits).
- `PagedKVCache` class — block allocator (alloc / grow / free / stats) +
  `write(seq_id, k, v)` + `gather(seq_ids)` + `attend(q, seq_ids)` glue.
  `attend()` is the SDPA fallback: it gathers the seq's K/V into
  contiguous tensors and runs the standard `mx.fast.scaled_dot_product_attention`.

All three kernels were ported from
[EricLBuehler/mistral.rs](https://github.com/EricLBuehler/mistral.rs/tree/master/mistralrs-paged-attn/src/metal/kernels)
to `mx.fast.metal_kernel` form (mlx auto-generates the function signature
from `input_names` / `output_names`, so the kernel BODIES are what we ship).

### Validation

Self-test in `paged_attn.py` does:
1. Allocate two sequences (lengths 5 and 17) into a 16-block × bs=8 pool.
2. Write K/V incrementally with interleaved sequences.
3. Gather both back into contiguous K/V → assert byte-perfect round-trip.
4. Run `attend()` with random Q against the paged cache → assert the
   output matches plain SDPA on the original K/V to within bf16 tolerance.
5. Exercise `free_seq` and verify blocks return to the pool.

All assertions pass. Production-scale numbers (256 blocks × bs=16,
8 KV heads, head_size=128, x=8 → 4096-token capacity in 16 MiB):

| op                                           | time      |
|----------------------------------------------|----------:|
| prefill 1280 tokens (1024 + 256, 2 seqs)     | 94 ms     |
| gather 1280 tokens back to contiguous        | 32 ms     |
| **decode-step write (1 new token)**          | **11.6 ms** |
| **decode-step attend (2 seqs, GQA 32→8, 1280 ctx)** | **5.8 ms** |

The 17 ms/step decode hot path (write + attend) is in dflash's neighborhood
(~22 ms/step at similar context) — but this is the *storage* layer only,
on top of vanilla `scaled_dot_product_attention`. The fused paged-attn
kernel below would knock the attend cost down further.

## What's NOT ported yet

The big one: `pagedattention.metal` — the fused kernel that runs
attention DIRECTLY on paged storage without gathering. 1434 lines of MSL
in mistral.rs's repo, with template parameters for head_size, block_size,
NUM_THREADS, NUM_SIMD_LANES, PARTITION_SIZE, plus FP8 / ALiBi / sinks
function constants.

Why we didn't port it in this turn:
- It's the *biggest* kernel and the *most numerically delicate* — a
  correctness regression in attention silently corrupts every output.
- Specializing to our exact config (bf16 + hs=128 + bs=16) and stripping
  templates would still be ~400-500 lines of MSL plus the helpers in
  `utils.metal` that do warp reductions and softmax accumulation.
- The current SDPA-fallback path is already correct AND in the same TPS
  ballpark; the win from the fused kernel mostly shows up at higher
  concurrency (4+ sequences) where the gather pass becomes a meaningful
  cost.

The rest of the paged-attn family (`gather_kv_cache`, `copy_blocks`,
`reshape_and_cache`, `kv_scale_update`, FP8 paths) are all small (60-160
lines) and ported / portable along the same lines as what's already
working.

## What to build next (priority order)

1. **Wire `PagedKVCache` into the Qwen3.6 attention class.** Touch the
   model's `forward()` to call `cache.write(seq_id, K, V)` after the
   QKV projection, then `cache.attend(Q, [seq_id])` in place of the
   in-line SDPA. Validate end-to-end output matches the current model
   token-for-token at greedy temperature. *Estimated: 1 day*.

2. **Build a paged scheduler that holds one `PagedKVCache` for the whole
   server.** Replace dflash's per-request KV ownership with a shared
   pool. Each chat / agent run claims an integer `seq_id`, writes its
   prompt, generates, and frees on completion. *Estimated: 1-2 days*.

3. **Port the fused `paged_attention` V1 kernel** specialized to bf16 +
   hs=128 + bs=16. Replace `cache.attend()` 's gather + SDPA with one
   kernel call. The expected win shows up at higher batch sizes —
   gather + SDPA scales O(B × L) in memory bandwidth; fused paged-attn
   is O(L). *Estimated: 2-3 days* including correctness validation.

4. **Port FP8 KV cache scales** (`kv_scale_update.metal` +
   `gather_kv_cache` with FP8 inputs). Halves KV memory at long
   context, similar to what dflash already does for non-paged storage.
   *Estimated: 0.5 day*.

## Trade-offs vs dflash today

| dimension                      | dflash (current default) | paged_attn (this work) |
|--------------------------------|--------------------------|------------------------|
| Single-request decode TPS      | 42 TPS (with spec dec)   | not yet integrated     |
| KV memory layout               | contiguous per-seq       | block-pooled           |
| Multi-seq concurrency          | serial (HOL blocking)    | natively concurrent    |
| Long-context fragmentation     | wastes per-seq tail      | none — block-aligned   |
| Spec decoding                  | DFlash block diffusion   | none (would lose)      |

So: **dflash wins for single-stream chat. The paged path is the right
foundation for true multi-tenant batching** — which is what the user's
agents-running-in-parallel workload actually needs once the integration
in Step 1 above is done.

## Snapshots taken before this work

- `qwen36_MTP_snapshots/qwen36_MTP-pre-paged-attention-20260504-200743.tar.gz`
  — pre-batching state (also pre-paged-kernels).
- `qwen36_MTP_snapshots/qwen36_MTP-pre-paged-kernels-20260504-203359.tar.gz`
  — state right before this turn started downloading kernels.

Restore either with:

```bash
tar -xzf <snapshot>.tar.gz -C /Users/abelobsenz/dev/qwen36_MTP/
```
