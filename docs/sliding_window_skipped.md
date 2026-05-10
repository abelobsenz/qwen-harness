# Sliding window attention — skipped (architectural mismatch)

## Decision
After investigating, do not add sliding window to Qwen3.6-35B-A3B. The model's architecture already implements the equivalent benefit, and restricting the remaining full-attention layers would degrade the model's design point.

## Why
Qwen3.6 uses a hybrid attention scheme (`model_type: qwen3_5_moe`):

| Layer kind | Count | Memory growth with seq | Role |
|---|---|---|---|
| GatedDeltaNet (linear/Mamba) | 30 of 40 | constant | local-ish, recurrent state |
| Full attention | 10 of 40 (idx 3,7,...,39) | linear in seq | global view |

The full-attention layers are spaced every 4 (`full_attention_interval=4`), giving the model 10 "hubs" where every position can attend to every other position. That's the model's mechanism for long-range reasoning — restricting them with a sliding window cripples the layer kind that was specifically designed for global context.

## Numbers that killed the case
- KV memory on the 10 full-attention layers at 9K context: 10 × 2 KV heads × 256 head_dim × 9000 × 1 byte (8-bit quant) = **~46 MB**. Already negligible.
- Prefill compute saved (10/40 layers, O(N²) → O(N·W)): ~2-5% of total wall at long context.
- Quality risk: unknown but real. Those 10 layers are the entire long-range capacity of the model.

## Architectural takeaway
The audit's recommendation to add sliding window came from intuition built on uniform-attention models (Llama, Mistral, plain Qwen3). Hybrid Mamba+attention models (Qwen3.6, Jamba, others) already paid the sliding-window-equivalent cost via Mamba layers and concentrated global attention into a few full-attn layers. Adding sliding window to those few full-attn layers fights the design.

## What would actually help instead
For long-context throughput on this architecture, the more productive direction is:
1. **Fused Metal kernel** for the GatedDeltaNet's recurrent update (if not already optimal).
2. **Parallel prefix scanning** on the linear-attention layers (mamba2-style chunked prefix-sum).
3. **mx.compile** on the spec loop (already in the audit) — affects both layer kinds.

## Status
Skipped. Re-evaluate only if a future model variant uses uniform full attention.
