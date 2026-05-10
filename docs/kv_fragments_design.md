# Per-message KV fragments — design (staged)

## Goal
Today: when `maybe_compact()` triggers (60K tokens), the agent summarizes prior turns and re-prefills the entire compacted prompt. That re-prefill costs seconds at long context.

Target: store per-message KV fragments at write time. On compaction, splice surviving messages' fragments back into the cache without re-prefill. ~10× faster compactions at long context.

## Why this is hard on Qwen3.6
The model has *hybrid* attention. Each layer's "cache" is one of three types:
- `KVCache` (full-attention layers) — keys + values per token, contiguous
- `RecurrentRollbackCache` (linear-attention layers) — recurrent state, NOT per-token
- `ArraysCache` (conv state in linear-attention)

Splicing recurrent state by message position is **mathematically wrong** for the linear layers — recurrent state at position N is a function of ALL prior tokens, not the last few. You can't "delete a middle message" from a recurrent state and reconstruct correctly.

This is the architectural blocker. KV fragments work cleanly on Llama/Mistral (uniform full-attention). They don't on Qwen3.6 without invalidating the linear layers.

## Three viable paths

### Path A — full re-prefill of linear-only state, splice full-attention KV
Hybrid: keep the per-message KV slices for the 10 full-attention layers; on compaction, REBUILD the linear-attention recurrent state from scratch by replaying surviving messages.

Re-prefill cost is 30/40 layers (linear ones). Saves 10/40 (full-attn) layers' prefill. Net savings: ~25% of prefill cost.

Implementation:
```python
@dataclass
class MessageFragment:
    msg_idx: int
    token_offset: int
    full_attn_kv: dict[layer_id, (k_slice, v_slice)]  # 10 layers
    msg_token_count: int
```
Snapshot at end-of-prefill for each message. On compaction:
- Take fragments of survivors
- Concatenate full_attn_kv slices in compacted order
- Reset linear-attention caches to empty
- Run forward over concatenated tokens with linear-only path active

This requires a new "linear-only forward" mode in the runtime. ~400 lines.

### Path B — fragment-aware compactor (no KV reuse, smarter prompt)
Cheaper alternative: don't reuse KV at all. Instead, the compactor extracts only the *most-relevant* prior turns and the resulting prompt is shorter, so re-prefill is fast even with cold cache.

Today's compactor summarizes via 4B model. Path B adds: per-turn relevance scoring (bm25 / embeddings) on the compactor input. Survivors that are uncited in the summary get dropped from the post-compact prompt.

This doesn't reduce re-prefill *time per token* but reduces *tokens to re-prefill*. ~150 lines, no runtime changes.

### Path C — disk-backed prefix cache for compactor outputs
Compactor outputs are deterministic-ish. Cache them on disk keyed by input-message-list-hash. Subsequent compactions hit cache. Saves the ~3s compactor latency per compaction event.

Doesn't address the re-prefill cost but is cheap (~100 lines).

## What I'd actually pick
**Path B + Path C combined** before attempting Path A.

Path A is the "real" KV-fragments win but requires runtime surgery, custom forward mode, and careful test coverage (~3 days). Path B+C is one day and addresses 80% of the user pain (compaction is slow, has fixed cost) without touching the runtime.

If after Path B+C the remaining compaction cost is still painful, then attempt Path A.

## Status
Design only. No code changes.

To pick up Path B+C in next session:
1. Add `compactor_disk_cache.py` keyed on message-hash + summarizer-version.
2. Modify `agent_tools.maybe_compact()` to consult cache first.
3. Add relevance scoring to the compactor's input prompt — bm25 over surviving tools/files.
