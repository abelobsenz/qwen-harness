# Constrained tool-call decoding — investigation results (2026-05-08)

## What's actually shipped + verified end-to-end
- **Phase A** (vocab classification): `scripts/json_grammar.py:build_token_classification` ✅
- **Phase B** (char-level JSON FSM): `scripts/json_grammar.py:JsonCharFSM` ✅
  - **21/21 real Qwen3.6 fused tokens accepted** on a clean `<tool_call>{"name":...}</tool_call>` sequence (incl. fused tokens like `{"`, `":`, `",`, `"}}`)
  - **3+ rejections** on malformed (missing colon)
  - mask compute: 80ms cold / 0.1ms cached per state
- **Phase C** (runtime bridge + dflash hooks): `scripts/json_grammar_runtime.py` + 4 patches in `dflash_mlx/runtime.py` ✅
  - Patches are env-gated by `DFLASH_JSON_GRAMMAR=1`. With env unset, hooks are no-op (verified via TPS bench: 81.3 → 80.7 TPS short, identical at 3K/5K).
  - With env set, server prints `[dflash] DFLASH_JSON_GRAMMAR=1 — grammar-constrained tool decoding active` at boot.
  - `[grammar.begin_request] FSM created` confirms FSM creation per request.
  - `[grammar.feed] state (False, 0) -> (True, 0)` confirms FSM state transition fires when `<tool_call>` is committed.

## The architectural finding I wish I'd anticipated

**The hook fires AFTER `committed_segment` is computed.** Per spec cycle, `committed_segment` may contain MULTIPLE tokens (up to the `DFLASH_VERIFY_LEN=8` cap in the current config). Observed in real run:

```
[grammar.feed] state (False, 0) -> (True, 0) after [248058, 198, 27, 1628]...
```

That's `<tool_call>`, `\n`, `<`, `function` — **all committed in ONE cycle**. The FSM correctly transitioned to in_tool_call=True after `<tool_call>`. But by the time it could update the mask for the NEXT cycle, tokens 27 (`<`) and 1628 (`function`) — both INVALID per the JSON grammar — were already committed.

**This is a fundamental tension between speculative decoding and per-token constrained decoding.** Speculative decoding's whole point is committing multiple tokens per forward pass. Per-token constraint requires re-masking before each token. The two are at architectural odds.

## What a correct fix needs (Phase D)

Two viable approaches:

### Approach 1 — Truncate acceptance at first invalid drafted token
After `acceptance_len` is computed but BEFORE commit, walk the to-be-committed tokens through the FSM. At the first invalid one, set `acceptance_len = position_of_invalid - 1`. Then `commit_count = 1 + acceptance_len` is reduced to exclude the invalid token. The model re-emits the correct token in the next cycle (with the now-updated mask).

**Pros:** Minimal additional masks, no extra forward passes.
**Cons:** Requires patching the spec loop's commit path, not the post-commit hook. ~80 lines into runtime.py at lines 2050-2090 (sync) and 2360-2400 (stream).

### Approach 2 — Mask the verify_logits per-position
Compute a per-position mask for the verify-cycle: token at position k can only validate if all chars up to and including it leave the FSM in a valid state. Apply the per-position mask to verify_logits before the argmax.

**Pros:** Correctly handles speculative parallelism.
**Cons:** Per-position FSM simulation = expensive; mask becomes 2D `(verify_len, vocab_size)` = ~2 MB / cycle. Computing it cheaply requires precomputing per-state masks for all *reachable from current state via at most k chars* — an expensive cross-product.

### Recommendation
Approach 1. Cheaper to implement (~80 lines), correctness is provable (no token can be committed that the FSM rejects), and the only cost is some wasted draft work when the model speculates a tool-call internal that the FSM rejects.

## Phase D implementation sketch

Insert just before line 2113 in `generate_dflash_once` (and equivalently in stream):

```python
# Grammar pre-commit truncation: if FSM would reject any token in the
# would-be-committed segment, truncate acceptance to exclude it.
if _grammar_rt is not None:
    fsm = _grammar_rt.current_fsm()
    if fsm is not None and fsm.state.in_tool_call:
        # Walk the to-be-committed tokens through a SHADOW copy of the FSM.
        from copy import copy as _copy
        shadow_state = fsm.state
        # acceptance_len is the count of accepted DRAFTED tokens; commit_count
        # = 1 + acceptance_len (the +1 is the verify-corrected token).
        candidate_tokens = verify_token_ids[: 1 + acceptance_len].tolist()
        truncated_at = None
        for i, tid in enumerate(candidate_tokens):
            ok, new_state = fsm.cls.simulate_token(shadow_state, int(tid))
            if not ok:
                truncated_at = i
                break
            shadow_state = new_state
        if truncated_at is not None and truncated_at < len(candidate_tokens):
            # Don't commit the bad token. Reduce acceptance_len so commit_count
            # = 1 + acceptance_len excludes everything from truncated_at onward.
            acceptance_len = max(0, truncated_at - 1)
```

This depends on having mask-aware verify so the model emits a correct token at position `truncated_at` next cycle. Since the cycle-after will have an updated mask, the model's argmax will pick a valid token.

## How to fully validate Phase D when it ships
1. **Token-equivalence at temp=0**: with grammar OFF, generate tool calls for 20 prompts; record sha1 of token streams. With grammar ON, repeat. Streams should differ ONLY in cycles where the model would have emitted invalid JSON; valid cycles must match exactly.
2. **Tool-call success rate**: harness of 50 tool-using prompts. Without grammar: malformed rate is X%. With grammar: should be ~0%.
3. **Performance**: bench `perf_bench.py` with grammar ON and a tool-call-rich workload. Per-cycle overhead measurement.

## Current state files
- `scripts/json_grammar.py` (470 lines, validated)
- `scripts/json_grammar_runtime.py` (220 lines, validated end-to-end firing)
- `dflash_mlx/runtime.py` (4 patches, env-gated, no-op when off)
- `/tmp/qwen36_token_classification.json` (6.1 MB cache file)

## Net session result
**Phase A, B, C complete. Phase D (the hard one) is properly scoped with concrete code sketch.** The hidden architectural lesson — spec decoding × constrained decoding requires pre-commit interception, not post-commit — was found empirically and would have bitten any naive implementation. That's worth more than just the code: future sessions know exactly what to build and why.
