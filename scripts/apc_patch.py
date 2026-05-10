"""apc_patch — monkey-patch dflash_mlx so multi-turn APC actually fires
on Qwen3.6's hybrid attention.

The bug. dflash saves cache keyed by `prompt_tokens + generated_tokens`.
mlx_lm's LRUPromptCache.fetch_nearest_cache has three retrieval paths:
  - exact match: rare (only when sending byte-identical prompt twice)
  - longer match → trim: only fires when ALL layers are trimmable;
    Qwen3.6 hybrid (KVCache + RecurrentRollbackCache + ArraysCache)
    fails this check, so this branch is dead.
  - shorter match → use + prefill suffix: never fires because we never
    save a strict-prefix entry.
Net: APC doesn't fire across turns on this model — confirmed by 1.0×
TTFT speedup measured in scripts/apc_multiturn.py.

The fix. ALSO save a snapshot keyed by `prompt_tokens` (post-prefill,
before generation mutates the recurrent state). Then turn N+1 — which
sends `prompt_N + assistant_N + new_user` — finds a strict-prefix match
for `prompt_N` and only prefills `[assistant_N + new_user]`.

Cost. One `copy.deepcopy(cache)` per request at end-of-prefill. For a
6k-token prompt on M-series with quantized KV the snapshot is a few
hundred MB; deepcopy adds ~50–200 ms to TTFT on the saving turn but
saves multiple seconds on the next turn. Memory cost is bounded by
DFLASH_PROMPT_CACHE_SLOTS × snapshot_bytes — 32 slots × ~500 MB = 16 GB
worst case, so users with tight memory should bump
DFLASH_PROMPT_CACHE_SLOTS down (override via env).

Install path. `bin/qwen` invokes `scripts/dflash_serve_patched.py`,
which calls `apc_patch.install()` before importing dflash_mlx.serve.
The patch survives upstream pip upgrades because we're not editing
the package.
"""
from __future__ import annotations
import copy
import sys


def install() -> bool:
    """Install the APC fix. Idempotent — calling twice is a no-op."""
    try:
        import dflash_mlx.runtime as rt
    except Exception as e:  # noqa: BLE001
        print(f"[apc-patch] dflash_mlx not importable: {e}", file=sys.stderr)
        return False
    if getattr(rt, "_apc_patch_installed", False):
        return True

    # Snapshots indexed by id(cache) since the cache list itself doesn't
    # accept attribute assignment. _save_prompt_cache pops here.
    _snapshots: dict[int, list] = {}

    # Optional: wrap _maybe_reuse_cache to trace cache hits/misses.
    # Only emits when DFLASH_APC_TRACE=1.
    _orig_reuse = rt._maybe_reuse_cache

    def _patched_reuse(target_model, prompt_tokens, **kw):
        cache, cached_prefix = _orig_reuse(target_model, prompt_tokens, **kw)
        if _APC_TRACE:
            _trace(f"reuse: prompt_len={len(prompt_tokens)} "
                   f"cached_prefix={cached_prefix} "
                   f"trie_size={len(rt._get_prompt_cache())}")
        return cache, cached_prefix

    # We replicate the body of dflash's _chunked_prefill so we can
    # snapshot the cache RIGHT BEFORE the last chunk is processed. Why:
    # Qwen3.6's chat template appends `<|im_start|>assistant\n<think>\n`
    # as a generation prompt suffix when add_generation_prompt=True.
    # Turn 1's prompt_tokens include those final tokens; turn 2's prompt
    # at the same position has assistant content instead. So a snapshot
    # at end-of-prefill produces a key that's NOT a strict prefix of
    # turn 2's tokens (divergence at the `<think>` position). Snapshotting
    # one chunk before the end gives us a key that's solidly in
    # common-prefix territory, with a few hundred tokens of suffix to
    # re-prefill on turn 2 — still far cheaper than the 4k+ token cold
    # prefill we pay today.
    import mlx.core as mx
    _orig_chunked_prefill = rt._chunked_prefill
    _PREFILL_CHUNK = rt._PREFILL_CHUNK

    # Trailing tokens to skip when picking the snapshot offset. Qwen3.6's
    # chat template with add_generation_prompt=True appends 5 tokens
    # (<|im_start|>assistant\n<think>\n). Turn N+1's prompt has assistant
    # content at the same position, so a snapshot covering the last K
    # tokens of turn N would not be a strict prefix of turn N+1's tokens.
    # K=8 gives a safe margin without sacrificing much cache benefit.
    import os as _os
    _TAIL_SKIP = int(_os.environ.get("DFLASH_APC_TAIL_SKIP", "8"))

    def _patched_chunked_prefill(target_model, prompt_array, cache):
        total_len = int(prompt_array.shape[1])
        if total_len == 0:
            return _orig_chunked_prefill(target_model, prompt_array, cache)
        # Strategy: process all-but-the-last-K tokens, snapshot, then
        # process the last K. The snapshot's offset is the cache state
        # at position (total_len - K) which is a safe strict-prefix
        # match for the next turn's prompt.
        # Read the cache's current offset (where prior cached prefix
        # already covers) so we can compute the absolute snapshot offset.
        prior_offset = 0
        for layer in cache:
            off = getattr(layer, "offset", None)
            if isinstance(off, int):
                prior_offset = off
                break
        boundaries = list(range(0, total_len, _PREFILL_CHUNK))
        if boundaries[-1] != total_len:
            boundaries.append(total_len)
        # Pick the last boundary we should reach BEFORE snapshotting.
        # Snapshot point = total_len - tail_skip (clamped to >= 0).
        snap_point = max(0, total_len - _TAIL_SKIP)
        snap = None
        last_logits = None
        # Process chunks until snap_point, then snapshot, then finish.
        i = 0
        # Phase 1: chunks fully under snap_point.
        while i + 1 < len(boundaries) and boundaries[i + 1] <= snap_point:
            chunk = prompt_array[:, boundaries[i]:boundaries[i + 1]]
            last_logits = target_model(chunk, cache=cache)
            mx.eval(last_logits)
            i += 1
        # Phase 2: chunk that crosses snap_point — split it.
        if i + 1 < len(boundaries):
            crossing_start = boundaries[i]
            crossing_end = boundaries[i + 1]
            if crossing_start < snap_point < crossing_end:
                # Process [crossing_start : snap_point]
                chunk_a = prompt_array[:, crossing_start:snap_point]
                if int(chunk_a.shape[1]) > 0:
                    last_logits = target_model(chunk_a, cache=cache)
                    mx.eval(last_logits)
                # Snapshot at exactly snap_point.
                if snap_point > 0:
                    try:
                        snap = copy.deepcopy(cache)
                    except Exception as e:  # noqa: BLE001
                        _trace(f"snapshot deepcopy at split: {e}")
                # Process [snap_point : crossing_end]
                chunk_b = prompt_array[:, snap_point:crossing_end]
                if int(chunk_b.shape[1]) > 0:
                    last_logits = target_model(chunk_b, cache=cache)
                    mx.eval(last_logits)
                i += 1
            elif crossing_start == snap_point and snap_point > 0:
                # Snapshot exactly here, then process the chunk normally.
                try:
                    snap = copy.deepcopy(cache)
                except Exception as e:  # noqa: BLE001
                    _trace(f"snapshot deepcopy at boundary: {e}")
                chunk = prompt_array[:, crossing_start:crossing_end]
                last_logits = target_model(chunk, cache=cache)
                mx.eval(last_logits)
                i += 1
        # Phase 3: any remaining chunks (shouldn't happen given boundaries).
        while i + 1 < len(boundaries):
            chunk = prompt_array[:, boundaries[i]:boundaries[i + 1]]
            last_logits = target_model(chunk, cache=cache)
            mx.eval(last_logits)
            i += 1
        if last_logits is None:
            # No chunks ran — fall back to original.
            return _orig_chunked_prefill(target_model, prompt_array, cache)
        if snap is not None:
            _snapshots[id(cache)] = snap
            _trace(f"chunked_prefill: snap saved at point={snap_point} "
                   f"prior_offset={prior_offset} total_len={total_len}")
        else:
            _trace(f"chunked_prefill: no snap (snap_point={snap_point} "
                   f"total_len={total_len})")
        return last_logits

    _orig_save = rt._save_prompt_cache

    # Tracing is opt-in via DFLASH_APC_TRACE=1. Off by default to avoid
    # disk churn on every request.
    import os as _os_for_trace
    _APC_TRACE = _os_for_trace.environ.get("DFLASH_APC_TRACE", "0") == "1"

    def _trace(msg: str):
        if not _APC_TRACE:
            return
        try:
            import os as _os
            with open(_os.path.expanduser("~/.qwen/.apc_patch_trace.log"),
                      "a") as _f:
                import time as _t
                _f.write(f"{_t.time():.3f} {msg}\n")
        except Exception:  # noqa: BLE001
            pass

    def _patched_save(target_model, prompt_tokens, cache):
        # Preserve the existing behavior (save key = prompt+gen).
        # Useful when next call repeats the same prompt+assistant exactly.
        _orig_save(target_model, prompt_tokens, cache)
        # New: save key = prompt_only via the deep-copied snapshot. This
        # is the entry the trie's "shorter match" path returns when the
        # next request extends the prompt.
        snap = _snapshots.pop(id(cache), None)
        if snap is None:
            _trace(f"save: no snapshot for cache id={id(cache)}")
            return
        try:
            snap_offset = None
            saw_offsets = []
            for i, layer in enumerate(snap):
                off = getattr(layer, "offset", None)
                if off is not None:
                    saw_offsets.append((i, type(layer).__name__, off))
                if isinstance(off, int) and off > 0 and snap_offset is None:
                    snap_offset = off
            _trace(f"save: offsets={saw_offsets[:6]} chosen={snap_offset} "
                   f"prompt_len={len(prompt_tokens)}")
            if snap_offset is None:
                return
            min_prefix = int(rt._PROMPT_CACHE_MIN_PREFIX)
            if snap_offset < min_prefix:
                _trace(f"save: skip (offset {snap_offset} < min {min_prefix})")
                return
            n_prompt = min(snap_offset, len(prompt_tokens))
            prefix_tokens = list(prompt_tokens)[:n_prompt]
            pc = rt._get_prompt_cache()
            try:
                pc.insert_cache(id(target_model), prefix_tokens, snap)
                _trace(f"save: ok inserted {len(prefix_tokens)} tokens, "
                       f"trie len now {len(pc)}")
            except Exception as e:  # noqa: BLE001
                _trace(f"save: insert_cache exc: {type(e).__name__}: {e}")
        except Exception as e:  # noqa: BLE001
            _trace(f"save: outer exc: {type(e).__name__}: {e}")

    rt._chunked_prefill = _patched_chunked_prefill
    rt._save_prompt_cache = _patched_save
    rt._maybe_reuse_cache = _patched_reuse
    rt._apc_patch_installed = True
    # Use a sentinel file as a heartbeat so we can verify install fired
    # from inside the daemon's subprocess where stderr may be detached.
    try:
        import os as _os
        sentinel = _os.path.expanduser("~/.qwen/.apc_patch_installed")
        _os.makedirs(_os.path.dirname(sentinel), exist_ok=True)
        with open(sentinel, "w") as _f:
            import time as _t
            _f.write(f"installed at {_t.time()}\npid={_os.getpid()}\n")
    except Exception:  # noqa: BLE001
        pass
    print("[apc-patch] installed", file=sys.stderr, flush=True)
    return True


if __name__ == "__main__":
    install()
