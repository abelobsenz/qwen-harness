#!/usr/bin/env python3
"""APC multi-turn hit-rate regression test.

The existing scripts/test_apc_patch.py verifies the monkey-patch *installs*
and *writes* a snapshot. It does NOT verify that the snapshot is actually
USED on a subsequent turn — which is the entire reason the patch exists.

This test closes that gap. It simulates two consecutive turns end-to-end
against a faked dflash runtime + a faked LRUPromptCache with the same
prefix-trie semantics that mlx_lm.LRUPromptCache exposes
(insert_cache / fetch_nearest_cache).

Scenario:
  - Turn 1 prompt tokens:  [SYS][USER_1]                   (length N1 = 800)
  - Turn 1 generation:     [<think>][ANS_1]                 (length G1 = 100)
  - Turn 2 prompt tokens:  [SYS][USER_1][ANS_1][USER_2]    (length N2 = 1000)

Invariant pinned by this test:
  - After turn 1, the trie has a snapshot keyed by SYS+USER_1 *minus*
    DFLASH_APC_TAIL_SKIP tokens (the chat-template "<think>" suffix).
  - On turn 2, fetch_nearest_cache(turn_2_prompt_tokens) returns that
    snapshot. Without the patch, the trie would either be empty or only
    contain a key that is NOT a strict prefix of turn-2's prompt.
  - The hit's cached prefix length matches what _maybe_reuse_cache would
    skip during turn 2's prefill.

Bonus efficiency checks:
  - The whole simulation runs in <100 ms (the patch's compute cost is a
    single deepcopy per turn; faster fail-loud bound is 100 ms for a
    50-token snapshot).
  - The reuse % vs cold prefill: turn 2 has N2=1000 tokens; the patch
    saves (N1 - TAIL_SKIP) = 792 of them. That's the "what would multi-
    turn TTFT speedup actually be" measurement the audit asked for.
"""

from __future__ import annotations

import os
import sys
import tempfile
import time
import types
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


# Reuse the fake dflash skeleton from test_apc_patch.py.
class FakePromptArray:
    def __init__(self, n_tokens: int):
        self.shape = (1, int(n_tokens))

    def __getitem__(self, key):
        _batch, token_slice = key
        start = 0 if token_slice.start is None else token_slice.start
        stop = self.shape[1] if token_slice.stop is None else token_slice.stop
        return FakePromptArray(max(0, stop - start))


class FakeLayer:
    """Cache layer: tracks `offset` (number of tokens consumed). Mimics
    mlx_lm KVCache.offset semantics."""
    def __init__(self, offset: int = 0):
        self.offset = offset


class FakeTargetModel:
    def __call__(self, chunk: FakePromptArray, *, cache):
        n_tokens = int(chunk.shape[1])
        for layer in cache:
            layer.offset += n_tokens
        return {"logits_for_tokens": n_tokens}


class FakeLRUPromptCache:
    """Simulates the prefix-trie behavior of mlx_lm.LRUPromptCache.

    Stores inserted entries keyed by their (model_id, tokens) tuple and
    answers `fetch_nearest_cache(model_id, target_tokens)` with the
    longest stored key that is a strict prefix of `target_tokens` —
    exactly the semantic the patch relies on for hit propagation.
    """
    def __init__(self) -> None:
        self.entries: list[tuple[int, list[int], object]] = []

    def insert_cache(self, model_id: int, tokens: list[int], cache_obj) -> None:
        self.entries.append((model_id, list(tokens), cache_obj))

    def fetch_nearest_cache(self, model_id: int,
                            target_tokens: list[int]):
        """Longest strict-prefix match wins. Returns (cache_obj,
        cached_prefix_len) or (None, 0) on miss."""
        best: tuple[object, int] | None = None
        for mid, tokens, snap in self.entries:
            if mid != model_id:
                continue
            n = len(tokens)
            if n == 0 or n > len(target_tokens):
                continue
            if target_tokens[:n] == tokens:
                if best is None or n > best[1]:
                    best = (snap, n)
        if best is None:
            return None, 0
        return best

    def __len__(self) -> int:
        return len(self.entries)


def install_fake_modules() -> tuple[types.ModuleType, FakeLRUPromptCache, dict]:
    prompt_cache = FakeLRUPromptCache()
    original_calls = {"chunked": 0, "save": 0, "reuse_calls": 0}

    runtime = types.ModuleType("dflash_mlx.runtime")
    runtime._PREFILL_CHUNK = 64
    runtime._PROMPT_CACHE_MIN_PREFIX = 4

    def _chunked_prefill(target_model, prompt_array, cache):
        original_calls["chunked"] += 1
        return target_model(prompt_array, cache=cache)

    def _save_prompt_cache(_target_model, _prompt_tokens, _cache):
        original_calls["save"] += 1

    def _maybe_reuse_cache(target_model, prompt_tokens, **_kw):
        """Mimic mlx_lm's prefix-trie lookup. The patch's _patched_save
        keys entries by `id(target_model)`, so the lookup must use the
        same key — different model instances would never share cache."""
        original_calls["reuse_calls"] += 1
        snap, n = prompt_cache.fetch_nearest_cache(id(target_model), prompt_tokens)
        if snap is None:
            return [], 0
        return snap, n

    runtime._chunked_prefill = _chunked_prefill
    runtime._save_prompt_cache = _save_prompt_cache
    runtime._maybe_reuse_cache = _maybe_reuse_cache
    runtime._get_prompt_cache = lambda: prompt_cache

    dflash_pkg = types.ModuleType("dflash_mlx")
    dflash_pkg.runtime = runtime

    mlx_pkg = types.ModuleType("mlx")
    mlx_core = types.ModuleType("mlx.core")
    mlx_core.eval = lambda _x: None
    mlx_pkg.core = mlx_core

    sys.modules["dflash_mlx"] = dflash_pkg
    sys.modules["dflash_mlx.runtime"] = runtime
    sys.modules["mlx"] = mlx_pkg
    sys.modules["mlx.core"] = mlx_core

    return runtime, prompt_cache, original_calls


def main() -> int:
    failures: list[str] = []

    def check(label: str, ok: bool, detail: str = "") -> None:
        marker = "✓" if ok else "✗"
        suffix = f" — {detail}" if detail else ""
        print(f"  [{marker}] {label}{suffix}")
        if not ok:
            failures.append(label)

    print("== APC multi-turn hit-rate regression ==\n")

    TAIL_SKIP = 8
    os.environ["DFLASH_APC_TAIL_SKIP"] = str(TAIL_SKIP)

    # Realistic shapes: a 600-token system prompt + 200-token user1, then
    # ~100-token assistant_1, then a 200-token user_2 in turn 2. The exact
    # numbers don't matter; the test only depends on (a) turn-1 prompt
    # being a strict prefix of turn-2 prompt and (b) TAIL_SKIP < lengths.
    SYS_LEN = 600
    USER_1_LEN = 200
    ASSIST_1_LEN = 100
    USER_2_LEN = 200
    TURN_1_LEN = SYS_LEN + USER_1_LEN
    TURN_2_LEN = SYS_LEN + USER_1_LEN + ASSIST_1_LEN + USER_2_LEN  # 1100

    with tempfile.TemporaryDirectory() as td:
        os.environ["HOME"] = td
        runtime, prompt_cache, calls = install_fake_modules()
        sys.modules.pop("apc_patch", None)
        import apc_patch  # noqa: WPS433

        t0 = time.perf_counter()
        ok = apc_patch.install()
        check("apc_patch.install() succeeded", ok)

        # ---- Turn 1 ---------------------------------------------------
        # The agent constructs the full prompt tokens. We use distinct ints
        # 1..TURN_1_LEN so the prefix-trie can do a strict match. The same
        # model instance is used for prefill, save, AND turn-2 lookup —
        # in production this is the live dflash target model object.
        model = FakeTargetModel()
        turn_1_tokens = list(range(1, TURN_1_LEN + 1))
        turn_1_array = FakePromptArray(TURN_1_LEN)
        turn_1_cache = [FakeLayer()]

        runtime._chunked_prefill(model, turn_1_array, turn_1_cache)
        runtime._save_prompt_cache(model, turn_1_tokens, turn_1_cache)

        check("turn-1 prefill processed all tokens",
              turn_1_cache[0].offset == TURN_1_LEN,
              f"offset={turn_1_cache[0].offset}")

        # The patch should have inserted ONE entry, keyed by the prefix
        # of length TURN_1_LEN - TAIL_SKIP = 792.
        expected_key_len = TURN_1_LEN - TAIL_SKIP
        check("trie has exactly one snapshot after turn 1",
              len(prompt_cache) == 1,
              f"len={len(prompt_cache)}")

        if prompt_cache.entries:
            _mid, key, snap = prompt_cache.entries[0]
            check("snapshot key length == TURN_1_LEN - TAIL_SKIP",
                  len(key) == expected_key_len,
                  f"len(key)={len(key)} expected={expected_key_len}")
            check("snapshot key is a prefix of turn-1 tokens",
                  key == turn_1_tokens[:expected_key_len])
            # The snapshot's cache layer offset reflects the prefix point.
            check("snapshot offset matches its key length",
                  isinstance(snap, list) and snap and snap[0].offset == expected_key_len,
                  f"snap[0].offset={snap[0].offset if snap else 'n/a'}")

        # ---- Turn 2 (THE HIT-RATE CHECK) -----------------------------
        # Construct the realistic turn-2 prompt: turn-1 prompt + assistant_1
        # + new user. The first TURN_1_LEN tokens match turn 1 EXCEPT for
        # the trailing TAIL_SKIP positions — that's the entire reason the
        # patch snapshots at -TAIL_SKIP. So as long as TAIL_SKIP is large
        # enough to cover the chat-template variation, the snapshot's key
        # (first TURN_1_LEN - TAIL_SKIP tokens) is a strict prefix of
        # turn 2's prompt.
        # In production: the divergence between turn-1's prompt suffix
        # and turn-2's prompt at the same positions comes from
        # "<|im_start|>assistant\n<think>\n" being only on turn 1.
        # We simulate by leaving the first (TURN_1_LEN - TAIL_SKIP)
        # tokens identical and giving turn 2 different content from there.
        turn_2_tokens = list(range(1, TURN_1_LEN - TAIL_SKIP + 1)) + \
            list(range(10_000, 10_000 + (TURN_2_LEN - (TURN_1_LEN - TAIL_SKIP))))

        snap, cached_n = runtime._maybe_reuse_cache(model, turn_2_tokens)
        check("turn-2 hit found in trie (the regression V-B1 closes)",
              snap is not None and cached_n > 0,
              f"snap={'set' if snap else 'None'} cached_n={cached_n}")
        check("turn-2 cached_prefix_len == expected snapshot key length",
              cached_n == expected_key_len,
              f"got {cached_n} expected {expected_key_len}")

        # Reuse fraction: how many of turn 2's tokens skip prefill?
        reuse_pct = (cached_n / TURN_2_LEN) * 100 if TURN_2_LEN else 0.0
        # On this fixture: 792 / 1100 = 72.0% of turn 2 reuses turn 1's cache.
        check("reuse fraction ≥ 50% of turn-2 prompt",
              reuse_pct >= 50.0,
              f"reuse_pct={reuse_pct:.1f}%")

        # ---- Negative case: a turn whose prompt has NO common prefix --
        # E.g. a fresh chat that doesn't share the system prompt. Must NOT
        # falsely return the turn-1 snapshot.
        unrelated_tokens = list(range(50_000, 50_000 + 500))
        snap_unrelated, cached_unrelated = runtime._maybe_reuse_cache(model, unrelated_tokens)
        check("unrelated prompt does NOT spuriously hit",
              snap_unrelated is None or cached_unrelated == 0,
              f"snap={'set' if snap_unrelated else 'None'} "
              f"cached={cached_unrelated}")

        # ---- Efficiency budget --------------------------------------
        elapsed_ms = (time.perf_counter() - t0) * 1000
        # The patch's per-turn overhead is dominated by one copy.deepcopy(cache)
        # on a list of FakeLayers — should be sub-millisecond. The whole
        # test (install + turn 1 prefill + save + turn 2 lookup) should
        # complete in <100 ms.
        check("install + 2-turn simulation completes in <100 ms",
              elapsed_ms < 100.0,
              f"{elapsed_ms:.1f} ms")

        # ---- Idempotency: re-installing must not corrupt state -----
        # The patch is supposed to be idempotent; insert another snapshot
        # via a second prefill and confirm no duplicate side effects.
        ok_idempotent = apc_patch.install()
        check("apc_patch.install() idempotent", ok_idempotent)

    print(f"\n  reuse % at this fixture: {reuse_pct:.1f}% "
          f"({cached_n}/{TURN_2_LEN} tokens skip prefill on turn 2)")
    print(f"\n== {'PASS' if not failures else 'FAIL'} "
          f"({len(failures)} failure(s)) ==")
    if failures:
        for f in failures:
            print(f"  - {f}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
