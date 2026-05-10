#!/usr/bin/env python3
"""Unit tests for apc_patch's monkey-patch wiring.

Uses a fake dflash_mlx.runtime module and fake mlx.core so this test validates
the patch logic without loading MLX, Metal, or a real model.
"""

from __future__ import annotations

import os
import sys
import tempfile
import types

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


class FakePromptArray:
    def __init__(self, n_tokens: int):
        self.shape = (1, int(n_tokens))

    def __getitem__(self, key):
        _batch, token_slice = key
        start = 0 if token_slice.start is None else token_slice.start
        stop = self.shape[1] if token_slice.stop is None else token_slice.stop
        return FakePromptArray(max(0, stop - start))


class FakeLayer:
    def __init__(self, offset: int = 0):
        self.offset = offset


class FakeTargetModel:
    def __call__(self, chunk: FakePromptArray, *, cache):
        n_tokens = int(chunk.shape[1])
        for layer in cache:
            layer.offset += n_tokens
        return {"logits_for_tokens": n_tokens}


class FakePromptCache:
    def __init__(self):
        self.inserted: list[tuple[int, list[int], list[FakeLayer]]] = []

    def insert_cache(self, model_id: int, tokens: list[int], cache) -> None:
        self.inserted.append((model_id, list(tokens), cache))

    def __len__(self):
        return len(self.inserted)


def install_fake_modules():
    prompt_cache = FakePromptCache()
    original_calls: dict[str, int] = {"chunked": 0, "save": 0}

    runtime = types.ModuleType("dflash_mlx.runtime")
    runtime._PREFILL_CHUNK = 8
    runtime._PROMPT_CACHE_MIN_PREFIX = 4

    def _chunked_prefill(target_model, prompt_array, cache):
        original_calls["chunked"] += 1
        return target_model(prompt_array, cache=cache)

    def _save_prompt_cache(_target_model, _prompt_tokens, _cache):
        original_calls["save"] += 1

    def _maybe_reuse_cache(_target_model, _prompt_tokens, **_kw):
        return [], 0

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

    print("== apc_patch tests ==\n")
    os.environ["DFLASH_APC_TAIL_SKIP"] = "8"
    with tempfile.TemporaryDirectory() as td:
        os.environ["HOME"] = td
        runtime, prompt_cache, original_calls = install_fake_modules()
        sys.modules.pop("apc_patch", None)
        import apc_patch

        check("install succeeds", apc_patch.install())
        check("install is idempotent", apc_patch.install())
        check("runtime sentinel set", bool(getattr(runtime, "_apc_patch_installed", False)))

        model = FakeTargetModel()
        cache = [FakeLayer()]
        logits = runtime._chunked_prefill(model, FakePromptArray(20), cache)
        check("patched prefill returns last logits",
              logits == {"logits_for_tokens": 4}, f"got {logits!r}")
        check("patched prefill processed all tokens",
              cache[0].offset == 20, f"offset={cache[0].offset}")

        runtime._save_prompt_cache(model, list(range(20)), cache)
        check("original save still called",
              original_calls["save"] == 1, f"calls={original_calls['save']}")
        check("prompt-only snapshot inserted",
              len(prompt_cache.inserted) == 1,
              f"insertions={len(prompt_cache.inserted)}")
        if prompt_cache.inserted:
            _mid, prefix_tokens, snap = prompt_cache.inserted[0]
            check("snapshot prefix trims chat-template tail",
                  len(prefix_tokens) == 12, f"len={len(prefix_tokens)}")
            check("snapshot preserves offset at prefix point",
                  snap[0].offset == 12, f"offset={snap[0].offset}")

        # Regression: zero-token prefill takes the original fallback path.
        zero_cache = [FakeLayer()]
        runtime._chunked_prefill(model, FakePromptArray(0), zero_cache)
        check("zero-token path falls back without NameError",
              original_calls["chunked"] == 1,
              f"calls={original_calls['chunked']}")

    print(f"\n== {'PASS' if not failures else 'FAIL'} "
          f"({len(failures)} failure(s)) ==")
    if failures:
        for f in failures:
            print(f"  - {f}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
