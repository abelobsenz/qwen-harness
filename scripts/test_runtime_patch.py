#!/usr/bin/env python3
"""Tests for runtime_patch.apply_repetition_penalty + RepetitionContext.

Verifies, against synthetic mlx logits:

  1. penalty=0 is a true no-op (returns input unchanged, by value)
  2. penalty>0 subtracts exactly `penalty` from each unique recent id
  3. argmax shifts AWAY from a recently-emitted token when its logit
     advantage is smaller than the penalty (the actual loop-breaking
     mechanism we want)
  4. argmax does NOT shift when the recent token's advantage exceeds
     the penalty (we don't aggressively rewrite the model's preferences)
  5. Empty recent list is a no-op
  6. Out-of-range ids are silently filtered (defensive)
  7. Works on both 1D (V,) and 2D (B, V) logits
  8. RepetitionContext.is_active reflects penalty + history_size
  9. RepetitionContext.observe + modify_logits round-trip works
 10. RepetitionContext with history_size=0 is inactive
 11. is_globally_enabled() reflects DEFAULT_PENALTY env

Skips gracefully if mlx isn't available (it is, in our venv, but the
test file is portable).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main() -> int:
    failures: list[str] = []

    # Force reload so DEFAULT_PENALTY picks up the env we set inside the test
    # rather than whatever was in the parent shell.
    os.environ["DFLASH_REP_PENALTY"] = "0.0"
    sys.modules.pop("runtime_patch", None)
    import runtime_patch as rp

    try:
        import mlx.core as mx
    except ImportError:
        print("mlx not available — skipping (this should never happen "
              "in the qwen36_MTP venv)")
        return 0

    print("== runtime_patch tests ==\n")

    def check(label: str, ok: bool, detail: str = "") -> None:
        marker = "✓" if ok else "✗"
        suffix = f" — {detail}" if detail else ""
        print(f"  [{marker}] {label}{suffix}")
        if not ok:
            failures.append(label)

    # 1. penalty=0 → no-op
    print("[1] penalty=0 is a no-op")
    logits = mx.array([1.0, 2.0, 3.0, 4.0, 5.0])
    out = rp.apply_repetition_penalty(logits, [0, 1, 2], penalty=0.0)
    check("returns input unchanged", bool(mx.array_equal(out, logits).item()))

    # 2. penalty>0 subtracts exactly that from each unique id
    print("\n[2] penalty>0 subtracts at recent ids")
    logits = mx.array([1.0, 2.0, 3.0, 4.0, 5.0])
    out = rp.apply_repetition_penalty(logits, [0, 1, 2], penalty=0.5)
    expected = mx.array([0.5, 1.5, 2.5, 4.0, 5.0])
    diff = float(mx.max(mx.abs(out - expected)).item())
    check("output matches expected (max diff < 1e-6)",
          diff < 1e-6, f"diff={diff:.2e}")

    # 3. argmax shifts away from a recent token when advantage < penalty
    print("\n[3] argmax shifts AWAY from recent token when advantage < penalty")
    logits = mx.array([2.0, 5.1, 5.0])  # token 1 is best by 0.1
    base_argmax = int(mx.argmax(logits).item())
    out = rp.apply_repetition_penalty(logits, [1], penalty=0.5)
    new_argmax = int(mx.argmax(out).item())
    check("baseline argmax = 1", base_argmax == 1, f"got {base_argmax}")
    check("after penalty argmax = 2 (different)",
          new_argmax == 2, f"got {new_argmax}")

    # 4. argmax does NOT shift when advantage > penalty
    print("\n[4] argmax stays put when advantage > penalty")
    logits = mx.array([2.0, 6.0, 5.0])  # token 1 is best by 1.0
    out = rp.apply_repetition_penalty(logits, [1], penalty=0.5)
    new_argmax = int(mx.argmax(out).item())
    check("argmax unchanged (still 1)",
          new_argmax == 1, f"got {new_argmax}")

    # 5. Empty recent list → no-op
    print("\n[5] empty recent list is a no-op")
    logits = mx.array([1.0, 2.0, 3.0])
    out = rp.apply_repetition_penalty(logits, [], penalty=0.5)
    check("returns input unchanged",
          bool(mx.array_equal(out, logits).item()))

    # 6. Out-of-range ids filtered
    print("\n[6] out-of-range ids filtered defensively")
    logits = mx.array([1.0, 2.0, 3.0])
    out = rp.apply_repetition_penalty(logits, [0, 99, -3, 1], penalty=1.0)
    expected = mx.array([0.0, 1.0, 3.0])  # only 0 and 1 are valid
    diff = float(mx.max(mx.abs(out - expected)).item())
    check("OOR ids ignored, in-range ids penalized",
          diff < 1e-6, f"diff={diff:.2e}")

    # 7. 2D logits work via broadcasting
    print("\n[7] 2D (B, V) logits work")
    logits2d = mx.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    out = rp.apply_repetition_penalty(logits2d, [0, 2], penalty=0.5)
    expected = mx.array([[0.5, 2.0, 2.5], [3.5, 5.0, 5.5]])
    diff = float(mx.max(mx.abs(out - expected)).item())
    check("2D broadcast applies penalty to all batch rows",
          diff < 1e-6, f"diff={diff:.2e}")

    # 8. is_active reflects penalty + history_size
    print("\n[8] RepetitionContext.is_active gates")
    ctx_off = rp.RepetitionContext(history_size=64, penalty=0.0)
    ctx_zero_history = rp.RepetitionContext(history_size=0, penalty=0.05)
    ctx_on = rp.RepetitionContext(history_size=64, penalty=0.05)
    check("penalty=0 → inactive", not ctx_off.is_active)
    check("history=0 → inactive", not ctx_zero_history.is_active)
    check("both > 0 → active", ctx_on.is_active)

    # 9. observe + modify_logits round-trip
    print("\n[9] RepetitionContext observe + modify_logits")
    ctx = rp.RepetitionContext(history_size=4, penalty=0.5)
    for tok in [1, 2, 1, 0]:  # last 4 (tied with maxlen) → {0, 1, 2}
        ctx.observe(tok)
    logits = mx.array([10.0, 10.0, 10.0, 10.0, 10.0])
    out = ctx.modify_logits(logits)
    expected = mx.array([9.5, 9.5, 9.5, 10.0, 10.0])
    diff = float(mx.max(mx.abs(out - expected)).item())
    check("observed tokens get penalty",
          diff < 1e-6, f"diff={diff:.2e}")

    # 10. inactive context: modify_logits is identity
    print("\n[10] inactive context: modify_logits is identity")
    ctx_off = rp.RepetitionContext(history_size=64, penalty=0.0)
    ctx_off.observe(0)
    ctx_off.observe(1)
    logits = mx.array([1.0, 2.0, 3.0])
    out = ctx_off.modify_logits(logits)
    check("inactive → identity",
          bool(mx.array_equal(out, logits).item()))

    # 11. is_globally_enabled reflects env
    print("\n[11] is_globally_enabled() honors DFLASH_REP_PENALTY env")
    # Reload with env set to a non-zero value
    os.environ["DFLASH_REP_PENALTY"] = "0.05"
    sys.modules.pop("runtime_patch", None)
    import runtime_patch as rp2
    check("env=0.05 → enabled", rp2.is_globally_enabled())
    os.environ["DFLASH_REP_PENALTY"] = "0.0"
    sys.modules.pop("runtime_patch", None)
    import runtime_patch as rp3
    check("env=0.0 → disabled", not rp3.is_globally_enabled())

    # 12. dflash runtime wiring is present (static audit; no model needed).
    print("\n[12] dflash runtime wiring")
    runtime_path = (
        Path(__file__).resolve().parents[1]
        / "venv/lib/python3.14/site-packages/dflash_mlx/runtime.py"
    )
    runtime_src = runtime_path.read_text()
    check("_make_repetition_context helper installed", "def _make_repetition_context" in runtime_src)
    check("baseline modifies logits before greedy", "_rep_modify_logits(rep_ctx, logits[:, -1, :])" in runtime_src)
    check("spec draft modifies logits", "_rep_modify_logits(rep_ctx, draft_logits)" in runtime_src)
    check("spec verify modifies logits", "_rep_modify_logits(rep_ctx, verify_logits[0])" in runtime_src)
    check("stream path honors lazy draft eval", runtime_src.count("DFLASH_LAZY_DRAFT_EVAL") >= 2)
    check("runtime returns repetition metric", runtime_src.count('"repetition_penalty"') >= 4)

    print(f"\n== {'PASS' if not failures else 'FAIL'} "
          f"({len(failures)} failure(s)) ==")
    if failures:
        for f in failures:
            print(f"  - {f}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
