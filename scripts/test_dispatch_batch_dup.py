#!/usr/bin/env python3
"""Tests for parallel-duplicate detection in CachedDispatcher.dispatch_batch.

Why: TJX iter 23 fired 8 byte-identical web_search queries in ONE turn (and
4 each in two later turns). The intra-turn dedup collapsed them to one
dispatch but returned 8 cached copies of the result, which the model read
as "still searching." After this fix:
  - Only ONE call actually dispatches.
  - The other 7 inputs get explicit `[REFUSED — parallel duplicate]`.
  - The cross-turn counter is charged for all 8 attempts, so the very next
    re-issue of the same query (even alone) hits the existing >=3 threshold
    and is REFUSED.
"""
from __future__ import annotations
import os, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import agent_tools  # noqa: E402


def main() -> int:
    failures = 0

    # ---- 1. 8 parallel-identical web_search → 1 dispatch + 7 refused ----
    print("[1] 8 byte-identical parallel web_search calls")
    cd = agent_tools.CachedDispatcher()
    calls = [("web_search", {"query": "X"})] * 8
    n_real = [0]
    real_dispatch = agent_tools.dispatch
    def mock(fn, args):
        n_real[0] += 1
        return f"result for {args.get('query', '')}"
    agent_tools.dispatch = mock
    try:
        results = cd.dispatch_batch(calls)
    finally:
        agent_tools.dispatch = real_dispatch
    n_refused = sum(1 for r, _ in results if r.startswith("[REFUSED"))
    if n_real[0] != 1:
        print(f"  [FAIL] expected 1 dispatch, got {n_real[0]}"); failures += 1
    elif n_refused != 7:
        print(f"  [FAIL] expected 7 refused inputs, got {n_refused}"); failures += 1
    elif results[0][0].startswith("[REFUSED"):
        print(f"  [FAIL] first input was refused — should have run"); failures += 1
    else:
        print("  [OK] 1 dispatch, 7 refused-in-batch, first ran")

    # ---- 2. Cross-turn counter blocks single re-issue ----
    print("[2] Single re-issue of same query on next turn → REFUSED")
    next_result, was_cached = cd.dispatch("web_search", {"query": "X"})
    if not next_result.startswith("[REFUSED"):
        print(f"  [FAIL] expected REFUSED, got {next_result[:80]!r}"); failures += 1
    else:
        print("  [OK] cross-turn re-issue refused")

    # ---- 3. Different queries in same batch are NOT refused ----
    print("[3] Two distinct web_search queries in one batch → both run")
    cd2 = agent_tools.CachedDispatcher()
    n_real[0] = 0
    agent_tools.dispatch = mock
    try:
        results = cd2.dispatch_batch([
            ("web_search", {"query": "alpha"}),
            ("web_search", {"query": "beta"}),
        ])
    finally:
        agent_tools.dispatch = real_dispatch
    if n_real[0] != 2 or any(r.startswith("[REFUSED") for r, _ in results):
        print(f"  [FAIL] expected both to dispatch, got {n_real[0]} dispatches, "
              f"results: {[r[:30] for r, _ in results]}"); failures += 1
    else:
        print("  [OK] both distinct queries dispatched normally")

    # ---- 4. Non-web duplicate (read_file) keeps prior shared-result ----
    print("[4] Non-web duplicates (read_file) share result, no refusal marker")
    cd3 = agent_tools.CachedDispatcher()
    n_real[0] = 0
    def mock2(fn, args):
        n_real[0] += 1
        return f"file content {args.get('path', '')}"
    agent_tools.dispatch = mock2
    try:
        results = cd3.dispatch_batch([
            ("read_file", {"path": "/tmp/x"}),
            ("read_file", {"path": "/tmp/x"}),
        ])
    finally:
        agent_tools.dispatch = real_dispatch
    if any(r.startswith("[REFUSED — parallel duplicate") for r, _ in results):
        print(f"  [FAIL] parallel-dup refusal fired on read_file; should be web-only"); failures += 1
    else:
        print("  [OK] non-web duplicates handled by existing cache, not the new refusal")

    # ---- 5. Three calls of same key: 1 dispatch, 2 refused, counter=3 ----
    print("[5] 3-burst of same query → counter=3 immediately")
    cd4 = agent_tools.CachedDispatcher()
    n_real[0] = 0
    agent_tools.dispatch = mock
    try:
        cd4.dispatch_batch([("web_search", {"query": "Z"})] * 3)
    finally:
        agent_tools.dispatch = real_dispatch
    key = ("web_search", agent_tools._arg_key({"query": "Z"}))
    count = cd4.web_call_counts.get(key, 0)
    if count != 3:
        print(f"  [FAIL] expected counter=3, got {count}"); failures += 1
    else:
        print(f"  [OK] counter at 3 after 3-burst (1 dispatch + 2 pre-charge)")

    if failures:
        print(f"\n== FAIL ({failures} failure(s)) ==")
        return 1
    print("\n== PASS (parallel-duplicate detection works) ==")
    return 0


if __name__ == "__main__":
    sys.exit(main())
