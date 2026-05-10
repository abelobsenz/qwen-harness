#!/usr/bin/env python3
"""Tiered test runner for qwen36_MTP.

Three modes:
  --fast (default): tests under ~5s each — the everyday inner loop
  --slow:           tests >5s — sustained-load + long-duration only
  --all:            everything

The tier is determined by filename pattern, not by measurement at run
time. The patterns are tuned to the tests that exist today (as of
Round 22, 20 test files):

  SLOW patterns:
    - test_proxy_long_*    sustained-load memory tests (idle/stream)

  FAST: everything else (currently 18 files; ~12s total wall).

Anything new that takes >5s should match a SLOW pattern OR be added
to the explicit slow list at the top of this file.

Why a runner instead of `pytest`: the project deliberately uses
stdlib-only test scripts so it can run inside the dflash venv without
extra deps. This runner preserves that — pure stdlib, ~80 lines.

Usage:
  scripts/run_tests.py             # fast tier (default)
  scripts/run_tests.py --slow      # slow tier only
  scripts/run_tests.py --all       # everything
  scripts/run_tests.py --list      # show classification, run nothing
  scripts/run_tests.py -v          # show test names as they run
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path


SCRIPT_DIR = Path(__file__).parent

# Patterns that mark a test as "slow tier". Add a glob-style prefix
# here when introducing new long-running tests.
SLOW_PATTERNS = (
    "test_proxy_long_",   # sustained-load memory: long_idle, long_stream
)

# Explicit fast-tier override: even if a name matches a slow pattern,
# a file listed here stays fast. Empty today; available for "this test
# is technically long-running but it's fast enough" exceptions.
EXPLICIT_FAST: tuple[str, ...] = ()


def classify(path: Path) -> str:
    """Return 'slow' or 'fast' for a given test file."""
    name = path.name
    if name in EXPLICIT_FAST:
        return "fast"
    for prefix in SLOW_PATTERNS:
        if name.startswith(prefix):
            return "slow"
    return "fast"


def discover() -> list[Path]:
    """Return sorted list of test files (test_*.py + the loop_guard
    self-test entry point)."""
    paths = sorted(SCRIPT_DIR.glob("test_*.py"))
    # loop_guard.py has its own self-test entry (`if __name__`) — keep
    # it in the suite as a fast test.
    self_test = SCRIPT_DIR / "loop_guard.py"
    if self_test.exists():
        paths.append(self_test)
    return paths


def run_one(path: Path, verbose: bool) -> tuple[bool, float, str]:
    """Run a single test. Returns (passed, elapsed_s, last_line)."""
    py = sys.executable
    t0 = time.perf_counter()
    proc = subprocess.run(
        [py, str(path)],
        cwd=str(path.parent.parent),  # project root, so bench_results/ etc resolve
        capture_output=True,
        text=True,
        timeout=600,
    )
    elapsed = time.perf_counter() - t0
    passed = proc.returncode == 0
    # Last non-empty stdout line is usually the PASS/FAIL summary.
    last = ""
    for line in reversed(proc.stdout.splitlines()):
        if line.strip():
            last = line.strip()
            break
    if verbose and not passed:
        # Surface the tail of stderr / stdout on failure so the user can
        # debug without re-running.
        print(f"\n--- {path.name} stdout (last 20 lines) ---")
        print("\n".join(proc.stdout.splitlines()[-20:]))
        if proc.stderr.strip():
            print(f"--- {path.name} stderr (last 20 lines) ---")
            print("\n".join(proc.stderr.splitlines()[-20:]))
    return passed, elapsed, last


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--fast", action="store_true",
                   help="run only fast-tier tests (default)")
    g.add_argument("--slow", action="store_true",
                   help="run only slow-tier tests")
    g.add_argument("--all", action="store_true",
                   help="run everything")
    g.add_argument("--list", action="store_true",
                   help="print classification and exit without running")
    ap.add_argument("-v", "--verbose", action="store_true",
                    help="show test name as it runs and surface failures")
    args = ap.parse_args()

    paths = discover()
    if not paths:
        print("no tests found")
        return 1

    by_tier: dict[str, list[Path]] = {"fast": [], "slow": []}
    for p in paths:
        by_tier[classify(p)].append(p)

    if args.list:
        for tier in ("fast", "slow"):
            print(f"\n[{tier}] ({len(by_tier[tier])}):")
            for p in by_tier[tier]:
                print(f"  {p.name}")
        return 0

    if args.slow:
        chosen = by_tier["slow"]
        label = "slow"
    elif args.all:
        chosen = paths
        label = "all"
    else:
        chosen = by_tier["fast"]
        label = "fast"

    print(f"== run_tests ({label}; {len(chosen)} tests) ==\n")

    failed: list[str] = []
    elapsed_total = 0.0
    for p in chosen:
        ok, elapsed, last = run_one(p, args.verbose)
        elapsed_total += elapsed
        marker = "✓" if ok else "✗"
        # Truncate the last-line summary so the runner output stays readable.
        last_short = last if len(last) <= 70 else last[:67] + "…"
        print(f"  [{marker}] {p.name:<40s} {elapsed:>6.2f}s  {last_short}")
        if not ok:
            failed.append(p.name)

    print(f"\n  total: {len(chosen)} tests, {elapsed_total:.1f}s wall, "
          f"{len(failed)} failed")
    if failed:
        print(f"  failures: {', '.join(failed)}")
        return 1
    print(f"  ✓ all tests pass")
    return 0


if __name__ == "__main__":
    sys.exit(main())
