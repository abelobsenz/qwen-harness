#!/usr/bin/env python3
"""Single-source-of-truth contract for the [REFUSED — ...] marker.

The cap-exhaustion nudge in scripts/agent.py:1003-1020 detects refusal
results by matching the literal `[REFUSED` prefix on tool-result content.
Any cap site that emits the marker with a different prefix (missing the
em-dash, lowercase variant, alternative dash glyph) silently bypasses the
nudge — the model burns turns repeating capped calls until the streaming
idle timeout (90s/turn) fires.

This test pins two invariants:

  1. EVERY cap site that emits a refusal in agent_tools.py uses the
     shared REFUSED_PREFIX constant. New cap sites added without going
     through the constant will fail this test.

  2. The detect-prefix at agent.py:885 and elsewhere is a strict prefix
     of REFUSED_PREFIX so .startswith(REFUSED_DETECT_PREFIX) catches
     every emit. (Belt-and-braces: if a future emit ever differed in the
     trailing dash/space, the detect-prefix still matches.)

Both checks run in <50 ms on a fast Python — they're pure source-level
greps + AST inspection, no model load, no HTTP, no subprocesses.
"""

from __future__ import annotations

import ast
import re
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from agent_tools import REFUSED_PREFIX, REFUSED_DETECT_PREFIX  # noqa: E402


def _find_literal_emit_sites(path: Path) -> list[tuple[int, str]]:
    """Find lines that emit the literal `[REFUSED — ` prefix INSIDE a
    string literal, anywhere except the REFUSED_PREFIX definition itself
    and docstring/comment references. Returns (lineno, snippet)."""
    out: list[tuple[int, str]] = []
    src = path.read_text()
    tree = ast.parse(src)
    for node in ast.walk(tree):
        # Plain string constants — these are the dangerous case.
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            if "[REFUSED — " in node.value:
                # The REFUSED_PREFIX assignment is the only legitimate
                # place where this literal appears in a string constant.
                # Skip docstrings and module-level assignment.
                # ast.Constant doesn't expose parent; cross-check by line.
                lineno = node.lineno
                # Re-read the source line; skip if it's the assignment.
                line = src.splitlines()[lineno - 1]
                if "REFUSED_PREFIX" in line and "=" in line:
                    continue
                # Skip if the line is purely a comment-like string (docstring or """).
                if line.lstrip().startswith(('"""', "'''", "#")):
                    continue
                out.append((lineno, line.strip()))
        # f-strings — check each string segment.
        elif isinstance(node, ast.JoinedStr):
            for v in node.values:
                if isinstance(v, ast.Constant) and isinstance(v.value, str):
                    if "[REFUSED — " in v.value:
                        out.append((node.lineno,
                                    src.splitlines()[node.lineno - 1].strip()))
                        break
    return out


def main() -> int:
    failures = 0
    t0 = time.perf_counter()

    print("== REFUSED_PREFIX contract ==\n")

    # 1. Confirm the constants exist with the expected literal values.
    expected_emit = "[REFUSED — "
    expected_detect = "[REFUSED"
    if REFUSED_PREFIX != expected_emit:
        print(f"  [✗] REFUSED_PREFIX changed: {REFUSED_PREFIX!r} != {expected_emit!r}")
        failures += 1
    else:
        print(f"  [✓] REFUSED_PREFIX == {REFUSED_PREFIX!r}")
    if REFUSED_DETECT_PREFIX != expected_detect:
        print(f"  [✗] REFUSED_DETECT_PREFIX changed: "
              f"{REFUSED_DETECT_PREFIX!r} != {expected_detect!r}")
        failures += 1
    else:
        print(f"  [✓] REFUSED_DETECT_PREFIX == {REFUSED_DETECT_PREFIX!r}")

    # 2. Detect-prefix must be a strict prefix of emit-prefix.
    if not REFUSED_PREFIX.startswith(REFUSED_DETECT_PREFIX):
        print(f"  [✗] REFUSED_PREFIX does not start with REFUSED_DETECT_PREFIX")
        failures += 1
    else:
        print("  [✓] REFUSED_DETECT_PREFIX is a strict prefix of REFUSED_PREFIX")

    # 3. No literal `[REFUSED — ` string constants outside the constant
    #    definition itself. AST-based so f-strings ARE caught.
    tools_path = HERE / "agent_tools.py"
    literal_sites = _find_literal_emit_sites(tools_path)
    if literal_sites:
        print(f"  [✗] {len(literal_sites)} literal '[REFUSED — ' "
              "site(s) outside REFUSED_PREFIX definition:")
        for ln, snippet in literal_sites:
            preview = snippet if len(snippet) <= 90 else snippet[:87] + "…"
            print(f"        line {ln}: {preview}")
        failures += 1
    else:
        print("  [✓] no literal '[REFUSED — ' emit sites remain "
              "(all refactored through REFUSED_PREFIX)")

    # 4. Confirm REFUSED_PREFIX is referenced from at least the expected
    #    number of cap sites (regression: if someone reverts the
    #    refactor on one site, this catches it).
    src = tools_path.read_text()
    n_refs = src.count("REFUSED_PREFIX")
    # 1 definition + 1 docstring comment-style mention isn't counted
    # by .count if it's worded differently. We expect at least 10 emit
    # sites + 1 definition + 1 doc string mention near agent.py-line 71.
    MIN_REFS = 11
    if n_refs < MIN_REFS:
        print(f"  [✗] REFUSED_PREFIX referenced {n_refs} times, "
              f"expected ≥ {MIN_REFS} (definition + ≥10 emit sites)")
        failures += 1
    else:
        print(f"  [✓] REFUSED_PREFIX referenced {n_refs} times "
              f"(definition + ≥10 emit sites)")

    # 5. Behavior: a CachedDispatcher instance refused by the search
    #    cap actually emits a string that starts with REFUSED_PREFIX.
    #    This is the end-to-end contract — if the constant assembly
    #    breaks (e.g. someone introduces a Unicode normalization bug),
    #    this catches it.
    from agent_tools import CachedDispatcher
    d = CachedDispatcher()
    # Force the cap by setting count above max
    d.web_search_count = d._web_search_max + 1
    refusal = d._check_web_search_cap()
    if refusal is None:
        print("  [✗] expected a refusal when count > max, got None")
        failures += 1
    elif not refusal.startswith(REFUSED_PREFIX):
        print(f"  [✗] refusal does not start with REFUSED_PREFIX: {refusal[:60]!r}")
        failures += 1
    else:
        print("  [✓] live cap-exhaustion emit uses REFUSED_PREFIX")

    elapsed = (time.perf_counter() - t0) * 1000
    print(f"\n  elapsed: {elapsed:.1f} ms")
    print(f"\n== {'PASS' if failures == 0 else 'FAIL'} "
          f"({failures} failure(s)) ==")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
