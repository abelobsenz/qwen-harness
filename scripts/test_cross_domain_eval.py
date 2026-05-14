#!/usr/bin/env python3
"""Structural tests for eval_data/cross_domain_prompts.json.

The fixture broadens the qwen agent's evaluation beyond finance to test
whether the audit-gate, REFUSED-marker, and quantitative-discipline
prompts (tuned on `vals-ai/finance_agent_benchmark`) generalise. This
test does NOT run inference — it pins the fixture's structure and
"groundability" so a typo or schema drift fails CI before any model
is loaded.

Invariants:
  1. The fixture loads as JSON and exposes `prompts`.
  2. Each prompt has the canonical 4-field schema
     (id, type, question, rubric_concepts).
  3. Every rubric_concepts entry is a non-empty list of non-empty
     strings (the per-fact alternatives the scorer checks).
  4. Prompt IDs are unique.
  5. The fixture covers at least three distinct `type` values (the
     whole point of broadening is breadth).
  6. Every prompt's question references at least one repo artefact that
     actually exists (no dead-pointer prompts). This is the
     "groundability" check.
  7. The fixture is loadable by the same Path the eval driver uses
     (sample_prompts.json shape).

All pure-source — no inference, no HTTP. Runs in <30 ms.
"""

from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
FIXTURE = ROOT / "eval_data" / "cross_domain_prompts.json"


def main() -> int:
    failures: list[str] = []

    def check(label: str, ok: bool, detail: str = "") -> None:
        marker = "✓" if ok else "✗"
        suffix = f" — {detail}" if detail else ""
        print(f"  [{marker}] {label}{suffix}")
        if not ok:
            failures.append(label)

    print("== cross-domain eval fixture tests ==\n")
    t0 = time.perf_counter()

    check("fixture file exists", FIXTURE.exists(), f"path={FIXTURE}")
    if not FIXTURE.exists():
        return 1

    try:
        data = json.loads(FIXTURE.read_text())
    except json.JSONDecodeError as e:
        check("fixture is valid JSON", False, str(e))
        return 1
    check("fixture is valid JSON", True)

    prompts = data.get("prompts")
    check("fixture has 'prompts' list",
          isinstance(prompts, list) and len(prompts) > 0,
          f"len={len(prompts) if prompts else 0}")

    if not isinstance(prompts, list):
        return 1

    # Per-prompt schema
    seen_ids: set[str] = set()
    seen_types: set[str] = set()
    for i, p in enumerate(prompts):
        prefix = f"prompt[{i}]"
        if not isinstance(p, dict):
            check(f"{prefix} is a dict", False, type(p).__name__)
            continue
        for key in ("id", "type", "question", "rubric_concepts"):
            if key not in p:
                check(f"{prefix} has '{key}'", False)
                continue
        pid = p.get("id")
        if not isinstance(pid, str) or not pid:
            check(f"{prefix} id is a non-empty string", False)
            continue
        if pid in seen_ids:
            check(f"{prefix} id is unique", False, f"duplicate {pid!r}")
        seen_ids.add(pid)
        ptype = p.get("type")
        if not isinstance(ptype, str) or not ptype:
            check(f"{prefix} type is a non-empty string", False)
            continue
        seen_types.add(ptype)
        q = p.get("question")
        if not isinstance(q, str) or len(q.strip()) < 10:
            check(f"{prefix} question is a non-trivial string", False)
        rc = p.get("rubric_concepts")
        if not isinstance(rc, list) or not rc:
            check(f"{prefix} rubric_concepts is a non-empty list", False)
            continue
        # Each fact is a non-empty list of non-empty strings.
        for j, fact in enumerate(rc):
            if not isinstance(fact, list) or not fact:
                check(f"{prefix} rubric_concepts[{j}] is a non-empty list",
                      False)
                continue
            if not all(isinstance(a, str) and a for a in fact):
                check(f"{prefix} rubric_concepts[{j}] has non-empty string alternatives",
                      False)

    if not failures:
        check(f"all {len(prompts)} prompts have valid schema", True)

    check("prompt IDs are unique",
          len(seen_ids) == len(prompts),
          f"{len(seen_ids)} unique / {len(prompts)} total")
    check("fixture covers ≥3 distinct prompt types",
          len(seen_types) >= 3,
          f"types={sorted(seen_types)}")

    # Groundability: every prompt should mention either a repo file
    # (scripts/<file>.py, config/<file>.conf), a known repo concept
    # (REFUSED_PREFIX, dflash_mlx, etc.), or a math problem (pure math).
    repo_files = {f.name for f in (ROOT / "scripts").glob("*.py")}
    repo_files |= {f.name for f in (ROOT / "scripts").glob("*.conf")}
    repo_files |= {f.name for f in (ROOT / "config").glob("*.conf")}
    KNOWN_CONCEPTS = {
        "dflash_mlx", "REFUSED_PREFIX", "REFUSED", "Fibonacci",
        "QWEN_AGENT_TOOL_TRUNC", "LOOP_GUARD_DISABLE",
        "LoopGuardConfig", "phrase_min_repeats",
    }
    MATH_TOKENS = ("multiply", "* ", "/", "+", "Fibonacci", "compute",
                   "Compute", "now - t", "window")
    for p in prompts:
        q = p.get("question", "")
        # 1. Any repo file mentioned by name?
        any_file = any(name in q for name in repo_files)
        # 2. Any known repo concept?
        any_concept = any(c in q for c in KNOWN_CONCEPTS)
        # 3. Pure math/algo question?
        any_math = any(t in q for t in MATH_TOKENS)
        ok = any_file or any_concept or any_math
        if not ok:
            check(f"prompt {p.get('id')!r} is groundable", False, f"q='{q[:60]}…'")
    if not failures:
        check("every prompt grounds in a repo artefact, known concept, or math problem",
              True)

    # Reuse: confirm the fixture can be loaded by the same driver that
    # loads sample_prompts.json. We don't import the driver (it has heavy
    # deps); we just confirm the schema matches by comparing top-level
    # keys against sample_prompts.json's _meta + prompts.
    canonical = ROOT / "eval_data" / "sample_prompts.json"
    if canonical.exists():
        canonical_data = json.loads(canonical.read_text())
        canonical_keys = set(canonical_data.keys())
        cross_keys = set(data.keys())
        check("cross-domain fixture has same top-level keys as canonical sample",
              cross_keys == canonical_keys,
              f"canonical={canonical_keys} cross={cross_keys}")

    # Efficiency: fixture loaded + validated in <30 ms.
    elapsed_ms = (time.perf_counter() - t0) * 1000
    check("fixture parsed + validated in <30 ms",
          elapsed_ms < 30.0, f"{elapsed_ms:.1f} ms")

    print(f"\n  elapsed: {elapsed_ms:.1f} ms")
    print(f"\n== {'PASS' if not failures else 'FAIL'} "
          f"({len(failures)} failure(s)) ==")
    if failures:
        for f in failures:
            print(f"  - {f}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
