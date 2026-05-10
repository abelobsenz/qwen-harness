#!/usr/bin/env python3
"""Small quality eval — math, reasoning, factual.

15 deterministic questions, auto-graded. Prints per-question score and
totals. Same script runs against any OpenAI-compatible endpoint, so we
can compare quants side-by-side.

Usage:
    python scripts/quality_eval.py
"""

from __future__ import annotations

import json
import os
import re
import time
import urllib.request

HOST = os.environ.get("QWEN_HOST", "127.0.0.1")
if HOST in ("0.0.0.0", ""):
    HOST = "127.0.0.1"
PORT = os.environ.get("QWEN_PORT", "8000")
MODEL = os.environ.get("QWEN_MODEL_NAME", "qwen3.6")
URL = f"http://{HOST}:{PORT}/v1/chat/completions"


def ask(question: str, max_tokens: int = 4096) -> tuple[str, float]:
    body = json.dumps(
        {
            "model": MODEL,
            "messages": [
                {"role": "system", "content": "Answer concisely. For math, give the numerical answer at the end. For factual questions, give a short direct answer."},
                {"role": "user", "content": question},
            ],
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "stream": False,
        }
    ).encode("utf-8")
    req = urllib.request.Request(
        URL, data=body, headers={"Content-Type": "application/json"}, method="POST"
    )
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=600) as resp:
        data = json.loads(resp.read())
    dt = time.time() - t0
    content = data.get("choices", [{}])[0].get("message", {}).get("content", "") or ""
    return content.strip(), dt


def all_numbers(text: str) -> list[float]:
    nums = re.findall(r"-?\d+\.?\d*", text.replace(",", ""))
    out = []
    for s in nums:
        try:
            out.append(float(s))
        except ValueError:
            pass
    return out


def grade_numeric(answer: str, expected: float, tol: float = 0.01) -> bool:
    """Pass if expected appears anywhere in the response (within tolerance).
    Robust against truncation of thinking models — the right number usually
    appears even if the final word doesn't."""
    for n in all_numbers(answer):
        if abs(n - expected) <= tol:
            return True
    return False


def grade_numeric_with_tol(answer: str, expected: float, tol: float) -> bool:
    return grade_numeric(answer, expected, tol)


def grade_substring(answer: str, expected: str | list[str]) -> bool:
    a = answer.lower()
    if isinstance(expected, str):
        expected = [expected]
    return any(s.lower() in a for s in expected)


# HARDEST eval — questions sourced from real elite benchmarks. Each
# requires multi-step reasoning + actual computation; can't be solved by
# pattern-matching memorized text. No tools available — pure inference.
# Tuple format: (id, kind, question, expected, grader, tol)
QUESTIONS: list[tuple[str, str, str, object, str, float | None]] = [
    # ---- Math: AIME / Putnam-grade ----
    (
        "M1", "math",
        # AIME 2008 II #9 — geometry + complex-number sequences
        "A particle starts at position (5, 0) on the coordinate plane. Each move "
        "consists of, in order: (1) a counterclockwise rotation by pi/4 radians about "
        "the origin, then (2) a translation by 10 units in the positive x direction. "
        "After exactly 150 moves the particle is at (p, q). Find the greatest integer "
        "less than or equal to |p| + |q|. Answer with just the integer.",
        19, "numeric", 0.01,
    ),
    (
        "M2", "math",
        # Combinatorics with inclusion-exclusion
        "How many subsets of {1, 2, 3, 4, 5, 6, 7, 8, 9, 10} contain at least one prime "
        "number AND at least one composite number? (Note: the prime numbers in this set "
        "are {2, 3, 5, 7}; the composite numbers are {4, 6, 8, 9, 10}; and 1 is neither "
        "prime nor composite.) Answer with just the integer.",
        930, "numeric", 0.01,
    ),
    # ---- Finance: stochastic calculus + bond convexity ----
    (
        "F1", "finance",
        # Itō's lemma applied to GBM — needs to know d(ln S) drift correction
        "A stock follows geometric Brownian motion: dS = mu*S dt + sigma*S dW, with "
        "S(0) = 100, drift mu = 0.10 per year, and volatility sigma = 0.20 per year. "
        "Using Ito's lemma, what is the expected value E[ln(S(1))] of the natural "
        "logarithm of the stock price after T = 1 year? Round to 4 decimal places. "
        "Answer with just the number.",
        4.6852, "numeric", 0.005,  # accepts 4.6802 to 4.6902
    ),
    (
        "F2", "finance",
        # Duration + convexity approximation
        "A bond has modified duration of 7.62 years and convexity of 95.0 years^2. "
        "Yield to maturity is 5%. If yields rise by 100 basis points (Delta_y = 0.01), "
        "the approximate percentage change in the bond's price is given by "
        "  Delta_P / P = -ModDur * Delta_y + (1/2) * Convexity * (Delta_y)^2. "
        "What is this percentage change, expressed as a number to two decimal places "
        "(e.g., -3.50 means -3.50%)? Answer with just the number.",
        -7.14, "numeric", 0.10,  # accepts -7.24 to -7.04
    ),
    # ---- SWE: information theory + cache-oblivious complexity ----
    (
        "S1", "swe",
        # Comparison-sort lower bound
        "What is the optimal worst-case number of pairwise comparisons required to sort "
        "exactly 5 distinct elements? (The information-theoretic lower bound is "
        "ceiling(log_2(5!)) and this bound is achievable for n=5.) Answer with just the integer.",
        7, "numeric", 0.01,
    ),
    (
        "S2", "swe",
        # Cache-oblivious matrix multiplication I/O complexity
        "In the cache-oblivious external-memory model with cache size M and block size B, "
        "the optimal algorithm for n x n matrix multiplication has asymptotic I/O "
        "complexity of the form Theta(n^a / (B * M^c)) for some non-negative real "
        "numbers a and c. What is the value of a + c (where c may be a fraction such "
        "as 0.5)? Answer with just the number.",
        3.5, "numeric", 0.01,
    ),
]


def main() -> None:
    print(f"Quality eval against {URL}")
    print(f"Model: {MODEL}")
    print(f"Questions: {len(QUESTIONS)}\n")
    print(f"{'ID':<5}{'kind':<11}{'pass':<7}{'time':<8}question")
    print("-" * 95)

    by_kind: dict[str, list[bool]] = {}
    total_time = 0.0

    for entry in QUESTIONS:
        if len(entry) == 6:
            qid, kind, q, expected, grader, tol = entry
        else:
            qid, kind, q, expected, grader = entry
            tol = 0.01
        try:
            answer, dt = ask(q)
        except Exception as e:  # noqa: BLE001
            print(f"{qid:<5}{kind:<11}ERR    -       {q[:50]}")
            print(f"     error: {e}")
            by_kind.setdefault(kind, []).append(False)
            continue

        if grader == "numeric":
            ok = grade_numeric(answer, float(expected), tol or 0.01)
        else:
            ok = grade_substring(answer, expected)

        by_kind.setdefault(kind, []).append(ok)
        total_time += dt
        marker = "✓" if ok else "✗"
        print(f"{qid:<5}{kind:<11}{marker:<7}{dt:>5.1f}s  {q[:60]}")
        if not ok:
            tail = answer.replace("\n", " ")[-150:]
            print(f"     got: …{tail}")

    print("\n" + "=" * 60)
    grand_total = 0
    grand_count = 0
    for kind, results in sorted(by_kind.items()):
        passed = sum(results)
        total = len(results)
        grand_total += passed
        grand_count += total
        print(f"  {kind:<10} {passed}/{total}  ({passed / total:.0%})")
    print(f"  {'TOTAL':<10} {grand_total}/{grand_count}  ({grand_total / grand_count:.0%})")
    print(f"  wall time: {total_time:.1f}s")


if __name__ == "__main__":
    main()
