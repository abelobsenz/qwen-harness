#!/usr/bin/env python3
"""Adversarial stress test for loop_guard.

Goes beyond the basic self-tests in loop_guard.py:

  - Code-like outputs that LOOK repetitive but are correct (must NOT trigger)
  - Markdown tables, list structures (must NOT trigger)
  - Mathematical sequences and step-by-step proofs (must NOT trigger)
  - Long-form prose with quoted repetition ("the thing said 'foo' three times")
  - Sticky-but-not-identical loops (paraphrase drift)
  - Deeply nested structural loops
  - Tool-call-style XML repeating with different parameter values
  - Streaming detection: how quickly does the guard catch various loop types?
  - Memory/perf: ensure constant memory under arbitrary stream length

Exits 0 only if every test classification is correct (no false positives,
no missed loops). Each scenario has a reason for its expected verdict so
threshold tuning has clear targets.
"""

from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from loop_guard import (
    LoopGuardConfig,
    StreamingLoopGuard,
    check_text,
)


_CASES: list[tuple[str, str, bool, str]] = [
    # (label, text, should_trigger, rationale)

    # ============ CLEAN: must NOT trigger ============
    (
        "long varied prose",
        ("The discovery of penicillin by Alexander Fleming in 1928 marked a "
         "turning point in modern medicine. Before antibiotics, even minor "
         "infections could be fatal. Fleming noticed that a mold growing in "
         "a Petri dish had killed the surrounding bacteria. The active "
         "compound was later isolated and mass-produced during World War II, "
         "saving countless lives. Today, antibiotic resistance presents a "
         "new challenge as overuse has selected for drug-resistant strains. "
         "Researchers are exploring phage therapy, antimicrobial peptides, "
         "and combination therapies to address this growing problem.") * 1,
        False,
        "varied factual prose, ~700 chars, no repeats",
    ),
    (
        "python module of varied functions",
        '''
import json
import os
import sys
from dataclasses import dataclass


@dataclass
class Config:
    host: str
    port: int
    timeout: float = 30.0


def load_config(path: str) -> Config:
    with open(path) as f:
        data = json.load(f)
    return Config(**data)


def save_config(cfg: Config, path: str) -> None:
    with open(path, "w") as f:
        json.dump(cfg.__dict__, f, indent=2)


def merge_configs(a: Config, b: Config) -> Config:
    return Config(
        host=b.host or a.host,
        port=b.port or a.port,
        timeout=b.timeout if b.timeout != 30.0 else a.timeout,
    )
''',
        False,
        "Python module with similar imports/decorators but distinct functions",
    ),
    (
        "markdown table",
        """
| Mathematician | Contribution | Year |
|---------------|--------------|------|
| Frey          | Frey curve   | 1985 |
| Ribet         | Epsilon thm  | 1986 |
| Wiles         | Modularity   | 1995 |
| Taylor        | Wiles patch  | 1995 |
| Diamond       | Extension    | 2001 |
""",
        False,
        "markdown table with consistent pipe structure but distinct content",
    ),
    (
        "numbered list of 60 items",
        "\n".join(f"{i}. Item number {i} has properties X{i}, Y{i*2}, Z{i*3}"
                  for i in range(1, 60)),
        False,
        "numbered list — structurally similar lines but content varies",
    ),
    (
        "math step-by-step proof",
        """
Theorem: For all positive integers n, the sum 1 + 2 + ... + n = n(n+1)/2.

Proof by induction.

Base case: n = 1. The sum is just 1, and n(n+1)/2 = 1·2/2 = 1. So the base case holds.

Inductive step: Assume the formula holds for some k ≥ 1, i.e., 1 + 2 + ... + k = k(k+1)/2.

We want to show it holds for k+1, i.e., that 1 + 2 + ... + k + (k+1) = (k+1)(k+2)/2.

Starting from the inductive hypothesis: 1 + 2 + ... + k = k(k+1)/2.

Adding (k+1) to both sides: 1 + 2 + ... + k + (k+1) = k(k+1)/2 + (k+1) = (k+1)(k/2 + 1) = (k+1)(k+2)/2.

This matches what we wanted to prove, so by induction the formula holds for all n ≥ 1. QED.
""",
        False,
        "rigorous math proof — looks 'formulaic' but content is genuinely varied",
    ),
    (
        "long JSON config",
        '{' + ', '.join(f'"key_{i}": {{"value": {i*7}, "label": "item {i}", "tags": ["t{i}", "t{i+1}"]}}' for i in range(40)) + '}',
        False,
        "valid JSON config with consistent shape but distinct keys/values",
    ),
    (
        "single legitimate tool_call",
        ("<think>Let me check the file first.</think>\n"
         "<tool_call><function=read_file>"
         "<parameter=path>/tmp/foo.txt</parameter>"
         "<parameter=offset>0</parameter>"
         "<parameter=limit>500</parameter>"
         "</function></tool_call>"
         "Now I'll analyze the contents and respond."),
        False,
        "exactly ONE tool_call — the format must not collide with the detector",
    ),
    (
        "many DIFFERENT tool_calls in one response",
        "\n".join(
            f"<tool_call><function=read_file>"
            f"<parameter=path>/src/file_{i}.py</parameter>"
            f"<parameter=offset>{i*100}</parameter>"
            f"</function></tool_call>"
            for i in range(10)
        ),
        False,
        "agent legitimately fanning out 10 distinct read_file calls — must not trigger",
    ),
    (
        "long markdown with frequent code blocks",
        "\n\n".join(
            f"## Section {i}\n\nHere is the code for step {i}:\n\n```python\n"
            f"def step_{i}(x): return x * {i}\n```\n\nThis demonstrates concept {i}."
            for i in range(15)
        ),
        False,
        "long markdown with code fences — has structural repetition but per-section content varies",
    ),

    # ============ LOOPS: should trigger ============
    (
        "exact 4× sentence loop (the user-reported bug)",
        ("I will use make_table now. Then the Mermaid code. I will not use "
         "any other tools. ") * 5,
        True,
        "EXACTLY the failure mode the user reported",
    ),
    (
        "code stuck in a function loop",
        "def foo(): return 1\n" * 30,
        True,
        "model emitted the same line 30× — clear collapse",
    ),
    (
        "5-token cycle 50× (extreme greedy collapse)",
        "alpha beta gamma delta epsilon " * 50,
        True,
        "tightest possible repeat — must catch instantly",
    ),
    (
        "paraphrase drift (near-loop)",
        ("The model says yes. The model said yes. The model is saying yes. "
         "The model thinks yes. The model agrees yes. ") * 6,
        True,
        "vocabulary recycling without identical bytes — churn detector territory",
    ),
    (
        "nested XML tool_call loop",
        ("<tool_call><function=foo><parameter=x>1</parameter></function>"
         "</tool_call>") * 15,
        True,
        "model spamming the same tool call — should abort",
    ),
    (
        "long phrase repeats with stutter",
        ("So the answer is essentially that the result depends on the "
         "context. So the answer is essentially that the result depends on "
         "the context. ") * 5,
        True,
        "long literal repeat — easy suffix detection",
    ),
    (
        "loop with growing tail (still loops in window)",
        "Hello world. " * 80,
        True,
        "trivially repetitive, should fire",
    ),
]


def run_all() -> int:
    failures: list[str] = []
    print("== Adversarial loop_guard tests ==\n")
    for label, text, should_trigger, rationale in _CASES:
        rep = check_text(text)
        ok = rep.triggered == should_trigger
        verdict = "TRIGGER" if rep.triggered else "clean"
        marker = "✓" if ok else "✗"
        print(f"  [{marker}] {label}")
        print(f"        len={len(text)} verdict={verdict} expected="
              f"{'TRIGGER' if should_trigger else 'clean'}")
        print(f"        rationale: {rationale}")
        if rep.triggered:
            print(f"        detail: {rep.reason} ({rep.detail})")
        if not ok:
            failures.append(label)
        print()

    # Streaming-mode latency test: how many chars does the guard need to
    # see before it triggers on each loop type? Smaller is better.
    print("\n== Streaming detection latency ==")
    for label, text, should_trigger, _ in _CASES:
        if not should_trigger:
            continue
        guard = StreamingLoopGuard()
        triggered_at = -1
        for i in range(0, len(text), 32):
            rep = guard.observe(text[i : i + 32])
            if rep.triggered:
                triggered_at = i + 32
                break
        if triggered_at < 0:
            failures.append(f"streaming-miss:{label}")
            print(f"  [✗] {label}: did NOT trigger in stream")
        else:
            pct = 100 * triggered_at / max(len(text), 1)
            print(f"  [✓] {label}: triggered after {triggered_at} chars "
                  f"({pct:.0f}% of full text)")

    # Memory bound: feed an arbitrarily long stream and ensure the guard's
    # internal buffer stays bounded.
    print("\n== Memory bound check (constant memory under long stream) ==")
    guard = StreamingLoopGuard()
    chunk = "varied non-repeating words gallop joyfully across paragraphs " * 4
    for _ in range(10000):
        guard.observe(chunk)
    sizes = sum(len(p) for p in guard._buf)  # pylint: disable=protected-access
    if sizes > guard._max_window * 4:  # pylint: disable=protected-access
        print(f"  [✗] memory unbounded: buffer={sizes} chars, max_window={guard._max_window}")
        failures.append("memory-bound")
    else:
        print(f"  [✓] memory bounded: buffer={sizes} chars (max_window="
              f"{guard._max_window})")

    # Performance: how slow is repeated check_text on a long buffer?
    print("\n== Performance: 100 calls × 600-char window ==")
    long_text = "Lorem ipsum dolor sit amet, consectetur adipiscing elit. " * 12
    t0 = time.perf_counter()
    for _ in range(100):
        check_text(long_text)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    print(f"  100 calls took {elapsed_ms:.2f} ms ({elapsed_ms/100:.3f} ms/call)")
    if elapsed_ms > 500:  # 5 ms/call is way too slow
        print("  [✗] performance regression")
        failures.append("performance")
    else:
        print(f"  [✓] performance acceptable")

    print(f"\n== {'PASS' if not failures else 'FAIL'} "
          f"({len(failures)} failure(s)) ==")
    if failures:
        for f in failures:
            print(f"  - {f}")
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(run_all())
