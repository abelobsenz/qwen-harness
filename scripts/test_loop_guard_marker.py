#!/usr/bin/env python3
"""Direct tests for `scripts/loop_guard_marker.py`.

Round 15 fixed a substring-only false-positive in agent.py. Round 16
extracted the detector into a shared module so agent_graph.py could
reuse it without re-introducing the bug. This test isolates the
helpers from any caller so future refactors of agent.py / agent_graph.py
can't silently break the detection contract.

Cases:
  - Real proxy abort markers (both non-stream and streaming forms) → True
  - Benign mentions in user-facing prose, code review, grep output → False
  - Suffix without substring (regex edge case) → False
  - Substring without suffix → False
  - Empty / None content → False
  - Reason extraction preserves hyphenated reason names
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main() -> int:
    from loop_guard_marker import (
        is_proxy_abort_marker,
        extract_reason,
        harness_nudge_message,
    )

    failures: list[str] = []
    print("== loop_guard_marker direct tests ==\n")

    # is_proxy_abort_marker — positive cases
    print("[1] Real abort markers — True")
    real_cases = [
        ("non-stream", "[loop-guard: low-churn (distinct/total=159/595 ratio=0.27 top-gram×14) — output stopped early]"),
        ("streaming",  "[loop-guard: aborted (suffix-extreme) — the model fell into a repetition loop. Try rephrasing or asking a more specific question.]"),
        ("with prefix prose", "I will now answer.\n\n[loop-guard: combined — output stopped early]"),
        ("mid-text suffix",   "Some content. [loop-guard: suffix-dominant (…)] then more. … fell into a repetition loop."),
    ]
    for label, txt in real_cases:
        ok = is_proxy_abort_marker(txt)
        marker = "✓" if ok else "✗"
        print(f"    [{marker}] {label}: got={ok} expected=True")
        if not ok:
            failures.append(f"real:{label}")

    # is_proxy_abort_marker — negative cases
    print("\n[2] Benign mentions — False")
    benign_cases = [
        ("user-explained",
         "The proxy emits a [loop-guard: <reason>] marker when it detects a loop."),
        ("code-review",
         "Looking at qwen_proxy.py:487, [loop-guard: ...] is logged."),
        ("grep-echo",
         "scripts/agent.py:308: _LOOP_GUARD_RE = re.compile(r'\\[loop-guard:...')"),
        ("substring no suffix",
         "We log [loop-guard: <reason>] when needed."),
        ("suffix no substring",
         "The output stopped early because the user pressed escape."),
        ("no substring at all",
         "Hello world."),
        ("empty string", ""),
    ]
    for label, txt in benign_cases:
        ok = is_proxy_abort_marker(txt) is False
        marker = "✓" if ok else "✗"
        print(f"    [{marker}] {label}: got_false={ok}")
        if not ok:
            failures.append(f"benign:{label}")

    # is_proxy_abort_marker — None handling (defensive)
    print("\n[3] Defensive: None content → False")
    try:
        ok = is_proxy_abort_marker(None) is False  # type: ignore[arg-type]
        print(f"    [{'✓' if ok else '✗'}] None → False")
        if not ok:
            failures.append("none-content")
    except Exception as e:  # noqa: BLE001
        print(f"    [✗] None raised {type(e).__name__}: {e}")
        failures.append("none-raises")

    # extract_reason — preserves hyphenated reason names
    print("\n[4] extract_reason preserves hyphenated reason names")
    reason_cases = [
        ("low-churn", "[loop-guard: low-churn — output stopped early]"),
        ("suffix-dominant", "[loop-guard: suffix-dominant — output stopped early]"),
        ("suffix-extreme", "[loop-guard: suffix-extreme — output stopped early]"),
        ("combined",   "[loop-guard: combined — output stopped early]"),
    ]
    for expected_reason, txt in reason_cases:
        got = extract_reason(txt)
        # The captured group may include trailing detail in parens; the
        # REASON KEYWORD must be present at the start.
        ok = got.startswith(expected_reason)
        marker = "✓" if ok else "✗"
        print(f"    [{marker}] {expected_reason}: got={got!r}")
        if not ok:
            failures.append(f"reason:{expected_reason}")

    # extract_reason — defensive fallback
    print("\n[5] extract_reason fallback when regex doesn't match")
    out = extract_reason("hello world")
    if out == "repetition loop":
        print(f"    [✓] no-match → fallback {out!r}")
    else:
        print(f"    [✗] no-match → {out!r} (expected 'repetition loop')")
        failures.append("reason-fallback")

    # harness_nudge_message — shape + content invariants
    print("\n[6] harness_nudge_message structure")
    msg = harness_nudge_message("low-churn (top-gram×14)")
    if msg.get("role") != "user":
        print(f"    [✗] role={msg.get('role')!r} expected 'user'")
        failures.append("nudge-role")
    elif "[HARNESS]" not in (msg.get("content") or ""):
        print(f"    [✗] content missing [HARNESS] marker")
        failures.append("nudge-marker")
    elif "low-churn" not in (msg.get("content") or ""):
        print(f"    [✗] content missing reason text")
        failures.append("nudge-reason")
    elif "different angle" not in msg["content"]:
        print(f"    [✗] content missing course-correction phrasing")
        failures.append("nudge-correction")
    else:
        print(f"    [✓] role=user, [HARNESS], reason, and correction all present")

    print(f"\n== {'PASS' if not failures else 'FAIL'} "
          f"({len(failures)} failure(s)) ==")
    if failures:
        for f in failures:
            print(f"  - {f}")
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
