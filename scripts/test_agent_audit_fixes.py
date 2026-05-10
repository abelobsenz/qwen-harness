#!/usr/bin/env python3
"""Tests for the audit-driven fixes (issues A, F).

Issue A — nudge prioritization: at most ONE nudge fires per turn so the
model never gets contradictory `[HARNESS]` messages. Tool-result-level
signals (cap-exhaustion / truncation / cache-loop) take priority over
step-count signals (read-loop / stale).

Issue F — read-loop nudge defers for investigation tasks: the standard
4-step nudge derails tasks like "audit X" or "summarize Y" where
reading IS the work. We now check `_task_expects_artifact` and switch
to a much higher threshold (16) when no artifact verb / path is present.
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))


def main() -> int:
    failures = 0

    import agent  # noqa: E402

    # ---- 1. _task_expects_artifact heuristic ----
    print("[1] _task_expects_artifact: artifact verb / named path / neither")
    cases = [
        ("Write a python script to /tmp/foo.py that prints hello", True),
        ("Create a markdown report on the bug", True),
        ("Build a CSV summary", True),
        ("Fix the bug in agent_tools.py", True),
        ("Save the answer to /tmp/answer.txt", True),
        ("Audit the codebase for security issues", False),
        ("Summarize the most recent arxiv paper on LLM agents", False),
        ("What's in /Users/abelobsenz/projects/foo?", False),  # path is input, not output
        ("How does the loop guard work?", False),
        ("Explore scripts/ and tell me what each file does", False),
    ]
    for question, expected in cases:
        msgs = [{"role": "user", "content": question}]
        got = agent._task_expects_artifact(msgs)
        tag = "[OK]" if got == expected else "[FAIL]"
        if got != expected:
            failures += 1
        print(f"    {tag} {question!r} → got={got} expected={expected}")

    # ---- 2. Threshold selection ----
    print("[2] read-loop threshold respects task type")
    expects = agent._task_expects_artifact(
        [{"role": "user", "content": "Write a report to /tmp/x.md"}])
    inv = agent._task_expects_artifact(
        [{"role": "user", "content": "Audit the codebase"}])
    artifact_threshold = (agent._NUDGE_AFTER_STEPS if expects
                          else agent._NUDGE_AFTER_STEPS_INVESTIGATIVE)
    inv_threshold = (agent._NUDGE_AFTER_STEPS if inv
                     else agent._NUDGE_AFTER_STEPS_INVESTIGATIVE)
    if (artifact_threshold == agent._NUDGE_AFTER_STEPS
            and inv_threshold == agent._NUDGE_AFTER_STEPS_INVESTIGATIVE
            and inv_threshold > artifact_threshold):
        print(f"    [OK] artifact tasks fire at step {artifact_threshold}, "
              f"investigations at step {inv_threshold}")
    else:
        print(f"    [FAIL] thresholds reversed or equal: "
              f"artifact={artifact_threshold} investigative={inv_threshold}")
        failures += 1

    # ---- 3. Nudge prioritization: cap-exhaustion preempts cache-loop ----
    # The nudge code runs ONLY in the tool-calls branch (after dispatch),
    # so the test needs a tool_call response + a dispatch_batch mock that
    # returns results matching the targeted counter pattern. [REFUSED]
    # results increment both _consecutive_all_refused_turns AND
    # _consecutive_all_cached_turns (was_cached=True), so a single
    # all-REFUSED turn drives both counters past threshold simultaneously
    # — the priority pick is what decides which nudge actually emits.
    print("[3] Cap-exhaustion preempts cache-loop in same turn")
    import io
    from contextlib import redirect_stdout
    import json
    from unittest.mock import patch

    def _resp_with_tool_call(name="web_search", args=None):
        return {
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [{
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": name,
                            "arguments": json.dumps(args or {"query": "x"}),
                        },
                    }],
                },
                "finish_reason": "tool_calls",
            }],
        }

    # Pre-load the cache-loop counter so this turn pushes it past
    # threshold; cap-exhaustion threshold is 1 by default so a single
    # all-REFUSED turn pushes it over too. Both nudges would fire
    # without prioritization; we expect only cap-exhaustion.
    agent._consecutive_all_refused_turns = 0
    agent._consecutive_missing_arg_turns = 0
    agent._consecutive_all_cached_turns = agent._LOOP_BREAK_THRESHOLD - 1
    agent._read_loop_nudged = False
    agent._stale_post_write_nudged = False
    messages = [
        {"role": "system", "content": agent.SYSTEM_PROMPT},
        {"role": "user", "content": "Write a finance summary"},
    ]
    buf = io.StringIO()
    with patch.object(agent._CACHED, "dispatch_batch",
                      return_value=[("[REFUSED — cap reached]", True)]), \
         patch.object(agent, "post_chat", return_value=_resp_with_tool_call()), \
         redirect_stdout(buf):
        agent.step(messages, step_num=2)
    out = buf.getvalue()
    if ("cap-exhaustion" in out
            and "loop detected" not in out
            and "truncation detected" not in out):
        print("    [OK] cap-exhaustion fired; cache-loop and truncation suppressed")
    else:
        print(f"    [FAIL] expected cap-exhaustion only; markers: "
              f"cap={'cap-exhaustion' in out} cache={'loop detected' in out} "
              f"trunc={'truncation detected' in out}")
        failures += 1

    # ---- 4. With no high-priority signal, cache-loop still fires ----
    # Use cached-but-not-refused results so all_cached counter ticks but
    # all_refused stays 0.
    print("[4] Cache-loop fires alone when no higher priority signal")
    agent._consecutive_all_refused_turns = 0
    agent._consecutive_missing_arg_turns = 0
    agent._consecutive_all_cached_turns = agent._LOOP_BREAK_THRESHOLD - 1
    agent._read_loop_nudged = False
    agent._stale_post_write_nudged = False
    messages2 = [
        {"role": "system", "content": agent.SYSTEM_PROMPT},
        {"role": "user", "content": "Write a finance summary"},
    ]
    buf2 = io.StringIO()
    with patch.object(agent._CACHED, "dispatch_batch",
                      return_value=[("cached result body", True)]), \
         patch.object(agent, "post_chat", return_value=_resp_with_tool_call()), \
         redirect_stdout(buf2):
        agent.step(messages2, step_num=2)
    out2 = buf2.getvalue()
    if "loop detected" in out2 and "cap-exhaustion" not in out2:
        print("    [OK] cache-loop fired when alone")
    else:
        print(f"    [FAIL] cache-loop should have fired alone. stdout tail:\n"
              f"{out2[-400:]}")
        failures += 1

    # ---- 5. Investigation task: read-loop nudge does NOT fire at step 4 ----
    print("[5] Investigation task at step 4: no read-loop nudge yet")
    agent._consecutive_all_refused_turns = 0
    agent._consecutive_missing_arg_turns = 0
    agent._consecutive_all_cached_turns = 0
    agent._read_loop_nudged = False
    messages3 = [
        {"role": "system", "content": agent.SYSTEM_PROMPT},
        {"role": "user", "content": "Audit the codebase for security issues"},
    ]

    def _resp_with_one_tool_call():
        # Returns a fake tool_call that won't actually dispatch (we mock
        # _CACHED.dispatch_batch below).
        return {
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [{
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "list_files",
                            "arguments": json.dumps({"path": "."}),
                        },
                    }],
                },
                "finish_reason": "tool_calls",
            }],
        }

    buf3 = io.StringIO()
    # Mock the dispatcher to avoid actual tool calls (which would write).
    with patch.object(agent._CACHED, "dispatch_batch",
                      return_value=[("(no matches)", False)]), \
         patch.object(agent, "post_chat", return_value=_resp_with_one_tool_call()), \
         redirect_stdout(buf3):
        agent.step(messages3, step_num=4)
    out3 = buf3.getvalue()
    if "no writes after 4 steps" not in out3 and not agent._read_loop_nudged:
        print(f"    [OK] investigation task didn't trigger read-loop nudge at step 4")
    else:
        print(f"    [FAIL] investigation task fired the nudge anyway")
        print(f"    stdout: {out3[-300:]!r}")
        failures += 1

    if failures:
        print(f"\n== FAIL ({failures} failure(s)) ==")
        return 1
    print("\n== PASS (audit-issue fixes work) ==")
    return 0


if __name__ == "__main__":
    sys.exit(main())
