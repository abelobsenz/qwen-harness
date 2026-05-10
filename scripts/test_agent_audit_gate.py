#!/usr/bin/env python3
"""Tests for the pre-done() self-audit gate in agent.py.

Why: the gate is a single-shot reliability fix that converts plausible-but-
wrong first answers (period / units / scope mismatches) into corrected
answers within the SAME query. Test that:
  1. Quantitative questions: first done() defers to an audit turn; the audit
     message is appended; step() returns False (continue).
  2. Quantitative questions: after audit, the second done() terminates
     normally (step() returns True).
  3. Qualitative / non-numeric questions: done() terminates immediately
     without firing the audit gate.
  4. Disabled with QWEN_AUDIT_GATE=0 → behaviour matches pre-gate flow.
  5. The quantitative-cue heuristic catches the common signals.
"""
from __future__ import annotations

import io
import json
import os
import sys
import tempfile
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import agent  # noqa: E402
import agent_tools  # noqa: E402


def _reset(workdir: str) -> None:
    """Clean global state between sub-tests so each one starts fresh."""
    agent_tools._session_writes_path = (
        f"/tmp/qwen_session_writes_test_{os.getpid()}.json"
    )
    agent_tools._done_sentinel_path = (
        f"/tmp/qwen_session_done_test_{os.getpid()}.txt"
    )
    for p in (agent_tools._session_writes_path, agent_tools._done_sentinel_path):
        try:
            os.unlink(p)
        except OSError:
            pass
    agent._audit_fired = False
    agent._loop_guard_nudge_fired = False
    agent._consecutive_all_cached_turns = 0


def _make_done_response(summary: str = "Wrote artifact.") -> dict:
    return {
        "choices": [{
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [{
                    "id": "call_done",
                    "type": "function",
                    "function": {
                        "name": "done",
                        "arguments": json.dumps({"summary": summary}),
                    },
                }],
            },
            "finish_reason": "tool_calls",
        }],
    }


def main() -> int:
    failures = 0
    with tempfile.TemporaryDirectory() as td:
        # ---- 1. Quantitative question: first done defers to audit ----
        print("[1] Quantitative question — first done() fires audit gate")
        _reset(td)
        agent_tools._track_write("/test_quant_artifact.md")
        messages = [
            {"role": "system", "content": agent.SYSTEM_PROMPT},
            {"role": "user", "content": "Calculate the % change in Q4 2024 vs Q4 2023 share repurchases."},
        ]
        fake = _make_done_response("First-pass answer with -32%.")
        buf = io.StringIO()
        with patch.object(agent, "post_chat", return_value=fake), \
             redirect_stdout(buf):
            ret = agent.step(messages, step_num=1)
        out = buf.getvalue()
        if ret is True:
            print(f"    [FAIL] step() returned True; expected False (audit defers done)")
            failures += 1
        elif "audit gate: deferring done" not in out:
            print(f"    [FAIL] audit-gate stdout marker missing")
            print(f"        stdout: {out[-300:]!r}")
            failures += 1
        elif not any("[HARNESS]" in (m.get("content") or "")
                     and "self-audit" in (m.get("content") or "")
                     for m in messages):
            print(f"    [FAIL] audit message not appended to messages")
            failures += 1
        elif agent._audit_fired is not True:
            print(f"    [FAIL] _audit_fired latch not set")
            failures += 1
        else:
            print(f"    [OK] audit deferred; gate set; message appended")

        # ---- 2. Second done after audit terminates normally ----
        print("[2] Second done() after audit terminates the session")
        # Don't reset — keep _audit_fired = True from sub-test 1.
        fake2 = _make_done_response("Audited answer: -78.72% (Q4 monthly tables).")
        buf = io.StringIO()
        with patch.object(agent, "post_chat", return_value=fake2), \
             redirect_stdout(buf):
            ret = agent.step(messages, step_num=2)
        out = buf.getvalue()
        if ret is not True:
            print(f"    [FAIL] step() returned {ret!r}; expected True after audit")
            failures += 1
        elif "done signal received" not in out:
            print(f"    [FAIL] missing 'done signal received' in stdout")
            failures += 1
        else:
            print(f"    [OK] post-audit done() terminated normally")

        # ---- 3. Qualitative question: no audit gate ----
        print("[3] Qualitative question — done() terminates without audit")
        _reset(td)
        agent_tools._track_write("/test_qual_artifact.md")
        messages = [
            {"role": "system", "content": agent.SYSTEM_PROMPT},
            {"role": "user", "content": "Hello, how are you doing today?"},
        ]
        fake = _make_done_response("Said hello back.")
        buf = io.StringIO()
        with patch.object(agent, "post_chat", return_value=fake), \
             redirect_stdout(buf):
            ret = agent.step(messages, step_num=1)
        out = buf.getvalue()
        if ret is not True:
            print(f"    [FAIL] qualitative done deferred unexpectedly (audit fired)")
            failures += 1
        elif "audit gate" in out:
            print(f"    [FAIL] audit gate fired on a qualitative question")
            failures += 1
        elif agent._audit_fired is True:
            print(f"    [FAIL] _audit_fired set on qualitative question")
            failures += 1
        else:
            print(f"    [OK] qualitative path unchanged")

        # ---- 4. Disabled gate → behaves like pre-gate flow ----
        print("[4] QWEN_AUDIT_GATE=0 disables gate even on quantitative questions")
        _reset(td)
        agent_tools._track_write("/test_disabled_artifact.md")
        original_flag = agent._AUDIT_GATE_ENABLED
        agent._AUDIT_GATE_ENABLED = False
        try:
            messages = [
                {"role": "system", "content": agent.SYSTEM_PROMPT},
                {"role": "user", "content": "Calculate the Q4 2024 revenue growth."},
            ]
            fake = _make_done_response("Quant answer.")
            buf = io.StringIO()
            with patch.object(agent, "post_chat", return_value=fake), \
                 redirect_stdout(buf):
                ret = agent.step(messages, step_num=1)
            out = buf.getvalue()
            if ret is not True:
                print(f"    [FAIL] disabled gate still deferred done()")
                failures += 1
            elif "audit gate" in out:
                print(f"    [FAIL] disabled gate still printed audit-gate marker")
                failures += 1
            else:
                print(f"    [OK] flag-off restores immediate-terminate behaviour")
        finally:
            agent._AUDIT_GATE_ENABLED = original_flag

        # ---- 5. Quantitative-cue heuristic spot checks ----
        print("[5] _question_is_quantitative heuristic")
        cases = [
            ("Calculate Q4 2024 revenue growth", True),
            ("How much debt does Boeing have?", True),
            ("What was the EBITDA margin?", True),
            ("Compute the % change between fiscal 2023 and 2024", True),
            ("$1,234,567 — what does this mean?", True),
            ("Hello, what time is it?", False),
            ("Write a poem about clouds", False),
            ("List the steps in the function", False),
            # iter 27 refinement: a bare year alone shouldn't fire the
            # audit gate on a purely qualitative question. BBSI iter 23
            # was the false positive that motivated the fix.
            ("In 2024, who was Nominated to Serve on BBSI's Board of Directors?", False),
            ("Who became CEO in 2023?", False),
            # But quantitative + year still fires.
            ("Calculate the revenue growth in 2024", True),
        ]
        for question, expected in cases:
            msgs = [{"role": "user", "content": question}]
            got = agent._question_is_quantitative(msgs)
            tag = "[OK]" if got == expected else "[FAIL]"
            if got != expected:
                failures += 1
            print(f"    {tag} {question!r} → got={got} expected={expected}")

        # ---- 6. Audit gate is single-fire per query ----
        print("[6] Audit gate fires only ONCE per query")
        _reset(td)
        agent_tools._track_write("/test_singlefire.md")
        messages = [
            {"role": "system", "content": agent.SYSTEM_PROMPT},
            {"role": "user", "content": "Calculate the EBITDA margin for Q3 2024."},
        ]
        fake = _make_done_response("First done.")
        with patch.object(agent, "post_chat", return_value=fake), \
             redirect_stdout(io.StringIO()):
            agent.step(messages, step_num=1)  # first done → audit fires
            ret_second = agent.step(messages, step_num=2)  # second done → terminate
            ret_third = agent.step(messages, step_num=3)  # would-be third → still terminate
        if ret_second is True and ret_third is True:
            print(f"    [OK] gate single-fired; subsequent dones terminate normally")
        else:
            print(f"    [FAIL] expected ret_second=True, ret_third=True; got "
                  f"{ret_second!r} {ret_third!r}")
            failures += 1

    if failures:
        print(f"\n== FAIL ({failures} failure(s)) ==")
        return 1
    print("\n== PASS (audit gate behaves correctly) ==")
    return 0


if __name__ == "__main__":
    sys.exit(main())
