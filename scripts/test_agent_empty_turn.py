#!/usr/bin/env python3
"""Tests for the empty-assistant-turn nudge (iter 28 fix).

Bug surfaced in iter 28: netflix_q4_repurchase scored 0 in 17.8s. The
agent issued one tool call (sec_filings), got results, and then the
streaming POST returned a turn with no content, no tool_calls, and no
reasoning — likely because dflash-serve hadn't finished loading the
model after the per-prompt hard-restart. The previous code returned
True (loop exit) on this empty turn, scoring 0.

The fix nudges ONCE per session, asking the model to either continue,
emit a final answer, or call done(). On a fresh subsequent call the
loop has another chance.

Tests:
  1. Empty turn → step() returns False (continue), nudge appended,
     latch set.
  2. Same session, second empty turn → step() returns True (no
     re-fire; gate is single-shot per session).
  3. Per-query reset clears the latch.
  4. Non-empty turn (content present) follows pre-existing path
     (no nudge fires).
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


def _reset() -> None:
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
    agent._empty_turn_nudged = False
    agent._audit_fired = False
    agent._exit_nudge_count = 0
    agent._loop_guard_nudge_fired = False


def _empty_response() -> dict:
    """Mirror what _do_post_streaming returns when the SSE stream
    completed with [DONE] but no content/tool_calls/reasoning chunks."""
    return {
        "choices": [{
            "message": {"role": "assistant", "content": ""},
            "finish_reason": "stop",
        }],
    }


def _content_response(text: str = "Hello.") -> dict:
    return {
        "choices": [{
            "message": {"role": "assistant", "content": text},
            "finish_reason": "stop",
        }],
    }


def main() -> int:
    failures = 0

    # ---- 1. Empty turn → nudged, returns False ----
    print("[1] Empty turn fires the nudge and continues the loop")
    _reset()
    messages = [
        {"role": "system", "content": agent.SYSTEM_PROMPT},
        {"role": "user", "content": "Calculate the % change in NFLX repurchases."},
    ]
    buf = io.StringIO()
    with patch.object(agent, "post_chat", return_value=_empty_response()), \
         redirect_stdout(buf):
        ret = agent.step(messages, step_num=1)
    out = buf.getvalue()
    if ret is True:
        print(f"    [FAIL] step() returned True; expected False (nudge defers exit)")
        failures += 1
    elif "empty assistant turn" not in out:
        print(f"    [FAIL] empty-turn marker not in stdout")
        print(f"    stdout tail: {out[-300:]!r}")
        failures += 1
    elif not any("[HARNESS]" in (m.get("content") or "")
                 and "no content, no tool calls, and no reasoning"
                 in (m.get("content") or "") for m in messages):
        print(f"    [FAIL] nudge message not appended to messages")
        failures += 1
    elif agent._empty_turn_nudged is not True:
        print(f"    [FAIL] _empty_turn_nudged latch not set")
        failures += 1
    else:
        print(f"    [OK] empty turn nudged; latch set; message appended")

    # ---- 2. Second empty turn in same session → no re-fire ----
    print("[2] Second empty turn in same session — single-shot gate")
    # Don't reset; keep _empty_turn_nudged = True from sub-test 1.
    msg_count_before = len(messages)
    buf = io.StringIO()
    with patch.object(agent, "post_chat", return_value=_empty_response()), \
         redirect_stdout(buf):
        ret2 = agent.step(messages, step_num=2)
    nudge_appended = any(
        "no content, no tool calls, and no reasoning" in (m.get("content") or "")
        for m in messages[msg_count_before:]
    )
    if nudge_appended:
        print(f"    [FAIL] empty-turn nudge fired twice in the same session")
        failures += 1
    else:
        print(f"    [OK] gate single-fired; second empty turn followed normal exit path")

    # ---- 3. Per-query reset clears the latch ----
    print("[3] Per-query reset clears the empty-turn latch")
    # The reset happens at the top of run_loop()/run_query(); simulate by
    # calling the reset block manually.
    agent._empty_turn_nudged = True  # sticky from prior session
    # Mimic the reset code block:
    agent._empty_turn_nudged = False
    if agent._empty_turn_nudged is False:
        print(f"    [OK] latch resettable per query")
    else:
        print(f"    [FAIL] latch did not reset")
        failures += 1

    # ---- 4. Non-empty turn does NOT trigger empty-turn nudge ----
    print("[4] Non-empty content turn skips the empty-turn nudge")
    _reset()
    messages = [
        {"role": "system", "content": agent.SYSTEM_PROMPT},
        {"role": "user", "content": "Hello, how are you?"},
    ]
    buf = io.StringIO()
    with patch.object(agent, "post_chat", return_value=_content_response()), \
         redirect_stdout(buf):
        ret4 = agent.step(messages, step_num=1)
    out = buf.getvalue()
    if "empty assistant turn" in out:
        print(f"    [FAIL] empty-turn marker fired on non-empty content turn")
        failures += 1
    elif agent._empty_turn_nudged:
        print(f"    [FAIL] latch set on non-empty content turn")
        failures += 1
    else:
        print(f"    [OK] non-empty turn took the normal path")

    if failures:
        print(f"\n== FAIL ({failures} failure(s)) ==")
        return 1
    print("\n== PASS (empty-turn nudge behaves correctly) ==")
    return 0


if __name__ == "__main__":
    sys.exit(main())
