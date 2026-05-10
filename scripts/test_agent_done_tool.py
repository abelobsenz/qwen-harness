#!/usr/bin/env python3
"""Adversarial unit test for the agent's `done(summary)` tool.

The `done` tool gates session completion: it refuses to accept "task done"
unless an artifact write was tracked this session, and on accept it drops a
sentinel file that agent.py's main loop watches. Both halves matter — a
false-accept means the agent can give up on a hard prompt; a false-refuse
means a finished session loops forever burning LLM rounds.

This test sweeps the `done` tool against:

  1. summary too short / empty / whitespace            → [error]
  2. no writes recorded                                → [refused]
  3. valid call after write                            → DONE accepted + sentinel
  4. refused → write → retry succeeds                  → recovers correctly
  5. sentinel content matches summary verbatim
  6. unicode + very long summary survive round-trip
  7. session-writes file corrupt (malformed JSON)      → treated as no writes
  8. multiple done calls: each accepts and overwrites the sentinel
  9. sentinel-write failure (read-only path)           → [error]
 10. agent.py recognizes "DONE accepted" → done_signaled = True
 11. agent.py does NOT mark done_signaled on a refused result
 12. simulated "loop-aborted then done": done after a loop-guard nudge
"""

from __future__ import annotations

import io
import json
import os
import sys
import tempfile
from contextlib import redirect_stdout
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def _reset(at):
    """Point both sentinel paths at fresh temp files and clear them."""
    import agent_tools
    agent_tools._session_writes_path = os.path.join(at, "writes.json")
    agent_tools._done_sentinel_path = os.path.join(at, "done.txt")
    for p in (agent_tools._session_writes_path,
              agent_tools._done_sentinel_path):
        try:
            os.unlink(p)
        except OSError:
            pass


def main() -> int:
    failures = 0
    print("== Agent done() tool adversarial test ==\n")

    import agent_tools

    with tempfile.TemporaryDirectory() as td:
        # -------- 1. summary too short / empty / whitespace ----------------
        print("[1] short / empty / whitespace summary → [error]")
        _reset(td)
        for bad in ("", "   ", "abc", "    \n  "):
            r = agent_tools.done(bad)
            if not r.startswith("[error]"):
                print(f"    [✗] summary={bad!r}: expected [error], got {r!r}")
                failures += 1
        if not failures:
            print("    [✓] all four short/empty cases rejected")

        # -------- 2. no writes recorded → [refused] ------------------------
        print("\n[2] no writes recorded → [refused]")
        _reset(td)
        r = agent_tools.done("Wrote nothing yet but claiming done.")
        if not r.startswith("[refused]"):
            print(f"    [✗] expected [refused], got {r!r}")
            failures += 1
        elif os.path.exists(agent_tools._done_sentinel_path):
            print("    [✗] [refused] but sentinel was still written!")
            failures += 1
        else:
            print(f"    [✓] refused with no sentinel side-effect")

        # -------- 3. valid call after write → DONE accepted ----------------
        print("\n[3] valid summary + writes recorded → DONE accepted")
        _reset(td)
        agent_tools._track_write("/some/artifact.md")
        r = agent_tools.done("Wrote /some/artifact.md with the summary.")
        if not r.startswith("DONE accepted"):
            print(f"    [✗] expected 'DONE accepted', got {r!r}")
            failures += 1
        elif not os.path.exists(agent_tools._done_sentinel_path):
            print("    [✗] sentinel file not created on accept")
            failures += 1
        else:
            print(f"    [✓] sentinel created; result: {r[:60]}...")

        # -------- 4. refused → write → retry recovers ----------------------
        print("\n[4] refused → write_file → retry succeeds")
        _reset(td)
        r1 = agent_tools.done("Premature claim.")
        agent_tools._track_write("/late/artifact.md")
        r2 = agent_tools.done("Now we have an artifact.")
        if not r1.startswith("[refused]"):
            print(f"    [✗] step 1 expected [refused], got {r1!r}")
            failures += 1
        elif not r2.startswith("DONE accepted"):
            print(f"    [✗] step 2 expected DONE accepted, got {r2!r}")
            failures += 1
        else:
            print("    [✓] recovery path works")

        # -------- 5. sentinel content matches summary verbatim --------------
        print("\n[5] sentinel content matches summary verbatim")
        _reset(td)
        agent_tools._track_write("/x.md")
        msg = "Specific message: file=/x.md, lines=42, summary inside."
        agent_tools.done(msg)
        with open(agent_tools._done_sentinel_path) as f:
            stored = f.read()
        if stored != msg:
            print(f"    [✗] sentinel mismatch:")
            print(f"        wrote: {msg!r}")
            print(f"        read:  {stored!r}")
            failures += 1
        else:
            print(f"    [✓] sentinel preserves summary exactly ({len(stored)} chars)")

        # -------- 6. unicode + very long summary round-trip ----------------
        print("\n[6] unicode emoji + 10 KB summary survive round-trip")
        _reset(td)
        agent_tools._track_write("/u.md")
        wide = "🎉 ünıcödé " + ("x" * 10000)
        r = agent_tools.done(wide)
        if not r.startswith("DONE accepted"):
            print(f"    [✗] long unicode summary rejected: {r[:80]}...")
            failures += 1
        else:
            with open(agent_tools._done_sentinel_path, encoding="utf-8") as f:
                stored = f.read()
            if stored != wide:
                print(f"    [✗] unicode round-trip mismatch (len {len(stored)} vs {len(wide)})")
                failures += 1
            else:
                print(f"    [✓] {len(wide)} chars unicode round-trip clean")

        # -------- 7. corrupted writes file → treat as no writes ------------
        print("\n[7] corrupt session-writes file → treated as no writes")
        _reset(td)
        # Write garbage that breaks JSON parsing
        with open(agent_tools._session_writes_path, "w") as f:
            f.write("{not-valid-json")
        r = agent_tools.done("Try done with corrupt state file.")
        if not r.startswith("[refused]"):
            print(f"    [✗] expected [refused] when state corrupt, got {r!r}")
            failures += 1
        else:
            print(f"    [✓] corrupt JSON treated as 'no writes' (refused)")

        # -------- 8. multiple done calls overwrite sentinel ----------------
        print("\n[8] multiple accepted done() calls → sentinel updates")
        _reset(td)
        agent_tools._track_write("/m.md")
        agent_tools.done("First accepted summary.")
        agent_tools.done("Second accepted summary, overwrites first.")
        with open(agent_tools._done_sentinel_path) as f:
            stored = f.read()
        if stored != "Second accepted summary, overwrites first.":
            print(f"    [✗] last write didn't overwrite, got: {stored!r}")
            failures += 1
        else:
            print(f"    [✓] second done() overwrites sentinel cleanly")

        # -------- 9. sentinel-write failure → [error] ----------------------
        print("\n[9] sentinel path unwritable → [error] surfaced")
        _reset(td)
        agent_tools._track_write("/q.md")
        # Point sentinel at an unreachable directory
        agent_tools._done_sentinel_path = "/nonexistent_dir_for_sentinel/done.txt"
        r = agent_tools.done("Should fail to write the sentinel.")
        if not r.startswith("[error]"):
            print(f"    [✗] expected [error] for unwritable path, got {r!r}")
            failures += 1
        else:
            print(f"    [✓] sentinel-write failure surfaced as [error]")

    # -------- 10–12. agent.py interaction with done results -----------------
    # Use a fresh temp dir for the interaction tests; agent_tools is shared.
    with tempfile.TemporaryDirectory() as td2:
        _reset(td2)
        import agent

        def _make_response_with_tool_call(name: str, args: dict) -> dict:
            return {
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [{
                            "id": "call_abc",
                            "type": "function",
                            "function": {
                                "name": name,
                                "arguments": json.dumps(args),
                            },
                        }],
                    },
                    "finish_reason": "tool_calls",
                }],
            }

        # ---- 10. accepted done → step() returns True (signals end) ----
        print("\n[10] agent.step() returns True when done is accepted")
        _reset(td2)
        agent_tools._track_write("/agent_test_artifact.md")
        agent._loop_guard_nudge_fired = False
        messages = [
            {"role": "system", "content": agent.SYSTEM_PROMPT},
            {"role": "user", "content": "do the task"},
        ]
        fake = _make_response_with_tool_call(
            "done", {"summary": "Wrote /agent_test_artifact.md as requested."},
        )
        buf = io.StringIO()
        with patch.object(agent, "post_chat", return_value=fake), \
             redirect_stdout(buf):
            ret = agent.step(messages, step_num=1)
        out = buf.getvalue()
        if ret is not True:
            print(f"    [✗] step() returned {ret!r}, expected True (done signal)")
            failures += 1
        elif "done signal received" not in out:
            print(f"    [✗] missing 'done signal received' in stdout")
            print(f"        stdout tail: {out[-300:]!r}")
            failures += 1
        else:
            print(f"    [✓] step() returned True; done signal logged")

        # ---- 11. refused done → step() returns False (continue session) ----
        print("\n[11] refused done → step() does NOT mark session complete")
        _reset(td2)
        # Intentionally no _track_write — done() will refuse.
        agent._loop_guard_nudge_fired = False
        messages = [
            {"role": "system", "content": agent.SYSTEM_PROMPT},
            {"role": "user", "content": "do the task"},
        ]
        fake = _make_response_with_tool_call(
            "done", {"summary": "Claiming done but no writes happened."},
        )
        buf = io.StringIO()
        with patch.object(agent, "post_chat", return_value=fake), \
             redirect_stdout(buf):
            ret = agent.step(messages, step_num=1)
        out = buf.getvalue()
        if ret is True:
            print(f"    [✗] refused done still returned True (would close session prematurely)")
            failures += 1
        elif "done signal received" in out:
            print(f"    [✗] 'done signal received' printed for a refused done")
            failures += 1
        else:
            # Verify the refusal message reached the tool result in messages
            tool_msgs = [m for m in messages if m.get("role") == "tool"]
            if not tool_msgs:
                print(f"    [✗] no tool result message appended")
                failures += 1
            elif not tool_msgs[-1].get("content", "").startswith("[refused]"):
                print(f"    [✗] last tool result not a refusal: {tool_msgs[-1]['content'][:120]!r}")
                failures += 1
            else:
                print(f"    [✓] refused-done correctly leaves session open")

        # ---- 12. done called immediately after a loop-guard-aborted turn ----
        # Round-26 question: "what if done() is called AFTER a loop-aborted
        # response?" The harness should still gate on the writes — a loop-
        # abort doesn't itself create a write. So a done() after pure
        # loop-abort with no write should still be refused.
        print("\n[12] done after loop-abort with no writes → refused")
        _reset(td2)
        agent._loop_guard_nudge_fired = False
        messages = [
            {"role": "system", "content": agent.SYSTEM_PROMPT},
            {"role": "user", "content": "produce X"},
        ]
        # Step A: loop-aborted response (no tool calls, just the marker)
        aborted = (
            "I'll start now. I'll start now. I'll start now. "
            "[loop-guard: low-churn (distinct/total=10/200 ratio=0.05 "
            "top-gram×40) — output stopped early]"
        )
        fake_a = {
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": aborted},
                "finish_reason": "stop",
            }],
        }
        buf = io.StringIO()
        with patch.object(agent, "post_chat", return_value=fake_a), \
             redirect_stdout(buf):
            ret_a = agent.step(messages, step_num=1)

        # Step B: agent now calls done() despite the abort
        fake_b = _make_response_with_tool_call(
            "done", {"summary": "I aborted but I'm calling done anyway."},
        )
        buf = io.StringIO()
        with patch.object(agent, "post_chat", return_value=fake_b), \
             redirect_stdout(buf):
            ret_b = agent.step(messages, step_num=2)
        out_b = buf.getvalue()

        if ret_b is True:
            print(f"    [✗] done() after loop-abort closed session (no writes!)")
            failures += 1
        else:
            tool_msgs = [m for m in messages if m.get("role") == "tool"]
            last_tool = tool_msgs[-1].get("content", "") if tool_msgs else ""
            if not last_tool.startswith("[refused]"):
                print(f"    [✗] post-abort done not refused: {last_tool[:80]!r}")
                failures += 1
            else:
                print(f"    [✓] post-abort done correctly refused (no writes)")

    # -------------------------------------------------------------------
    print()
    if failures:
        print(f"FAIL: {failures} assertion(s)")
        return 1
    print("PASS: all done() edge cases handled correctly")
    return 0


if __name__ == "__main__":
    sys.exit(main())
