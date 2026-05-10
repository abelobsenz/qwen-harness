#!/usr/bin/env python3
"""Verify agent.py detects [loop-guard:] markers in proxy responses and
injects a course-correction nudge for the next turn.

This is an in-process unit test that monkeypatches `agent.post_chat` to
return a synthetic loop-guard-aborted response, then checks that
`agent.step()`:

  1. Prints a colored loop-guard notice (visible in stdout)
  2. Appends a `[HARNESS] Your previous response was cut off…` user
     message to the message list
  3. Returns False (don't exit; let next step course-correct)
  4. Single-fires within one top-level query (run_query reset)

This is the mirror image of the proxy-side test
(test_loop_guard_disabled.py) — together they verify the full chain:

  upstream loops → proxy aborts → agent surfaces + nudges → next turn
"""

from __future__ import annotations

import io
import os
import sys
from contextlib import redirect_stdout
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def _make_response(content: str, finish: str = "stop",
                   tool_calls: list[dict] | None = None) -> dict:
    """Build a synthetic OpenAI-shape chat completion response."""
    msg = {"role": "assistant", "content": content}
    if tool_calls:
        msg["tool_calls"] = tool_calls
    return {
        "choices": [{
            "index": 0,
            "message": msg,
            "finish_reason": finish,
        }],
    }


def main() -> int:
    failures = 0
    print("== Agent loop-guard nudge unit test ==\n")

    import agent

    def reset_loop_guard_state() -> None:
        """Mirror the per-query loop-guard reset in agent.run_query().

        The production harness tracks both a single-fire nudge latch and a
        hard abort counter. These synthetic scenarios are independent, so
        reset the full state between them instead of only the latch.
        """
        agent._loop_guard_nudge_fired = False
        agent._loop_guard_abort_count = 0
        agent._loop_guard_force_terminate = False

    # --- Test 1: loop-guard marker triggers nudge + returns False ---
    print("[1] Single loop-guard response triggers nudge")
    reset_loop_guard_state()
    messages = [
        {"role": "system", "content": agent.SYSTEM_PROMPT},
        {"role": "user", "content": "tell me about XYZ"},
    ]
    aborted_content = (
        "I will use make_table now. Then the Mermaid code. I will use "
        "make_table now. Then the Mermaid code. I will use make_table now. "
        "\n\n[loop-guard: low-churn (distinct/total=159/595 ratio=0.27 "
        "top-gram×14) — output stopped early]"
    )
    fake_resp = _make_response(aborted_content)
    buf = io.StringIO()
    with patch.object(agent, "post_chat", return_value=fake_resp), \
         redirect_stdout(buf):
        ret = agent.step(messages, step_num=1)
    output = buf.getvalue()

    print(f"    step() returned: {ret} (expect False)")
    print(f"    output snippet: {repr(output[-200:])[:200]}")
    if ret is not False:
        print("    [✗] step() should return False (don't exit)")
        failures += 1
    if "loop-guard fired" not in output:
        print("    [✗] missing colored loop-guard notice in stdout")
        failures += 1
    nudge_msgs = [m for m in messages if m.get("role") == "user"
                  and "[HARNESS]" in (m.get("content") or "")]
    if not nudge_msgs:
        print("    [✗] no [HARNESS] nudge appended to messages")
        failures += 1
    else:
        nudge = nudge_msgs[-1]["content"]
        if "loop guard" not in nudge.lower():
            print("    [✗] nudge text doesn't mention loop guard")
            failures += 1
        else:
            print(f"    [✓] nudge appended ({len(nudge)} chars, mentions loop guard)")

    # --- Test 2: second occurrence within same query does NOT re-fire ---
    print("\n[2] Second loop-guard within same query — single-fire")
    pre_count = len(messages)
    fake_resp2 = _make_response(aborted_content)
    buf = io.StringIO()
    with patch.object(agent, "post_chat", return_value=fake_resp2), \
         redirect_stdout(buf):
        ret = agent.step(messages, step_num=2)
    post_count = len(messages)
    # The second response gets appended (the assistant message from step 2),
    # but the harness nudge should NOT fire again.
    new_harness = sum(1 for m in messages[pre_count:]
                      if m.get("role") == "user"
                      and "[HARNESS]" in (m.get("content") or ""))
    if new_harness > 0:
        print(f"    [✗] second loop-guard fired ANOTHER nudge (would spam)")
        failures += 1
    else:
        print(f"    [✓] single-fire: no extra HARNESS message added")

    # --- Test 3: clean response after reset does fire fresh ---
    print("\n[3] After reset, loop-guard fires again on next user query")
    reset_loop_guard_state()
    messages2 = [
        {"role": "system", "content": agent.SYSTEM_PROMPT},
        {"role": "user", "content": "another question"},
    ]
    fake_resp3 = _make_response(aborted_content)
    buf = io.StringIO()
    with patch.object(agent, "post_chat", return_value=fake_resp3), \
         redirect_stdout(buf):
        ret = agent.step(messages2, step_num=1)
    nudge_msgs2 = [m for m in messages2 if m.get("role") == "user"
                   and "[HARNESS]" in (m.get("content") or "")]
    if not nudge_msgs2:
        print("    [✗] reset didn't re-arm the nudge")
        failures += 1
    else:
        print(f"    [✓] post-reset: nudge fires again on fresh query")

    # --- Test 4: clean response (no marker) doesn't fire nudge ---
    print("\n[4] Clean response (no marker) — no false-fire")
    reset_loop_guard_state()
    messages3 = [
        {"role": "system", "content": agent.SYSTEM_PROMPT},
        {"role": "user", "content": "what's 2+2?"},
    ]
    fake_resp4 = _make_response("2+2 is 4. This is basic arithmetic.")
    buf = io.StringIO()
    with patch.object(agent, "post_chat", return_value=fake_resp4), \
         redirect_stdout(buf):
        ret = agent.step(messages3, step_num=1)
    nudge_msgs4 = [m for m in messages3 if m.get("role") == "user"
                   and "[HARNESS]" in (m.get("content") or "")]
    if nudge_msgs4:
        print(f"    [✗] FALSE POSITIVE: nudge fired on clean response")
        failures += 1
    else:
        print(f"    [✓] clean response leaves messages alone")
    # And step() should return True (it's a normal exit on no tool_calls)
    if ret is not True:
        print(f"    [✗] step() on clean response should return True, got {ret}")
        failures += 1
    else:
        print(f"    [✓] step() returned True (normal exit)")

    # --- Test 5: marker + tool_calls — nudge still fires ---
    # The proxy can emit the marker AFTER a tool_call in content if the
    # model started a tool_call then began looping. Without the early
    # check, this case would be silent because the nudge logic only ran
    # in the no-tool-calls branch.
    print("\n[5] Tool_calls present + marker — nudge still fires")
    reset_loop_guard_state()
    messages5 = [
        {"role": "system", "content": agent.SYSTEM_PROMPT},
        {"role": "user", "content": "read the file"},
    ]
    aborted_with_tool = (
        "I'll read the file.\n\n[loop-guard: low-churn (distinct/total=159/595 "
        "ratio=0.27 top-gram×14) — output stopped early]"
    )
    fake_resp5 = _make_response(
        aborted_with_tool,
        tool_calls=[{
            "id": "call_0",
            "type": "function",
            "function": {"name": "read_file",
                         "arguments": '{"path": "/tmp/foo.txt"}'},
        }],
    )
    # Need to patch dispatch path too so step() doesn't actually try to
    # read /tmp/foo.txt. Patch CachedDispatcher.dispatch_batch instead.
    from agent_tools import CachedDispatcher
    buf = io.StringIO()
    with patch.object(agent, "post_chat", return_value=fake_resp5), \
         patch.object(CachedDispatcher, "dispatch_batch",
                      return_value=[("file contents here", False)]), \
         redirect_stdout(buf):
        ret = agent.step(messages5, step_num=1)
    output = buf.getvalue()
    nudge_msgs5 = [m for m in messages5 if m.get("role") == "user"
                   and "[HARNESS]" in (m.get("content") or "")]
    if "loop-guard fired" not in output:
        print("    [✗] missing colored notice when both tool_calls + marker present")
        failures += 1
    if not nudge_msgs5:
        print("    [✗] no nudge appended despite marker in content")
        failures += 1
    else:
        print(f"    [✓] nudge fired even with tool_calls present")

    # --- Test 6: benign mention of [loop-guard:] does NOT false-fire ---
    # Real-world false-positive risk: the model legitimately mentions
    # the substring while explaining the system to the user, OR a tool
    # result echoes config / log text containing the marker. None of
    # these have the proxy's specific abort suffix ("output stopped
    # early" / "fell into a repetition loop"), so the tightened
    # detector should leave them alone.
    print("\n[6] Benign [loop-guard:] mentions — no false-positive")
    benign_cases = [
        ("user-explained",
         "The proxy emits a [loop-guard: <reason>] marker when it detects "
         "a loop. You can disable this with LOOP_GUARD_DISABLE=1."),
        ("code-review",
         "Looking at qwen_proxy.py line 487, you can see [loop-guard: ...] "
         "is logged via logger.warning when the abort fires."),
        ("grep-output",
         "The grep found these matches:\n"
         "  scripts/agent.py:308: _LOOP_GUARD_RE = re.compile(r'\\[loop-guard:...')\n"
         "  scripts/qwen_proxy.py:487: print(f'[loop-guard: {reason}]')"),
    ]
    for label, content in benign_cases:
        reset_loop_guard_state()
        msgs = [
            {"role": "system", "content": agent.SYSTEM_PROMPT},
            {"role": "user", "content": "explain how the loop guard works"},
        ]
        fake = _make_response(content)
        buf = io.StringIO()
        with patch.object(agent, "post_chat", return_value=fake), \
             redirect_stdout(buf):
            ret = agent.step(msgs, step_num=1)
        output = buf.getvalue()
        nudges = [m for m in msgs if m.get("role") == "user"
                  and "[HARNESS]" in (m.get("content") or "")]
        if nudges:
            print(f"    [✗] {label}: FALSE POSITIVE — nudge fired on benign content")
            failures += 1
        elif "loop-guard fired" in output:
            print(f"    [✗] {label}: FALSE POSITIVE — printed colored notice on benign content")
            failures += 1
        else:
            print(f"    [✓] {label}: no false-fire")

    # --- Test 7: helper invariant ---
    # _is_proxy_abort_marker(content) should be True iff substring AND suffix.
    print("\n[7] _is_proxy_abort_marker helper invariant")
    cases = [
        ("real abort (non-stream)",
         "[loop-guard: low-churn — output stopped early]", True),
        ("real abort (streaming)",
         "[loop-guard: aborted (suffix-extreme) — the model fell into a repetition loop. Try again.]", True),
        ("benign mention",
         "We log [loop-guard: <reason>] when aborts happen.", False),
        ("substring but no suffix",
         "The string '[loop-guard:' appears in this comment.", False),
        ("no substring at all",
         "Hello world.", False),
    ]
    for label, content, expected in cases:
        got = agent._is_proxy_abort_marker(content)
        ok = got == expected
        marker = "✓" if ok else "✗"
        print(f"    [{marker}] {label}: got={got} expected={expected}")
        if not ok:
            failures += 1

    print(f"\n== {'PASS' if failures == 0 else 'FAIL'} ({failures} failure(s)) ==")
    return failures


if __name__ == "__main__":
    sys.exit(main())
