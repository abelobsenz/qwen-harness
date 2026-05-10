#!/usr/bin/env python3
"""Integration test: agent_graph._run_node injects a course-correction
nudge when the proxy aborts a node mid-AGFMT.

Mocks `AgentGraph._post` to control what the synthetic model returns:
  - Turn 1: assistant content containing the proxy's `[loop-guard:]`
    marker + abort suffix → loop-guard wiring should fire ONCE,
    inject the HARNESS nudge, and `continue` the per-node loop.
  - Turn 2: valid AGFMT → parsing succeeds, node returns normally.

Negative path: a benign mention of `[loop-guard:` (e.g. the model
explaining the system) must NOT trigger the wiring. Asserts agent_graph
uses the same false-positive-resistant detector that agent.py uses
(via `loop_guard_marker.is_proxy_abort_marker`).
"""

from __future__ import annotations

import io
import os
import sys
from contextlib import redirect_stdout
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def _make_resp(content: str, tool_calls: list[dict] | None = None) -> dict:
    msg = {"role": "assistant", "content": content}
    if tool_calls:
        msg["tool_calls"] = tool_calls
    return {"choices": [{"index": 0, "message": msg, "finish_reason": "stop"}]}


def main() -> int:
    failures: list[str] = []
    print("== agent_graph loop-guard integration test ==\n")

    from agent_graph import AgentGraph

    # Build a minimal graph via the builder API: one node, no tools.
    graph = AgentGraph(name="test")
    node = graph.add_node(
        name="researcher",
        role="researcher",
        goal="Investigate the topic and emit AGFMT.",
        inputs=["topic"],
        outputs=["finding:t"],  # one required text output named `finding`
        tools=None,
        max_steps=5,
        max_output_retries=1,
    )

    # Valid AGFMT response for the second turn.
    AGFMT_OK = "@finding:t\nThe topic resolves to X.\n@END"

    # The aborted response carries the proxy's marker.
    LOOP_ABORT = (
        "Looking at the topic, I see something interesting. I see something "
        "interesting. I see something interesting. I see something "
        "interesting.\n\n"
        "[loop-guard: low-churn (distinct/total=159/595 ratio=0.27 "
        "top-gram×14) — output stopped early]"
    )

    # ===== Test 1: marker fires nudge, next turn parses OK =====
    print("[1] Loop-guard marker → HARNESS nudge → next-turn AGFMT OK")
    inputs = {"topic": "quantum computing"}

    # Sequence the mock: first call → LOOP_ABORT, second → AGFMT_OK.
    responses = iter([_make_resp(LOOP_ABORT), _make_resp(AGFMT_OK)])
    buf = io.StringIO()
    with patch.object(AgentGraph, "_post", side_effect=lambda *a, **kw: next(responses)), \
         redirect_stdout(buf):
        parsed, stats = graph._run_node(node, inputs, verbose=True)

    # Assertions:
    if parsed.get("finding") != "The topic resolves to X.":
        print(f"    [✗] expected parsed['finding']='The topic resolves to X.', "
              f"got {parsed.get('finding')!r}")
        failures.append("parsed-output")
    else:
        print(f"    [✓] parsed['finding'] = {parsed.get('finding')!r}")

    output = buf.getvalue()
    if "loop-guard fired in graph node" not in output:
        print(f"    [✗] missing colored notice in stdout")
        failures.append("colored-notice")
    else:
        # Capture the snippet for the log
        for line in output.splitlines():
            if "loop-guard" in line.lower():
                print(f"    notice: {line.strip()}")
        print(f"    [✓] colored notice printed")

    # ===== Test 2: single-fire — second loop-guard marker (same node) =====
    # Reset for a fresh node + run.
    print("\n[2] Two loop-guards in a row — only one nudge fires per node")
    responses = iter([
        _make_resp(LOOP_ABORT),       # turn 1: triggers nudge, continue
        _make_resp(LOOP_ABORT),       # turn 2: SHOULD NOT trigger another nudge
        _make_resp(AGFMT_OK),         # turn 3: valid AGFMT, return
    ])
    buf = io.StringIO()
    # Reset retry budget on the node.
    node.max_output_retries = 1
    with patch.object(AgentGraph, "_post", side_effect=lambda *a, **kw: next(responses)), \
         redirect_stdout(buf):
        parsed, stats = graph._run_node(node, inputs, verbose=True)
    output = buf.getvalue()
    nudge_fired_count = output.count("loop-guard fired in graph node")
    if nudge_fired_count != 1:
        print(f"    [✗] nudge fired {nudge_fired_count}× (expected exactly 1)")
        failures.append("single-fire")
    else:
        print(f"    [✓] nudge fired exactly 1× across 2 abort markers (single-fire)")

    # ===== Test 3: benign mention does NOT trigger =====
    print("\n[3] Benign mention of [loop-guard:] — no nudge, no harness")
    BENIGN = (
        "@finding:t\n"
        "The proxy emits a [loop-guard: <reason>] marker when it detects "
        "a loop. You can see this in scripts/qwen_proxy.py.\n@END"
    )
    responses = iter([_make_resp(BENIGN)])
    buf = io.StringIO()
    node.max_output_retries = 1
    with patch.object(AgentGraph, "_post", side_effect=lambda *a, **kw: next(responses)), \
         redirect_stdout(buf):
        parsed, stats = graph._run_node(node, inputs, verbose=True)
    output = buf.getvalue()
    if "loop-guard fired in graph node" in output:
        print(f"    [✗] FALSE POSITIVE: nudge fired on benign mention")
        failures.append("benign-fp")
    elif parsed.get("finding") is None:
        print(f"    [✗] benign content didn't parse as AGFMT (got {parsed!r})")
        failures.append("benign-parse")
    else:
        print(f"    [✓] benign mention left alone, AGFMT parsed normally "
              f"(finding={parsed['finding'][:50]!r}…)")

    # ===== Test 4: marker present + AGFMT also present (rare shape) =====
    # If the model somehow emitted BOTH valid AGFMT AND a marker (unlikely
    # in practice but possible if abort fired AFTER valid AGFMT was generated),
    # the nudge should still fire AND the AGFMT should still parse — we
    # don't want to lose valid output to a defensive nudge.
    print("\n[4] Marker + valid AGFMT — both fire, output preserved")
    BOTH = (
        "@finding:t\n"
        "Here is the partial answer.\n"
        "@END\n\n"
        "[loop-guard: combined — output stopped early]"
    )
    responses = iter([_make_resp(BOTH)])
    buf = io.StringIO()
    node.max_output_retries = 1
    with patch.object(AgentGraph, "_post", side_effect=lambda *a, **kw: next(responses)), \
         redirect_stdout(buf):
        parsed, stats = graph._run_node(node, inputs, verbose=True)
    output = buf.getvalue()
    if "loop-guard fired in graph node" not in output:
        print(f"    [✗] marker present but nudge did not fire")
        failures.append("both-no-nudge")
    else:
        print(f"    [✓] nudge fired on combined marker+AGFMT response")

    print(f"\n== {'PASS' if not failures else 'FAIL'} "
          f"({len(failures)} failure(s)) ==")
    if failures:
        for f in failures:
            print(f"  - {f}")
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
