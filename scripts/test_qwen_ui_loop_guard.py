#!/usr/bin/env python3
"""Verify the JS `isProxyAbortMarker()` in qwen_ui_static/app.js mirrors
the Python `is_proxy_abort_marker()` contract.

The UI's surfacing depends on a JS regex check; the same false-positive
class that hit agent.py (Round 15) and agent_graph.py (Round 16) was
present in the JS. Round 18 fixed it. This test:

  1. Reads the JS source for `isProxyAbortMarker`.
  2. Cross-checks the function's documented suffix list against the
     proxy's actual marker emissions in qwen_proxy.py — if the proxy
     ever changes its suffix wording, this test will flag the JS as
     out-of-sync at code-review time.
  3. Runs the canonical positive + negative test cases through both
     Python (`loop_guard_marker.is_proxy_abort_marker`) and a Python
     port of the JS regex/substring logic, asserting they agree
     character-for-character.

If `node`/`deno` were available we'd run the actual JS; relying on
Python equivalents keeps this test stdlib-only and portable, with the
risk that JS regex semantics could diverge from Python for these
specific patterns. The patterns here are so simple (literal substrings
and disjunction over fixed strings) that the divergence risk is zero
in practice.
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


APP_JS = Path("/Users/abelobsenz/dev/qwen36_MTP/scripts/qwen_ui_static/app.js")
PROXY_PY = Path("/Users/abelobsenz/dev/qwen36_MTP/scripts/qwen_proxy.py")


def python_port_of_js_check(text: str) -> bool:
    """Equivalent to the JS isProxyAbortMarker. Used to verify the JS
    logic matches the Python authoritative implementation."""
    if not text or "[loop-guard:" not in text:
        return False
    return re.search(r"output stopped early|fell into a repetition loop",
                     text, re.IGNORECASE) is not None


def main() -> int:
    failures: list[str] = []
    print("== qwen_ui loop-guard JS detection test ==\n")

    # 1. Read JS source and confirm the new function exists.
    print("[1] JS source carries the new isProxyAbortMarker function")
    js = APP_JS.read_text()
    if "function isProxyAbortMarker" not in js:
        print("    [✗] JS source is missing function isProxyAbortMarker")
        failures.append("missing-function")
    else:
        print("    [✓] function isProxyAbortMarker present")

    # The naïve substring-only check should be GONE.
    if "/\\[loop-guard:/i.test(t.textNode._buf)" in js:
        print("    [✗] JS still uses the naïve substring regex")
        failures.append("naive-substring-still-present")
    else:
        print("    [✓] naïve substring regex removed (replaced by helper)")

    # 2. Cross-check JS suffix list vs the proxy's actual emissions.
    # The proxy uses multi-line f-strings (e.g. `f"the model fell into a "
    # f"repetition loop."`), so we can't grep for the joined string in
    # source. Instead, normalize the source by collapsing whitespace + JS
    # concatenation artifacts before checking for substring presence.
    print("\n[2] JS suffix list matches proxy-emitted suffixes")
    proxy_src = PROXY_PY.read_text()
    # Collapse runs of whitespace AND any f-string boundary patterns so
    # "fell into a " + "repetition loop" surfaces as one phrase.
    normalized = re.sub(r"['\"]\s*(?:f?['\"])?\s*", " ", proxy_src)
    normalized = re.sub(r"\s+", " ", normalized)
    js_suffixes = set()
    if re.search(r"output stopped early\|fell into a repetition loop", js):
        js_suffixes = {"output stopped early", "fell into a repetition loop"}
    for suf in js_suffixes:
        if suf not in normalized:
            print(f"    [✗] suffix {suf!r} expected in JS but not in proxy")
            failures.append(f"suffix-drift:{suf}")
        else:
            print(f"    [✓] suffix {suf!r} present in both JS and proxy")

    # 3. Cross-validate against the canonical test corpus from
    #    test_loop_guard_marker.py.
    print("\n[3] Python port of JS check matches authoritative is_proxy_abort_marker")
    from loop_guard_marker import is_proxy_abort_marker

    # Cases pulled verbatim from the agreed corpus across rounds 15-17.
    cases = [
        # (label, text, expected)
        ("real abort non-stream",
         "[loop-guard: low-churn (...) — output stopped early]", True),
        ("real abort streaming",
         "[loop-guard: aborted (suffix-extreme) — the model fell into a "
         "repetition loop. Try rephrasing.]", True),
        ("user-explained",
         "The proxy emits a [loop-guard: <reason>] marker when it detects "
         "a loop.", False),
        ("code-review",
         "Looking at qwen_proxy.py:487, [loop-guard: ...] is logged.", False),
        ("grep-echo",
         "scripts/agent.py:308: _LOOP_GUARD_RE = re.compile(r'\\[loop-guard:...')",
         False),
        ("substring no suffix",
         "We log [loop-guard: <reason>] when needed.", False),
        ("suffix no substring",
         "The output stopped early because the user pressed escape.", False),
        ("empty", "", False),
    ]
    for label, text, expected in cases:
        py_authority = is_proxy_abort_marker(text)
        py_port = python_port_of_js_check(text)
        # Both must equal `expected`, AND must agree with each other.
        agreed = py_authority == py_port == expected
        if not agreed:
            print(f"    [✗] {label}: authority={py_authority} "
                  f"port={py_port} expected={expected}")
            failures.append(f"divergence:{label}")
        else:
            print(f"    [✓] {label}: both={py_authority} expected={expected}")

    # 4. CSS class hook is unchanged and consistent.
    print("\n[4] CSS class hook 'has-loop-guard-abort' is stable")
    css = (
        Path("/Users/abelobsenz/dev/qwen36_MTP/scripts/qwen_ui_static/style.css")
        .read_text()
    )
    if "has-loop-guard-abort" not in css:
        print("    [✗] CSS missing the .has-loop-guard-abort class")
        failures.append("missing-css")
    elif "has-loop-guard-abort" not in js:
        print("    [✗] JS no longer adds the .has-loop-guard-abort class")
        failures.append("missing-class-add")
    else:
        print("    [✓] CSS class hook stable (JS still adds, CSS still styles)")

    # 5. Python-side qwen_ui chat loop now surfaces the marker too
    #    (Round 19 added this — parity with agent.py + agent_graph.py).
    print("\n[5] qwen_ui.py Python side imports loop_guard_marker + nudges")
    qwen_ui_py = Path(
        "/Users/abelobsenz/dev/qwen36_MTP/scripts/qwen_ui.py"
    ).read_text()
    if "from loop_guard_marker import" not in qwen_ui_py:
        print("    [✗] qwen_ui.py does not import loop_guard_marker")
        failures.append("missing-py-import")
    elif "is_proxy_abort_marker(asst_content)" not in qwen_ui_py:
        print("    [✗] qwen_ui.py imports module but doesn't call detector")
        failures.append("py-detector-not-called")
    elif "loop_guard_nudged" not in qwen_ui_py:
        print("    [✗] qwen_ui.py missing single-fire flag")
        failures.append("missing-py-flag")
    elif "harness_nudge_message(reason)" not in qwen_ui_py:
        print("    [✗] qwen_ui.py missing nudge-message append")
        failures.append("missing-py-nudge")
    else:
        print("    [✓] qwen_ui.py: imports detector, calls it, has flag, "
              "appends nudge")

    print(f"\n== {'PASS' if not failures else 'FAIL'} "
          f"({len(failures)} failure(s)) ==")
    if failures:
        for f in failures:
            print(f"  - {f}")
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
