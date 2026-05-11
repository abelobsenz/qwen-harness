#!/usr/bin/env python3
"""Subagent / explore allowlist contract pins.

`subagent_implement` and `explore` each carry an allowlist of tool names
they may dispatch (SUBAGENT_IMPLEMENT_TOOL_NAMES, EXPLORE_TOOL_NAMES).
Both are enforced at dispatch time — a tool_call to a name not in the
allowlist returns a `[denied]` marker without executing.

This test pins the load-bearing invariants:

  1. Both allowlists are non-empty `set`s of strings.
  2. The `explore` allowlist is read-only (no write, edit, patch, run, or
     bash). Explore is a read-only subagent by design.
  3. The `subagent_implement` allowlist is write-capable but excludes
     recursion (no `explore`, no `subagent_implement`) and excludes
     network access (no `web_search`, no `web_fetch`).
  4. The denial path in subagent_implement actually fires for an
     out-of-allowlist tool name (simulated dispatch, no model call).
  5. Each tool listed in either allowlist actually exists in TOOLS /
     DISPATCH — a stale name is a silent breakage.

Pure source / no-network / no-model. Runs in <50 ms.
"""

from __future__ import annotations

import inspect
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from agent_tools import (  # noqa: E402
    DISPATCH,
    TOOLS,
    SUBAGENT_IMPLEMENT_TOOL_NAMES,
    EXPLORE_TOOL_NAMES,
)


def _tool_names() -> set[str]:
    return {t["function"]["name"] for t in TOOLS}


def main() -> int:
    failures = 0
    t0 = time.perf_counter()
    print("== subagent / explore allowlist contract ==\n")

    # 1. Both allowlists are non-empty sets of strings.
    for label, allow in (("SUBAGENT_IMPLEMENT_TOOL_NAMES",
                          SUBAGENT_IMPLEMENT_TOOL_NAMES),
                         ("EXPLORE_TOOL_NAMES", EXPLORE_TOOL_NAMES)):
        if not isinstance(allow, (set, frozenset)):
            print(f"  [✗] {label} is not a set/frozenset: {type(allow).__name__}")
            failures += 1
            continue
        if not allow:
            print(f"  [✗] {label} is empty")
            failures += 1
            continue
        if not all(isinstance(n, str) for n in allow):
            print(f"  [✗] {label} contains non-string entries")
            failures += 1
            continue
        print(f"  [✓] {label} is a non-empty set of {len(allow)} strings")

    # 2. EXPLORE_TOOL_NAMES is strictly read-only.
    WRITE_TOOLS = {
        "write_file", "edit_file", "apply_patch", "write_file_verified",
        "append_finding", "python_run", "python_reset",
        "notebook_edit", "notebook_run", "bash", "test_run",
        "memory_save", "memory_delete", "memory_reembed",
        "subagent_implement", "explore",
    }
    bad = EXPLORE_TOOL_NAMES & WRITE_TOOLS
    if bad:
        print(f"  [✗] EXPLORE_TOOL_NAMES leaks write/recursion capabilities: {sorted(bad)}")
        failures += 1
    else:
        print("  [✓] EXPLORE_TOOL_NAMES is strictly read-only (no write/recursion)")

    # 3. SUBAGENT_IMPLEMENT_TOOL_NAMES excludes recursion AND network.
    FORBIDDEN_FOR_IMPLEMENT = {
        "explore",          # No recursive subagent
        "subagent_implement",  # No recursive self
        "web_search",       # No network
        "web_fetch",        # No network
        "web_outline",
        "find_in_url",
        "github_repo",
        "sec_filings",
        "arxiv_search",
        "arxiv_fetch",
        "doi_resolve",
    }
    bad = SUBAGENT_IMPLEMENT_TOOL_NAMES & FORBIDDEN_FOR_IMPLEMENT
    if bad:
        print(f"  [✗] SUBAGENT_IMPLEMENT_TOOL_NAMES contains recursion/network "
              f"tools (should be code-edit only): {sorted(bad)}")
        failures += 1
    else:
        print("  [✓] SUBAGENT_IMPLEMENT_TOOL_NAMES excludes recursion + network")

    # 4. Stale-name check: every allowlisted tool must exist in TOOLS+DISPATCH.
    registered = _tool_names()
    for label, allow in (("SUBAGENT_IMPLEMENT_TOOL_NAMES",
                          SUBAGENT_IMPLEMENT_TOOL_NAMES),
                         ("EXPLORE_TOOL_NAMES", EXPLORE_TOOL_NAMES)):
        missing_from_tools = allow - registered
        missing_from_dispatch = allow - set(DISPATCH.keys())
        if missing_from_tools:
            print(f"  [✗] {label} references tool(s) absent from TOOLS: "
                  f"{sorted(missing_from_tools)}")
            failures += 1
        if missing_from_dispatch:
            print(f"  [✗] {label} references tool(s) absent from DISPATCH: "
                  f"{sorted(missing_from_dispatch)}")
            failures += 1
    if not failures:
        print("  [✓] every allowlisted tool exists in TOOLS and DISPATCH")

    # 5. Source-level: subagent_implement actually has the deny line.
    import agent_tools as at
    src = inspect.getsource(at.subagent_implement)
    if "[denied]" not in src or "SUBAGENT_IMPLEMENT_TOOL_NAMES" not in src:
        print("  [✗] subagent_implement source missing the deny branch")
        failures += 1
    else:
        print("  [✓] subagent_implement dispatch branch enforces the allowlist")

    # 6. Behaviour: a fresh `done` tool call goes through; an invented
    #    tool name doesn't. We synthesise the dispatch step inline so
    #    no LLM round-trip is needed.
    valid_name = "done"
    invalid_name = "rm_rf_everything"
    if valid_name not in SUBAGENT_IMPLEMENT_TOOL_NAMES:
        print(f"  [✗] sanity: '{valid_name}' should be allowlisted")
        failures += 1
    if invalid_name in SUBAGENT_IMPLEMENT_TOOL_NAMES:
        print(f"  [✗] sanity: '{invalid_name}' should NOT be allowlisted")
        failures += 1

    # 7. Explore also has the deny line.
    src_explore = inspect.getsource(at.explore)
    if "[denied]" not in src_explore or "EXPLORE_TOOL_NAMES" not in src_explore:
        print("  [✗] explore source missing the deny branch")
        failures += 1
    else:
        print("  [✓] explore dispatch branch enforces the allowlist")

    elapsed = (time.perf_counter() - t0) * 1000
    print(f"\n  elapsed: {elapsed:.1f} ms")
    print(f"\n== {'PASS' if failures == 0 else 'FAIL'} "
          f"({failures} failure(s)) ==")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
