#!/usr/bin/env python3
"""Static audit: every main-model LLM call in scripts/ goes through the proxy.

Walks scripts/ AST-style for any `urlopen(...)`/`httpx.get(...)`/
`httpx.post(...)`/`requests.{get,post}(...)` whose URL contains
`/v1/chat/completions` or `/v1/models`, then confirms the URL is built
from `QWEN_HOST` + `QWEN_PORT` (the proxy's env vars) and not a
hard-coded port that bypasses the proxy.

Why this matters: the loop_guard's reach is exactly "all calls that go
through qwen-proxy". Any direct call to `127.0.0.1:8002` (dflash-serve
upstream) would bypass the guard. Today we believe nothing does this,
but a future commit could add such a path silently. This test catches
that on every run rather than at incident time.

The optional compactor sidecar is allowed to call its own mlx_lm.server
directly. It runs a fixed summarization prompt, has no tools, and is already
documented as a deliberate side port in `_qwen_daemon.py`; it is not a
main-model bypass.

Approach: regex-based grep, not full AST. Cheaper, and false negatives
matter more than false positives here — we'd rather flag something
suspicious for human review than miss it.

Skips:
- venv/ (third-party code)
- .snapshots/ (frozen historical copies)
- __pycache__/
- this file itself (it lists banned strings as test data)
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path


SCRIPT_DIR = Path("/Users/abelobsenz/dev/qwen36_MTP/scripts")
THIS_FILE = Path(__file__).resolve()

# Endpoints that ARE the LLM. Any call hitting these MUST be via the proxy.
LLM_PATHS = ("/v1/chat/completions", "/v1/models")

# The proxy port. dflash-serve internal port is 8002.
PROXY_PORT_DEFAULT = "8000"
UPSTREAM_PORT_DEFAULT = "8002"

# Regexes
_HTTP_CALL_RE = re.compile(
    r"\b(?:urlopen|urllib\.request\.Request|httpx\.get|httpx\.post|requests\.(?:get|post))\s*\("
)
_DIRECT_PORT_RE = re.compile(r"127\.0\.0\.1:(\d+)")
_HARDCODED_8002 = re.compile(r"127\.0\.0\.1:8002")


# Files where the audit is a no-op:
#  - qwen_proxy.py: IS the proxy. Naturally serves /v1/* endpoints.
#  - _qwen_daemon.py: launcher. Health-checks upstream:8002 by design;
#    that's not an LLM-bypass, it's "is dflash up yet?".
#  - test_*.py: each test spawns a fake upstream on a randomly-allocated
#    port and exercises the proxy in isolation. Hard-coded ports there
#    are the test fixtures, not production code paths.
_AUDIT_SKIP = {
    "qwen_proxy.py",
    "_qwen_daemon.py",
}


def audit_file(path: Path) -> list[str]:
    """Return a list of issue strings for this file."""
    issues: list[str] = []
    if path.name.startswith("test_") or path.name in _AUDIT_SKIP:
        return issues
    try:
        src = path.read_text()
    except (OSError, UnicodeDecodeError) as e:
        return [f"{path}: cannot read ({e})"]

    # Hard rule: nothing that hits the LLM should hard-code :8002.
    # That's the dflash-serve upstream port — bypassing the proxy means
    # bypassing the loop_guard.
    for ln_no, line in enumerate(src.splitlines(), start=1):
        if _HARDCODED_8002.search(line) and any(p in line for p in LLM_PATHS):
            issues.append(
                f"{path}:{ln_no}: hard-codes 127.0.0.1:8002 with an LLM "
                f"path — would bypass the proxy loop_guard"
            )

    # Soft rule: any urlopen/httpx call that mentions /v1/chat/completions
    # should be in the same lexical neighborhood as a proxy-rooted base
    # URL (QWEN_PORT / UPSTREAM_BASE / URL = …f"…{PORT}…").
    #
    # We only flag a path mention if it appears NEAR an actual HTTP call
    # — otherwise comments, docstrings, and string literals describing
    # the architecture get reported as false positives. The "actual call"
    # signal is one of urlopen/httpx.{get,post}/requests.{get,post}
    # within ±3 lines.
    lines = src.splitlines()
    for i, line in enumerate(lines):
        for path_str in LLM_PATHS:
            if path_str not in line:
                continue
            # Skip pure-comment lines.
            stripped = line.lstrip()
            if stripped.startswith("#"):
                continue
            # Require an actual HTTP-call site in the immediate vicinity.
            call_window_start = max(0, i - 3)
            call_window_end = min(len(lines), i + 4)
            call_window = "\n".join(lines[call_window_start:call_window_end])
            if not _HTTP_CALL_RE.search(call_window):
                # Path mentioned in a docstring or unrelated string; not a call.
                continue
            # Now check the wider window for proxy-rooted URL builders.
            ctx_start = max(0, i - 12)
            ctx_end = min(len(lines), i + 3)
            window = "\n".join(lines[ctx_start:ctx_end])
            compactor_sidecar = (
                path_str in LLM_PATHS
                and "_compactor_base" in window
            )
            if compactor_sidecar:
                continue
            proxy_rooted = bool(re.search(
                r"QWEN_PORT|PROXY_PORT|UPSTREAM_BASE|UPSTREAM_PORT"
                r"|_llm_endpoint|\bURL\s*=|\bMODELS_URL\s*=", window))
            if not proxy_rooted:
                issues.append(
                    f"{path}:{i+1}: '{path_str}' used in an HTTP call "
                    f"with no nearby proxy-rooted URL constant — verify "
                    f"this routes through the proxy"
                )
    return issues


def main() -> int:
    print(f"== Proxy routing audit (scripts/) ==\n")
    if not SCRIPT_DIR.exists():
        print(f"  scripts dir missing: {SCRIPT_DIR}")
        return 1

    failures: list[str] = []
    files_scanned = 0
    files_with_calls = 0
    for path in sorted(SCRIPT_DIR.rglob("*.py")):
        if "__pycache__" in path.parts or path.resolve() == THIS_FILE:
            continue
        files_scanned += 1
        src = path.read_text(errors="ignore")
        if _HTTP_CALL_RE.search(src):
            files_with_calls += 1
            issues = audit_file(path)
            if issues:
                failures.extend(issues)

    print(f"  files scanned: {files_scanned}")
    print(f"  files with HTTP calls: {files_with_calls}")
    print(f"  issues: {len(failures)}\n")
    if failures:
        for issue in failures:
            print(f"  [✗] {issue}")
        print(f"\n== FAIL ==")
        return 1

    # Bonus check: confirm QWEN_PORT default is the proxy port,
    # NOT the upstream port. Reading from agent_tools.py + agent.py
    # + qwen_ui.py + chat.py.
    print(f"  Default-port check (must default to proxy {PROXY_PORT_DEFAULT}, "
          f"not upstream {UPSTREAM_PORT_DEFAULT}):")
    bad = []
    for f in ("agent.py", "agent_graph.py", "agent_tools.py", "chat.py", "qwen_ui.py"):
        p = SCRIPT_DIR / f
        if not p.exists():
            continue
        src = p.read_text(errors="ignore")
        m = re.search(r"QWEN_PORT.*?\"(\d+)\"", src)
        if m:
            port = m.group(1)
            ok = port == PROXY_PORT_DEFAULT
            mark = "✓" if ok else "✗"
            print(f"    [{mark}] {f}: QWEN_PORT default = {port}")
            if not ok:
                bad.append(f)
    if bad:
        print(f"\n== FAIL ({len(bad)} file(s) default to wrong port) ==")
        return 1
    print(f"\n== PASS ==")
    return 0


if __name__ == "__main__":
    sys.exit(main())
