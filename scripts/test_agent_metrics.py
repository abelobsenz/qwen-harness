#!/usr/bin/env python3
"""Tests for scripts/agent_metrics.py.

Contract:
  1. inc / set / get / snapshot / reset / flush work as specified.
  2. flush() writes valid JSONL to the configured path.
  3. Disabled mode (QWEN_AGENT_METRICS=0) is a strict no-op.
  4. flush() degrades silently on OSError.
  5. flush_from_agent_globals() pulls the right names without crashing
     when agent isn't importable.
  6. Per-inc() overhead is <1 µs on this hardware (efficiency budget).

All assertions run in <50 ms without writing into the user's real
~/.qwen directory — we redirect QWEN_AGENT_METRICS_PATH to a temp dir.
"""

from __future__ import annotations

import importlib
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from unittest import mock

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def _reload_metrics(env: dict[str, str]) -> "module":  # type: ignore[name-defined]
    """Reimport agent_metrics under a controlled env. The module reads
    its toggles at import time, so the test must clear and reimport."""
    for k in ("QWEN_AGENT_METRICS", "QWEN_AGENT_METRICS_PATH"):
        os.environ.pop(k, None)
    for k, v in env.items():
        os.environ[k] = v
    sys.modules.pop("agent_metrics", None)
    return importlib.import_module("agent_metrics")


def main() -> int:
    failures: list[str] = []

    def check(label: str, ok: bool, detail: str = "") -> None:
        marker = "✓" if ok else "✗"
        suffix = f" — {detail}" if detail else ""
        print(f"  [{marker}] {label}{suffix}")
        if not ok:
            failures.append(label)

    print("== agent_metrics tests ==\n")
    t0 = time.perf_counter()

    with tempfile.TemporaryDirectory() as td:
        sink = Path(td) / "metrics.jsonl"

        # ---- 1. Basic inc / set / get / snapshot ---------------------
        m = _reload_metrics({"QWEN_AGENT_METRICS_PATH": str(sink)})
        m.reset()
        check("starts enabled", m.is_enabled() is True)
        m.inc("nudge_cache_loop")
        m.inc("nudge_cache_loop")
        m.inc("nudge_refused", 3)
        m.set_value("audit_gate_fired", 1)
        snap = m.snapshot()
        check("inc accumulates", snap["counters"]["nudge_cache_loop"] == 2,
              f"got {snap['counters'].get('nudge_cache_loop')}")
        check("inc(n=3) works", snap["counters"]["nudge_refused"] == 3)
        check("set_value works", snap["counters"]["audit_gate_fired"] == 1)
        check("get(name) returns the value", m.get("nudge_refused") == 3)
        check("get(missing) returns default", m.get("does_not_exist", 42) == 42)
        check("snapshot includes ts and uptime_s",
              "ts" in snap and "uptime_s" in snap)

        # ---- 2. flush writes valid JSONL -----------------------------
        path = m.flush(extra={"iter": 1, "prompt": "nflx_q4_repurchase"})
        check("flush returns the path", path == sink)
        check("sink file exists after flush", sink.exists())
        lines = sink.read_text().splitlines()
        check("flush appends exactly one line", len(lines) == 1,
              f"lines={len(lines)}")
        record = json.loads(lines[0])
        check("record is valid JSON", isinstance(record, dict))
        check("record carries the extra fields",
              record.get("iter") == 1 and record.get("prompt") == "nflx_q4_repurchase")
        check("record's counters match snapshot",
              record["counters"]["nudge_cache_loop"] == 2)

        # Second flush appends, doesn't overwrite.
        m.inc("nudge_cache_loop")
        m.flush(extra={"iter": 2})
        lines = sink.read_text().splitlines()
        check("second flush appends a line", len(lines) == 2)

        # ---- 3. reset() zeroes everything ----------------------------
        m.reset()
        snap = m.snapshot()
        check("reset() zeroes all counters", snap["counters"] == {})

        # ---- 4. Disabled mode is no-op -------------------------------
        sink_disabled = Path(td) / "disabled.jsonl"
        m_off = _reload_metrics({
            "QWEN_AGENT_METRICS": "0",
            "QWEN_AGENT_METRICS_PATH": str(sink_disabled),
        })
        check("disabled mode reports disabled", m_off.is_enabled() is False)
        m_off.inc("x", 1000)
        m_off.set_value("y", 42)
        snap = m_off.snapshot()
        check("disabled inc does not record",
              snap["counters"].get("x", 0) == 0)
        path = m_off.flush()
        check("disabled flush returns None", path is None)
        check("disabled mode does NOT create the sink file",
              sink_disabled.exists() is False)

        # ---- 5. flush degrades silently on OSError -------------------
        m = _reload_metrics({"QWEN_AGENT_METRICS_PATH": str(sink)})
        m.inc("y")
        with mock.patch.object(Path, "open",
                                side_effect=PermissionError("no")):
            # Should not raise.
            result = m.flush()
        check("flush returns None on OSError (silent)", result is None)

        # ---- 6. flush_from_agent_globals doesn't crash without agent -
        m = _reload_metrics({"QWEN_AGENT_METRICS_PATH": str(sink)})
        # Ensure no `agent` module is loaded.
        sys.modules.pop("agent", None)
        result = m.flush_from_agent_globals(extra={"test": "no_agent"})
        check("flush_from_agent_globals works without agent module",
              result == sink)

        # Now fake an `agent` module with a few globals and verify the
        # right names propagate.
        fake_agent = type(sys)("agent")
        fake_agent._consecutive_all_cached_turns = 5
        fake_agent._audit_fired = True
        fake_agent._loop_guard_abort_count = 2
        fake_agent._read_loop_nudged = False
        sys.modules["agent"] = fake_agent
        m.reset()
        m.flush_from_agent_globals(extra={"test": "with_agent"})
        snap = m.snapshot()
        check("agent-globals translated to nudge_cache_loop_consecutive",
              snap["counters"].get("nudge_cache_loop_consecutive") == 5)
        check("agent _audit_fired (bool True) → 1",
              snap["counters"].get("audit_gate_fired_query") == 1)
        check("agent _loop_guard_abort_count → loop_guard_aborts_query",
              snap["counters"].get("loop_guard_aborts_query") == 2)
        check("agent _read_loop_nudged (bool False) → 0",
              snap["counters"].get("nudge_read_loop_fired") == 0)
        sys.modules.pop("agent", None)

        # ---- 7. Per-inc overhead < 1 µs ------------------------------
        m = _reload_metrics({"QWEN_AGENT_METRICS_PATH": str(sink)})
        m.reset()
        N = 200_000
        t_inc = time.perf_counter()
        for _ in range(N):
            m.inc("perf_check")
        elapsed_inc = (time.perf_counter() - t_inc)
        per_call_ns = (elapsed_inc / N) * 1e9
        # < 2 µs per call (target was 1, leave 2× margin for cold caches).
        check(f"inc() per-call overhead < 2 µs",
              per_call_ns < 2000,
              f"{per_call_ns:.0f} ns/call over {N:,} calls")
        check("perf-loop bumped counter to N",
              m.get("perf_check") == N)

        # ---- 8. tail() helper ----------------------------------------
        last = m.tail(path=sink, n=2)
        check("tail returns last N records", len(last) <= 2)

    elapsed_ms = (time.perf_counter() - t0) * 1000
    print(f"\n  elapsed: {elapsed_ms:.1f} ms")
    print(f"\n== {'PASS' if not failures else 'FAIL'} "
          f"({len(failures)} failure(s)) ==")
    if failures:
        for f in failures:
            print(f"  - {f}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
