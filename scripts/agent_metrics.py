#!/usr/bin/env python3
"""Agent-side metrics: lightweight counter dict + JSONL sink.

The proxy already exposes `/debug/metrics` (qwen_proxy.py:994-1002) but
the AGENT layer has no equivalent: audit-gate fires, nudge counts by
type, retry counts, and loop-guard course-correction fires only show in
the agent's stdout. This module fills that gap with a tiny standalone
counter container that:

  - has zero dependencies (stdlib only)
  - is cheap enough to call on every turn (sub-microsecond per inc)
  - serialises to ~/.qwen/agent_metrics.jsonl on flush()
  - can be safely no-op'd by setting QWEN_AGENT_METRICS=0

Integration model: agent.py imports and calls flush() at the end of
each run_query, wrapped in try/except so any defect here can't crash a
real session. The sink path can be overridden via QWEN_AGENT_METRICS_PATH.

Counter naming convention (snake_case):
  nudge_<class>_fired          : count of nudge fires by class
  audit_gate_fired             : count of audit-gate deferrals
  audit_gate_unsupported_nums  : sum of unsupported-number flags raised
  loop_guard_aborts            : per-query loop-guard hard-cap fires
  retries_5xx                  : count of 5xx retries
  retries_url_error            : count of urllib.error.URLError retries
  compactions                  : count of context compactions triggered

Counters are global per-process and survive across queries unless
`reset()` is called. The intended use is "increment freely, reset on
some boundary (end of query, end of eval iteration)."

Time complexity:
  - inc()    O(1)  — single dict update
  - snapshot() O(N) — N = number of distinct counter names (~10)
  - flush()  O(N)   — one JSONL line append
"""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Any


# Process-global counter dict. The lock guards against concurrent
# increments from threading-enabled call sites (qwen_proxy is threaded;
# agent.py is single-threaded but the agent_graph executor may have
# overlapping nodes that share an interpreter even if dflash is
# semaphored).
_LOCK = threading.Lock()
_COUNTERS: dict[str, int] = {}
_META: dict[str, Any] = {"start_ts": time.time()}

# Toggle and sink path. Read once at import; set ENV before importing
# to override.
_ENABLED = os.environ.get("QWEN_AGENT_METRICS", "1") not in ("0", "false", "False")

_DEFAULT_SINK = Path(os.path.expanduser(
    os.environ.get("QWEN_AGENT_METRICS_PATH",
                    "~/.qwen/agent_metrics.jsonl")))


def is_enabled() -> bool:
    return _ENABLED


def inc(name: str, n: int = 1) -> None:
    """Increment a counter by n. Fast path: <1µs per call on stock laptops.
    Silent no-op when disabled so caller doesn't need to check is_enabled."""
    if not _ENABLED:
        return
    with _LOCK:
        _COUNTERS[name] = int(_COUNTERS.get(name, 0)) + int(n)


def set_value(name: str, value: int) -> None:
    """Set a counter to an absolute value (e.g. when reading agent.py's
    global counters at end-of-query)."""
    if not _ENABLED:
        return
    with _LOCK:
        _COUNTERS[name] = int(value)


def get(name: str, default: int = 0) -> int:
    with _LOCK:
        return int(_COUNTERS.get(name, default))


def snapshot() -> dict[str, Any]:
    """Return a copy of the current counters + metadata. Safe to call
    while another thread is incrementing — copies under the lock."""
    with _LOCK:
        return {
            "ts": time.time(),
            "uptime_s": round(time.time() - _META["start_ts"], 3),
            "counters": dict(_COUNTERS),
        }


def reset() -> None:
    """Zero all counters. Typically called at the start of a top-level
    query so per-query metrics are clean."""
    with _LOCK:
        _COUNTERS.clear()
        _META["start_ts"] = time.time()


def flush(path: Path | str | None = None,
          extra: dict[str, Any] | None = None) -> Path | None:
    """Append one JSONL record to the sink and return the path. Returns
    None when disabled. The directory is created lazily.

    `extra` is merged into the record at the TOP level so callers can
    tag a row with arbitrary identifiers (eval iter id, prompt id, etc.)
    without polluting the counters namespace.
    """
    if not _ENABLED:
        return None
    record = snapshot()
    if extra:
        record.update(extra)
    target = Path(path) if path else _DEFAULT_SINK
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False, default=str))
            f.write("\n")
    except OSError:
        # Disk full, read-only home, permission denied: stay silent.
        # The agent must never fail because metrics couldn't write.
        return None
    return target


def flush_from_agent_globals(extra: dict[str, Any] | None = None) -> Path | None:
    """Convenience: read the relevant globals from `agent` and emit one
    record. Used by agent.py at end-of-query.

    Wrapped here (not in agent.py) so import + attribute resolution
    failures don't crash the agent. If agent isn't importable for any
    reason (test harness, partial install), the call no-ops.
    """
    if not _ENABLED:
        return None
    try:
        import sys as _sys
        agent_mod = _sys.modules.get("agent") or _sys.modules.get("__main__")
        if agent_mod is None:
            return flush(extra=extra)
        for src_name, metric_name in (
            ("_consecutive_all_cached_turns", "nudge_cache_loop_consecutive"),
            ("_consecutive_missing_arg_turns", "nudge_missing_arg_consecutive"),
            ("_consecutive_all_refused_turns", "nudge_refused_consecutive"),
            ("_total_all_refused_turns", "nudge_refused_total"),
            ("_audit_fired", "audit_gate_fired_query"),
            ("_loop_guard_abort_count", "loop_guard_aborts_query"),
            ("_loop_guard_force_terminate", "loop_guard_force_terminated"),
            ("_read_loop_nudged", "nudge_read_loop_fired"),
            ("_stale_post_write_nudged", "nudge_stale_post_write_fired"),
            ("_empty_turn_nudged", "nudge_empty_turn_fired"),
            ("_loop_guard_nudge_fired", "nudge_loop_guard_fired"),
        ):
            val = getattr(agent_mod, src_name, None)
            if isinstance(val, bool):
                set_value(metric_name, 1 if val else 0)
            elif isinstance(val, int):
                set_value(metric_name, val)
    except Exception:  # noqa: BLE001
        # Never let metrics crash the agent.
        return None
    return flush(extra=extra)


# Make it easy for a CLI inspector to dump the latest record without
# importing this module's process state.
def tail(path: Path | str | None = None, n: int = 1) -> list[dict[str, Any]]:
    """Return the last `n` JSONL records (oldest → newest). Useful for
    `python -c "from agent_metrics import tail; print(tail())"`."""
    target = Path(path) if path else _DEFAULT_SINK
    if not target.exists():
        return []
    out: list[dict[str, Any]] = []
    # Memory-efficient: read backwards by lines. For sinks expected to
    # stay small (<10 MB), reading the file once is fine and avoids the
    # complexity of seek-by-byte.
    with target.open("r", encoding="utf-8") as f:
        lines = f.readlines()
    for line in lines[-n:]:
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


if __name__ == "__main__":
    # CLI entry: `python agent_metrics.py` dumps the latest record.
    import sys
    rec = tail(n=1)
    if not rec:
        print("(no metrics yet)")
        sys.exit(0)
    print(json.dumps(rec[0], indent=2, default=str))
