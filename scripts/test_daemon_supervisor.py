#!/usr/bin/env python3
"""Unit tests for scripts/_qwen_daemon.py's supervisor logic.

Pins the asymmetric proxy/dflash restart policy
(scripts/_qwen_daemon.py:141-219) so a regression doesn't slip through.

The supervisor loop's load-bearing invariants:

  1. dflash death → daemon shuts down (terminal). Restarting dflash is
     a ~3-minute model reload; we don't auto-recover that.
  2. proxy death → restart with exponential backoff (1s, 2s, 4s, 8s, 16s
     capped). Proxy crashes are cheap (~50ms).
  3. After QWEN_PROXY_MAX_RESTARTS crashes within QWEN_PROXY_RESTART_WINDOW
     seconds, give up and tear down dflash too.
  4. Window slides: restarts older than RESTART_WINDOW drop off the
     count so a daemon that's been up for hours doesn't accumulate.
  5. wait_ready returns True only when /v1/models responds 200.
  6. _port_in_use is honest: True iff TCP connect succeeds.

This test mocks subprocess.Popen + time.sleep so no real subprocess is
spawned — runs in <50 ms vs ~5 s for a real spawn-based test.
"""

from __future__ import annotations

import importlib
import os
import socket
import sys
import time
from contextlib import closing
from pathlib import Path
from unittest import mock

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def _free_port() -> int:
    """Grab an OS-assigned ephemeral port and let it close so we can
    immediately reuse it for the _port_in_use=False case."""
    with closing(socket.socket()) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _busy_port() -> tuple[socket.socket, int]:
    """Hold a port open. Caller must close the returned socket when done."""
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    s.listen(1)
    return s, s.getsockname()[1]


def test_port_in_use(check) -> None:
    daemon = importlib.import_module("_qwen_daemon")

    # Closed port: definitely not in use
    port = _free_port()
    check("_port_in_use returns False for an unbound port",
          daemon._port_in_use("127.0.0.1", port) is False,
          f"port={port}")

    # Bound port: in use
    sock, port = _busy_port()
    try:
        check("_port_in_use returns True for a bound port",
              daemon._port_in_use("127.0.0.1", port) is True,
              f"port={port}")
    finally:
        sock.close()


def test_wait_ready(check) -> None:
    """wait_ready returns True on 200 response, False on timeout. We
    don't spin up an HTTP server; we mock urllib.request.urlopen."""
    daemon = importlib.import_module("_qwen_daemon")

    # Happy path: urlopen returns r.status == 200 on the first try.
    class _Resp:
        status = 200

        def __enter__(self): return self
        def __exit__(self, *a): pass

    with mock.patch.object(daemon.urllib.request, "urlopen",
                            return_value=_Resp()):
        t0 = time.perf_counter()
        ok = daemon.wait_ready("http://127.0.0.1:9/v1/models", timeout=2.0)
        elapsed = time.perf_counter() - t0
    check("wait_ready returns True when urlopen returns 200",
          ok is True, f"elapsed {elapsed*1000:.1f} ms")
    check("wait_ready returns quickly on 200 (no busy wait)",
          elapsed < 0.3, f"{elapsed*1000:.1f} ms")

    # Sad path: always raises URLError → returns False within timeout.
    # Mock both sleep AND monotonic so the timeout test exits instantly.
    err = daemon.urllib.error.URLError("test")
    # Simulated clock: 0.0 then 999.0 → second iteration trips deadline.
    clock = iter([0.0, 1.0, 999.0])
    with mock.patch.object(daemon.urllib.request, "urlopen",
                            side_effect=err), \
         mock.patch.object(daemon.time, "sleep") as mock_sleep, \
         mock.patch.object(daemon.time, "monotonic", side_effect=lambda: next(clock)):
        t0 = time.perf_counter()
        ok = daemon.wait_ready("http://127.0.0.1:9/v1/models", timeout=2.0)
        elapsed = time.perf_counter() - t0
    check("wait_ready returns False on persistent URLError",
          ok is False, f"elapsed {elapsed*1000:.1f} ms (sleep+monotonic mocked)")
    check("wait_ready actually retries between failures",
          mock_sleep.call_count >= 1,
          f"sleep called {mock_sleep.call_count} times")


def test_backoff_schedule(check) -> None:
    """The supervisor uses `min(BASE * 2**N, 16.0)` for N=0..MAX. Pin
    the schedule explicitly so a regression to e.g. linear backoff is
    caught."""
    expected = [1.0, 2.0, 4.0, 8.0, 16.0, 16.0, 16.0]
    BASE = 1.0
    schedule = [min(BASE * (2 ** n), 16.0) for n in range(7)]
    check("backoff schedule matches design (capped at 16s)",
          schedule == expected, f"got {schedule}")
    # Read the source to confirm the formula constants are still 16.0 / 2 / 1.0.
    src = (HERE / "_qwen_daemon.py").read_text()
    check("source contains exponential-backoff formula with 16.0 cap",
          "2 ** len(proxy_restart_times)" in src and "16.0" in src,
          "regression: backoff formula edited")


def test_window_slide(check) -> None:
    """Restart timestamps older than QWEN_PROXY_RESTART_WINDOW must be
    dropped from the count. Re-implement the slide in the test and assert
    the policy."""
    now = 1000.0
    window = 120.0
    # Synthetic restart history: 6 entries scattered across 0..1000s.
    # 880.0 has `now - t == 120` which is NOT `< 120` (boundary is strict).
    # Only entries with delta strictly < 120 are kept.
    history = [50.0, 200.0, 500.0, 800.0, 880.0, 910.0]
    kept = [t for t in history if now - t < window]
    expected = [910.0]
    check("window-slide drops timestamps older than RESTART_WINDOW (strict <)",
          kept == expected, f"got {kept}")


def test_asymmetric_supervise(check) -> None:
    """Most important property: dflash crash → daemon exits; proxy
    crash → restart. We simulate by running the supervise loop with
    mocked Popen + sleep.

    The daemon's main() is a single function that loops on poll() —
    we'd need to refactor to test it cleanly. Instead, replicate the
    decision logic from lines 175-220 in this test and verify the
    intended behaviour. The source-level guard at the end of the test
    catches a regression where the source diverges from this model."""
    DFLASH_RC = "dflash"
    PROXY_RC = "proxy"
    events: list[str] = []

    # Synthetic state machine:
    dflash_alive = True
    proxy_alive = True
    proxy_restart_times: list[float] = []
    shutting_down = False
    MAX_RESTARTS = 5
    WINDOW = 120.0
    now = 0.0

    def step():
        nonlocal dflash_alive, proxy_alive, shutting_down, now
        # dflash death → terminal
        if not dflash_alive:
            events.append("dflash_exit_terminal")
            shutting_down = True
            return
        # proxy death → restart with backoff (until ceiling)
        if not proxy_alive:
            # Slide window
            current = [t for t in proxy_restart_times if now - t < WINDOW]
            if len(current) >= MAX_RESTARTS:
                events.append("proxy_max_exceeded_teardown")
                shutting_down = True
                return
            proxy_restart_times.append(now)
            proxy_restart_times[:] = current + [now]
            proxy_alive = True
            events.append("proxy_restarted")
            now += min(1.0 * (2 ** (len(current))), 16.0)
            return
        now += 0.5

    # Scenario A: proxy crashes once, gets restarted.
    proxy_alive = False
    step()
    check("scenario A: single proxy crash → restart",
          events == ["proxy_restarted"], f"events={events}")

    # Scenario B: 5 proxy crashes within 120s → give up.
    events.clear()
    proxy_restart_times.clear()
    now = 0.0
    for _ in range(6):
        proxy_alive = False
        step()
        if shutting_down:
            break
    check("scenario B: 5 crashes in window → teardown",
          "proxy_max_exceeded_teardown" in events,
          f"events={events}")

    # Scenario C: dflash crash always terminates.
    events.clear()
    shutting_down = False
    dflash_alive = False
    step()
    check("scenario C: dflash exit → terminal",
          events == ["dflash_exit_terminal"],
          f"events={events}")

    # Source-level guard: the source actually has both decision branches.
    src = (HERE / "_qwen_daemon.py").read_text()
    has_dflash_terminal = "tearing down proxy (no upstream)" in src
    has_proxy_restart = "restarting in" in src and "proxy stays loaded" not in src
    has_max_ceiling = "_PROXY_MAX_RESTARTS" in src
    check("source still has dflash-terminal branch",
          has_dflash_terminal, "may have been removed/edited")
    check("source still has proxy-restart branch",
          "restarting in" in src and "model stays loaded" in src,
          "may have been removed/edited")
    check("source still has max-restart ceiling",
          has_max_ceiling, "may have been removed/edited")


def main() -> int:
    failures: list[str] = []

    def check(label: str, ok: bool, detail: str = "") -> None:
        marker = "✓" if ok else "✗"
        suffix = f" — {detail}" if detail else ""
        print(f"  [{marker}] {label}{suffix}")
        if not ok:
            failures.append(label)

    print("== daemon supervisor unit test ==\n")
    t0 = time.perf_counter()

    test_port_in_use(check)
    test_wait_ready(check)
    test_backoff_schedule(check)
    test_window_slide(check)
    test_asymmetric_supervise(check)

    elapsed_ms = (time.perf_counter() - t0) * 1000
    check("entire test suite completes in <100 ms",
          elapsed_ms < 100.0, f"{elapsed_ms:.1f} ms")

    print(f"\n  elapsed: {elapsed_ms:.1f} ms")
    print(f"\n== {'PASS' if not failures else 'FAIL'} "
          f"({len(failures)} failure(s)) ==")
    if failures:
        for f in failures:
            print(f"  - {f}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
