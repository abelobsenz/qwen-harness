#!/usr/bin/env python3
"""Long-running low-rate stress test for the proxy.

The existing memory tests cover three pathologies:
  - `test_proxy_memory.py`: 1000 short requests, 50ms apart (~3s wall)
  - `test_proxy_long_stream_memory.py`: 30 long-stream cancels (~5s wall)
  - `test_proxy_concurrent.py`: 20× concurrent burst (~1s wall)

All three finish in seconds. This test fills a different niche:
**low-rate, long-duration**. 1 request per second for 90 seconds, sampling
proxy RSS every 10 seconds. Detects leak shapes that 1000-fast doesn't:
  - Per-connection cleanup that delays GC behind a timer
  - Logger handler buildup
  - Thread-pool churn
  - Connection state in `socket` / `urllib` that accumulates across
    long inter-request gaps

Each sample is its own short request; we don't reuse connections (no
explicit keepalive pool), so each one creates a fresh handler thread on
the proxy side and a fresh socket. After 90 cycles we'll have spawned 90
threads, all of which should be reaped cleanly. If they don't, RSS will
drift up.

Stdlib only; cross-platform via `ps -o rss=`.
"""

from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import sys
import threading
import time
import urllib.request
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, HTTPServer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class _Quiet(HTTPServer):
    def handle_error(self, request, client_address):
        import sys as _s
        if isinstance(_s.exc_info()[1], (BrokenPipeError, ConnectionResetError, OSError)):
            return
        super().handle_error(request, client_address)


# Trivial fake upstream — instant healthy reply so the test isolates the
# proxy's per-request lifecycle, not upstream timing.
_HEALTHY_TEXT = (
    "Quantum systems exhibit superposition until measured. Entanglement "
    "links the states of multiple qubits regardless of distance. "
)


class FastUpstream(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.0"

    def log_message(self, fmt, *args):  # noqa: A003
        return

    def do_GET(self):  # noqa: N802
        if self.path == "/v1/models":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"data":[{"id":"test","object":"model"}]}')
            return
        self.send_response(404); self.end_headers()

    def do_POST(self):  # noqa: N802
        length = int(self.headers.get("Content-Length", "0"))
        _ = self.rfile.read(length) if length else b""
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        self._send_frame({"choices": [{"delta": {"role": "assistant"}}]})
        try:
            for i in range(0, len(_HEALTHY_TEXT), 32):
                self._send_frame({
                    "choices": [{"delta": {"content": _HEALTHY_TEXT[i:i+32]}}]
                })
            self._send_frame({"choices": [{"delta": {}, "finish_reason": "stop"}]})
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass

    def _send_frame(self, obj):
        self.wfile.write(f"data: {json.dumps(obj)}\n\n".encode())
        self.wfile.flush()


@contextmanager
def background_server(handler_cls, port):
    server = _Quiet(("127.0.0.1", port), handler_cls)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()


def _rss_kb(pid: int) -> int | None:
    try:
        out = subprocess.check_output(["ps", "-o", "rss=", "-p", str(pid)],
                                      stderr=subprocess.DEVNULL).strip()
        return int(out) if out else None
    except (subprocess.CalledProcessError, ValueError):
        return None


def _open_thread_count(pid: int) -> int | None:
    """Return current open thread count for the process; None if unsupported."""
    try:
        # macOS: -M lists threads
        out = subprocess.check_output(["ps", "-M", "-p", str(pid)],
                                      stderr=subprocess.DEVNULL).decode("utf-8", "ignore")
        # Header line + N thread lines.
        return max(0, len(out.strip().splitlines()) - 1)
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def _issue_one(proxy_port: int) -> None:
    body = json.dumps({"messages": [{"role": "user", "content": "hi"}],
                       "stream": False}).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{proxy_port}/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=10) as resp:
        resp.read()


def main() -> int:
    if shutil.which("ps") is None:
        print("ps(1) not on PATH — skipping memory test")
        return 0

    upstream_port = _free_port()
    proxy_port = _free_port()
    py = "/Users/abelobsenz/dev/qwen36_MTP/venv/bin/python"
    proxy_cmd = [
        py,
        "/Users/abelobsenz/dev/qwen36_MTP/scripts/qwen_proxy.py",
        "--listen-host", "127.0.0.1",
        "--listen-port", str(proxy_port),
        "--upstream", f"http://127.0.0.1:{upstream_port}",
    ]
    proc = subprocess.Popen(proxy_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    pid = proc.pid

    failures = 0
    try:
        with background_server(FastUpstream, upstream_port):
            # Wait for proxy
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline:
                try:
                    with urllib.request.urlopen(
                        f"http://127.0.0.1:{proxy_port}/v1/models", timeout=0.5
                    ) as r:
                        if r.status == 200:
                            break
                except Exception:  # noqa: BLE001
                    time.sleep(0.05)
            else:
                print("[✗] proxy did not start"); return 1

            print(f"== Proxy long-idle stress (pid {pid}, "
                  f"1 req/sec for 90 s) ==\n")

            # Warmup
            for _ in range(3):
                _issue_one(proxy_port)
            time.sleep(0.3)
            warmup_rss = _rss_kb(pid) or 0
            warmup_threads = _open_thread_count(pid)
            print(f"  post-warmup RSS: {warmup_rss/1024:.1f} MB"
                  f"{' threads=' + str(warmup_threads) if warmup_threads else ''}")

            # Drive at 1 req/sec for ~90 seconds, sampling every 10 sec
            duration_s = 90
            samples: list[tuple[int, int, int | None]] = [
                (0, warmup_rss, warmup_threads)
            ]
            start = time.monotonic()
            target = start + 1.0
            t_count = 0
            while time.monotonic() - start < duration_s:
                # Wait until target tick (yields scheduler so the proxy's
                # cleanup paths can run).
                now = time.monotonic()
                if now < target:
                    time.sleep(target - now)
                _issue_one(proxy_port)
                t_count += 1
                target += 1.0
                # Sample every 10 ticks (~10 seconds)
                if t_count % 10 == 0:
                    elapsed = int(time.monotonic() - start)
                    rss = _rss_kb(pid) or 0
                    threads = _open_thread_count(pid)
                    samples.append((elapsed, rss, threads))
                    print(f"  t={elapsed:3d}s  reqs={t_count:3d}  "
                          f"RSS={rss/1024:.1f} MB  "
                          f"threads={threads}  "
                          f"Δwarmup={((rss - warmup_rss)/1024):+.1f} MB")

            time.sleep(0.5)  # let GC settle
            final_rss = _rss_kb(pid) or 0
            final_threads = _open_thread_count(pid)
            peak = max(s[1] for s in samples)
            peak_threads = max(
                (s[2] for s in samples if s[2] is not None), default=None)
            print(f"\n  total requests: {t_count}")
            print(f"  final RSS: {final_rss/1024:.1f} MB "
                  f"(Δwarmup {((final_rss - warmup_rss)/1024):+.1f} MB)")
            print(f"  peak RSS:  {peak/1024:.1f} MB")
            if peak_threads is not None:
                print(f"  peak threads: {peak_threads}")

            growth_mb = (final_rss - warmup_rss) / 1024
            if growth_mb > 5:
                print(f"\n  [✗] RSS grew by {growth_mb:.1f} MB across "
                      f"{t_count} requests over {duration_s}s "
                      f"(gate: 5 MB)")
                failures += 1
            else:
                print(f"\n  [✓] RSS growth {growth_mb:+.1f} MB ≤ 5 MB gate")
            # Thread count gate: should not climb unbounded. The proxy uses
            # ThreadingHTTPServer with daemon threads, so we expect threads
            # to come and go. Peak should be < 30 (warmup + a few in-flight).
            if peak_threads is not None and peak_threads > 30:
                print(f"  [✗] peak thread count {peak_threads} > 30 — "
                      f"thread-pool churn / leak")
                failures += 1
            elif peak_threads is not None:
                print(f"  [✓] peak threads {peak_threads} ≤ 30 (no thread leak)")
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            proc.kill()

    print(f"\n== {'PASS' if failures == 0 else 'FAIL'} ({failures} failure(s)) ==")
    return failures


if __name__ == "__main__":
    sys.exit(main())
