#!/usr/bin/env python3
"""Mid-rate sustained-load stress: 4 concurrent workers, 30 seconds.

Existing memory tests cover four patterns:
  - test_proxy_memory.py: 1000 sequential reqs in ~3s (~333 r/s burst)
  - test_proxy_concurrent.py: 20× concurrent in one burst (~1s wall)
  - test_proxy_long_idle_memory.py: 1 r/s × 90s sequential
  - test_proxy_long_stream_memory.py: 30 long-stream cancels

Gap: **sustained moderate concurrency**. The burst test runs <3s
with low parallelism; the concurrent test runs <1s with high
parallelism; the idle test runs 90s with no parallelism. None
exercises "4 simultaneous clients hammering the proxy for 30
seconds" — exactly what a busy interactive user generates with a
multi-tab session.

This test:
  - 4 worker threads, each issuing requests as fast as the proxy
    will respond, for 30 seconds wall-clock.
  - Sample proxy RSS + thread count every 5 seconds.
  - Gate: RSS growth ≤ 10 MB, peak threads ≤ 50.

Threads are higher than other tests because 4 concurrent clients
expect 4-8 in-flight handler threads on the proxy at any moment.

Stdlib only.
"""

from __future__ import annotations

import concurrent.futures
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


_HEALTHY_TEXT = (
    "Quantum systems exhibit superposition until measured. "
    "Entanglement links qubit states regardless of distance. "
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
    try:
        out = subprocess.check_output(["ps", "-M", "-p", str(pid)],
                                      stderr=subprocess.DEVNULL).decode("utf-8", "ignore")
        return max(0, len(out.strip().splitlines()) - 1)
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def _worker_loop(proxy_port: int, deadline: float, counter: list[int],
                 errors: list[str], idx: int) -> None:
    """One worker thread: issue requests as fast as the proxy responds
    until `deadline` (monotonic time)."""
    body = json.dumps({"messages": [{"role": "user",
                                     "content": f"hi from worker {idx}"}],
                       "stream": False}).encode()
    while time.monotonic() < deadline:
        req = urllib.request.Request(
            f"http://127.0.0.1:{proxy_port}/v1/chat/completions",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                resp.read()
            counter[idx] += 1
        except Exception as e:  # noqa: BLE001
            errors.append(f"worker {idx}: {type(e).__name__}: {e}")
            return  # one failure terminates that worker so we don't spam


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
            deadline_boot = time.monotonic() + 5
            while time.monotonic() < deadline_boot:
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

            print(f"== Proxy mid-rate sustained-load (pid {pid}, "
                  f"4 workers × 30 s) ==\n")

            # Brief warmup
            for _ in range(5):
                req = urllib.request.Request(
                    f"http://127.0.0.1:{proxy_port}/v1/chat/completions",
                    data=b'{"messages":[{"role":"user","content":"warmup"}],"stream":false}',
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=10) as r:
                    r.read()
            time.sleep(0.3)
            warmup_rss = _rss_kb(pid) or 0
            warmup_threads = _open_thread_count(pid) or 0
            print(f"  post-warmup RSS: {warmup_rss/1024:.1f} MB  threads={warmup_threads}")

            # Run 4 workers × 30s
            n_workers = 4
            duration_s = 30
            counters = [0] * n_workers
            errors: list[str] = []
            run_deadline = time.monotonic() + duration_s

            with concurrent.futures.ThreadPoolExecutor(
                    max_workers=n_workers) as pool:
                for i in range(n_workers):
                    pool.submit(_worker_loop, proxy_port, run_deadline,
                                counters, errors, i)

                # Sample proxy RSS + threads every 5s while workers run
                samples: list[tuple[int, int, int | None]] = [
                    (0, warmup_rss, warmup_threads)
                ]
                start = time.monotonic()
                next_sample = start + 5.0
                while time.monotonic() < run_deadline:
                    sleep_for = min(0.5, max(0, run_deadline - time.monotonic()))
                    time.sleep(sleep_for)
                    if time.monotonic() >= next_sample:
                        elapsed = int(time.monotonic() - start)
                        rss = _rss_kb(pid) or 0
                        threads = _open_thread_count(pid)
                        samples.append((elapsed, rss, threads))
                        n_done = sum(counters)
                        rate = n_done / max(elapsed, 1)
                        print(f"  t={elapsed:2d}s  total_reqs={n_done:5d}  "
                              f"rate={rate:5.1f} r/s  RSS={rss/1024:.1f} MB  "
                              f"threads={threads}")
                        next_sample += 5.0
                # Workers exit when run_deadline passes; ThreadPoolExecutor
                # context manager waits for them.

            time.sleep(0.5)
            final_rss = _rss_kb(pid) or 0
            final_threads = _open_thread_count(pid) or 0
            total_done = sum(counters)
            print(f"\n  total requests: {total_done} (per worker: {counters})")
            print(f"  errors: {len(errors)}")
            print(f"  final RSS: {final_rss/1024:.1f} MB "
                  f"(Δwarmup {((final_rss - warmup_rss)/1024):+.1f} MB)")
            peak_rss = max(s[1] for s in samples)
            peak_threads = max(
                (s[2] for s in samples if s[2] is not None),
                default=warmup_threads)
            print(f"  peak RSS:  {peak_rss/1024:.1f} MB")
            print(f"  peak threads: {peak_threads}")
            print(f"  effective rate: {total_done/duration_s:.1f} r/s")

            if errors:
                print(f"\n  [✗] {len(errors)} worker errors (first 3):")
                for e in errors[:3]:
                    print(f"      {e}")
                failures += 1

            growth_mb = (final_rss - warmup_rss) / 1024
            if growth_mb > 10:
                print(f"\n  [✗] RSS grew by {growth_mb:.1f} MB across "
                      f"{total_done} concurrent reqs (gate: 10 MB)")
                failures += 1
            else:
                print(f"\n  [✓] RSS growth {growth_mb:+.1f} MB ≤ 10 MB gate")

            if peak_threads > 50:
                print(f"  [✗] peak threads {peak_threads} > 50 — possible "
                      f"thread leak under sustained concurrency")
                failures += 1
            else:
                print(f"  [✓] peak threads {peak_threads} ≤ 50 (bounded "
                      f"under {n_workers}-way concurrent load)")

            # Sanity: we should have done at least 100 reqs in 30s × 4 workers
            min_expected = n_workers * 25  # ~0.83 r/s/worker minimum
            if total_done < min_expected:
                print(f"  [✗] only {total_done} reqs in {duration_s}s — "
                      f"expected at least {min_expected}")
                failures += 1
            else:
                print(f"  [✓] throughput sanity: "
                      f"{total_done/duration_s:.0f} r/s ≥ "
                      f"{min_expected/duration_s:.0f} r/s baseline")
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
