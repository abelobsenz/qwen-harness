#!/usr/bin/env python3
"""Long-stream cancel-midway memory pathology test.

Companion to:
  - `test_proxy_memory.py`: many short requests, sustained-load RSS
  - `test_proxy_backpressure.py`: one long stream, abrupt cancel,
    propagation timing

This test combines the two: many long streams, each cancelled midway,
measuring the proxy's RSS across cycles. The pathology we want to rule
out: when a client cancels mid-stream, does the proxy retain the
partially-buffered SSE content (in `_StreamAssembler._content_parts`,
`StreamingLoopGuard._buf`, or some intermediate variable) past the end
of `do_POST`?

Setup:
  - Slow fake upstream emits 500 chunks, 5 ms apart.
  - Test client opens a stream, reads 30 chunks (~150 ms in), then
    aggressively closes the connection.
  - Repeat 30 times.
  - Sample proxy RSS every 5 cycles via `ps -o rss=`.

Gates:
  - RSS growth from post-warmup to end-of-test ≤ 30 MB
  - Peak RSS ≤ 75 MB (sanity bound vs the ~35 MB baseline established
    in `test_proxy_memory.py`)

Stdlib only. Cross-platform via `ps`.
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


class SlowUpstream(BaseHTTPRequestHandler):
    """Streams 500 distinct content chunks, 5ms apart. Distinct content
    keeps the proxy's loop_guard from firing — we want to test memory
    bound under partially-consumed long streams, not loop-abort paths."""

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
        try:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            self._send_frame({"choices": [{"delta": {"role": "assistant"}}]})
            for i in range(500):
                # Distinct content per chunk — varied prose so loop_guard
                # never fires. ~120 chars per frame = ~60 KB total if the
                # full stream were consumed.
                content = (f"chunk {i:04d} contains varied factual content "
                           f"about topic number {i*7 % 13} and aspect {i % 5}; "
                           f"semantically distinct from neighbors. ")
                try:
                    self._send_frame({"choices": [{"delta": {"content": content}}]})
                except (BrokenPipeError, ConnectionResetError, OSError):
                    return
                time.sleep(0.005)
            try:
                self._send_frame({"choices": [{"delta": {}, "finish_reason": "stop"}]})
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError, OSError):
                return
        except (BrokenPipeError, ConnectionResetError, OSError):
            return

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


def _open_and_cancel(proxy_port: int, chunks_to_read: int) -> int:
    """Open a streaming request, read `chunks_to_read` SSE content
    frames, then aggressively close the connection. Returns the number
    of chunks actually read before the read loop broke (for diagnostics)."""
    body = json.dumps({
        "messages": [{"role": "user", "content": "stream a long answer"}],
        "stream": True,
    }).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{proxy_port}/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    resp = urllib.request.urlopen(req, timeout=10)
    received = 0
    try:
        for raw in resp:
            line = raw.decode("utf-8", errors="ignore").strip()
            if line.startswith("data:"):
                payload = line[5:].strip()
                if payload and payload != "[DONE]":
                    try:
                        obj = json.loads(payload)
                        delta = obj.get("choices", [{}])[0].get("delta", {})
                        if delta.get("content"):
                            received += 1
                            if received >= chunks_to_read:
                                break
                    except json.JSONDecodeError:
                        pass
    finally:
        try:
            resp.close()
        except Exception:  # noqa: BLE001
            pass
        try:
            resp.fp.close()  # type: ignore[attr-defined]
        except Exception:  # noqa: BLE001
            pass
    return received


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
        with background_server(SlowUpstream, upstream_port):
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

            print(f"== Proxy long-stream-cancel memory test (pid {pid}) ==\n")

            # Warmup: 3 cancelled streams to settle allocator
            for _ in range(3):
                _open_and_cancel(proxy_port, chunks_to_read=30)
            time.sleep(0.2)
            warmup_rss = _rss_kb(pid) or 0
            print(f"  post-warmup RSS: {warmup_rss/1024:.1f} MB")

            # 30 cycles of cancel-midway
            samples: list[tuple[int, int]] = [(0, warmup_rss)]
            cycles = 30
            for i in range(cycles):
                received = _open_and_cancel(proxy_port, chunks_to_read=30)
                if (i + 1) % 5 == 0:
                    rss = _rss_kb(pid) or 0
                    samples.append((i + 1, rss))
                    print(f"  cycle {i+1:2d}: read {received} chunks, "
                          f"RSS={rss/1024:.1f} MB "
                          f"(Δwarmup {((rss - warmup_rss)/1024):+.1f} MB)")

            # Give the proxy a beat to GC the dropped connections.
            time.sleep(0.5)
            final_rss = _rss_kb(pid) or 0
            peak = max(s[1] for s in samples)
            print(f"\n  final RSS: {final_rss/1024:.1f} MB "
                  f"(Δwarmup {((final_rss - warmup_rss)/1024):+.1f} MB)")
            print(f"  peak RSS:  {peak/1024:.1f} MB")
            print(f"  total cycles: {cycles}")

            growth_mb = (final_rss - warmup_rss) / 1024
            peak_mb = peak / 1024
            if growth_mb > 30:
                print(f"\n  [✗] RSS grew by {growth_mb:.1f} MB across "
                      f"{cycles} cancel cycles (gate: 30 MB)")
                failures += 1
            else:
                print(f"\n  [✓] RSS growth {growth_mb:+.1f} MB ≤ 30 MB gate")
            if peak_mb > 75:
                print(f"  [✗] peak RSS {peak_mb:.1f} MB exceeds 75 MB sanity bound")
                failures += 1
            else:
                print(f"  [✓] peak RSS {peak_mb:.1f} MB ≤ 75 MB sanity bound")
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
