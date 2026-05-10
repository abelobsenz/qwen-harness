#!/usr/bin/env python3
"""Slow-body upload tests — variant of Round 26's slowloris case.

Round 26 case 9 covered "client sends partial headers, never finishes"
(slowloris-style headers). Round 27 added a body-size cap that
short-circuits dishonest Content-Length declarations BEFORE reading.

What's still untested: legitimately-sized body sent SLOWLY. Two
sub-cases matter:

  A. Slow but finishing — 1 KB body sent at 100 bytes/100ms (~1 s
     total). The proxy should accept it. This is the realistic
     case where a user's network is bad but they're still
     uploading.

  B. Stalled — client sends partial body, then stops. The proxy's
     thread should not be held forever; either an OS-level
     timeout reaps it, or the client-side close eventually
     surfaces.

If sub-case (A) passes but (B) hangs the proxy beyond a few
seconds, that's a real production concern: a single hostile client
can monopolize a handler thread by sending data 1 byte at a time.

Findings flow into either a clean-bill-of-health log entry or a
follow-up hardening task.
"""

from __future__ import annotations

import json
import os
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


class FakeUpstream(BaseHTTPRequestHandler):
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
        try:
            self.wfile.write(b'data: {"choices":[{"delta":{"role":"assistant"}}]}\n\n')
            self.wfile.write(b'data: {"choices":[{"delta":{"content":"ok"}}]}\n\n')
            self.wfile.write(b'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\n')
            self.wfile.write(b'data: [DONE]\n\n')
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass


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


def _probe_alive(proxy_port: int, timeout: float = 2.0) -> bool:
    try:
        with urllib.request.urlopen(
            f"http://127.0.0.1:{proxy_port}/v1/models", timeout=timeout
        ) as r:
            return r.status == 200
    except Exception:  # noqa: BLE001
        return False


def _slow_body_request(proxy_port: int, body_bytes: bytes,
                       chunk_size: int, delay_s: float,
                       timeout: float) -> tuple[int | None, float]:
    """Open a connection, send headers + Content-Length declaring
    `len(body_bytes)`, then send the body in `chunk_size` slices with
    `delay_s` between each. Returns (status_code, wall_seconds)."""
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(timeout)
    t0 = time.monotonic()
    try:
        s.connect(("127.0.0.1", proxy_port))
        # Headers
        head = (
            f"POST /v1/chat/completions HTTP/1.0\r\n"
            f"Content-Type: application/json\r\n"
            f"Content-Length: {len(body_bytes)}\r\n"
            f"\r\n"
        ).encode("latin-1")
        s.sendall(head)
        # Body in chunks
        for i in range(0, len(body_bytes), chunk_size):
            s.sendall(body_bytes[i : i + chunk_size])
            if delay_s > 0:
                time.sleep(delay_s)
        # Read response
        chunks: list[bytes] = []
        while True:
            try:
                ch = s.recv(65536)
            except socket.timeout:
                break
            if not ch:
                break
            chunks.append(ch)
        elapsed = time.monotonic() - t0
        full = b"".join(chunks)
        try:
            status_line = full.split(b"\r\n", 1)[0].decode("latin-1")
            return int(status_line.split(" ", 2)[1]), elapsed
        except (IndexError, ValueError):
            return None, elapsed
    except (socket.error, OSError):
        return None, time.monotonic() - t0
    finally:
        s.close()


def main() -> int:
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
    proc = subprocess.Popen(proxy_cmd, stdout=subprocess.DEVNULL,
                            stderr=subprocess.PIPE)

    failures: list[str] = []
    try:
        with background_server(FakeUpstream, upstream_port):
            # Wait for proxy
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline:
                if _probe_alive(proxy_port, timeout=0.5):
                    break
                time.sleep(0.05)
            else:
                print("[✗] proxy did not start"); return 1

            print(f"== Proxy slow-body upload tests (pid {proc.pid}) ==\n")

            # Sub-case A: realistic slow upload (1 KB at 100 B / 100 ms)
            print("[A] realistic slow upload (1 KB at 100 B / 100 ms ≈ 1 s)")
            body = json.dumps({
                "messages": [{"role": "user", "content": "hi" * 100}],
            }).encode()
            # Pad to ~1 KB for predictable chunking
            body = body + b" " * max(0, 1024 - len(body))
            code, elapsed = _slow_body_request(
                proxy_port, body, chunk_size=100, delay_s=0.1, timeout=10.0)
            print(f"    status={code} elapsed={elapsed:.2f}s "
                  f"(body_size={len(body)})")
            if code != 200:
                print(f"    [✗] expected 200, got {code}")
                failures.append("A-status")
            elif elapsed > 5.0:
                print(f"    [✗] slow but finishing upload took {elapsed:.1f}s "
                      f"(expected ~1s, gate 5s)")
                failures.append("A-elapsed")
            else:
                print(f"    [✓] realistic slow upload accepted in {elapsed:.2f}s")

            # Daemon must still be alive
            if not _probe_alive(proxy_port):
                print("    [✗] daemon died")
                failures.append("A-daemon-dead")

            # Sub-case B: very slow upload (10 KB at 100 B / 50 ms ≈ 5 s)
            # Tests a slower-but-still-finishing body to verify the proxy
            # waits patiently rather than timing out prematurely.
            print("\n[B] very slow upload (10 KB at 100 B / 50 ms ≈ 5 s)")
            big_body = json.dumps({
                "messages": [{"role": "user", "content": "x" * 9000}],
            }).encode()
            big_body = big_body + b" " * max(0, 10240 - len(big_body))
            code, elapsed = _slow_body_request(
                proxy_port, big_body, chunk_size=100, delay_s=0.05, timeout=15.0)
            print(f"    status={code} elapsed={elapsed:.2f}s "
                  f"(body_size={len(big_body)})")
            if code != 200:
                print(f"    [✗] expected 200, got {code}")
                failures.append("B-status")
            elif elapsed > 10.0:
                print(f"    [✗] very-slow upload took {elapsed:.1f}s "
                      f"(expected ~5s, gate 10s)")
                failures.append("B-elapsed")
            else:
                print(f"    [✓] very-slow upload accepted in {elapsed:.2f}s")

            if not _probe_alive(proxy_port):
                print("    [✗] daemon died")
                failures.append("B-daemon-dead")

            # Sub-case C: stalled body — declare 1000 bytes, send 100, then
            # close abruptly. The proxy's read should hit EOF / connection
            # reset, raise, and the request handler should die without
            # killing the daemon.
            print("\n[C] stalled body — declare 1000, send 100, close")
            partial = json.dumps({"messages": []}).encode()[:100]
            t0 = time.monotonic()
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(5.0)
            try:
                s.connect(("127.0.0.1", proxy_port))
                head = (
                    "POST /v1/chat/completions HTTP/1.0\r\n"
                    "Content-Type: application/json\r\n"
                    "Content-Length: 1000\r\n"
                    "\r\n"
                ).encode("latin-1")
                s.sendall(head + partial)  # 100 bytes of the declared 1000
                # Don't send the rest — close.
                s.close()
            except (socket.error, OSError):
                pass
            elapsed_c = time.monotonic() - t0
            print(f"    client closed after {elapsed_c*1000:.0f}ms with 100/1000 "
                  f"bytes sent")
            # Give the proxy a moment to clean up the half-read request,
            # then check it's still alive.
            time.sleep(0.5)
            if not _probe_alive(proxy_port, timeout=2.0):
                print("    [✗] daemon died after stalled-body abuse")
                failures.append("C-daemon-dead")
            else:
                print("    [✓] daemon survived a half-uploaded request")

            # Sub-case D: many concurrent slow uploads — does the proxy
            # remain responsive to OTHER clients while N are uploading
            # slowly? Tests that one slow client doesn't starve the
            # ThreadingHTTPServer.
            print("\n[D] concurrent slow uploads × 4 + healthy probe")
            results = {"slow": [], "alive_during": False}

            def slow_uploader(idx: int) -> None:
                code, elapsed = _slow_body_request(
                    proxy_port, body, chunk_size=50, delay_s=0.05,
                    timeout=20.0)
                results["slow"].append((idx, code, elapsed))

            threads = [threading.Thread(target=slow_uploader, args=(i,))
                       for i in range(4)]
            for t in threads:
                t.start()
            time.sleep(0.3)  # let the slow uploads get going
            # While the slow uploads are happening, hit /v1/models
            results["alive_during"] = _probe_alive(proxy_port, timeout=2.0)
            for t in threads:
                t.join()

            n_ok = sum(1 for _, code, _ in results["slow"] if code == 200)
            print(f"    {n_ok}/4 slow uploads succeeded, "
                  f"alive-during-slow={results['alive_during']}")
            if n_ok != 4:
                print(f"    [✗] expected all 4 to succeed")
                failures.append("D-not-all-ok")
            elif not results["alive_during"]:
                print(f"    [✗] proxy was unresponsive while slow uploads "
                      f"were in flight (thread starvation)")
                failures.append("D-unresponsive")
            else:
                print(f"    [✓] all 4 slow uploads succeeded, proxy stayed "
                      f"responsive throughout")
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            proc.kill()

    print(f"\n== {'PASS' if not failures else 'FAIL'} "
          f"({len(failures)} failure(s)) ==")
    if failures:
        for f in failures:
            print(f"  - {f}")
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
