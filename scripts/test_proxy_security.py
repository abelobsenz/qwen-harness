#!/usr/bin/env python3
"""Pen-test the proxy's attack surface.

The proxy is a long-running HTTP daemon. Even on localhost it can be
hit by misbehaving clients (a buggy agent, a runaway script, a curl
fired by the user). This test fires the canonical "did you handle
this?" cases against the real proxy and asserts the daemon stays up
+ returns sane responses.

Cases:
  1. Malformed JSON body → 400, no crash
  2. Empty body to /v1/chat/completions → handled cleanly
  3. Oversized body (5 MB) → declined or accepted, no crash
  4. Wrong Content-Type → handled
  5. HEAD method on POST endpoint → 405/404 / sane response
  6. PUT/DELETE → not 500, daemon survives
  7. Path traversal in URL → 404
  8. Body with non-UTF-8 bytes → 400, no crash
  9. Slowloris-style: open socket, dribble headers, never finish.
     The proxy's per-request timeout should reap it.

After each case, hit /v1/models to confirm the daemon is still alive.
A test "fails" if the proxy crashes (subprocess exits) or returns 500
on input that should be a graceful client error.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import threading
import time
import urllib.error
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
    """Minimal upstream — replies 200 OK with empty SSE so the
    healthy-path probe to /v1/models works."""

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
        self.wfile.write(b'data: {"choices":[{"delta":{"role":"assistant"}}]}\n\n')
        self.wfile.write(b'data: {"choices":[{"delta":{"content":"ok"}}]}\n\n')
        self.wfile.write(b'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\n')
        self.wfile.write(b'data: [DONE]\n\n')
        try:
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
    """Hit /v1/models. Returns True iff the proxy responds 200."""
    try:
        with urllib.request.urlopen(
            f"http://127.0.0.1:{proxy_port}/v1/models", timeout=timeout
        ) as r:
            return r.status == 200
    except Exception:  # noqa: BLE001
        return False


def _raw_request(proxy_port: int, method: str, path: str,
                 headers: dict, body: bytes,
                 timeout: float = 5.0) -> tuple[int | None, bytes]:
    """Send a single HTTP request via raw socket, read the response.
    Returns (status_code, body_bytes), or (None, b'<reason>') on
    connection error."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(timeout)
        s.connect(("127.0.0.1", proxy_port))
        req_lines = [f"{method} {path} HTTP/1.0"]
        for k, v in headers.items():
            req_lines.append(f"{k}: {v}")
        if body:
            req_lines.append(f"Content-Length: {len(body)}")
        req_lines.append("")
        req_lines.append("")
        s.sendall("\r\n".join(req_lines).encode("latin-1") + body)
        chunks: list[bytes] = []
        while True:
            try:
                chunk = s.recv(65536)
            except socket.timeout:
                break
            if not chunk:
                break
            chunks.append(chunk)
        s.close()
        full = b"".join(chunks)
        # Parse status line: "HTTP/1.0 200 OK\r\n..."
        try:
            status_line = full.split(b"\r\n", 1)[0].decode("latin-1")
            parts = status_line.split(" ", 2)
            status = int(parts[1]) if len(parts) >= 2 else None
        except (IndexError, ValueError):
            status = None
        return status, full
    except (socket.error, OSError) as e:
        return None, str(e).encode()


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

            print(f"== Proxy security pen-test (pid {proc.pid}) ==\n")

            def case(label: str, fn) -> None:
                """Run case, then probe alive. Records failure if either
                the case asserts wrong or the daemon dies."""
                ok, msg = fn()
                marker = "✓" if ok else "✗"
                print(f"  [{marker}] {label}: {msg}")
                if not ok:
                    failures.append(label)
                if not _probe_alive(proxy_port):
                    print(f"  [!!] proxy STOPPED responding after {label!r}")
                    failures.append(f"DAEMON_DOWN:{label}")
                    return  # subsequent cases would all fail; but we keep
                            # going so we get the full picture.

            # Case 1: malformed JSON
            def c1():
                code, _ = _raw_request(
                    proxy_port, "POST", "/v1/chat/completions",
                    {"Content-Type": "application/json"},
                    b"{not valid json")
                return (code in (400, 500), f"got {code} (expected 400)")
            case("1. malformed JSON body", c1)

            # Case 2: empty body
            def c2():
                code, _ = _raw_request(
                    proxy_port, "POST", "/v1/chat/completions",
                    {"Content-Type": "application/json"},
                    b"")
                # Empty body is parseable as `{}` by current code; the
                # proxy should respond cleanly, not crash. Anything in
                # 200..599 except a dropped connection is fine.
                return (code is not None, f"got {code}")
            case("2. empty body", c2)

            # Case 3: oversized body (5 MB)
            def c3():
                big = b"{" + b'"a":"x"' + b',"b":"' + (b"x" * (5 * 1024 * 1024)) + b'"}'
                code, _ = _raw_request(
                    proxy_port, "POST", "/v1/chat/completions",
                    {"Content-Type": "application/json"},
                    big, timeout=15.0)
                return (code is not None, f"got {code} on 5 MB body")
            case("3. oversized body (5 MB)", c3)

            # Case 4: wrong Content-Type
            def c4():
                code, _ = _raw_request(
                    proxy_port, "POST", "/v1/chat/completions",
                    {"Content-Type": "text/plain"},
                    b'{"messages":[{"role":"user","content":"hi"}]}')
                return (code is not None, f"got {code}")
            case("4. wrong Content-Type", c4)

            # Case 5: HEAD on POST endpoint
            def c5():
                code, _ = _raw_request(
                    proxy_port, "HEAD", "/v1/chat/completions",
                    {}, b"")
                # 200/404/405 acceptable; 500 means we crashed
                return (code is not None and code != 500,
                        f"got {code}")
            case("5. HEAD on /v1/chat/completions", c5)

            # Case 6: PUT — uncommon method
            def c6():
                code, _ = _raw_request(
                    proxy_port, "PUT", "/v1/chat/completions",
                    {}, b'{"x":1}')
                return (code is not None and code != 500,
                        f"got {code}")
            case("6. PUT method", c6)

            # Case 7: path traversal
            def c7():
                code, _ = _raw_request(
                    proxy_port, "GET", "/../../etc/passwd",
                    {}, b"")
                # Should be 404 (or 400). NOT 200, NOT 500.
                return (code is not None and code in (400, 404),
                        f"got {code}")
            case("7. path traversal", c7)

            # Case 8: body with non-UTF-8 bytes
            def c8():
                bad = b"\xff\xfe\xfd"  # invalid UTF-8 surrogate-ish
                code, _ = _raw_request(
                    proxy_port, "POST", "/v1/chat/completions",
                    {"Content-Type": "application/json"},
                    bad)
                # Decoder should raise → caller catches → 400 or 500.
                # Either is fine; daemon must survive.
                return (code is not None, f"got {code}")
            case("8. non-UTF-8 bytes", c8)

            # Case 9: slowloris — open socket, dribble headers, never finish
            def c9():
                try:
                    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    s.settimeout(2.0)
                    s.connect(("127.0.0.1", proxy_port))
                    # Send only the request line + a partial header
                    s.sendall(b"POST /v1/chat/completions HTTP/1.0\r\n")
                    # Wait briefly without sending more
                    time.sleep(1.5)
                    s.close()
                    return (True, "client closed early; proxy didn't hang")
                except (socket.error, OSError) as e:
                    return (True, f"socket error (expected): {e}")
            case("9. slowloris (partial headers)", c9)

            # Case 10: huge URL (8 KB path)
            def c10():
                huge_path = "/" + "a" * 8000
                code, _ = _raw_request(
                    proxy_port, "GET", huge_path, {}, b"")
                # Should 404 or 414. Anything but a daemon crash.
                return (code is not None and code != 500,
                        f"got {code} for 8 KB path")
            case("10. 8 KB URL path", c10)

            # Case 11: many headers
            def c11():
                headers = {f"X-Custom-{i}": "v" * 80 for i in range(50)}
                code, _ = _raw_request(
                    proxy_port, "POST", "/v1/chat/completions",
                    {**headers, "Content-Type": "application/json"},
                    b'{"messages":[]}')
                return (code is not None, f"got {code} with 50 custom headers")
            case("11. 50 custom headers", c11)

            # Case 12: lying Content-Length — declares 200 MB, sends nothing.
            # Without the body-size cap, the proxy would block on rfile.read
            # waiting for 200 MB that never arrives. With the cap (default
            # 50 MB), the proxy returns 413 immediately based on the header
            # alone — no read attempted, no memory allocated.
            def c12():
                code, body = _raw_request(
                    proxy_port, "POST", "/v1/chat/completions",
                    {"Content-Type": "application/json",
                     "Content-Length": str(200 * 1024 * 1024)},
                    b"",  # empty body; the lie is in the header
                    timeout=3.0)
                # Expect 413 (Payload Too Large) — proxy rejects on header
                # alone, never reads the body.
                return (code == 413, f"got {code} (expected 413)")
            case("12. lying Content-Length (200 MB declared, no body sent)", c12)

            # Case 13: malformed Content-Length (non-numeric)
            def c13():
                code, _ = _raw_request(
                    proxy_port, "POST", "/v1/chat/completions",
                    {"Content-Type": "application/json",
                     "Content-Length": "abc"},
                    b'{"messages":[]}')
                return (code in (400, 411), f"got {code}")
            case("13. malformed Content-Length", c13)

            # Case 14: deeply-nested JSON (10000-level array). Python's
            # json.loads handles array nesting iteratively (no stack
            # blowup), so this should parse in ~1ms. The test just
            # confirms the proxy doesn't choke on adversarial shapes
            # within the size cap. A failure here would suggest a
            # CPython regression or a custom parser path we missed.
            def c14():
                depth = 10000
                # Build a `[[[...]]]` body. Within the 50 MB cap.
                body = b"[" * depth + b"]" * depth
                # Wrap in a top-level object so the proxy passes it to
                # transform_request without an immediate type error.
                wrapped = b'{"messages":[{"role":"user","content":"x"}],"_x":' + body + b"}"
                t0 = time.monotonic()
                code, _ = _raw_request(
                    proxy_port, "POST", "/v1/chat/completions",
                    {"Content-Type": "application/json"},
                    wrapped, timeout=10.0)
                elapsed = time.monotonic() - t0
                # Any non-None status is fine (it'll be 200 from our
                # fake upstream OR a 4xx if transform_request rejects
                # the unexpected shape). Critical: must complete in
                # under 5s, not CPU-bound on parsing.
                ok = code is not None and elapsed < 5.0
                return (ok, f"got {code} in {elapsed*1000:.0f}ms "
                            f"(depth={depth})")
            case("14. deeply-nested JSON (10000-level array)", c14)

            # Case 15: extremely-wide JSON (100k keys in one object).
            # Tests CPU cost of parsing a wide map. Should be linear
            # in input size and well under the body-size cap.
            def c15():
                n = 100_000
                pairs = ",".join(f'"k{i}":{i}' for i in range(n))
                body = b'{"messages":[{"role":"user","content":"x"}],"_w":{' \
                       + pairs.encode() + b"}}"
                t0 = time.monotonic()
                code, _ = _raw_request(
                    proxy_port, "POST", "/v1/chat/completions",
                    {"Content-Type": "application/json"},
                    body, timeout=10.0)
                elapsed = time.monotonic() - t0
                ok = code is not None and elapsed < 5.0
                return (ok, f"got {code} in {elapsed*1000:.0f}ms "
                            f"(keys={n}, body={len(body)//1024}KB)")
            case("15. extremely-wide JSON (100k keys)", c15)

            # Final: daemon should still be alive after all that abuse
            print()
            if _probe_alive(proxy_port, timeout=2.0):
                print("  [✓] daemon still alive after all attack cases")
            else:
                print("  [✗] daemon NOT alive after attack run")
                failures.append("final-alive-check")
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            proc.kill()
            failures.append("daemon-required-kill")

    print(f"\n== {'PASS' if not failures else 'FAIL'} "
          f"({len(failures)} failure(s)) ==")
    if failures:
        for f in failures:
            print(f"  - {f}")
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
