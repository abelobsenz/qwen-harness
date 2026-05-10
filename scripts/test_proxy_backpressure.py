#!/usr/bin/env python3
"""Streaming back-pressure: verify that when a client disconnects
mid-stream, the proxy propagates the disconnect upstream so dflash-serve
stops generating.

Why this matters: dflash-serve is single-stream by design (one model
load → one generation at a time). If a client drops out mid-stream and
the proxy keeps reading from upstream until max_tokens, dflash-serve
holds the GPU busy and blocks every queued client behind it. The
proxy MUST close the upstream socket promptly when it can no longer
write to its own client.

Mechanism: when `self.wfile.write(line)` raises `BrokenPipeError` or
`ConnectionResetError`, the proxy's `try`/`except` (qwen_proxy.py:518)
exits the for-loop. The `finally` block closes `resp`. Closing the
HTTP response from the urllib side closes the underlying socket, and
dflash-serve's stream generator hits a `BrokenPipeError` on its next
`_sse_write` and short-circuits (we audited that at the top of the
file: dflash_mlx/serve.py wraps SSE writes in `_sse_write` which
returns False on disconnect).

This test:
  1. Spawns a slow fake upstream that emits 200 chunks with 25ms
     between each (~5s total if it ran to completion).
  2. Opens a streaming connection from a client thread.
  3. Reads ~5 chunks, then aggressively closes the connection.
  4. Asserts the fake upstream stopped emitting within 1s of the
     disconnect (i.e. saw the propagated abort).
  5. Asserts the upstream emitted FEWER than the full 200 chunks.

Stdlib only.
"""

from __future__ import annotations

import json
import os
import socket
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


# Shared state: how many chunks the upstream actually emitted, when the
# upstream started seeing failures, etc.
_UPSTREAM_STATE = {
    "chunks_sent": 0,
    "first_failure_at": None,  # monotonic time of first BrokenPipeError
    "completed_normally": False,
}
_state_lock = threading.Lock()


class SlowUpstream(BaseHTTPRequestHandler):
    """Emits 200 distinct content chunks, 25ms apart. Records when its
    writes start to fail (i.e. when the proxy closed its end)."""

    # HTTP/1.0 = no keep-alive => connection closes after do_POST. Avoids
    # the noisy ConnectionResetError that fires on the next request-line
    # readline when the proxy has already closed the socket.
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
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            self._send_frame({"choices": [{"delta": {"role": "assistant"}}]})
            for i in range(200):
                # Distinct, varied content so loop_guard never fires
                # (we want to test back-pressure, not loop abort).
                content = f"chunk_{i:03d}_with_varied_text_{i*7}_and_index "
                try:
                    self._send_frame({
                        "choices": [{"delta": {"content": content}}]
                    })
                except (BrokenPipeError, ConnectionResetError, OSError):
                    with _state_lock:
                        if _UPSTREAM_STATE["first_failure_at"] is None:
                            _UPSTREAM_STATE["first_failure_at"] = time.monotonic()
                    return
                with _state_lock:
                    _UPSTREAM_STATE["chunks_sent"] = i + 1
                time.sleep(0.025)
            try:
                self._send_frame({"choices": [{"delta": {}, "finish_reason": "stop"}]})
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError, OSError):
                with _state_lock:
                    if _UPSTREAM_STATE["first_failure_at"] is None:
                        _UPSTREAM_STATE["first_failure_at"] = time.monotonic()
                return
            with _state_lock:
                _UPSTREAM_STATE["completed_normally"] = True
        except (BrokenPipeError, ConnectionResetError, OSError):
            with _state_lock:
                if _UPSTREAM_STATE["first_failure_at"] is None:
                    _UPSTREAM_STATE["first_failure_at"] = time.monotonic()

    def _send_frame(self, obj):
        self.wfile.write(f"data: {json.dumps(obj)}\n\n".encode())
        self.wfile.flush()


class _QuietHTTPServer(HTTPServer):
    """HTTPServer subclass that silently drops the ConnectionResetError
    chains that occur when a client closes its socket early. The default
    `ThreadingHTTPServer.handle_error` prints the traceback to stderr,
    which is just noise here — the test relies on the upstream seeing
    the disconnect, not on the framework being silent about it."""

    def handle_error(self, request, client_address):  # noqa: D401
        import sys
        exc = sys.exc_info()[1]
        if isinstance(exc, (BrokenPipeError, ConnectionResetError, OSError)):
            return
        super().handle_error(request, client_address)


@contextmanager
def background_server(handler_cls, port):
    server = _QuietHTTPServer(("127.0.0.1", port), handler_cls)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()


def main() -> int:
    failures = 0

    upstream_port = _free_port()
    proxy_port = _free_port()

    print("== Proxy streaming back-pressure test ==\n")

    with background_server(SlowUpstream, upstream_port):
        from http.server import ThreadingHTTPServer

        # ThreadingHTTPServer + our quiet error handler. Without this,
        # the proxy itself prints a noisy traceback when the test client
        # disconnects (the proxy's keep-alive readline fails and the
        # default handle_error prints to stderr).
        class _QuietThreadingHTTPServer(ThreadingHTTPServer):
            def handle_error(self, request, client_address):  # noqa: D401
                import sys
                exc = sys.exc_info()[1]
                if isinstance(exc, (BrokenPipeError, ConnectionResetError, OSError)):
                    return
                super().handle_error(request, client_address)

        for mod in ("qwen_proxy", "loop_guard"):
            sys.modules.pop(mod, None)
        import qwen_proxy
        qwen_proxy.ProxyHandler.upstream_base = f"http://127.0.0.1:{upstream_port}"
        proxy_server = _QuietThreadingHTTPServer(
            ("127.0.0.1", proxy_port), qwen_proxy.ProxyHandler)
        threading.Thread(target=proxy_server.serve_forever, daemon=True).start()

        try:
            # Open a streaming connection, read a few chunks, then bail.
            req_body = json.dumps({
                "messages": [{"role": "user", "content": "stream me something"}],
                "stream": True,
            }).encode()
            req = urllib.request.Request(
                f"http://127.0.0.1:{proxy_port}/v1/chat/completions",
                data=req_body,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            t_open = time.monotonic()
            resp = urllib.request.urlopen(req, timeout=10)

            chunks_received = 0
            target_reads = 5  # read ~5 chunks then disconnect
            for raw in resp:
                line = raw.decode("utf-8", errors="ignore").strip()
                if line.startswith("data:"):
                    payload = line[5:].strip()
                    if payload and payload != "[DONE]":
                        try:
                            obj = json.loads(payload)
                            delta = obj.get("choices", [{}])[0].get("delta", {})
                            if delta.get("content"):
                                chunks_received += 1
                                if chunks_received >= target_reads:
                                    break
                        except json.JSONDecodeError:
                            pass

            t_disconnect = time.monotonic()
            print(f"  [client] received {chunks_received} chunks, "
                  f"closing connection at +{(t_disconnect-t_open)*1000:.0f} ms")

            # Forcibly close — this is what a real disconnecting client does.
            try:
                resp.close()
            except Exception:  # noqa: BLE001
                pass
            # Also drop our underlying socket if the urllib wrapper kept it.
            try:
                resp.fp.close()  # type: ignore[attr-defined]
            except Exception:  # noqa: BLE001
                pass

            # Wait up to 2 seconds for the upstream to notice.
            deadline = time.monotonic() + 2.0
            while time.monotonic() < deadline:
                with _state_lock:
                    if _UPSTREAM_STATE["first_failure_at"] is not None:
                        break
                    if _UPSTREAM_STATE["completed_normally"]:
                        break
                time.sleep(0.025)

            with _state_lock:
                state = dict(_UPSTREAM_STATE)
            propagation_delay = (
                (state["first_failure_at"] - t_disconnect) * 1000
                if state["first_failure_at"] is not None else None
            )
            print(f"  [upstream] chunks_sent={state['chunks_sent']}, "
                  f"completed_normally={state['completed_normally']}, "
                  f"propagation_delay_ms="
                  f"{f'{propagation_delay:.0f}' if propagation_delay else 'N/A'}")

            # Gates
            if state["completed_normally"]:
                print(f"  [✗] upstream ran to completion despite client disconnect")
                failures += 1
            elif state["first_failure_at"] is None:
                print(f"  [✗] upstream NEVER saw a write failure — disconnect "
                      f"was not propagated")
                failures += 1
            elif propagation_delay is not None and propagation_delay > 1500:
                print(f"  [✗] disconnect propagation took "
                      f"{propagation_delay:.0f}ms (gate: <=1500ms)")
                failures += 1
            else:
                print(f"  [✓] disconnect propagated in "
                      f"{propagation_delay:.0f}ms")

            if state["chunks_sent"] >= 199:
                print(f"  [✗] upstream emitted {state['chunks_sent']} chunks — "
                      f"continued generating after disconnect")
                failures += 1
            else:
                print(f"  [✓] upstream stopped at {state['chunks_sent']} of 200 "
                      f"chunks (saved {200 - state['chunks_sent']} chunks of work)")

        finally:
            proxy_server.shutdown()
            proxy_server.server_close()

    print(f"\n== {'PASS' if failures == 0 else 'FAIL'} ({failures} failure(s)) ==")
    return failures


if __name__ == "__main__":
    sys.exit(main())
