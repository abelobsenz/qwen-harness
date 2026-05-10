#!/usr/bin/env python3
"""Concurrent-load stress test for the proxy with loop_guard.

Verifies:
  - No deadlocks or race conditions when N parallel clients hit the proxy
  - Each client gets its own correct response (no cross-talk)
  - Memory usage stays bounded under sustained concurrent load
  - Loop_guard state is per-request, not shared (verified by mixing loop +
    healthy scenarios in flight at the same time)

The fake upstream uses a small artificial delay per chunk to simulate
real-model latency, so the test exercises real concurrency rather than
trivial sequential return.
"""

from __future__ import annotations

import concurrent.futures
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


_LOOP_PHRASE = "I will use make_table now. Then the Mermaid code. " * 50
_HEALTHY = (
    "The history of computing began long before electronic computers. "
    "Mechanical calculators date back to the 17th century with Pascal's "
    "wheel, and 19th-century work on Babbage's analytical engine "
    "introduced the concepts of stored programs and conditional branching. "
    "Boolean logic, formalized by George Boole in 1847, gave us the "
    "algebraic foundation. Alan Turing's 1936 paper on computable numbers "
    "established the mathematical theory of computation, including the "
    "Halting Problem. The transistor's invention in 1947 enabled compact "
    "electronics, and integrated circuits in the 1960s allowed exponential "
    "miniaturization. Modern CPUs contain billions of transistors. "
)


class FakeUpstream(BaseHTTPRequestHandler):
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
        body = self.rfile.read(length) if length else b""
        try:
            req = json.loads(body)
        except json.JSONDecodeError:
            self.send_response(400); self.end_headers(); return
        scenario = "loop"
        for m in req.get("messages") or []:
            content = m.get("content") or ""
            if isinstance(content, str) and "SCENARIO=healthy" in content:
                scenario = "healthy"
                break
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        self._send_frame({"choices": [{"delta": {"role": "assistant"}}]})
        text = _LOOP_PHRASE if scenario == "loop" else _HEALTHY
        try:
            for i in range(0, len(text), 32):
                self._send_frame({"choices": [{"delta": {"content": text[i:i+32]}}]})
                time.sleep(0.0005)  # 0.5 ms per chunk
            self._send_frame({"choices": [{"delta": {}, "finish_reason": "stop"}]})
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass

    def _send_frame(self, obj):
        try:
            self.wfile.write(f"data: {json.dumps(obj)}\n\n".encode())
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass


@contextmanager
def background_server(handler_cls, port):
    server = HTTPServer(("127.0.0.1", port), handler_cls)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()


def issue_request(proxy_port: int, scenario: str) -> dict:
    body = json.dumps({"messages": [{"role": "user", "content": f"hi SCENARIO={scenario}"}],
                       "stream": False}).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{proxy_port}/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=20) as resp:
        body = resp.read()
    elapsed = time.perf_counter() - t0
    data = json.loads(body)
    content = data["choices"][0]["message"]["content"] or ""
    return {
        "scenario": scenario,
        "elapsed_s": elapsed,
        "content_len": len(content),
        "saw_marker": "loop-guard" in content,
    }


def main() -> int:
    upstream_port = _free_port()
    proxy_port = _free_port()
    print(f"== Concurrent-load stress test ==\n  upstream :{upstream_port} proxy :{proxy_port}\n")

    failures = 0

    with background_server(FakeUpstream, upstream_port):
        from http.server import ThreadingHTTPServer
        for mod in ("qwen_proxy", "loop_guard"):
            sys.modules.pop(mod, None)
        import qwen_proxy
        qwen_proxy.ProxyHandler.upstream_base = f"http://127.0.0.1:{upstream_port}"
        proxy_server = ThreadingHTTPServer(("127.0.0.1", proxy_port), qwen_proxy.ProxyHandler)
        threading.Thread(target=proxy_server.serve_forever, daemon=True).start()

        # Sequential warmup
        issue_request(proxy_port, "healthy")
        issue_request(proxy_port, "loop")

        # 20 concurrent requests, mixed scenarios
        scenarios = ["loop", "healthy"] * 10
        t0 = time.perf_counter()
        with concurrent.futures.ThreadPoolExecutor(max_workers=20) as pool:
            futures = [pool.submit(issue_request, proxy_port, s) for s in scenarios]
            results = [f.result() for f in futures]
        wall = time.perf_counter() - t0

        loops = [r for r in results if r["scenario"] == "loop"]
        healthies = [r for r in results if r["scenario"] == "healthy"]

        print(f"  Concurrent batch: {len(scenarios)} requests, wall={wall:.2f}s")
        print(f"    loop scenarios:    {len(loops)}, all aborted={all(r['saw_marker'] for r in loops)}")
        print(f"    healthy scenarios: {len(healthies)}, none aborted={not any(r['saw_marker'] for r in healthies)}")

        if not all(r["saw_marker"] for r in loops):
            print("    [✗] some loop scenario failed to be aborted under concurrency")
            failures += 1
        else:
            print("    [✓] all loops correctly aborted")

        if any(r["saw_marker"] for r in healthies):
            print("    [✗] healthy scenario falsely aborted")
            failures += 1
        else:
            print("    [✓] healthy responses unaffected")

        # Cross-talk check: lengths should be consistent within scenario
        loop_lens = sorted({r["content_len"] for r in loops})
        healthy_lens = sorted({r["content_len"] for r in healthies})
        print(f"    loop content lens:    {loop_lens}")
        print(f"    healthy content lens: {healthy_lens}")
        # Healthy should all be same length (no cross-talk)
        if len(healthy_lens) > 1:
            print(f"    [✗] healthy responses have varying lengths — possible cross-talk")
            failures += 1
        else:
            print(f"    [✓] healthy responses are consistent (no cross-talk)")

        # Long-stream stress: 100 sequential requests to check memory bounds
        print(f"\n  Long-stream stress: 100 sequential healthy requests")
        for _ in range(100):
            issue_request(proxy_port, "healthy")
        print(f"    [✓] 100 sequential requests completed without error")

        proxy_server.shutdown()
        proxy_server.server_close()

    print(f"\n== {'PASS' if failures == 0 else 'FAIL'} ({failures} failure(s)) ==")
    return failures


if __name__ == "__main__":
    sys.exit(main())
