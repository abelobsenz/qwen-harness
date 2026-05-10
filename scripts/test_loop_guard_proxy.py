#!/usr/bin/env python3
"""End-to-end test: spin up a fake dflash-serve that loops, point qwen-proxy
at it, send a chat completion, check that the proxy aborts.

Run with: python scripts/test_loop_guard_proxy.py

The fake upstream supports two regimes selected by URL path:
  /v1/chat/completions  → emits a pathological loop (~5000 chars of repeats)
  /v1/healthy           → emits varied prose that should NOT trigger

We then start qwen_proxy.main() in a thread on a different port and POST
test requests to it. The proxy should:
  - On the loop endpoint: abort the response (finish_reason=length, content
    contains the loop-guard marker)
  - On the healthy endpoint: pass content through unchanged

Stdlib only. No real model loaded.
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


_LOOP_PHRASE = (
    "I will use make_table now. Then the Mermaid code. I will not use any "
    "other tools. I will just output the text. Wait, I need to use "
    "`make_table` for the table. I will do that now. "
)
_HEALTHY_TEXT = (
    "Greedy decoding picks the most probable token at each step. To avoid "
    "loops we need either sampling (temperature, top-p) or repetition "
    "penalty. The proxy-side guard is a defence-in-depth that catches "
    "loops regardless of the underlying decoder. It uses two detectors: a "
    "literal-suffix repeat detector and an n-gram churn detector. Both "
    "have low overhead and run on a sliding window of the streamed output."
)


class FakeUpstream(BaseHTTPRequestHandler):
    """Streams an SSE response that either loops or is healthy."""

    def log_message(self, fmt, *args):  # noqa: A003
        return

    def do_GET(self):  # noqa: N802
        if self.path == "/v1/models":
            body = json.dumps({"data": [{"id": "test", "object": "model"}]})
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(body.encode())
            return
        self.send_response(404)
        self.end_headers()

    def do_POST(self):  # noqa: N802
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length) if length else b""
        try:
            req = json.loads(body)
        except json.JSONDecodeError:
            self.send_response(400)
            self.end_headers()
            return
        # Only chat completions
        if not self.path.endswith("/chat/completions"):
            self.send_response(404)
            self.end_headers()
            return
        # Pull the scenario from the user message content (the proxy doesn't
        # forward arbitrary headers so we encode it in-band).
        scenario = "loop"
        for m in req.get("messages") or []:
            content = m.get("content")
            if isinstance(content, str) and "SCENARIO=" in content:
                if "SCENARIO=healthy" in content:
                    scenario = "healthy"
                else:
                    scenario = "loop"
                break
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        # First frame
        self._send_frame({"choices": [{"index": 0, "delta": {"role": "assistant"}}]})
        text = _LOOP_PHRASE * 50 if scenario == "loop" else _HEALTHY_TEXT
        # Stream tokens 16 chars at a time so the guard sees granular chunks
        self._tokens_aborted = 0
        try:
            for i in range(0, len(text), 16):
                chunk = text[i : i + 16]
                self._send_frame({
                    "choices": [{"index": 0, "delta": {"content": chunk}}]
                })
                time.sleep(0.001)  # yield-friendly
                self._tokens_aborted = i + 16
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, OSError):
            # Proxy aborted us — exactly what we want when a loop is detected.
            return
        try:
            self._send_frame({"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]})
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass

    def _send_frame(self, obj: dict) -> None:
        line = f"data: {json.dumps(obj)}\n\n"
        self.wfile.write(line.encode())
        self.wfile.flush()


@contextmanager
def background_server(handler_cls, port: int):
    server = HTTPServer(("127.0.0.1", port), handler_cls)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()


def main() -> int:
    upstream_port = _free_port()
    proxy_port = _free_port()

    failures = 0

    print(f"== loop-guard proxy integration test ==")
    print(f"  upstream :{upstream_port}, proxy :{proxy_port}")

    with background_server(FakeUpstream, upstream_port):
        # Lazily import the proxy module so its main() can be patched.
        import qwen_proxy

        # Configure handler before we start the server
        qwen_proxy.ProxyHandler.upstream_base = f"http://127.0.0.1:{upstream_port}"

        from http.server import ThreadingHTTPServer
        proxy_server = ThreadingHTTPServer(
            ("127.0.0.1", proxy_port), qwen_proxy.ProxyHandler
        )
        proxy_thread = threading.Thread(target=proxy_server.serve_forever, daemon=True)
        proxy_thread.start()

        try:
            # Test 1: loop scenario, non-stream
            print("\n[1] Loop scenario, non-streaming")
            req_body = json.dumps({"messages": [{"role": "user", "content": "hi SCENARIO=loop"}],
                                   "stream": False}).encode()
            req = urllib.request.Request(
                f"http://127.0.0.1:{proxy_port}/v1/chat/completions",
                data=req_body,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            t0 = time.monotonic()
            with urllib.request.urlopen(req, timeout=20) as resp:
                resp_body = resp.read()
            elapsed = time.monotonic() - t0
            data = json.loads(resp_body)
            content = data["choices"][0]["message"]["content"] or ""
            print(f"    elapsed {elapsed:.2f}s, content len={len(content)}")
            print(f"    finish_reason={data['choices'][0].get('finish_reason')}")
            print(f"    tail: {content[-120:]!r}")
            if "loop-guard" in content and len(content) < 4000:
                print("    [✓] non-stream loop scenario aborted as expected")
            else:
                print("    [✗] non-stream loop scenario NOT properly aborted")
                failures += 1

            # Test 2: healthy scenario, non-stream
            print("\n[2] Healthy scenario, non-streaming")
            req_body = json.dumps({"messages": [{"role": "user", "content": "hi SCENARIO=healthy"}],
                                   "stream": False}).encode()
            req = urllib.request.Request(
                f"http://127.0.0.1:{proxy_port}/v1/chat/completions",
                data=req_body,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=20) as resp:
                resp_body = resp.read()
            data = json.loads(resp_body)
            content = data["choices"][0]["message"]["content"] or ""
            print(f"    content len={len(content)}, "
                  f"contains-marker={'loop-guard' in content}")
            if "loop-guard" not in content and len(content) >= len(_HEALTHY_TEXT) * 0.9:
                print("    [✓] healthy scenario passed through cleanly")
            else:
                print("    [✗] healthy scenario broken (false positive or truncated)")
                failures += 1

            # Test 3: loop scenario, streaming
            print("\n[3] Loop scenario, streaming")
            req_body = json.dumps({"messages": [{"role": "user", "content": "hi SCENARIO=loop"}],
                                   "stream": True}).encode()
            req = urllib.request.Request(
                f"http://127.0.0.1:{proxy_port}/v1/chat/completions",
                data=req_body,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            chunks: list[str] = []
            saw_marker = False
            with urllib.request.urlopen(req, timeout=20) as resp:
                for raw in resp:
                    line = raw.decode("utf-8", errors="ignore").strip()
                    if not line.startswith("data:"):
                        continue
                    payload = line[5:].strip()
                    if payload == "[DONE]":
                        break
                    try:
                        obj = json.loads(payload)
                    except json.JSONDecodeError:
                        continue
                    delta = obj.get("choices", [{}])[0].get("delta", {})
                    if delta.get("content"):
                        chunks.append(delta["content"])
                        if "loop-guard" in delta["content"]:
                            saw_marker = True
            total = "".join(chunks)
            print(f"    streamed len={len(total)}, saw loop-guard marker={saw_marker}")
            if saw_marker and len(total) < 4000:
                print("    [✓] streaming loop scenario aborted via SSE")
            else:
                print("    [✗] streaming loop scenario did NOT abort properly")
                failures += 1

            # Test 4: healthy scenario, streaming
            print("\n[4] Healthy scenario, streaming")
            req_body = json.dumps({"messages": [{"role": "user", "content": "hi SCENARIO=healthy"}],
                                   "stream": True}).encode()
            req = urllib.request.Request(
                f"http://127.0.0.1:{proxy_port}/v1/chat/completions",
                data=req_body,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            chunks2: list[str] = []
            with urllib.request.urlopen(req, timeout=20) as resp:
                for raw in resp:
                    line = raw.decode("utf-8", errors="ignore").strip()
                    if not line.startswith("data:"):
                        continue
                    payload = line[5:].strip()
                    if payload == "[DONE]":
                        break
                    try:
                        obj = json.loads(payload)
                    except json.JSONDecodeError:
                        continue
                    delta = obj.get("choices", [{}])[0].get("delta", {})
                    if delta.get("content"):
                        chunks2.append(delta["content"])
            total2 = "".join(chunks2)
            print(f"    streamed len={len(total2)}, contains marker="
                  f"{'loop-guard' in total2}")
            if "loop-guard" not in total2 and len(total2) >= len(_HEALTHY_TEXT) * 0.9:
                print("    [✓] healthy streaming passed through cleanly")
            else:
                print("    [✗] healthy streaming broken")
                failures += 1
        finally:
            proxy_server.shutdown()
            proxy_server.server_close()

    print(f"\n== {'PASS' if failures == 0 else 'FAIL'} ({failures} failure(s)) ==")
    return failures


if __name__ == "__main__":
    sys.exit(main())
