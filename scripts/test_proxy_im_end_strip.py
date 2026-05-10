#!/usr/bin/env python3
"""Verify the proxy strips `<|im_end|>` from streaming SSE content.

When the Qwen3 model finishes its turn, its last content token is the
chat-template marker `<|im_end|>` (vocabulary id 151645). dflash-serve
emits that token's literal decoded form into the SSE content delta.
The proxy is responsible for hiding it — otherwise streaming clients
(chat REPL, qwen-ui) display the literal "<|im_end|>" string at the
end of every assistant response.

Two cases:
  A. Streaming — proxy must rewrite the SSE chunk to strip the marker
  B. Non-streaming — proxy already strips via split_reasoning's
     IM_END_RE; this test confirms that stays working.

Plus a healthy-baseline check (no marker → no rewrite).
"""

from __future__ import annotations

import json
import os
import socket
import sys
import threading
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


# Upstream emits the model's response with the chat-template control
# token at the end, mirroring how dflash-serve actually behaves when
# Qwen3 finishes a turn.
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
        self._send({"choices": [{"delta": {"role": "assistant"}}]})
        # Healthy content
        self._send({"choices": [{"delta": {"content": "Hi! "}}]})
        self._send({"choices": [{"delta": {"content": "How can I help "}}]})
        self._send({"choices": [{"delta": {"content": "you today?"}}]})
        # Final content frames include each chat-template control token
        # the proxy is supposed to strip. Round 32 broadened the strip
        # set from <|im_end|> alone to also <|im_start|> and
        # <|endoftext|>, both of which surface in some Qwen3 chat-
        # template variants and finetunes.
        self._send({"choices": [{"delta": {"content": "<|im_end|>"}}]})
        self._send({"choices": [{"delta": {"content": "<|im_start|>"}}]})
        self._send({"choices": [{"delta": {"content": "<|endoftext|>"}}]})
        self._send({"choices": [{"delta": {}, "finish_reason": "stop"}]})
        try:
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass

    def _send(self, obj):
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


def main() -> int:
    failures = 0
    upstream_port = _free_port()
    proxy_port = _free_port()

    print("== Proxy <|im_end|> strip test ==\n")

    with background_server(FakeUpstream, upstream_port):
        from http.server import ThreadingHTTPServer
        for mod in ("qwen_proxy", "loop_guard"):
            sys.modules.pop(mod, None)
        import qwen_proxy
        qwen_proxy.ProxyHandler.upstream_base = f"http://127.0.0.1:{upstream_port}"

        class _QuietT(ThreadingHTTPServer):
            def handle_error(self, request, client_address):
                import sys as _s
                if isinstance(_s.exc_info()[1], (BrokenPipeError, ConnectionResetError, OSError)):
                    return
                super().handle_error(request, client_address)

        proxy_server = _QuietT(("127.0.0.1", proxy_port), qwen_proxy.ProxyHandler)
        threading.Thread(target=proxy_server.serve_forever, daemon=True).start()

        try:
            # ----- Case A: streaming -----
            print("[A] Streaming — <|im_end|> stripped from SSE chunks")
            req_body = json.dumps({
                "messages": [{"role": "user", "content": "say hi"}],
                "stream": True,
            }).encode()
            req = urllib.request.Request(
                f"http://127.0.0.1:{proxy_port}/v1/chat/completions",
                data=req_body,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            assembled = []
            with urllib.request.urlopen(req, timeout=10) as resp:
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
                        assembled.append(delta["content"])
            full = "".join(assembled)
            print(f"    streamed: {full!r}")
            for tok in ("<|im_end|>", "<|im_start|>", "<|endoftext|>"):
                if tok in full:
                    print(f"    [✗] streamed content still contains {tok}")
                    failures += 1
                else:
                    print(f"    [✓] {tok} stripped")
            if "Hi! How can I help you today?" not in full:
                print(f"    [✗] expected content body got mangled")
                failures += 1
            else:
                print(f"    [✓] healthy text preserved through stripping")

            # ----- Case B: non-streaming (regression check) -----
            print("\n[B] Non-streaming — already-stripping path stays working")
            req_body = json.dumps({
                "messages": [{"role": "user", "content": "say hi"}],
                "stream": False,
            }).encode()
            req = urllib.request.Request(
                f"http://127.0.0.1:{proxy_port}/v1/chat/completions",
                data=req_body,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read())
            content = (data.get("choices") or [{}])[0].get("message", {}).get("content") or ""
            print(f"    response: {content!r}")
            for tok in ("<|im_end|>", "<|im_start|>", "<|endoftext|>"):
                if tok in content:
                    print(f"    [✗] non-stream content still contains {tok}")
                    failures += 1
                else:
                    print(f"    [✓] {tok} stripped from non-stream")
        finally:
            proxy_server.shutdown()
            proxy_server.server_close()

    print(f"\n== {'PASS' if failures == 0 else 'FAIL'} ({failures} failure(s)) ==")
    return failures


if __name__ == "__main__":
    sys.exit(main())
