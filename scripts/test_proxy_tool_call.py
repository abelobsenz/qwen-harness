#!/usr/bin/env python3
"""End-to-end test: proxy preserves tool_call parsing through the loop_guard.

Simulates a realistic Qwen3.6 response that mixes <think>, normal text,
and an XML <tool_call>. The proxy should:

  - Not abort (response is healthy)
  - Strip <think> into reasoning_content
  - Parse <tool_call> into structured tool_calls
  - Strip the XML from .content
  - Set finish_reason=tool_calls

This is the path the agent depends on. If we broke it, agent.py would get
unparsed XML tools, no reasoning_content, etc.
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


# Realistic varied response with <think>, plain text, and a tool_call.
_REALISTIC = (
    "Looking at the user's request, I need to fetch the README first to "
    "understand the project structure before I can answer the question.\n\n"
    "</think>\n\n"
    "Let me check the README first.\n\n"
    "<tool_call>\n"
    "<function=read_file>\n"
    "<parameter=path>/Users/abelobsenz/dev/qwen36_MTP/README.md</parameter>\n"
    "<parameter=offset>0</parameter>\n"
    "<parameter=limit>200</parameter>\n"
    "</function>\n"
    "</tool_call>"
)


class FakeUpstream(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):  # noqa: A003
        return

    def do_GET(self):  # noqa: N802
        if self.path == "/v1/models":
            body = b'{"data":[{"id":"test","object":"model"}]}'
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(body)
            return
        self.send_response(404); self.end_headers()

    def do_POST(self):  # noqa: N802
        length = int(self.headers.get("Content-Length", "0"))
        _ = self.rfile.read(length) if length else b""
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        # First role frame
        self._send_frame({"choices": [{"delta": {"role": "assistant"}}]})
        # Stream the realistic response
        for i in range(0, len(_REALISTIC), 16):
            chunk = _REALISTIC[i : i + 16]
            self._send_frame({"choices": [{"delta": {"content": chunk}}]})
        self._send_frame({"choices": [{"delta": {}, "finish_reason": "stop"}]})
        try:
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _send_frame(self, obj):
        try:
            self.wfile.write(f"data: {json.dumps(obj)}\n\n".encode())
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
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


def main() -> int:
    upstream_port = _free_port()
    proxy_port = _free_port()
    failures = 0
    print("== End-to-end tool_call parsing test ==\n")
    with background_server(FakeUpstream, upstream_port):
        from http.server import ThreadingHTTPServer
        # fresh import
        for mod in ("qwen_proxy", "loop_guard"):
            sys.modules.pop(mod, None)
        import qwen_proxy
        qwen_proxy.ProxyHandler.upstream_base = f"http://127.0.0.1:{upstream_port}"
        proxy_server = ThreadingHTTPServer(("127.0.0.1", proxy_port), qwen_proxy.ProxyHandler)
        threading.Thread(target=proxy_server.serve_forever, daemon=True).start()

        try:
            req_body = json.dumps({
                "messages": [{"role": "user", "content": "explain the project"}],
                "tools": [{"type": "function", "function": {
                    "name": "read_file",
                    "description": "Read a file",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string"},
                            "offset": {"type": "integer"},
                            "limit": {"type": "integer"},
                        },
                        "required": ["path"],
                    },
                }}],
                "stream": False,
            }).encode()
            req = urllib.request.Request(
                f"http://127.0.0.1:{proxy_port}/v1/chat/completions",
                data=req_body,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=20) as resp:
                resp_body = resp.read()
            data = json.loads(resp_body)
            msg = data["choices"][0]["message"]
            tool_calls = msg.get("tool_calls") or []
            content = msg.get("content")
            reasoning = msg.get("reasoning_content")
            finish = data["choices"][0].get("finish_reason")

            print(f"  finish_reason: {finish}")
            print(f"  content: {content!r}")
            print(f"  reasoning_content (first 80): {(reasoning or '')[:80]!r}")
            print(f"  tool_calls count: {len(tool_calls)}")
            if tool_calls:
                tc = tool_calls[0]
                print(f"  tool_call[0].function.name: {tc['function']['name']}")
                print(f"  tool_call[0].function.arguments: {tc['function']['arguments']}")

            # Check expectations
            if not tool_calls:
                print("    [✗] no tool_calls parsed")
                failures += 1
            elif tool_calls[0]["function"]["name"] != "read_file":
                print(f"    [✗] wrong tool name")
                failures += 1
            else:
                print("    [✓] tool_call parsed")
            if "<tool_call>" in (content or ""):
                print("    [✗] XML still in content")
                failures += 1
            else:
                print("    [✓] XML stripped from content")
            if not reasoning or len(reasoning) < 30:
                print("    [✗] reasoning_content not extracted")
                failures += 1
            else:
                print("    [✓] reasoning_content extracted")
            if finish != "tool_calls":
                print(f"    [✗] finish_reason should be tool_calls, got {finish}")
                failures += 1
            else:
                print("    [✓] finish_reason=tool_calls")

            print("\n[stream=True] structured streamed tool_calls")
            stream_body = json.dumps({
                "messages": [{"role": "user", "content": "explain the project"}],
                "tools": [{"type": "function", "function": {
                    "name": "read_file",
                    "description": "Read a file",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string"},
                            "offset": {"type": "integer"},
                            "limit": {"type": "integer"},
                        },
                        "required": ["path"],
                    },
                }}],
                "stream": True,
            }).encode()
            req = urllib.request.Request(
                f"http://127.0.0.1:{proxy_port}/v1/chat/completions",
                data=stream_body,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            stream_frames = []
            raw_stream = b""
            with urllib.request.urlopen(req, timeout=20) as resp:
                for raw in resp:
                    raw_stream += raw
                    line = raw.decode("utf-8", errors="replace").strip()
                    if not line.startswith("data:"):
                        continue
                    payload = line[5:].strip()
                    if payload == "[DONE]":
                        break
                    if not payload:
                        continue
                    stream_frames.append(json.loads(payload))
            leaked_xml = b"<tool_call>" in raw_stream or b"</tool_call>" in raw_stream
            streamed_calls = []
            for frame in stream_frames:
                choice = (frame.get("choices") or [{}])[0]
                delta = choice.get("delta") or {}
                streamed_calls.extend(delta.get("tool_calls") or [])
            if leaked_xml:
                print("    [✗] streamed XML leaked to client")
                failures += 1
            else:
                print("    [✓] streamed XML suppressed")
            if not streamed_calls:
                print("    [✗] no streamed tool_calls emitted")
                failures += 1
            elif streamed_calls[0]["function"]["name"] != "read_file":
                print(f"    [✗] streamed wrong tool name")
                failures += 1
            else:
                print("    [✓] streamed tool_call parsed")

            metrics_req = urllib.request.Request(
                f"http://127.0.0.1:{proxy_port}/debug/metrics",
                method="GET",
            )
            with urllib.request.urlopen(metrics_req, timeout=5) as resp:
                metrics = json.loads(resp.read())
            counters = metrics.get("counters") or {}
            if counters.get("stream_tool_calls_parsed", 0) < 1:
                print("    [✗] stream_tool_calls_parsed metric not incremented")
                failures += 1
            else:
                print("    [✓] stream tool-call metric incremented")
        finally:
            proxy_server.shutdown()
            proxy_server.server_close()

    print(f"\n== {'PASS' if failures == 0 else 'FAIL'} ({failures} failure(s)) ==")
    return failures


if __name__ == "__main__":
    sys.exit(main())
