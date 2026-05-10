#!/usr/bin/env python3
"""Tests for streaming agent.py:_do_post + watchdog.

Why: iter 23 had two TIMEOUTs (NFLX ARPU + TSM) where the non-stream POST
buffered for 480s with zero assistant turns logged. Streaming + a per-byte
watchdog converts those silent hangs into URLErrors that post_chat()
retries instead of letting the driver hit its subprocess timeout.

Verifies:
  1. Successful SSE stream rebuilds the same dict shape as non-stream.
  2. First-byte timeout aborts and raises URLError.
  3. Idle (mid-stream) timeout aborts and raises URLError.
  4. tool_calls deltas are accumulated.
  5. reasoning_content deltas are accumulated separately.
  6. QWEN_AGENT_STREAM=0 falls back to the legacy blocking path.
"""
from __future__ import annotations
import json
import os
import socket
import sys
import threading
import time
import urllib.error
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


class _Handler(BaseHTTPRequestHandler):
    """Mock SSE server controllable via class-level scenario dict."""
    SCENARIO = "happy"
    SLEEP_BEFORE_FIRST = 0.0
    SLEEP_BETWEEN = 0.0

    def log_message(self, *_args, **_kw) -> None:
        pass  # silence access logs

    def do_POST(self) -> None:  # noqa: N802
        # Drain the body without parsing — we don't care about the request
        # for these tests.
        n = int(self.headers.get("Content-Length", "0"))
        if n:
            self.rfile.read(n)

        if _Handler.SCENARIO == "blocking":
            # Simulate the legacy non-streaming path: send an OpenAI-shape
            # response. Used by the QWEN_AGENT_STREAM=0 fallback test.
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            body = json.dumps({
                "choices": [{
                    "message": {"role": "assistant", "content": "blocking-mode reply"},
                    "finish_reason": "stop",
                }],
            }).encode()
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()

        if _Handler.SLEEP_BEFORE_FIRST > 0:
            time.sleep(_Handler.SLEEP_BEFORE_FIRST)

        if _Handler.SCENARIO == "first_byte_timeout":
            # Sleep past the watchdog's first-byte threshold without
            # writing anything. Watchdog must close the socket; the
            # client will see a URLError.
            time.sleep(10)
            return

        chunks: list[dict] = []
        if _Handler.SCENARIO == "happy":
            chunks = [
                {"choices": [{"delta": {"role": "assistant", "content": "Hello "}}]},
                {"choices": [{"delta": {"content": "world"}}]},
                {"choices": [{"delta": {}, "finish_reason": "stop"}]},
            ]
        elif _Handler.SCENARIO == "tool_calls":
            chunks = [
                {"choices": [{"delta": {
                    "tool_calls": [{
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "web_search",
                            "arguments": json.dumps({"query": "test"}),
                        },
                    }]
                }}]},
                {"choices": [{"delta": {}, "finish_reason": "tool_calls"}]},
            ]
        elif _Handler.SCENARIO == "reasoning":
            chunks = [
                {"choices": [{"delta": {"reasoning_content": "Let me think... "}}]},
                {"choices": [{"delta": {"reasoning": "still thinking"}}]},
                {"choices": [{"delta": {"content": "answer"}}]},
                {"choices": [{"delta": {}, "finish_reason": "stop"}]},
            ]
        elif _Handler.SCENARIO == "idle_timeout":
            # First byte fine, then long silence to trigger the
            # idle-timeout watchdog branch.
            chunks = [
                {"choices": [{"delta": {"content": "starting..."}}]},
                # then we sleep below
            ]
        for i, c in enumerate(chunks):
            data = f"data: {json.dumps(c)}\n\n".encode()
            self.wfile.write(data)
            self.wfile.flush()
            if i + 1 < len(chunks) and _Handler.SLEEP_BETWEEN > 0:
                time.sleep(_Handler.SLEEP_BETWEEN)
        if _Handler.SCENARIO == "idle_timeout":
            # Stall after the first chunk to trigger the idle watchdog.
            time.sleep(10)
        # Standard SSE terminator.
        self.wfile.write(b"data: [DONE]\n\n")
        try:
            self.wfile.flush()
        except Exception:  # noqa: BLE001
            pass


def _start_mock_server() -> tuple[ThreadingHTTPServer, str]:
    """Start a mock /v1/chat/completions server on a free port."""
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    server = ThreadingHTTPServer(("127.0.0.1", port), _Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, f"http://127.0.0.1:{port}/v1/chat/completions"


def main() -> int:
    failures = 0
    server, url = _start_mock_server()
    try:
        # Lower watchdog timeouts for fast tests. Must be set BEFORE
        # importing agent so the module-level reads pick them up.
        os.environ["QWEN_AGENT_FIRST_BYTE_TIMEOUT"] = "3"
        os.environ["QWEN_AGENT_IDLE_TIMEOUT"] = "3"
        os.environ["QWEN_AGENT_STREAM"] = "1"
        # Force the agent module's URL constant to point at our mock.
        os.environ["QWEN_HOST"] = "127.0.0.1"
        os.environ["QWEN_PORT"] = url.split(":")[2].split("/")[0]

        # Avoid loading the real tool registry (heavy imports).
        sys.modules.pop("agent", None)
        import agent  # noqa: E402
        agent.URL = url
        agent._STREAM_FIRST_BYTE_TIMEOUT = 3.0
        agent._STREAM_IDLE_TIMEOUT = 3.0
        # Don't try to filter tools — return empty list to avoid loading.
        agent._filtered_tools = lambda: []

        msgs = [{"role": "user", "content": "test"}]

        # ---- 1. Happy path ----
        print("[1] Happy path: SSE stream rebuilds dict shape")
        _Handler.SCENARIO = "happy"
        _Handler.SLEEP_BEFORE_FIRST = 0
        _Handler.SLEEP_BETWEEN = 0
        try:
            resp = agent._do_post_streaming(msgs)
            content = resp["choices"][0]["message"]["content"]
            if content == "Hello world":
                print(f"    [OK] reassembled content: {content!r}")
            else:
                print(f"    [FAIL] expected 'Hello world', got {content!r}")
                failures += 1
        except Exception as e:  # noqa: BLE001
            print(f"    [FAIL] {type(e).__name__}: {e}")
            failures += 1

        # ---- 2. First-byte timeout ----
        print("[2] First-byte timeout → URLError")
        _Handler.SCENARIO = "first_byte_timeout"
        t0 = time.monotonic()
        try:
            agent._do_post_streaming(msgs)
            print(f"    [FAIL] expected URLError, got success")
            failures += 1
        except urllib.error.URLError as e:
            elapsed = time.monotonic() - t0
            if "first-byte" in str(e) and elapsed < 6:
                print(f"    [OK] aborted in {elapsed:.1f}s with first-byte URLError")
            else:
                print(f"    [FAIL] wrong error or too slow: elapsed={elapsed:.1f}s err={e}")
                failures += 1
        except Exception as e:  # noqa: BLE001
            print(f"    [FAIL] expected URLError, got {type(e).__name__}: {e}")
            failures += 1

        # ---- 3. Idle (mid-stream) timeout ----
        print("[3] Idle mid-stream timeout → URLError")
        _Handler.SCENARIO = "idle_timeout"
        t0 = time.monotonic()
        try:
            agent._do_post_streaming(msgs)
            print(f"    [FAIL] expected URLError, got success")
            failures += 1
        except urllib.error.URLError as e:
            elapsed = time.monotonic() - t0
            if "idle" in str(e) and elapsed < 7:
                print(f"    [OK] aborted in {elapsed:.1f}s with idle URLError")
            else:
                print(f"    [FAIL] wrong error or too slow: elapsed={elapsed:.1f}s err={e}")
                failures += 1
        except Exception as e:  # noqa: BLE001
            print(f"    [FAIL] expected URLError, got {type(e).__name__}: {e}")
            failures += 1

        # ---- 4. Tool calls deltas ----
        print("[4] Tool-call deltas → tool_calls in reassembled msg")
        _Handler.SCENARIO = "tool_calls"
        try:
            resp = agent._do_post_streaming(msgs)
            tcs = resp["choices"][0]["message"].get("tool_calls", [])
            if (len(tcs) == 1 and (tcs[0].get("function") or {}).get("name") == "web_search"):
                print(f"    [OK] tool_call rebuilt")
            else:
                print(f"    [FAIL] tool_calls not rebuilt: {tcs}")
                failures += 1
        except Exception as e:  # noqa: BLE001
            print(f"    [FAIL] {type(e).__name__}: {e}")
            failures += 1

        # ---- 5. Reasoning_content accumulation ----
        print("[5] reasoning_content deltas → reasoning_content in msg")
        _Handler.SCENARIO = "reasoning"
        try:
            resp = agent._do_post_streaming(msgs)
            r = resp["choices"][0]["message"].get("reasoning_content", "")
            if "Let me think" in r and "still thinking" in r:
                print(f"    [OK] reasoning_content reassembled: {r[:50]!r}")
            else:
                print(f"    [FAIL] reasoning_content missing or wrong: {r!r}")
                failures += 1
        except Exception as e:  # noqa: BLE001
            print(f"    [FAIL] {type(e).__name__}: {e}")
            failures += 1

        # ---- 6. QWEN_AGENT_STREAM=0 falls back to blocking ----
        print("[6] QWEN_AGENT_STREAM=0 → uses blocking path")
        os.environ["QWEN_AGENT_STREAM"] = "0"
        _Handler.SCENARIO = "blocking"
        try:
            resp = agent._do_post(msgs)
            content = resp["choices"][0]["message"]["content"]
            if content == "blocking-mode reply":
                print(f"    [OK] blocking fallback returned non-stream response")
            else:
                print(f"    [FAIL] unexpected content: {content!r}")
                failures += 1
        except Exception as e:  # noqa: BLE001
            print(f"    [FAIL] {type(e).__name__}: {e}")
            failures += 1

    finally:
        server.shutdown()

    if failures:
        print(f"\n== FAIL ({failures} failure(s)) ==")
        return 1
    print("\n== PASS (streaming + watchdog work correctly) ==")
    return 0


if __name__ == "__main__":
    sys.exit(main())
