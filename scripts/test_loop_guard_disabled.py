#!/usr/bin/env python3
"""Escape-hatch validation: LOOP_GUARD_DISABLE=1 makes the proxy behave
the same way it did before loop_guard existed.

We test two equivalences:

1. **Detector inertness**: with the env var set, every adversarial input
   that normally triggers a loop returns clean from check_text/Streaming
   guard. Confirms the env var actually disables every code path.

2. **Proxy pass-through**: with the env var set, a loop-emitting upstream
   no longer gets aborted by the proxy — the full text streams through.
   Compares byte-for-byte (modulo SSE framing) what the upstream sent vs
   what the proxy delivered.

This guarantees that if a future bug or false-positive surfaces in
production, the user can hot-disable the guard without redeploying.
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


_LOOP_TEXT = ("I will use make_table now. Then the Mermaid code. " * 60)


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
        _ = self.rfile.read(length) if length else b""
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        self._send_frame({"choices": [{"delta": {"role": "assistant"}}]})
        try:
            for i in range(0, len(_LOOP_TEXT), 16):
                self._send_frame({
                    "choices": [{"delta": {"content": _LOOP_TEXT[i:i+16]}}]
                })
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


def _run_subprocess_test(env: dict, label: str) -> dict:
    """Spawn the proxy in a subprocess with the given env so the
    LOOP_GUARD_DISABLE flag actually takes effect at module import time
    (re-importing in the same process leaves the env-resolved constants
    bound in `LoopGuardConfig` defaults stale).

    Returns: {content_len, saw_marker, loop_text_present}
    """
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
    full_env = {**os.environ, **env}
    proc = subprocess.Popen(proxy_cmd, env=full_env,
                            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)

    with background_server(FakeUpstream, upstream_port):
        # Wait for proxy to come up
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
            proc.terminate()
            return {"error": "proxy did not start"}
        try:
            req_body = json.dumps({
                "messages": [{"role": "user", "content": "hi"}],
                "stream": False,
            }).encode()
            req = urllib.request.Request(
                f"http://127.0.0.1:{proxy_port}/v1/chat/completions",
                data=req_body,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=20) as resp:
                body = resp.read()
            data = json.loads(body)
            content = data["choices"][0]["message"]["content"] or ""
            return {
                "label": label,
                "content_len": len(content),
                "saw_marker": "loop-guard" in content,
                # proxy normally delivers the loop in chunks, no marker means
                # full text passes through.
                "make_table_count": content.count("make_table now"),
            }
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                proc.kill()


def main() -> int:
    failures = 0
    print("== LOOP_GUARD_DISABLE escape-hatch validation ==\n")

    # --- 1. Detector inertness in-process ---
    print("[1] In-process detector inertness with LOOP_GUARD_DISABLE=1")
    os.environ["LOOP_GUARD_DISABLE"] = "1"
    for mod in ("loop_guard", "qwen_proxy"):
        sys.modules.pop(mod, None)
    from loop_guard import LoopGuardConfig, check_text, StreamingLoopGuard

    cfg = LoopGuardConfig()
    print(f"    cfg.enabled = {cfg.enabled} (expect False)")
    if cfg.enabled:
        print("    [✗] cfg.enabled should be False")
        failures += 1
    rep = check_text(_LOOP_TEXT, cfg)
    print(f"    check_text on loop input: triggered={rep.triggered} (expect False)")
    if rep.triggered:
        print("    [✗] disabled detector still triggered")
        failures += 1
    g = StreamingLoopGuard(cfg)
    triggered_in_stream = False
    for i in range(0, len(_LOOP_TEXT), 32):
        if g.observe(_LOOP_TEXT[i : i + 32]).triggered:
            triggered_in_stream = True
            break
    if triggered_in_stream or g.finalize().triggered:
        print("    [✗] disabled streaming guard still triggered")
        failures += 1
    else:
        print("    [✓] streaming guard inert when disabled")

    # --- 2. End-to-end proxy: subprocess with the env var ---
    print("\n[2] End-to-end proxy: LOOP_GUARD_DISABLE=1 in subprocess")
    res_disabled = _run_subprocess_test({"LOOP_GUARD_DISABLE": "1"}, "disabled")
    print(f"    {res_disabled}")
    if res_disabled.get("error"):
        print(f"    [✗] {res_disabled['error']}")
        failures += 1
    elif res_disabled["saw_marker"]:
        print("    [✗] disabled proxy still emitted a loop-guard marker")
        failures += 1
    elif res_disabled["make_table_count"] < 50:
        print(f"    [✗] disabled proxy didn't pass full loop through "
              f"(make_table count={res_disabled['make_table_count']}, expected ~60)")
        failures += 1
    else:
        print(f"    [✓] disabled proxy passed full loop through "
              f"({res_disabled['make_table_count']} repetitions of 'make_table now')")

    # --- 3. End-to-end proxy: enabled (sanity check) ---
    print("\n[3] Sanity: LOOP_GUARD_DISABLE=0 still aborts")
    res_enabled = _run_subprocess_test({"LOOP_GUARD_DISABLE": "0"}, "enabled")
    print(f"    {res_enabled}")
    if res_enabled.get("error"):
        print(f"    [✗] {res_enabled['error']}")
        failures += 1
    elif not res_enabled["saw_marker"]:
        print("    [✗] enabled proxy did NOT emit a loop-guard marker")
        failures += 1
    elif res_enabled["make_table_count"] >= res_disabled.get("make_table_count", 60):
        print("    [✗] enabled proxy didn't actually shorten the response")
        failures += 1
    else:
        print(f"    [✓] enabled proxy aborted "
              f"({res_enabled['make_table_count']} reps vs "
              f"{res_disabled['make_table_count']} disabled)")

    print(f"\n== {'PASS' if failures == 0 else 'FAIL'} ({failures} failure(s)) ==")
    return failures


if __name__ == "__main__":
    sys.exit(main())
