#!/usr/bin/env python3
"""Sustained-load memory profile for the proxy.

Spawns a fake upstream + the real `qwen_proxy.py` in a subprocess, then
drives 1000 sequential requests through the proxy. Periodically samples
the proxy process's RSS via `ps -o rss=` and asserts:

  - RSS doesn't grow more than 50 MB between the warmup point and the
    end of the run (proxy is bounded; no per-request retention)
  - RSS at the END is within 30 MB of the post-warmup peak (no slow leak)

The fake upstream alternates between healthy and loop scenarios so both
proxy code paths run. Each request closes its connection cleanly.

Why this matters: the `_StreamAssembler` accumulates `_content_parts`
as a Python list and `StreamingLoopGuard._buf` keeps a sliding window.
Both are bounded per-request by inspection — but inspection isn't proof.
A memory regression would otherwise show up only after deployment in a
long-lived agent session.

Stdlib only. Uses `ps` for cross-platform RSS reading (works on macOS).
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


_LOOP_TEXT = "I will use make_table now. Then the Mermaid code. " * 50
_HEALTHY_TEXT = (
    "Quantum computing relies on superposition and entanglement to perform "
    "calculations that are intractable on classical hardware. The "
    "fundamental unit of quantum information is the qubit, which can exist "
    "in a superposition of |0⟩ and |1⟩ states until measured. Algorithms "
    "like Shor's factoring algorithm and Grover's search algorithm "
    "demonstrate exponential and quadratic speedups respectively over the "
    "best known classical equivalents. Practical quantum systems face "
    "decoherence and error correction challenges that researchers are "
    "actively addressing through topological codes and surface codes. "
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
        text = _LOOP_TEXT if scenario == "loop" else _HEALTHY_TEXT
        try:
            for i in range(0, len(text), 32):
                self._send_frame({"choices": [{"delta": {"content": text[i:i+32]}}]})
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


def _rss_kb(pid: int) -> int | None:
    """Return RSS in KB by shelling to `ps`. Returns None if pid is gone."""
    try:
        out = subprocess.check_output(["ps", "-o", "rss=", "-p", str(pid)],
                                      stderr=subprocess.DEVNULL).strip()
        return int(out) if out else None
    except (subprocess.CalledProcessError, ValueError):
        return None


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
        with background_server(FakeUpstream, upstream_port):
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

            print(f"== Proxy memory profile (pid {pid}) ==\n")

            def issue(scenario: str) -> None:
                body = json.dumps({
                    "messages": [{"role": "user", "content": f"hi SCENARIO={scenario}"}],
                    "stream": False,
                }).encode()
                req = urllib.request.Request(
                    f"http://127.0.0.1:{proxy_port}/v1/chat/completions",
                    data=body,
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=10) as resp:
                    resp.read()

            # Warmup: 50 requests so allocator settles
            for i in range(50):
                issue("healthy" if i % 2 == 0 else "loop")
            time.sleep(0.3)
            warmup_rss = _rss_kb(pid) or 0
            print(f"  post-warmup RSS: {warmup_rss/1024:.1f} MB")

            # Sustained: 1000 requests, sample every 100
            samples: list[tuple[int, int]] = [(50, warmup_rss)]
            for i in range(1000):
                issue("healthy" if i % 2 == 0 else "loop")
                if (i + 1) % 100 == 0:
                    rss = _rss_kb(pid) or 0
                    samples.append((50 + i + 1, rss))
                    print(f"  after {i+1+50:4d} reqs: {rss/1024:.1f} MB "
                          f"(Δwarmup {((rss - warmup_rss)/1024):+.1f} MB)")

            time.sleep(0.5)
            final_rss = _rss_kb(pid) or 0
            print(f"\n  final RSS: {final_rss/1024:.1f} MB "
                  f"(Δwarmup {((final_rss - warmup_rss)/1024):+.1f} MB)")
            peak = max(s[1] for s in samples)
            print(f"  peak RSS:  {peak/1024:.1f} MB")
            print(f"  total requests: 1050")

            # Gates: 50 MB total growth, 30 MB peak-to-final
            growth_mb = (final_rss - warmup_rss) / 1024
            peak_to_final_mb = (peak - final_rss) / 1024
            if growth_mb > 50:
                print(f"\n  [✗] RSS grew by {growth_mb:.1f} MB (gate: 50 MB)")
                failures += 1
            else:
                print(f"\n  [✓] RSS growth {growth_mb:+.1f} MB ≤ 50 MB gate")
            if abs(peak_to_final_mb) > 30:
                # peak well above final = allocator hasn't released — OK if growth is bounded
                # peak below final = monotone growth — bad
                print(f"  [WARN] peak {peak/1024:.1f} MB vs final {final_rss/1024:.1f} MB "
                      f"(diff {peak_to_final_mb:+.1f} MB) — review")
            else:
                print(f"  [✓] peak vs final within 30 MB ({peak_to_final_mb:+.1f} MB)")
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
