#!/usr/bin/env python3
"""Smoke test for bench_rep_penalty.py: run the bench against a fake
upstream that responds with a known loop, verify the JSON artifact has
the expected shape, then run the diff command to verify it formats.

This doesn't validate that the rep-penalty WORKS (that needs a real
model). It validates that the bench harness:
  - Issues requests through the proxy
  - Detects loops via the loop_guard
  - Saves valid JSON
  - The diff command parses + formats correctly
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


_LOOP_TEXT = "I will use make_table now. Then the Mermaid code. " * 50
_HEALTHY_TEXT = (
    "Frey constructed an elliptic curve from a hypothetical Fermat "
    "counter-example. Ribet proved the curve couldn't be modular. "
    "Wiles then proved modularity for semistable elliptic curves, "
    "completing the contradiction. " * 2
)


class _Quiet(HTTPServer):
    def handle_error(self, request, client_address):
        import sys as _s
        if isinstance(_s.exc_info()[1], (BrokenPipeError, ConnectionResetError, OSError)):
            return
        super().handle_error(request, client_address)


class FakeUpstream(BaseHTTPRequestHandler):
    """Streams either a loop or healthy text based on a global toggle."""

    scenario = "loop"  # mutated between bench labels
    protocol_version = "HTTP/1.0"

    def log_message(self, fmt, *args):  # noqa: A003
        return

    def do_GET(self):  # noqa: N802
        if self.path == "/v1/models":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"data":[{"id":"fake-model","object":"model"}]}')
            return
        self.send_response(404); self.end_headers()

    def do_POST(self):  # noqa: N802
        length = int(self.headers.get("Content-Length", "0"))
        _ = self.rfile.read(length) if length else b""
        text = _LOOP_TEXT if FakeUpstream.scenario == "loop" else _HEALTHY_TEXT
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        self._send_frame({"choices": [{"delta": {"role": "assistant"}}]})
        try:
            for i in range(0, len(text), 32):
                self._send_frame({"choices": [{"delta": {"content": text[i:i+32]}}]})
            self._send_frame({"choices": [{"delta": {}, "finish_reason": "stop"}]})
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, OSError):
            return

    def _send_frame(self, obj):
        try:
            self.wfile.write(f"data: {json.dumps(obj)}\n\n".encode())
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, OSError):
            raise


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

    print("== bench_rep_penalty smoke test ==\n")

    with background_server(FakeUpstream, upstream_port):
        from http.server import ThreadingHTTPServer

        class _QuietT(ThreadingHTTPServer):
            def handle_error(self, request, client_address):
                import sys as _s
                if isinstance(_s.exc_info()[1], (BrokenPipeError, ConnectionResetError, OSError)):
                    return
                super().handle_error(request, client_address)

        for mod in ("qwen_proxy", "loop_guard", "bench_rep_penalty"):
            sys.modules.pop(mod, None)
        import qwen_proxy
        qwen_proxy.ProxyHandler.upstream_base = f"http://127.0.0.1:{upstream_port}"
        proxy_server = _QuietT(("127.0.0.1", proxy_port), qwen_proxy.ProxyHandler)
        threading.Thread(target=proxy_server.serve_forever, daemon=True).start()

        try:
            # Point the bench at our test proxy
            os.environ["QWEN_HOST"] = "127.0.0.1"
            os.environ["QWEN_PORT"] = str(proxy_port)
            sys.modules.pop("bench_rep_penalty", None)
            import bench_rep_penalty as bench

            # Run with the loop scenario active — bench should detect.
            FakeUpstream.scenario = "loop"
            from pathlib import Path
            bench_dir = Path("./bench_results")
            for stale in bench_dir.glob("smoke_*.json"):
                stale.unlink()

            print("[1] bench --label smoke_loop with loop scenario")
            out_loop = bench.run("smoke_loop", requests=2, max_tokens=512, use_proxy=True)
            sa = out_loop["summary"]
            if sa.get("successful", 0) != 2:
                print(f"    [✗] expected 2 successful, got {sa.get('successful')}")
                failures += 1
            elif sa.get("proxy_aborts", 0) < 2:
                print(f"    [✗] expected 2 proxy aborts on loop scenario, got "
                      f"{sa.get('proxy_aborts')}")
                failures += 1
            else:
                print(f"    [✓] {sa['proxy_aborts']} proxy aborts on loop scenario")
            if not (bench_dir / "smoke_loop.json").exists():
                print("    [✗] smoke_loop.json not saved")
                failures += 1
            else:
                print("    [✓] smoke_loop.json saved")

            # Switch to healthy scenario — should NOT abort
            FakeUpstream.scenario = "healthy"
            print("\n[2] bench --label smoke_healthy with healthy scenario")
            out_h = bench.run("smoke_healthy", requests=2, max_tokens=512, use_proxy=True)
            sb = out_h["summary"]
            if sb.get("proxy_aborts", 0) > 0:
                print(f"    [✗] healthy scenario should have 0 proxy aborts, got "
                      f"{sb.get('proxy_aborts')}")
                failures += 1
            else:
                print(f"    [✓] 0 proxy aborts on healthy scenario")
            if sb.get("loops_detected", 0) > 0:
                print(f"    [✗] healthy false-positive loops_detected="
                      f"{sb.get('loops_detected')}")
                failures += 1
            else:
                print(f"    [✓] 0 false-positive loop detections on healthy")

            # Diff — must run cleanly and report non-zero loops_drop in
            # smoke_loop direction (since smoke_healthy has 0 loops).
            print("\n[3] bench --diff smoke_loop smoke_healthy")
            import io, contextlib
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                ret = bench.diff("smoke_loop", "smoke_healthy")
            output = buf.getvalue()
            print(f"    diff returned: {ret}")
            if ret != 0:
                print("    [✗] diff returned non-zero")
                failures += 1
            elif "LOOPS:" not in output or "TPS:" not in output:
                print("    [✗] diff output missing LOOPS or TPS line")
                failures += 1
            else:
                # Show a snippet of the diff
                for line in output.splitlines():
                    if "LOOPS:" in line or "TPS:" in line or "loops_detected" in line:
                        print(f"    diff: {line.strip()}")
                print("    [✓] diff formatted correctly")

            # Streaming variant — must measure TTFT and produce identical
            # loop-detection signal.
            print("\n[4] bench --label smoke_stream_healthy with --stream")
            FakeUpstream.scenario = "healthy"
            out_s = bench.run("smoke_stream_healthy", requests=2,
                              max_tokens=512, use_proxy=True, streaming=True)
            ss = out_s["summary"]
            if "ttft_mean_s" not in ss:
                print(f"    [✗] streaming run didn't record TTFT")
                failures += 1
            elif ss["ttft_mean_s"] <= 0:
                print(f"    [✗] TTFT = {ss['ttft_mean_s']:.3f}s — should be > 0")
                failures += 1
            else:
                print(f"    [✓] streaming TTFT recorded: "
                      f"{ss['ttft_mean_s']*1000:.0f}ms mean")
            if ss.get("proxy_aborts", 0) > 0:
                print(f"    [✗] streaming healthy false-positive proxy abort")
                failures += 1
            else:
                print(f"    [✓] streaming healthy: 0 proxy aborts")

            print("\n[5] bench --label smoke_stream_loop with --stream + loop")
            FakeUpstream.scenario = "loop"
            out_sl = bench.run("smoke_stream_loop", requests=2,
                               max_tokens=512, use_proxy=True, streaming=True)
            ssl = out_sl["summary"]
            if ssl.get("proxy_aborts", 0) < 2:
                print(f"    [✗] streaming loop scenario only got "
                      f"{ssl.get('proxy_aborts')} aborts (expected 2)")
                failures += 1
            else:
                print(f"    [✓] streaming loop: {ssl['proxy_aborts']} aborts")

            # --- [6] bench --list shows the saved runs in a table ---
            print("\n[6] bench --list parses saved runs and prints a table")
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                ret = bench.list_runs()
            output = buf.getvalue()
            if ret != 0:
                print("    [✗] list_runs returned non-zero")
                failures += 1
            else:
                # All four smoke labels we just saved must appear in the
                # list output. Order doesn't matter (sorted by mtime).
                expected = ["smoke_loop", "smoke_healthy",
                            "smoke_stream_healthy", "smoke_stream_loop"]
                missing = [lbl for lbl in expected if lbl not in output]
                if missing:
                    print(f"    [✗] missing labels in --list output: {missing}")
                    failures += 1
                else:
                    # Spot-check that header columns exist
                    if not all(col in output for col in
                               ("label", "when", "TPS", "loops", "aborts")):
                        print("    [✗] --list output missing expected columns")
                        failures += 1
                    else:
                        print(f"    [✓] --list shows all {len(expected)} "
                              f"smoke runs with header columns")
        finally:
            proxy_server.shutdown()
            proxy_server.server_close()

    # --- [7] list_runs handles empty/missing bench_results directory ---
    print("\n[7] bench --list with no bench_results dir is a no-op")
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        cwd_orig = os.getcwd()
        os.chdir(tmp)
        try:
            sys.modules.pop("bench_rep_penalty", None)
            import bench_rep_penalty as bench2
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                ret = bench2.list_runs()
            output = buf.getvalue()
            if ret != 0 or "no" not in output.lower():
                print(f"    [✗] empty-dir case ret={ret} output={output!r}")
                failures += 1
            else:
                print("    [✓] empty-dir handled gracefully")
        finally:
            os.chdir(cwd_orig)

    print(f"\n== {'PASS' if failures == 0 else 'FAIL'} ({failures} failure(s)) ==")
    return failures


if __name__ == "__main__":
    sys.exit(main())
