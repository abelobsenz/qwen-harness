#!/usr/bin/env python3
"""Internal daemon launcher: starts dflash-serve + qwen-proxy as one unit.

Invoked by bin/qwen for `start -d`. The two processes are siblings inside
a single process group (we call os.setpgrp early), so a SIGINT to the
parent group cleanly takes them both down.

Layout:
    parent (this script, sets pgrp)
       ├── dflash-serve  (internal port)
       └── qwen-proxy    (public port)

The proxy parses dflash-serve's <tool_call><function=...> output back into
OpenAI-format tool_calls, so the agent never sees raw XML.
"""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request


def wait_ready(url: str, timeout: float) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2) as r:
                if r.status == 200:
                    return True
        except (urllib.error.URLError, ConnectionError, TimeoutError):
            pass
        time.sleep(1)
    return False


def _port_in_use(host: str, port: int) -> bool:
    """Quick TCP connect probe — True if something is already bound to
    `host:port`. Used as a pre-flight check so we fail fast with a clear
    message instead of letting the child crash silently on EADDRINUSE."""
    import socket
    s = socket.socket()
    s.settimeout(0.5)
    try:
        s.connect((host, port))
        return True
    except (ConnectionRefusedError, socket.timeout, OSError):
        return False
    finally:
        try:
            s.close()
        except OSError:
            pass


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--public-port", type=int, default=8000)
    ap.add_argument("--upstream-port", type=int, default=8002)
    ap.add_argument("--proxy-script", required=True,
                    help="absolute path to qwen_proxy.py")
    ap.add_argument("--public-host", default="127.0.0.1")
    ap.add_argument("--ready-timeout", type=float, default=180.0)
    # Everything after `--` goes verbatim to dflash-serve.
    ap.add_argument("dflash_args", nargs=argparse.REMAINDER)
    args = ap.parse_args()

    # Strip an explicit "--" separator if present.
    dflash_args = [a for a in args.dflash_args if a != "--"]

    os.setpgrp()

    # Pre-flight port checks so we fail fast with a clear error if
    # 8000 or 8002 is already held. Without this the child crashes on
    # EADDRINUSE and the daemon exits with no obvious cause — silently
    # fatal at startup. See audit issue G.
    for label, host, port in (
        ("dflash", "127.0.0.1", args.upstream_port),
        ("proxy", args.public_host, args.public_port),
    ):
        if _port_in_use(host, port):
            print(f"_qwen_daemon: {label} port {host}:{port} is already in use. "
                  f"Run `bin/qwen stop` to clear the prior daemon, or "
                  f"`lsof -i :{port}` to find the holder.", file=sys.stderr)
            return 3

    # Use our patched launcher so apc_patch.install() runs before
    # dflash_mlx.serve.main(). Falls back to plain dflash-serve if the
    # launcher script is missing.
    patched_launcher = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "dflash_serve_patched.py",
    )
    if os.path.exists(patched_launcher):
        # Use the same Python that's running this daemon — guarantees
        # the right venv even if the launcher's shebang resolves
        # differently.
        dflash_cmd = [
            sys.executable, patched_launcher,
            "--port", str(args.upstream_port), *dflash_args,
        ]
    else:
        dflash_cmd = [
            "dflash-serve", "--port", str(args.upstream_port), *dflash_args
        ]
    print(f"_qwen_daemon: starting dflash-serve on :{args.upstream_port}")
    sys.stdout.flush()
    dflash = subprocess.Popen(dflash_cmd)

    upstream_url = f"http://127.0.0.1:{args.upstream_port}"
    if not wait_ready(upstream_url + "/v1/models", args.ready_timeout):
        print(f"_qwen_daemon: upstream {upstream_url} not ready in "
              f"{args.ready_timeout}s — aborting", file=sys.stderr)
        dflash.terminate()
        try:
            dflash.wait(timeout=10)
        except subprocess.TimeoutExpired:
            dflash.kill()
        return 2

    proxy_cmd = [
        sys.executable, args.proxy_script,
        "--listen-host", args.public_host,
        "--listen-port", str(args.public_port),
        "--upstream", upstream_url,
    ]

    def _start_proxy() -> subprocess.Popen:
        print(f"_qwen_daemon: starting qwen-proxy on {args.public_host}:{args.public_port}")
        sys.stdout.flush()
        return subprocess.Popen(proxy_cmd)

    proxy = _start_proxy()

    # Independent supervision (audit issue D fix). The OLD behaviour
    # killed BOTH children when EITHER exited. That meant a crashed
    # proxy (cheap to restart, ~50ms) would force a full dflash reload
    # (~3min for the 35B). New policy:
    #   * dflash exit → tear down proxy too (no upstream to talk to,
    #     proxy would just 502 every request).
    #   * proxy exit → restart the proxy with exponential backoff. The
    #     model stays loaded; the user sees a brief interruption while
    #     the proxy comes back up.
    # If the proxy keeps crashing past _PROXY_MAX_RESTARTS within
    # _PROXY_RESTART_WINDOW seconds, treat it as a deeper bug and tear
    # down the daemon so the operator notices.
    _PROXY_MAX_RESTARTS = int(os.environ.get("QWEN_PROXY_MAX_RESTARTS", "5"))
    _PROXY_RESTART_WINDOW = float(os.environ.get(
        "QWEN_PROXY_RESTART_WINDOW", "120"))
    _PROXY_BACKOFF_BASE = float(os.environ.get(
        "QWEN_PROXY_BACKOFF_BASE", "1.0"))
    proxy_restart_times: list[float] = []

    shutting_down = [False]

    def shutdown(_signo=None, _frame=None):
        shutting_down[0] = True
        for p in (proxy, dflash):
            if p.poll() is None:
                try:
                    p.terminate()
                except (OSError, ProcessLookupError):
                    pass

    signal.signal(signal.SIGTERM, shutdown)
    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGHUP, shutdown)

    try:
        while not shutting_down[0]:
            # dflash death → terminal: nothing to serve from anyway.
            dflash_rc = dflash.poll()
            if dflash_rc is not None:
                print(f"_qwen_daemon: dflash pid={dflash.pid} exited rc={dflash_rc} — "
                      "tearing down proxy (no upstream).")
                sys.stdout.flush()
                shutdown()
                break
            # Proxy death → recoverable: restart with backoff. The model
            # remains hot; only ~50-200ms downtime per restart.
            proxy_rc = proxy.poll()
            if proxy_rc is not None:
                now = time.monotonic()
                # Slide the window: drop restart timestamps older than
                # _PROXY_RESTART_WINDOW so a daemon that's been up for
                # hours doesn't accumulate stale restart counts.
                proxy_restart_times = [
                    t for t in proxy_restart_times
                    if now - t < _PROXY_RESTART_WINDOW
                ]
                if len(proxy_restart_times) >= _PROXY_MAX_RESTARTS:
                    print(f"_qwen_daemon: proxy crashed {len(proxy_restart_times)}× "
                          f"in {_PROXY_RESTART_WINDOW:.0f}s — giving up and tearing "
                          f"down dflash. This usually means a config issue or a "
                          f"genuine bug in qwen_proxy.py.", file=sys.stderr)
                    sys.stderr.flush()
                    shutdown()
                    break
                # Exponential backoff: 1s, 2s, 4s, 8s, … capped at 16s.
                backoff = min(
                    _PROXY_BACKOFF_BASE * (2 ** len(proxy_restart_times)),
                    16.0,
                )
                print(f"_qwen_daemon: proxy pid={proxy.pid} exited rc={proxy_rc} — "
                      f"restarting in {backoff:.1f}s "
                      f"(attempt {len(proxy_restart_times) + 1}/"
                      f"{_PROXY_MAX_RESTARTS}; dflash stays loaded).")
                sys.stdout.flush()
                time.sleep(backoff)
                if shutting_down[0]:
                    break
                proxy_restart_times.append(now)
                proxy = _start_proxy()
                continue
            time.sleep(0.5)
    except KeyboardInterrupt:
        shutdown()

    # Final wait on both.
    for p in (proxy, dflash):
        try:
            p.wait(timeout=10)
        except subprocess.TimeoutExpired:
            p.kill()
    # Return code policy: dflash exit propagates as the daemon's exit
    # code (it's the load-bearing process). Proxy crashes that we
    # successfully recovered from don't affect the exit code.
    if dflash.returncode is not None:
        return dflash.returncode
    if proxy.returncode is not None:
        return proxy.returncode
    return 0


if __name__ == "__main__":
    sys.exit(main())
