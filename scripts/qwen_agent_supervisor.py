#!/usr/bin/env python3
"""qwen agent supervisor — runs scheduled agent prompts on intervals.

Single process, separate from the UI server, so agents keep running after
the user closes the browser/UI. Listens on 127.0.0.1:8003 for control HTTP
from qwen_ui.py.

Design constraints (memory-bound on M4 Pro 48 GB):
  * Single-flight: only ONE agent runs at a time. Concurrent agents would
    collide with the inference server's KV budget. Queue extras.
  * Stdlib only — same constraint as the rest of the project.
  * Persistent state lives entirely on disk under `~/.qwen/agents/`,
    so a supervisor restart picks up exactly where it left off.

Config layout (all per-user, in `~/.qwen/agents/<agent_id>/`):
    agent.json            — config + last-run summary, written via atomic rename
    runs/<run_id>.jsonl   — session log produced by `agent.py --headless`
    runs/<run_id>.meta.json
                          — {started_at, ended_at, status, exit_code, prompt}

HTTP API (control plane — UI proxies these):
    GET  /status                       — daemon health, current run, queue size
    GET  /agents                       — list all agents w/ computed status
    GET  /agents/<id>                  — single agent w/ recent runs
    GET  /agents/<id>/runs             — list runs (paginated)
    GET  /agents/<id>/runs/<run_id>    — full session log for one run
    POST /agents/<id>/run-now          — schedule immediate run
    POST /agents/<id>/cancel-run       — SIGTERM the running subprocess
                                          if the named agent is the active one
    POST /reload                       — rescan agents dir
    POST /shutdown                     — graceful stop

Note: agent CRUD (create/update/delete) is done by the UI by editing
config files directly. The supervisor watches the directory and picks up
changes on the next scan tick (≤5 s). This avoids putting business logic
in two places.
"""
from __future__ import annotations

import argparse
import contextlib
import datetime as _dt
import json
import logging
import os
import queue
import re
import signal
import subprocess
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

UI_HOME = Path(os.environ.get("QWEN_UI_HOME", str(Path.home() / ".qwen")))
AGENTS_DIR = UI_HOME / "agents"
LOG_PATH = UI_HOME / "agent_supervisor.log"
PID_PATH = UI_HOME / "agent_supervisor.pid"

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PYTHON = os.environ.get("QWEN_PYTHON", str(PROJECT_ROOT / "venv" / "bin" / "python"))
AGENT_PY = str(PROJECT_ROOT / "scripts" / "agent.py")

# Port choice: dflash-serve already binds 8000 (the qwen-proxy front) and
# 8002 (the dflash-mlx backend). The qwen-ui server lives on 8001. We
# take 8003 by default.
DEFAULT_PORT = int(os.environ.get("QWEN_SUPERVISOR_PORT", "8003"))
SCAN_INTERVAL_SEC = float(os.environ.get("QWEN_SUPERVISOR_SCAN_SEC", "5"))
RUN_TIMEOUT_SEC = int(os.environ.get("QWEN_SUPERVISOR_RUN_TIMEOUT", "1800"))  # 30 min
JITTER_FRAC = 0.20  # ±20 % so multiple agents don't lock-step

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("agent-supervisor")


# ---------------------------------------------------------------------------
# state
# ---------------------------------------------------------------------------

# Single mutex for the small bits of shared state we touch from both the
# scheduler thread and the HTTP handlers.
_lock = threading.RLock()
_force_run: set[str] = set()           # agent_ids the user clicked "run now"
_running_agent: dict | None = None     # {agent_id, run_id, started_at, popen}
_runs_log: list[dict] = []              # rolling buffer of run summaries (in-memory)
_started_at: float = time.time()
_supervisor_should_exit = threading.Event()


# ---------------------------------------------------------------------------
# config IO
# ---------------------------------------------------------------------------

def _ensure_dirs() -> None:
    AGENTS_DIR.mkdir(parents=True, exist_ok=True)


def _atomic_write_json(path: Path, data: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    tmp.replace(path)


def _read_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except FileNotFoundError:
        return None
    except json.JSONDecodeError as e:
        log.warning("bad config %s: %s", path, e)
        return None


def _scan_agents() -> list[dict]:
    """Read every `~/.qwen/agents/*/agent.json`. Returns a list of dicts
    with at minimum `id`. Skips broken configs (logged once)."""
    out: list[dict] = []
    if not AGENTS_DIR.is_dir():
        return out
    for child in sorted(AGENTS_DIR.iterdir()):
        if not child.is_dir():
            continue
        cfg = _read_json(child / "agent.json")
        if cfg is None:
            continue
        cfg.setdefault("id", child.name)
        out.append(cfg)
    return out


def _agent_runs(agent_id: str, limit: int = 20) -> list[dict]:
    runs_dir = AGENTS_DIR / agent_id / "runs"
    if not runs_dir.is_dir():
        return []
    metas: list[dict] = []
    for p in sorted(runs_dir.glob("*.meta.json"), reverse=True):
        m = _read_json(p)
        if m is None:
            continue
        m["run_id"] = p.stem.removesuffix(".meta")
        metas.append(m)
        if len(metas) >= limit:
            break
    return metas


# ---------------------------------------------------------------------------
# scheduling
# ---------------------------------------------------------------------------

def _now_ts() -> float:
    return time.time()


def _next_run_for(cfg: dict) -> float:
    """Compute when this agent should next fire. Force-run wins."""
    if cfg.get("id") in _force_run:
        return 0.0
    if not cfg.get("enabled", True):
        return float("inf")
    last = float(cfg.get("last_run_at") or 0)
    interval = float(cfg.get("interval_seconds") or 0)
    if interval <= 0:
        return float("inf")
    if last <= 0:
        return _now_ts()  # never run before — fire ASAP
    base = last + interval
    if base <= _now_ts():
        return base
    # ±20 % jitter so a fleet doesn't sync. Deterministic per-agent so
    # restarts don't shift the schedule each time.
    h = abs(hash((cfg.get("id"), int(last))))
    j = (h % 1000) / 1000.0  # 0.0 .. 1.0
    delta = (j - 0.5) * 2 * JITTER_FRAC * interval
    return base + delta


def _pick_next() -> dict | None:
    candidates = []
    for cfg in _scan_agents():
        t = _next_run_for(cfg)
        candidates.append((t, cfg))
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0])
    t, cfg = candidates[0]
    if t > _now_ts():
        return None
    return cfg


def _spawn_run(cfg: dict) -> int:
    """Run `agent.py --headless` for this agent, blocking until done.
    Updates the agent's `last_*` fields atomically. Returns exit code."""
    global _running_agent

    agent_id = cfg["id"]
    run_id = _dt.datetime.now().strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
    runs_dir = AGENTS_DIR / agent_id / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    session_log = runs_dir / f"{run_id}.jsonl"
    meta_path = runs_dir / f"{run_id}.meta.json"

    prompt = cfg.get("prompt") or ""
    cwd = cfg.get("cwd") or str(Path.home())
    if not Path(cwd).is_dir():
        cwd = str(Path.home())

    started_at = _now_ts()

    # Write meta in `running` state so the UI can show in-flight status.
    meta = {
        "agent_id": agent_id,
        "run_id": run_id,
        "started_at": started_at,
        "status": "running",
        "exit_code": None,
        "prompt": prompt,
        "cwd": cwd,
    }
    _atomic_write_json(meta_path, meta)

    env = os.environ.copy()
    env["QWEN_SESSION_LOG"] = "on"
    env["QWEN_SESSION_LOG_DIR"] = str(runs_dir)
    # Force the session log file to use our run_id so we can find it.
    # Achieved by pre-creating the path and letting agent.py append. But
    # agent.py picks its own filename (timestamp + pid). Easier: have it
    # write into runs_dir and we reconcile the filename after exit.

    cmd = [PYTHON, AGENT_PY, "--headless", "--prompt", prompt]
    log.info("[run %s] starting agent=%s cwd=%s", run_id, agent_id, cwd)

    try:
        proc = subprocess.Popen(
            cmd,
            cwd=cwd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
    except OSError as e:
        log.error("[run %s] spawn failed: %s", run_id, e)
        meta.update(status="failed", ended_at=_now_ts(), exit_code=-1, error=str(e))
        _atomic_write_json(meta_path, meta)
        return -1

    with _lock:
        _running_agent = {
            "agent_id": agent_id, "run_id": run_id,
            "started_at": started_at, "popen": proc,
        }

    try:
        out, _ = proc.communicate(timeout=RUN_TIMEOUT_SEC)
        rc = proc.returncode
        status = "ok" if rc == 0 else "fail"
    except subprocess.TimeoutExpired:
        proc.terminate()
        try:
            out, _ = proc.communicate(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            out, _ = proc.communicate()
        rc = -1
        status = "timeout"
    except Exception as e:  # noqa: BLE001
        log.exception("[run %s] unexpected error", run_id)
        out = ""
        rc = -1
        status = "error"
        meta["error"] = str(e)
    finally:
        with _lock:
            _running_agent = None

    ended_at = _now_ts()
    # Reconcile the session log path: agent.py writes its own filename
    # in QWEN_SESSION_LOG_DIR; we move the most-recent jsonl in there
    # to our run_id-named file.
    _claim_session_log(runs_dir, started_at, session_log)

    meta.update(status=status, ended_at=ended_at, exit_code=rc)
    if out:
        # Truncate captured stdout — used for the run summary tooltip.
        meta["stdout_tail"] = out[-2000:]
    _atomic_write_json(meta_path, meta)

    # Update the agent config with last-run info.
    _bump_last_run(agent_id, ended_at, status, run_id)

    log.info("[run %s] finished status=%s rc=%d in %.1fs",
             run_id, status, rc, ended_at - started_at)
    with _lock:
        _runs_log.append({"agent_id": agent_id, "run_id": run_id,
                          "status": status, "ended_at": ended_at})
        if len(_runs_log) > 200:
            _runs_log[:] = _runs_log[-200:]
        _force_run.discard(agent_id)
    return rc


def _claim_session_log(runs_dir: Path, started_at: float, target: Path) -> None:
    """Pick the freshest agent-*.jsonl in runs_dir that was created at or
    after `started_at` and rename it to `target`. agent.py names files
    `agent-YYYYMMDD-HHMMSS-PID.jsonl`."""
    if target.exists():
        return
    candidates = []
    for p in runs_dir.glob("agent-*.jsonl"):
        try:
            if p.stat().st_mtime >= started_at - 2:
                candidates.append((p.stat().st_mtime, p))
        except OSError:
            continue
    if not candidates:
        return
    _, src = max(candidates)
    try:
        src.replace(target)
    except OSError as e:
        log.warning("could not rename %s -> %s: %s", src, target, e)


def _bump_last_run(agent_id: str, ended_at: float, status: str, run_id: str) -> None:
    cfg_path = AGENTS_DIR / agent_id / "agent.json"
    cfg = _read_json(cfg_path)
    if cfg is None:
        return
    cfg["last_run_at"] = ended_at
    cfg["last_run_status"] = status
    cfg["last_run_id"] = run_id
    _atomic_write_json(cfg_path, cfg)


def _scheduler_loop() -> None:
    log.info("scheduler loop started; agents dir = %s", AGENTS_DIR)
    while not _supervisor_should_exit.is_set():
        try:
            cfg = _pick_next()
        except Exception:
            log.exception("pick_next failed")
            cfg = None
        if cfg is None:
            _supervisor_should_exit.wait(SCAN_INTERVAL_SEC)
            continue
        try:
            _spawn_run(cfg)
        except Exception:
            log.exception("spawn_run failed for %s", cfg.get("id"))
            time.sleep(2)
    log.info("scheduler loop exiting")


# ---------------------------------------------------------------------------
# HTTP control plane
# ---------------------------------------------------------------------------

def _send_json(handler: BaseHTTPRequestHandler, status: int, body) -> None:
    raw = json.dumps(body, ensure_ascii=False, default=str).encode()
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(raw)))
    handler.send_header("Access-Control-Allow-Origin", "*")
    handler.end_headers()
    handler.wfile.write(raw)


def _agent_status(cfg: dict) -> str:
    aid = cfg.get("id")
    if _running_agent and _running_agent.get("agent_id") == aid:
        return "running"
    if aid in _force_run:
        return "queued"
    if not cfg.get("enabled", True):
        return "paused"
    return "idle"


def _agent_summary(cfg: dict) -> dict:
    out = {
        "id": cfg.get("id"),
        "name": cfg.get("name") or cfg.get("id"),
        "enabled": cfg.get("enabled", True),
        "interval_seconds": cfg.get("interval_seconds"),
        "cwd": cfg.get("cwd"),
        "last_run_at": cfg.get("last_run_at"),
        "last_run_status": cfg.get("last_run_status"),
        "last_run_id": cfg.get("last_run_id"),
        "next_run_at": _next_run_for(cfg),
        "status": _agent_status(cfg),
        "created_at": cfg.get("created_at"),
    }
    if out["next_run_at"] == float("inf"):
        out["next_run_at"] = None
    return out


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):  # noqa: A003
        log.info("%s - %s", self.address_string(), fmt % args)

    def do_GET(self):  # noqa: N802
        path = self.path.split("?", 1)[0]
        if path == "/status":
            with _lock:
                running = None
                if _running_agent:
                    running = {
                        "agent_id": _running_agent["agent_id"],
                        "run_id": _running_agent["run_id"],
                        "started_at": _running_agent["started_at"],
                    }
                queued = sorted(_force_run)
                recent = list(_runs_log[-20:])
            _send_json(self, 200, {
                "ok": True,
                "pid": os.getpid(),
                "started_at": _started_at,
                "agents_dir": str(AGENTS_DIR),
                "running": running,
                "queued": queued,
                "recent_runs": recent,
            })
            return
        if path == "/agents":
            agents = [_agent_summary(c) for c in _scan_agents()]
            _send_json(self, 200, agents)
            return
        m = re.fullmatch(r"/agents/([\w\-]+)", path)
        if m:
            aid = m.group(1)
            cfg = _read_json(AGENTS_DIR / aid / "agent.json")
            if not cfg:
                _send_json(self, 404, {"error": "no such agent"})
                return
            cfg.setdefault("id", aid)
            summary = _agent_summary(cfg)
            summary["prompt"] = cfg.get("prompt", "")
            summary["recent_runs"] = _agent_runs(aid)
            _send_json(self, 200, summary)
            return
        m = re.fullmatch(r"/agents/([\w\-]+)/runs", path)
        if m:
            _send_json(self, 200, _agent_runs(m.group(1), limit=200))
            return
        m = re.fullmatch(r"/agents/([\w\-]+)/runs/([\w\-]+)", path)
        if m:
            aid, rid = m.group(1), m.group(2)
            jsonl_path = AGENTS_DIR / aid / "runs" / f"{rid}.jsonl"
            meta_path = AGENTS_DIR / aid / "runs" / f"{rid}.meta.json"
            meta = _read_json(meta_path) or {}
            events = []
            if jsonl_path.exists():
                try:
                    for line in jsonl_path.read_text().splitlines():
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            events.append(json.loads(line))
                        except json.JSONDecodeError:
                            pass
                except OSError as e:
                    _send_json(self, 500, {"error": str(e)})
                    return
            _send_json(self, 200, {"meta": meta, "events": events})
            return
        _send_json(self, 404, {"error": "not found", "path": path})

    def do_POST(self):  # noqa: N802
        path = self.path.split("?", 1)[0]
        if path == "/reload":
            _send_json(self, 200, {"ok": True})
            return
        if path == "/shutdown":
            _send_json(self, 200, {"ok": True, "msg": "shutting down"})
            _supervisor_should_exit.set()
            return
        m = re.fullmatch(r"/agents/([\w\-]+)/run-now", path)
        if m:
            aid = m.group(1)
            cfg = _read_json(AGENTS_DIR / aid / "agent.json")
            if not cfg:
                _send_json(self, 404, {"error": "no such agent"})
                return
            with _lock:
                _force_run.add(aid)
            _send_json(self, 202, {"ok": True, "queued": aid})
            return
        m = re.fullmatch(r"/agents/([\w\-]+)/cancel-run", path)
        if m:
            aid = m.group(1)
            with _lock:
                running = _running_agent
            if running and running.get("agent_id") == aid:
                proc = running.get("popen")
                if proc and proc.poll() is None:
                    proc.terminate()
                _send_json(self, 200, {"ok": True, "terminated": True})
                return
            with _lock:
                _force_run.discard(aid)
            _send_json(self, 200, {"ok": True, "terminated": False})
            return
        _send_json(self, 404, {"error": "not found", "path": path})


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------

def _write_pid() -> None:
    PID_PATH.parent.mkdir(parents=True, exist_ok=True)
    PID_PATH.write_text(str(os.getpid()))


def _clear_pid() -> None:
    with contextlib.suppress(OSError):
        PID_PATH.unlink()


def _setup_file_logging() -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(LOG_PATH)
    fh.setFormatter(logging.Formatter(
        "%(asctime)s %(levelname)s %(message)s",
        "%Y-%m-%d %H:%M:%S",
    ))
    log.addHandler(fh)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    ap.add_argument("--foreground", action="store_true",
                    help="don't daemonize / log to stderr only")
    args = ap.parse_args(argv)

    _ensure_dirs()
    _setup_file_logging()
    _write_pid()

    def _on_signal(signum, _frame):
        log.info("signal %s — shutting down", signum)
        _supervisor_should_exit.set()

    signal.signal(signal.SIGINT, _on_signal)
    signal.signal(signal.SIGTERM, _on_signal)

    # Pre-warm APC with the agent's static system prompt before the first
    # agent run fires. The supervisor often starts seconds before the first
    # scheduled agent, so warming up-front means the agent's first turn pays
    # ~40ms less TTFT instead of cold prefill of the ~600-token static prefix.
    # Best-effort — failure (server not yet up) is OK; the next agent run
    # will pay normal cold-cache cost.
    if not os.environ.get("QWEN_WARM_DISABLE"):
        def _warm_in_thread():
            try:
                warm_path = Path(__file__).parent / "warm_prompts.py"
                if not warm_path.exists():
                    return
                # Best-effort, single attempt; ignore failures.
                rc = subprocess.call(
                    [PYTHON, str(warm_path)],
                    timeout=20,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                log.info("warm_prompts.py exited rc=%d", rc)
            except Exception as e:  # noqa: BLE001
                log.info("warm_prompts.py skipped: %s", e)
        threading.Thread(target=_warm_in_thread, name="warm", daemon=True).start()

    sched_thread = threading.Thread(target=_scheduler_loop, name="sched", daemon=True)
    sched_thread.start()

    httpd = ThreadingHTTPServer((args.host, args.port), _Handler)
    log.info("supervisor listening on http://%s:%d", args.host, args.port)

    # Watcher thread: when the exit flag flips, tell the HTTP server to
    # break out of serve_forever(). serve_forever blocks otherwise.
    def _shutdown_watcher():
        _supervisor_should_exit.wait()
        log.info("shutdown watcher firing")
        httpd.shutdown()

    threading.Thread(target=_shutdown_watcher, name="shutdown", daemon=True).start()

    try:
        httpd.serve_forever(poll_interval=0.5)
    finally:
        log.info("supervisor stopping")
        httpd.server_close()
        _supervisor_should_exit.set()
        sched_thread.join(timeout=5)
        with _lock:
            r = _running_agent
        if r and r.get("popen") and r["popen"].poll() is None:
            log.info("terminating in-flight run %s", r["run_id"])
            r["popen"].terminate()
        _clear_pid()
    return 0


if __name__ == "__main__":
    sys.exit(main())
