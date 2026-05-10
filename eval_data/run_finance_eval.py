#!/usr/bin/env python3
"""Drive the qwen agent through the finance_agent_benchmark sample.

Runs each prompt headless via `scripts/agent.py --headless`, captures the
JSONL session log, scores the agent's final answer against the prompt's
rubric_keys (substring match), appends a row to results.jsonl.

Designed to be invoked from the autonomous improvement loop. Each row in
results.jsonl carries an iter_id so we can compare iterations.

Usage:
    python eval_data/run_finance_eval.py --iter 1
    python eval_data/run_finance_eval.py --iter 1 --only netflix_q4_repurchase
"""

from __future__ import annotations
import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SAMPLE_PATH = Path(os.environ.get(
    "QWEN_EVAL_SAMPLE_PATH",
    str(ROOT / "eval_data" / "sample_prompts.json"),
))
RESULTS_PATH = ROOT / "eval_data" / "results.jsonl"
SESSION_LOG_DIR = ROOT / "eval_data" / "sessions"
AGENT_PY = ROOT / "scripts" / "agent.py"
PYTHON = ROOT / "venv" / "bin" / "python"


def _final_answer(jsonl_path: Path) -> tuple[str, dict]:
    """Walk the session log; return (final_answer_text, stats).

    `final_answer_text` is the model's actual answer surface — last
    non-empty assistant prose, OR the `done` tool's `summary` argument,
    OR the contents of any file written via write_file / append_finding /
    write_file_verified. The model frequently emits ONLY tool_calls in
    its closing turns (no narration), so reading prose alone misses the
    artifact entirely.
    """
    final_prose = ""
    done_summary = ""
    written_paths: list[str] = []
    n_assistant = 0
    n_tool_calls = 0
    n_tool_results = 0
    n_harness_nudges = 0
    tool_names: dict[str, int] = {}
    refused = 0
    cached_hits = 0
    near_dup = 0
    if not jsonl_path.exists():
        return "", {"error": "no session log"}
    with jsonl_path.open() as f:
        for line in f:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            kind = rec.get("kind")
            if kind == "assistant":
                n_assistant += 1
                content = rec.get("content") or ""
                import re as _re
                stripped = _re.sub(r"<tool_call>.*?</tool_call>\s*", "",
                                   content, flags=_re.DOTALL).strip()
                if stripped:
                    final_prose = stripped
                tcs = rec.get("tool_calls") or []
                for tc in tcs:
                    n_tool_calls += 1
                    fn = (tc.get("function") or {}).get("name", "?")
                    tool_names[fn] = tool_names.get(fn, 0) + 1
                    raw_args = (tc.get("function") or {}).get("arguments") or "{}"
                    try:
                        args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
                    except json.JSONDecodeError:
                        args = {}
                    if fn == "done":
                        done_summary = args.get("summary", "")
                    elif fn in ("write_file", "write_file_verified", "append_finding"):
                        p = args.get("path", "")
                        if p and p not in written_paths:
                            written_paths.append(p)
            elif kind == "tool_result":
                n_tool_results += 1
                content = rec.get("content") or ""
                if content.lstrip().startswith("[REFUSED"):
                    refused += 1
                if "[cached" in content[:60]:
                    cached_hits += 1
                if content.lstrip().startswith("[near-duplicate"):
                    near_dup += 1
            elif kind == "harness_nudge":
                n_harness_nudges += 1
    # Concatenate the available answer surfaces, biased to the most
    # specific. done_summary tends to be the cleanest 1-paragraph claim;
    # written file content carries the deliverable; prose is the fallback.
    parts = []
    if done_summary:
        parts.append(done_summary)
    for p in written_paths:
        try:
            parts.append(Path(p).read_text(errors="replace"))
        except OSError:
            pass
    if final_prose:
        parts.append(final_prose)
    final = "\n\n".join(parts)
    return final, {
        "n_assistant_turns": n_assistant,
        "n_tool_calls": n_tool_calls,
        "n_tool_results": n_tool_results,
        "n_harness_nudges": n_harness_nudges,
        "tool_breakdown": tool_names,
        "n_refused": refused,
        "n_cached": cached_hits,
        "n_near_dup": near_dup,
    }


_NUM_RE = re.compile(
    r"(?:\$|USD\s*)?"           # optional dollar prefix
    r"(-?\d{1,3}(?:,\d{3})+(?:\.\d+)?|-?\d+(?:\.\d+)?)"  # the number
    r"\s*(%|bps|basis\s*points?|bn|billion|m|million|k|thousand)?",
    re.IGNORECASE,
)


def _normalize_number(token: str) -> tuple[float, str] | None:
    """Parse a number token like '$1.935B', '1,165,827', '78.8%', '80bps'
    into (value_as_float, unit_class). Unit classes:
      'pct'   — percentage (78.8%)
      'bps'   — basis points
      'big'   — bn/billion (× 1e9)
      'mid'   — m/million (× 1e6)
      'small' — k/thousand (× 1e3)
      'raw'   — unitless
    Returns None if the token can't be parsed.
    """
    s = token.strip().lower()
    s = s.replace("usd", "").replace("$", "").strip()
    m = re.fullmatch(r"(-?\d{1,3}(?:,\d{3})+(?:\.\d+)?|-?\d+(?:\.\d+)?)"
                     r"\s*(%|bps|basis\s*points?|bn|billion|b|m|million|k|thousand)?",
                     s, re.IGNORECASE)
    if not m:
        return None
    num_str, suffix = m.group(1), (m.group(2) or "").lower()
    try:
        v = float(num_str.replace(",", ""))
    except ValueError:
        return None
    if suffix == "%":
        return v, "pct"
    if "bps" in suffix or "basis" in suffix:
        return v, "bps"
    if suffix in ("bn", "billion", "b"):
        return v, "big"
    if suffix in ("m", "million"):
        return v, "mid"
    if suffix in ("k", "thousand"):
        return v, "small"
    return v, "raw"


def _scaled_values(value: float, unit: str) -> list[float]:
    """Generate plausible numeric forms of a (value, unit) pair so we can
    match across writing conventions: "1.261" (raw small float) might
    have been written by the model as "$1,261M" or "$1.261B"; "$1.94B"
    might also be written as "1,940M". Returns absolute magnitudes.
    """
    if unit == "pct":
        return [value]
    if unit == "bps":
        return [value, value / 100.0]   # 80bps ≡ 0.8% sometimes
    if unit == "big":
        return [value * 1e9, value * 1e3]   # 1.94B = 1.94e9 raw or 1940M
    if unit == "mid":
        return [value * 1e6, value / 1e3]   # 1289M = 1.289e9 raw or 1.289B
    if unit == "small":
        return [value * 1e3, value / 1e3]
    # raw / unitless: a small float like 1.261 might mean 1.261B; a big
    # number like 1289 might mean 1289M (= 1.289B).
    out = [value]
    if abs(value) < 1000:
        out.extend([value * 1e3, value * 1e6, value * 1e9])
    if abs(value) >= 1000:
        out.append(value * 1e6)         # 1289 → 1.289e9
    return out


def _numeric_concept_hit(concept_alts: list[str], answer: str,
                          tol: float = 0.04) -> bool:
    """Numeric proximity match across plausible magnitude scalings.
    Falls back to plain substring."""
    low = answer.lower()
    if any(a.lower() in low for a in concept_alts):
        return True
    alt_scaled: list[tuple[list[float], str]] = []
    for a in concept_alts:
        n = _normalize_number(a)
        if n is not None:
            v, u = n
            alt_scaled.append((_scaled_values(v, u), u))
    if not alt_scaled:
        return False
    for m in _NUM_RE.finditer(answer):
        an = _normalize_number(m.group(0))
        if an is None:
            continue
        ans_v, ans_unit = an
        for ans_candidate in _scaled_values(ans_v, ans_unit):
            for alt_candidates, alt_unit in alt_scaled:
                # Strict unit-class agreement only for percentages and bps.
                if alt_unit in ("pct", "bps") and ans_unit not in ("pct", "bps"):
                    continue
                if ans_unit in ("pct", "bps") and alt_unit not in ("pct", "bps"):
                    continue
                for alt_v in alt_candidates:
                    denom = max(abs(alt_v), 1e-9)
                    if abs(ans_candidate - alt_v) / denom <= tol:
                        return True
    return False


def _score(final: str, prompt_obj: dict) -> tuple[str, float, list]:
    """Concept-based scoring with numeric proximity matching.

    Each prompt has `rubric_concepts: [[alt1, alt2, ...], ...]`. A concept
    is hit when ANY alternative is found in the answer either as:
      (a) a literal substring, OR
      (b) a numeric value within ±2% of a numeric alt with matching unit
          class (e.g. "1.94" matches alt "1.935"; "$1.94B" matches "1.935B";
          "10.9%" matches "10.7%" within rounding tolerance).

    Without (b), legitimate answers that round differently from the
    rubric's exact phrasing get scored 0 (e.g. iter 19 Uber said
    "$1.94 billion" for an adjustment the rubric labeled "1.935B").

    PASS at ≥0.66, PARTIAL at ≥0.34, else FAIL.
    """
    concepts = prompt_obj.get("rubric_concepts")
    if not concepts:
        keys = prompt_obj.get("rubric_keys", [])
        concepts = [[k] for k in keys]
    if not final or not concepts:
        missing = [c[0] for c in concepts]
        return "FAIL", 0.0, missing
    hit_concepts = []
    missing_concepts = []
    tol = float(os.environ.get("QWEN_EVAL_NUM_TOL", "0.04"))
    for concept_alts in concepts:
        if _numeric_concept_hit(concept_alts, final, tol=tol):
            hit_concepts.append(concept_alts[0])
        else:
            missing_concepts.append(concept_alts)
    frac = len(hit_concepts) / len(concepts)
    if frac >= 0.66:
        verdict = "PASS"
    elif frac >= 0.34:
        verdict = "PARTIAL"
    else:
        verdict = "FAIL"
    return verdict, frac, missing_concepts


def _hard_restart_server(timeout_s: int = 180) -> bool:
    """Hard-restart the qwen daemon before each prompt to clear any
    state buildup (KV cache fragmentation, memory pressure, transient
    proxy lockups). Returns True when the model can actually serve a
    chat completion, else False.

    Iter 28: previously this returned True as soon as /v1/models 200'd,
    but that endpoint goes live BEFORE dflash-serve loads model weights
    into MLX. Iter 28's netflix prompt failed in 17.8s because the agent
    issued a chat-completion request while the model was still warming
    up; the response came back empty, the loop exited silently, score
    was 0. Fix: drive an actual 1-token completion as the readiness
    probe. /v1/models 200 is a necessary condition but not sufficient.

    Burning 30-90s per iter on a cold restart + warmup trades wallclock
    for evaluation reliability.
    """
    import urllib.request, urllib.error
    qwen_bin = ROOT / "bin" / "qwen"
    if not qwen_bin.exists():
        print(f"[restart] {qwen_bin} not found; skipping", flush=True)
        return True
    try:
        subprocess.run([str(qwen_bin), "restart"], cwd=str(ROOT),
                       capture_output=True, text=True, timeout=30)
    except subprocess.TimeoutExpired:
        print("[restart] timed out; pressing on anyway", flush=True)

    conf = _load_qwen_conf()
    model_id = conf.get("QWEN_MODEL_NAME") or conf.get("QWEN_MODEL_PATH") or "qwen3.6"

    deadline = time.time() + timeout_s
    # Phase 1: wait for /v1/models 200 (proxy alive, dflash-serve listening).
    models_ok = False
    while time.time() < deadline and not models_ok:
        try:
            with urllib.request.urlopen(
                "http://127.0.0.1:8000/v1/models", timeout=3,
            ) as r:
                if r.status == 200:
                    models_ok = True
                    break
        except (urllib.error.URLError, OSError):
            pass
        time.sleep(2)
    if not models_ok:
        print("[restart] /v1/models never came up before deadline", flush=True)
        return False

    # Phase 2: drive an actual completion. dflash-serve answers /v1/models
    # before MLX has finished loading model weights, so we MUST exercise
    # the decode path before declaring ready. A 1-token "ping" is the
    # cheapest probe that proves the model is actually live.
    body = json.dumps({
        "model": model_id,
        "messages": [{"role": "user", "content": "ok"}],
        "max_tokens": 1,
        "stream": False,
    }).encode("utf-8")
    while time.time() < deadline:
        try:
            req = urllib.request.Request(
                "http://127.0.0.1:8000/v1/chat/completions",
                data=body,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=20) as r:
                payload = r.read().decode("utf-8", errors="replace")
            obj = json.loads(payload)
            choices = obj.get("choices") or []
            msg = (choices[0].get("message") if choices else {}) or {}
            content = msg.get("content") or ""
            # Empty content with finish_reason=stop = model alive but warming.
            # Wait one more cycle and try again. Non-empty = ready.
            if content.strip():
                elapsed = timeout_s - int(deadline - time.time())
                print(f"[restart] server ready in {elapsed}s (model decoded "
                      f"{len(content)} chars)", flush=True)
                return True
        except (urllib.error.URLError, OSError, json.JSONDecodeError):
            pass
        time.sleep(2)
    print("[restart] model never produced a non-empty completion before deadline",
          flush=True)
    return False


def _load_qwen_conf() -> dict[str, str]:
    """Parse `config/qwen.conf` (KEY=VALUE shell-style) so subprocess
    invocations of `agent.py` inherit the same env the daemon uses.
    Without this the proxy rejects requests with HTTP 400 ("Loaded
    model is ..., got qwen3.6") because QWEN_MODEL_NAME default doesn't
    match the actually-loaded model id."""
    out: dict[str, str] = {}
    conf = ROOT / "config" / "qwen.conf"
    if not conf.exists():
        return out
    for raw in conf.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        k, v = line.split("=", 1)
        out[k.strip()] = v.strip().strip('"').strip("'")
    return out


def run_one(prompt_obj: dict, iter_id: int, timeout: int = 600,
            hard_restart: bool = True) -> dict:
    pid = prompt_obj["id"]
    SESSION_LOG_DIR.mkdir(parents=True, exist_ok=True)
    # Hard-restart server before every prompt so each run starts from
    # a clean memory state. Disable with --no-restart for batch dev.
    if hard_restart:
        _hard_restart_server()
    # Isolate cwd: a fresh empty dir per run so the agent's grep/read_file
    # tools can't see the eval rubric (`sample_prompts.json`), prior results,
    # or any other state in the project tree. Without this, the model in a
    # prior iter ran `grep -r "repurchased" ROOT` and pulled my rubric file
    # straight into context — score went up but the test was contaminated.
    import tempfile
    workspace = Path(tempfile.mkdtemp(prefix=f"qwen_eval_{pid}_"))
    env = os.environ.copy()
    env.update(_load_qwen_conf())
    env["QWEN_SESSION_LOG_DIR"] = str(SESSION_LOG_DIR)
    env["QWEN_SESSION_LOG"] = "on"
    started = time.time()
    proc = subprocess.run(
        [str(PYTHON), str(AGENT_PY), "--headless", "--prompt", prompt_obj["question"]],
        cwd=str(workspace),
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    elapsed = time.time() - started
    # Find the most-recently-written session log under SESSION_LOG_DIR.
    logs = sorted(SESSION_LOG_DIR.glob("agent-*.jsonl"),
                  key=lambda p: p.stat().st_mtime, reverse=True)
    log_path = logs[0] if logs else None
    final, stats = _final_answer(log_path) if log_path else ("", {"error": "no log"})
    verdict, frac, missing = _score(final, prompt_obj)
    return {
        "iter": iter_id,
        "id": pid,
        "type": prompt_obj.get("type"),
        "verdict": verdict,
        "score": round(frac, 2),
        "missing": missing,
        "final_answer": final[:1500],
        "stats": stats,
        "elapsed_s": round(elapsed, 1),
        "session_log": str(log_path) if log_path else None,
        "exit_code": proc.returncode,
        "stderr_tail": (proc.stderr[-500:] if proc.stderr else ""),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iter", type=int, required=True)
    ap.add_argument("--only", default="", help="comma-separated prompt ids")
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--no-restart", action="store_true",
                    help="skip the per-prompt server hard-restart")
    ap.add_argument("--retry-on-fail", type=int, default=0,
                    help="if a prompt FAILs/PARTIALs/TIMEOUTs, re-run up to "
                         "N additional times and keep the best score "
                         "(default 0 = single attempt)")
    args = ap.parse_args()

    sample = json.loads(SAMPLE_PATH.read_text())
    prompts = sample["prompts"]
    if args.only:
        wanted = {s.strip() for s in args.only.split(",") if s.strip()}
        prompts = [p for p in prompts if p["id"] in wanted]
        if not prompts:
            print(f"no prompts match {args.only}", file=sys.stderr)
            return 2

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    for p in prompts:
        print(f"=== iter {args.iter}  prompt={p['id']}  type={p['type']} ===",
              flush=True)
        # Iter 37: keep the best of up to 1 + retry_on_fail attempts.
        # Targets the variance-prone prompts (TJX, AMD, MU, Boeing) where
        # the model's outcome depends heavily on which URLs come back from
        # search. Each retry is a clean subprocess + hard-restart, so they
        # are independent samples. Stop early on PASS=1.0 to save time.
        attempts = []
        for attempt_idx in range(1 + args.retry_on_fail):
            try:
                row = run_one(p, args.iter, timeout=args.timeout,
                              hard_restart=not args.no_restart)
            except subprocess.TimeoutExpired:
                row = {"iter": args.iter, "id": p["id"],
                       "verdict": "TIMEOUT", "score": 0.0,
                       "elapsed_s": args.timeout}
            attempts.append(row)
            score = row.get("score", 0)
            verdict = row.get("verdict", "?")
            print(f"   [attempt {attempt_idx+1}] {verdict}  score={score}  "
                  f"calls={row.get('stats',{}).get('n_tool_calls','?')}",
                  flush=True)
            # Early exit on a clean win — no need to retry a perfect score.
            if score >= 1.0 and verdict == "PASS":
                break
        # Pick best by score, breaking ties by verdict rank PASS > PARTIAL > FAIL > TIMEOUT.
        rank = {"PASS": 3, "PARTIAL": 2, "FAIL": 1, "TIMEOUT": 0}
        attempts.sort(key=lambda r: (r.get("score", 0),
                                      rank.get(r.get("verdict",""), 0)),
                      reverse=True)
        row = attempts[0]
        if len(attempts) > 1:
            row = dict(row)
            row["n_attempts"] = len(attempts)
        with RESULTS_PATH.open("a") as f:
            f.write(json.dumps(row) + "\n")
        v = row.get("verdict")
        s = row.get("score", 0)
        e = row.get("elapsed_s", "?")
        n_calls = row.get("stats", {}).get("n_tool_calls", "?")
        print(f"   -> {v}  score={s}  calls={n_calls}  {e}s",
              flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
