#!/usr/bin/env python3
"""bench_rep_penalty: A/B harness for the runtime rep-penalty patch.

Companion to `scripts/runtime_patch.py`. The runtime hook is wired but
off by default; this harness validates a candidate `DFLASH_REP_PENALTY`
value on a real model session before promoting it to "default on."

Usage (when dflash-serve is up):

    # Baseline: no patch installed
    python scripts/bench_rep_penalty.py --label baseline

    # With runtime hook enabled:
    DFLASH_REP_PENALTY=0.05 python scripts/bench_rep_penalty.py --label rep05

    # Diff:
    python scripts/bench_rep_penalty.py --diff baseline rep05

The bench:
  1. Issues N requests to dflash-serve (or the proxy) with the canonical
     "make_table" prompt that's known to loop on bare greedy decoding.
  2. Records, per request: wall_seconds, generated_chars, generated_tokens
     (estimated via len/4), loop-guard reason (if any), final response shape.
  3. Computes per-request TPS estimates plus aggregate summary.
  4. Saves results as JSON to ./bench_results/{label}.json so subsequent
     `--diff` runs can compare.
  5. The diff command prints a side-by-side table of the two labels'
     metrics, with deltas.

Defaults are tuned to be cheap (3 requests by default; max_tokens=512).
The user can crank N via --requests for tighter statistics on long runs.

The bench respects the SAME env vars as the proxy + agent so it's
deployable in the existing daemon environment without surprises:

  QWEN_HOST       127.0.0.1
  QWEN_PORT       8000   # proxy by default; --raw-upstream switches to 8002
  QWEN_MODEL_NAME qwen3.6 (or whatever resolve_model_id returns)
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


CANONICAL_LOOP_PROMPT = (
    "Make me a table summarizing the contributions of Frey, Ribet, and "
    "Wiles to Fermat's Last Theorem. Then add a mind-map-like graph showing "
    "how their contributions connect. Be thorough but concise."
)
# Why this specific prompt: it's the one that originally surfaced the
# 14 KB "make_table now / Mermaid code" loop in the user-reported bug.
# Any rep-penalty value that prevents this loop on this prompt is a
# meaningful signal that the patch generalizes.


def _llm_endpoint(use_proxy: bool) -> str:
    host = os.environ.get("QWEN_HOST", "127.0.0.1")
    if host in ("0.0.0.0", ""):
        host = "127.0.0.1"
    if use_proxy:
        port = os.environ.get("QWEN_PORT", "8000")
    else:
        # Direct upstream port from config/qwen.conf.
        port = os.environ.get("DFLASH_PORT", "8002")
    return f"http://{host}:{port}/v1/chat/completions"


def _model_id() -> str:
    """Best-effort: ask /v1/models for the loaded model id."""
    host = os.environ.get("QWEN_HOST", "127.0.0.1")
    port = os.environ.get("QWEN_PORT", "8000")
    url = f"http://{host}:{port}/v1/models"
    try:
        with urllib.request.urlopen(url, timeout=4) as r:
            data = json.loads(r.read())
        items = data.get("data") or []
        if items:
            return str(items[0].get("id"))
    except Exception:  # noqa: BLE001
        pass
    return os.environ.get("QWEN_MODEL_NAME", "qwen3.6")


def issue_one(endpoint: str, model: str, max_tokens: int,
              streaming: bool = False) -> dict:
    """Issue a single request. When `streaming=True`, opens an SSE
    stream and records time-to-first-token in addition to total wall.

    Returns dict with metrics; on error sets `error` field.
    """
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": CANONICAL_LOOP_PROMPT}],
        "stream": bool(streaming),
        "max_tokens": max_tokens,
    }).encode("utf-8")
    req = urllib.request.Request(
        endpoint,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.perf_counter()
    if not streaming:
        # Non-streaming: single read of the assembled response.
        try:
            with urllib.request.urlopen(req, timeout=600) as resp:
                raw = resp.read()
        except urllib.error.HTTPError as e:
            return {"error": f"HTTP {e.code}", "elapsed_s": time.perf_counter() - t0}
        except urllib.error.URLError as e:
            return {"error": f"URLError: {e}", "elapsed_s": time.perf_counter() - t0}
        elapsed = time.perf_counter() - t0
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return {"error": "non-json response", "elapsed_s": elapsed}
        msg = (data.get("choices") or [{}])[0].get("message") or {}
        content = msg.get("content") or ""
        finish = (data.get("choices") or [{}])[0].get("finish_reason")
        usage = data.get("usage") or {}
        completion_tokens = usage.get("completion_tokens") or len(content) // 4
        ttft_s = None
    else:
        # Streaming: open SSE stream, measure TTFT separately.
        try:
            resp = urllib.request.urlopen(req, timeout=600)
        except urllib.error.HTTPError as e:
            return {"error": f"HTTP {e.code}", "elapsed_s": time.perf_counter() - t0}
        except urllib.error.URLError as e:
            return {"error": f"URLError: {e}", "elapsed_s": time.perf_counter() - t0}
        chunks: list[str] = []
        ttft_s: float | None = None
        finish: str | None = None
        try:
            for raw_line in resp:
                line = raw_line.decode("utf-8", errors="ignore").strip()
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    break
                try:
                    obj = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                delta = (obj.get("choices") or [{}])[0].get("delta") or {}
                # TTFT measured at the first non-empty content frame
                # (NOT the role frame — that's just SSE overhead).
                content_chunk = delta.get("content") or ""
                if content_chunk and ttft_s is None:
                    ttft_s = time.perf_counter() - t0
                if content_chunk:
                    chunks.append(content_chunk)
                fr = (obj.get("choices") or [{}])[0].get("finish_reason")
                if fr:
                    finish = fr
        finally:
            try:
                resp.close()
            except Exception:  # noqa: BLE001
                pass
        elapsed = time.perf_counter() - t0
        content = "".join(chunks)
        completion_tokens = len(content) // 4
    # Run the loop_guard offline against the response to flag whether the
    # text DID contain a loop pattern. Independent of whether the proxy
    # aborted (proxy might not be in the loop in raw-upstream mode).
    from loop_guard import check_text  # noqa: E402
    loop_check = check_text(content)
    out = {
        "elapsed_s": elapsed,
        "completion_tokens": completion_tokens,
        "tps": completion_tokens / elapsed if elapsed > 0 else 0.0,
        "content_len": len(content),
        "finish_reason": finish,
        "proxy_aborted": "[loop-guard:" in content,
        "offline_loop_detected": loop_check.triggered,
        "offline_loop_reason": loop_check.reason if loop_check.triggered else "",
        "content_tail": content[-200:] if content else "",
    }
    if streaming:
        out["ttft_s"] = ttft_s
    return out


def run(label: str, requests: int, max_tokens: int, use_proxy: bool,
        streaming: bool = False) -> dict:
    endpoint = _llm_endpoint(use_proxy)
    model = _model_id()
    print(f"== bench_rep_penalty (label={label!r}) ==")
    print(f"  endpoint:    {endpoint}")
    print(f"  model:       {model}")
    print(f"  requests:    {requests}")
    print(f"  max_tokens:  {max_tokens}")
    print(f"  streaming:   {streaming}")
    print(f"  rep_penalty env: DFLASH_REP_PENALTY="
          f"{os.environ.get('DFLASH_REP_PENALTY', '0.0 (default)')}")
    print()
    results: list[dict] = []
    for i in range(requests):
        r = issue_one(endpoint, model, max_tokens, streaming=streaming)
        results.append(r)
        if r.get("error"):
            print(f"  [{i+1}/{requests}] ERROR: {r['error']}")
            continue
        marker = "🚨" if r.get("offline_loop_detected") else "✓ "
        proxy = " [proxy-aborted]" if r.get("proxy_aborted") else ""
        ttft_str = ""
        if streaming and r.get("ttft_s") is not None:
            ttft_str = f" ttft={r['ttft_s']*1000:.0f}ms"
        print(f"  [{i+1}/{requests}] {marker} "
              f"{r['completion_tokens']:>4d} tok in "
              f"{r['elapsed_s']:>5.2f}s = {r['tps']:>5.1f} TPS"
              f"{ttft_str}"
              f" finish={r['finish_reason']}"
              f" loop={r['offline_loop_reason'] or '-'}{proxy}")

    summary = _summarize(results)
    out = {
        "label": label,
        "endpoint": endpoint,
        "model": model,
        "requests": requests,
        "max_tokens": max_tokens,
        "streaming": streaming,
        "env": {
            k: os.environ.get(k, "")
            for k in ("DFLASH_REP_PENALTY", "DFLASH_REP_HISTORY",
                      "LOOP_GUARD_DISABLE", "QWEN_PROXY_COMPACT_SCHEMA")
        },
        "summary": summary,
        "results": results,
    }
    bench_dir = Path("./bench_results")
    bench_dir.mkdir(exist_ok=True)
    path = bench_dir / f"{label}.json"
    path.write_text(json.dumps(out, indent=2))
    print(f"\n  saved: {path}")
    print(_format_summary(summary))
    return out


def _summarize(results: list[dict]) -> dict:
    ok = [r for r in results if not r.get("error")]
    if not ok:
        return {"successful": 0}
    tps = [r["tps"] for r in ok]
    elapsed = [r["elapsed_s"] for r in ok]
    completion_tokens = [r["completion_tokens"] for r in ok]
    # TTFT is only set when the run was streaming. None values are
    # filtered out so this works for both streaming + non-streaming.
    ttfts = [r["ttft_s"] for r in ok
             if r.get("ttft_s") is not None]
    summary = {
        "successful": len(ok),
        "errored": len(results) - len(ok),
        "loops_detected": sum(1 for r in ok if r.get("offline_loop_detected")),
        "proxy_aborts": sum(1 for r in ok if r.get("proxy_aborted")),
        "tps_mean": statistics.mean(tps) if tps else 0.0,
        "tps_p50": statistics.median(tps) if tps else 0.0,
        "tps_min": min(tps) if tps else 0.0,
        "tps_max": max(tps) if tps else 0.0,
        "elapsed_total_s": sum(elapsed),
        "completion_tokens_mean": (statistics.mean(completion_tokens)
                                   if completion_tokens else 0),
    }
    if ttfts:
        summary["ttft_mean_s"] = statistics.mean(ttfts)
        summary["ttft_p50_s"] = statistics.median(ttfts)
        summary["ttft_min_s"] = min(ttfts)
        summary["ttft_max_s"] = max(ttfts)
    return summary


def _format_summary(s: dict) -> str:
    if s.get("successful", 0) == 0:
        return "  (no successful requests — endpoint unreachable or all errored)"
    out = (
        f"\n  -- summary --\n"
        f"  successful:      {s['successful']}\n"
        f"  errored:         {s.get('errored', 0)}\n"
        f"  loops detected:  {s.get('loops_detected', 0)} of {s['successful']} "
        f"({100*s.get('loops_detected', 0)/s['successful']:.0f}%)\n"
        f"  proxy aborts:    {s.get('proxy_aborts', 0)}\n"
        f"  TPS mean:        {s['tps_mean']:.1f}\n"
        f"  TPS p50/min/max: {s['tps_p50']:.1f} / {s['tps_min']:.1f} / "
        f"{s['tps_max']:.1f}\n"
        f"  elapsed total:   {s['elapsed_total_s']:.1f}s"
    )
    if "ttft_mean_s" in s:
        out += (
            f"\n  TTFT mean:       {s['ttft_mean_s']*1000:.0f}ms\n"
            f"  TTFT p50/min/max: "
            f"{s['ttft_p50_s']*1000:.0f}ms / "
            f"{s['ttft_min_s']*1000:.0f}ms / "
            f"{s['ttft_max_s']*1000:.0f}ms"
        )
    return out


def diff(label_a: str, label_b: str) -> int:
    pa = Path("./bench_results") / f"{label_a}.json"
    pb = Path("./bench_results") / f"{label_b}.json"
    if not pa.exists() or not pb.exists():
        print(f"  missing: {pa.exists() and 'have' or 'MISSING'} {pa}")
        print(f"  missing: {pb.exists() and 'have' or 'MISSING'} {pb}")
        return 2
    a = json.loads(pa.read_text())
    b = json.loads(pb.read_text())
    sa, sb = a["summary"], b["summary"]

    def cell(metric: str, fmt: str = "{:.1f}") -> str:
        va = sa.get(metric, 0)
        vb = sb.get(metric, 0)
        try:
            d = vb - va
            # Delta gets a leading sign. Format depends on whether the
            # base format is integer or float; mirroring the format
            # string keeps columns aligned across rows.
            if isinstance(d, int) and "{:d}" in fmt:
                d_str = f"{d:+d}"
            else:
                d_str = f"{d:+.1f}"
            return f"{fmt.format(va):>10s}  {fmt.format(vb):>10s}  {d_str:>10s}"
        except (TypeError, ValueError):
            return f"{va!s:>10s}  {vb!s:>10s}  {'?':>10s}"

    print(f"== bench_rep_penalty diff ==")
    print(f"  {a['label']!r} vs {b['label']!r}\n")
    print(f"  {'metric':<22}{label_a:>10}  {label_b:>10}  {'delta':>10}")
    print(f"  {'-'*22}{'-'*10}  {'-'*10}  {'-'*10}")
    print(f"  {'successful':<22}{cell('successful', '{:d}')}")
    print(f"  {'loops_detected':<22}{cell('loops_detected', '{:d}')}")
    print(f"  {'proxy_aborts':<22}{cell('proxy_aborts', '{:d}')}")
    print(f"  {'tps_mean':<22}{cell('tps_mean')}")
    print(f"  {'tps_p50':<22}{cell('tps_p50')}")
    print(f"  {'completion_tokens':<22}{cell('completion_tokens_mean')}")
    print(f"  {'elapsed_total_s':<22}{cell('elapsed_total_s')}")
    # TTFT only present when both labels were streaming runs.
    if "ttft_mean_s" in sa or "ttft_mean_s" in sb:
        print(f"  {'ttft_mean_s':<22}{cell('ttft_mean_s', '{:.3f}')}")
        print(f"  {'ttft_p50_s':<22}{cell('ttft_p50_s', '{:.3f}')}")

    # Headline judgement
    print()
    loops_drop = sa.get("loops_detected", 0) - sb.get("loops_detected", 0)
    tps_change_pct = 0.0
    if sa.get("tps_mean", 0) > 0:
        tps_change_pct = (sb.get("tps_mean", 0) - sa["tps_mean"]) / sa["tps_mean"] * 100
    if loops_drop > 0:
        print(f"  LOOPS: {label_b} caught {loops_drop} fewer loops than {label_a}")
    elif loops_drop < 0:
        print(f"  LOOPS: {label_b} caught {-loops_drop} MORE loops than {label_a}")
    else:
        print(f"  LOOPS: same count in both")
    print(f"  TPS:   {tps_change_pct:+.1f}% (mean) — "
          f"{'regression' if tps_change_pct < -3 else 'stable' if abs(tps_change_pct) < 3 else 'improvement'}")
    return 0


def list_runs() -> int:
    """Print a table of all saved bench runs (label, run-time, key metrics).

    Reads `bench_results/*.json`. Skips files that don't parse as the
    expected shape (corrupt or hand-written). Sorted by file mtime
    descending — newest first — so the user sees recent runs at the top.

    Useful as `bench_rep_penalty.py --list` between A/B sessions to
    see what labels exist and recall historical numbers without
    re-running.
    """
    import datetime
    bench_dir = Path("./bench_results")
    if not bench_dir.exists():
        print("(no ./bench_results directory yet — run a bench first)")
        return 0
    rows: list[tuple[float, dict]] = []
    for path in bench_dir.glob("*.json"):
        try:
            data = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if "label" not in data or "summary" not in data:
            continue
        rows.append((path.stat().st_mtime, data))
    if not rows:
        print(f"(no parseable runs in {bench_dir.resolve()})")
        return 0
    rows.sort(key=lambda r: r[0], reverse=True)

    # Header
    cols = ("label", "when", "stream", "rep_pen", "TPS", "TTFT", "loops",
            "aborts", "n")
    widths = (24, 16, 6, 7, 8, 7, 5, 6, 4)
    fmt = "  " + "  ".join(f"{{:<{w}}}" for w in widths)
    print(fmt.format(*cols))
    print("  " + "  ".join("-" * w for w in widths))

    for mtime, data in rows:
        s = data.get("summary") or {}
        env = data.get("env") or {}
        when = datetime.datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M")
        rep_pen = env.get("DFLASH_REP_PENALTY") or "-"
        # Truncate float strings so the column stays narrow
        if rep_pen and rep_pen not in ("-", ""):
            try:
                rep_pen = f"{float(rep_pen):.3f}".rstrip("0").rstrip(".")
            except ValueError:
                pass
        tps = f"{s.get('tps_mean', 0):.0f}" if s.get("tps_mean") else "-"
        ttft = (f"{s['ttft_mean_s']*1000:.0f}ms"
                if s.get("ttft_mean_s") is not None else "-")
        row = (
            data.get("label", "?")[:widths[0]],
            when,
            "yes" if data.get("streaming") else "no",
            rep_pen,
            tps,
            ttft,
            str(s.get("loops_detected", "-")),
            str(s.get("proxy_aborts", "-")),
            str(s.get("successful", "-")),
        )
        print(fmt.format(*row))
    print(f"\n  {len(rows)} runs in {bench_dir.resolve()}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--label", default="baseline",
                    help="name for this run; saved to bench_results/{label}.json")
    ap.add_argument("--requests", type=int, default=3)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--raw-upstream", action="store_true",
                    help="hit dflash-serve directly (bypass proxy + loop_guard)")
    ap.add_argument("--stream", action="store_true",
                    help="use SSE streaming and report time-to-first-token "
                         "in addition to overall TPS")
    ap.add_argument("--diff", nargs=2, metavar=("A", "B"),
                    help="compare two saved label files instead of running")
    ap.add_argument("--list", action="store_true",
                    help="show all saved bench runs (label, date, key metrics)")
    args = ap.parse_args()

    if args.list:
        return list_runs()

    if args.diff:
        return diff(args.diff[0], args.diff[1])
    out = run(args.label, args.requests, args.max_tokens,
              use_proxy=not args.raw_upstream,
              streaming=args.stream)
    return 0 if out["summary"].get("successful", 0) > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
