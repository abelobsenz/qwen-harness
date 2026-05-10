#!/usr/bin/env python3
"""Repeatable decode-TPS bench for the local dflash daemon.

Mirrors the SPEED_TUNING.md methodology: 4 prompt sizes, 100-token
completions, multi-run for noise control. Reports median + range.

Plus a separate quality probe (5 deterministic short-answer items)
that any speed change must NOT regress.

Usage:
    python scripts/perf_bench.py                # full bench + quality
    python scripts/perf_bench.py --tps          # TPS only
    python scripts/perf_bench.py --quality      # quality only
    python scripts/perf_bench.py --runs 5       # more runs per size
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
import urllib.request


URL = "http://127.0.0.1:8000/v1/chat/completions"
MODEL_URL = "http://127.0.0.1:8000/v1/models"


def _resolve_model() -> str:
    with urllib.request.urlopen(MODEL_URL, timeout=4) as r:
        items = json.loads(r.read()).get("data") or []
    if not items:
        return "qwen3.6"
    return items[0].get("id") or "qwen3.6"


def _post(model: str, messages: list, max_tokens: int,
          timeout: float = 600.0) -> dict:
    # NOTE: We deliberately mirror /tmp/bench_safe.py's request shape exactly
    # — no `tools`, no `tool_choice`, no `chat_template_kwargs`. Adding any
    # of those changes the proxy/dflash-serve code path enough to halve TPS
    # on this stack (verified empirically). The bench measures ONLY decode
    # rate, so we want the leanest possible request body.
    body = json.dumps({
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "stream": False,
        "temperature": 0.0,
    }).encode("utf-8")
    req = urllib.request.Request(
        URL, data=body, headers={"Content-Type": "application/json"}, method="POST")
    t0 = time.monotonic()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read())
    wall = time.monotonic() - t0
    usage = data.get("usage") or {}
    content = (data.get("choices") or [{}])[0].get("message", {}).get("content") or ""
    return {
        "wall": wall,
        "prompt_tokens": int(usage.get("prompt_tokens") or 0),
        "completion_tokens": int(usage.get("completion_tokens") or 0),
        "content": content,
    }


# ----------------------------- TPS bench -----------------------------------

# Sized so the prompt-token totals land near the SPEED_TUNING bands.
# Each "filler" repetition adds roughly 3-4 tokens, so:
FILLER = "Some context. "  # ~3 tokens per repetition

PROMPT_BANDS = [
    ("short_~80",  FILLER * 6   + "\nReply with just OK."),
    ("p_3k",       FILLER * 700 + "\nReply with just OK."),
    ("p_5k",       FILLER * 1300 + "\nReply with just OK."),
    ("p_9k",       FILLER * 2400 + "\nReply with just OK."),
]


def run_tps_bench(model: str, runs: int = 3, max_tokens: int = 100,
                  *, warm_up: bool = True) -> dict:
    if warm_up:
        # Tiny request to trigger any cold-path JIT.
        _post(model, [{"role": "user", "content": "hi"}], max_tokens=4)

    summary: dict[str, dict] = {}
    print(f"# bench  model={model}  runs/size={runs}  max_tokens={max_tokens}")
    for label, prompt in PROMPT_BANDS:
        tps_runs: list[float] = []
        prompt_tok_runs: list[int] = []
        for r in range(runs):
            try:
                res = _post(model, [{"role": "user", "content": prompt}],
                             max_tokens=max_tokens)
            except Exception as e:  # noqa: BLE001
                print(f"  [{label}] run {r+1} failed: {e}", file=sys.stderr)
                continue
            tps = res["completion_tokens"] / max(res["wall"], 1e-6)
            tps_runs.append(tps)
            prompt_tok_runs.append(res["prompt_tokens"])
        if not tps_runs:
            summary[label] = {"tps_median": None, "tps_runs": []}
            print(f"  [{label}] all runs failed")
            continue
        med = statistics.median(tps_runs)
        rng = (min(tps_runs), max(tps_runs))
        ptok = statistics.median(prompt_tok_runs)
        summary[label] = {
            "tps_median": med,
            "tps_runs": tps_runs,
            "tps_min": rng[0],
            "tps_max": rng[1],
            "prompt_tokens_median": ptok,
        }
        print(f"  [{label:>10s}] prompt≈{int(ptok):>5d}tok  TPS "
              f"med={med:5.1f}  range=[{rng[0]:5.1f}..{rng[1]:5.1f}]  "
              f"runs={['%.1f' % t for t in tps_runs]}")
    return summary


# ----------------------------- quality probe --------------------------------

# Five deterministic items spanning math / reasoning / code / fact.
# Graded with a substring/numeric check; fragile to formatting but the model
# is consistent at temp=0. The point is to catch regressions, not measure
# absolute quality — same items used across all variants.

QUALITY_ITEMS = [
    {
        "q": "Compute 17 * 23. Output only the number.",
        "check": lambda c: "391" in c,
        "label": "math:17*23",
    },
    {
        "q": "What's the 7th prime number? Output only the number.",
        "check": lambda c: "17" in c and "171" not in c.replace("17", "_", 1),
        "label": "math:7th_prime",
    },
    {
        "q": ("Without writing any code: a list contains [3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5]. "
              "How many times does 5 appear? Output only the number."),
        "check": lambda c: "3" in c and not any(d in c for d in ["13", "23", "33", "43", "53", "63", "73", "83", "93"]),
        "label": "count:5_in_list",
    },
    {
        "q": "What's the capital of Australia? Output only the city name.",
        "check": lambda c: "canberra" in c.lower(),
        "label": "fact:au_capital",
    },
    {
        "q": ("Write a Python one-liner that returns the sum of squares of [1,2,3,4,5]. "
              "Output only the code, no markdown."),
        "check": lambda c: "55" in c or ("sum" in c.lower() and "**" in c),
        "label": "code:sum_squares",
    },
]


def run_quality(model: str) -> dict:
    print(f"# quality probe  model={model}  items={len(QUALITY_ITEMS)}")
    results = []
    n_pass = 0
    for it in QUALITY_ITEMS:
        try:
            res = _post(model, [
                {"role": "system", "content": "Answer concisely. Output only what was asked."},
                {"role": "user", "content": it["q"]},
            ], max_tokens=128)
        except Exception as e:  # noqa: BLE001
            print(f"  [{it['label']}] FAIL (request error): {e}")
            results.append({"label": it["label"], "pass": False, "reason": str(e)})
            continue
        ok = bool(it["check"](res["content"]))
        n_pass += int(ok)
        snippet = " ".join(res["content"].split())[:80]
        print(f"  [{it['label']}] {'✓' if ok else '✗'}  →  {snippet!r}")
        results.append({"label": it["label"], "pass": ok, "snippet": snippet,
                        "wall": res["wall"]})
    print(f"  → quality: {n_pass}/{len(QUALITY_ITEMS)} pass")
    return {"n_pass": n_pass, "n_total": len(QUALITY_ITEMS), "results": results}


# ----------------------------- main ----------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--max-tokens", type=int, default=100)
    ap.add_argument("--tps", action="store_true",
                    help="TPS bench only (skip quality)")
    ap.add_argument("--quality", action="store_true",
                    help="quality probe only (skip TPS)")
    ap.add_argument("--label", default="run",
                    help="label printed at the start of the report")
    args = ap.parse_args()

    print(f"=== {args.label} ===")
    try:
        model = _resolve_model()
    except Exception as e:  # noqa: BLE001
        print(f"[error] cannot reach upstream: {e}", file=sys.stderr)
        return 2

    full_report: dict = {"label": args.label, "model": model,
                         "ts": time.strftime("%Y-%m-%d %H:%M:%S")}

    if not args.quality:
        full_report["tps"] = run_tps_bench(model, runs=args.runs,
                                            max_tokens=args.max_tokens)
        print()

    if not args.tps:
        full_report["quality"] = run_quality(model)
        print()

    # Summary one-liner — easy to copy/paste between runs.
    if not args.quality:
        bands = full_report.get("tps") or {}
        med_str = "  ".join(
            f"{k}={v['tps_median']:.1f}" if v.get("tps_median") is not None else f"{k}=fail"
            for k, v in bands.items())
        print(f"SUMMARY[{args.label}]  TPS  {med_str}")
    if not args.tps:
        q = full_report.get("quality") or {}
        print(f"SUMMARY[{args.label}]  quality  {q.get('n_pass')}/{q.get('n_total')} pass")

    return 0


if __name__ == "__main__":
    sys.exit(main())
