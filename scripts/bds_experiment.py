"""Backward Digit Span benchmark for the local Qwen 3.6 model.

Replicates Diak et al. (CogSci 2026) "Backward Digit Span Benchmarks Working
Memory in LLMs" against the local dflash-served Qwen3.6-35B-A3B model.

Conditions:
  - No tools, reasoning ("thinking") enabled.
  - Identical prompt wording to the paper (forward = "Reprint", backward =
    "Reverse"; output must be comma-separated digits only).
  - Per trial: one random uniform-with-replacement digit sequence is sampled
    and used for both forward and backward conditions (sequence content held
    constant within trial, as in the paper).
  - Bypasses the qwen-proxy (port 8000) and talks to dflash-serve directly
    (port 8002) — the proxy's loop-guard treats the comma-digit output as a
    hallucination loop and aborts mid-response.
  - Strictly single-stream (no client-side concurrency) — dflash deadlocks
    under concurrent requests.
  - Scoring: exact match of the digit sequence extracted from the model's
    POST-thinking answer text (anything before `</think>` is reasoning and
    is excluded from scoring).
"""

from __future__ import annotations

import csv
import json
import os
import random
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

HOST = os.environ.get("QWEN_HOST", "127.0.0.1")
# Through the qwen-proxy on 8000 now that loop_guard is patched: it
# correctly lets diverse comma-separated digit sequences through (BDS
# answers, JSON numeric arrays) while still firing on stuck-digit loops.
# Bonus: the proxy splits `<think>...</think>` into `reasoning_content` and
# leaves `content` as the answer-only string, so scoring is clean.
PORT = int(os.environ.get("QWEN_PORT", "8000"))
MODEL = os.environ.get("QWEN_MODEL_NAME", "./models/Qwen3.6-35B-A3B-OptiQ-4bit")
URL = f"http://{HOST}:{PORT}/v1/chat/completions"

OUT_DIR = Path(__file__).resolve().parent.parent / "eval_data" / "bds_qwen36"
OUT_DIR.mkdir(parents=True, exist_ok=True)
TRIALS_PATH = OUT_DIR / "trials.jsonl"
SUMMARY_PATH = OUT_DIR / "summary.csv"
PLOT_PATH = OUT_DIR / "qwen36_bds.png"

# Verbatim from Diak et al. (CogSci 2026), §Methods, Task Design.
FORWARD_PROMPT = (
    "Reprint the following sequence of digits:\n"
    "{sequence}\n"
    "Your answer must ONLY contain the printed digits, comma-separated."
)
BACKWARD_PROMPT = (
    "Reverse the following sequence of digits:\n"
    "{sequence}\n"
    "Your answer must ONLY contain the reversed digits, comma-separated."
)

THINK_END = "</think>"
CONTROL_TOKENS = ("<|im_end|>", "<|im_start|>", "<|endoftext|>")

SET_SIZES = [2, 3, 4, 5, 6, 7, 8, 10, 12, 15, 20, 30]
TRIALS_PER_CONDITION = 10
MAX_TOKENS = 8192
REQUEST_TIMEOUT_S = 600


def fmt_sequence(digits: list[int]) -> str:
    return ", ".join(str(d) for d in digits)


def extract_answer_text(content: str) -> str:
    if THINK_END in content:
        content = content.rsplit(THINK_END, 1)[1]
    for tok in CONTROL_TOKENS:
        content = content.replace(tok, "")
    return content.strip()


def extract_digit_sequence(text: str) -> list[int]:
    return [int(c) for c in text if c.isdigit()]


@dataclass
class Trial:
    n: int
    trial_idx: int
    sequence: list[int]
    direction: str
    expected: list[int]
    response_content: str
    response_reasoning_chars: int
    parsed: list[int]
    correct: bool
    elapsed_s: float
    error: str | None


def call_model(prompt: str) -> tuple[str, int, float, str | None]:
    body = json.dumps({
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": MAX_TOKENS,
        "stream": False,
    }).encode("utf-8")
    req = urllib.request.Request(
        URL,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT_S) as resp:
            data = json.load(resp)
    except urllib.error.URLError as e:
        return "", 0, time.time() - t0, f"URLError: {e}"
    except Exception as e:
        return "", 0, time.time() - t0, f"{type(e).__name__}: {e}"
    msg = data["choices"][0]["message"]
    content = msg.get("content") or ""
    reasoning = msg.get("reasoning_content") or msg.get("reasoning") or ""
    return content, len(reasoning), time.time() - t0, None


def run_one(direction: str, sequence: list[int], trial_idx: int) -> Trial:
    seq_str = fmt_sequence(sequence)
    if direction == "forward":
        prompt = FORWARD_PROMPT.format(sequence=seq_str)
        expected = list(sequence)
    else:
        prompt = BACKWARD_PROMPT.format(sequence=seq_str)
        expected = list(reversed(sequence))
    content, rlen, elapsed, err = call_model(prompt)
    answer = extract_answer_text(content)
    parsed = extract_digit_sequence(answer)
    return Trial(
        n=len(sequence),
        trial_idx=trial_idx,
        sequence=list(sequence),
        direction=direction,
        expected=expected,
        response_content=content,
        response_reasoning_chars=rlen,
        parsed=parsed,
        correct=(err is None and parsed == expected),
        elapsed_s=elapsed,
        error=err,
    )


def write_trial(fp, t: Trial) -> None:
    answer = extract_answer_text(t.response_content) if t.response_content else ""
    fp.write(json.dumps({
        "n": t.n,
        "trial_idx": t.trial_idx,
        "direction": t.direction,
        "sequence": t.sequence,
        "expected": t.expected,
        "parsed": t.parsed,
        "correct": t.correct,
        "elapsed_s": round(t.elapsed_s, 3),
        "reasoning_chars": t.response_reasoning_chars,
        "error": t.error,
        "answer_text": answer[:512],
        "content": (t.response_content[:4096] if t.response_content else ""),
    }) + "\n")
    fp.flush()


def run_experiment() -> dict:
    rng = random.Random(20260509)
    results = {}
    fp = TRIALS_PATH.open("w")
    try:
        for n in SET_SIZES:
            block_t0 = time.time()
            fwd_correct = 0
            bwd_correct = 0
            fwd_elapsed = 0.0
            bwd_elapsed = 0.0
            for i in range(TRIALS_PER_CONDITION):
                seq = [rng.randint(0, 9) for _ in range(n)]
                fwd = run_one("forward", seq, i)
                write_trial(fp, fwd)
                bwd = run_one("backward", seq, i)
                write_trial(fp, bwd)
                fwd_correct += int(fwd.correct)
                bwd_correct += int(bwd.correct)
                fwd_elapsed += fwd.elapsed_s
                bwd_elapsed += bwd.elapsed_s
                # per-trial heartbeat so progress is visible mid-block
                print(
                    f"  N={n} trial {i+1}/{TRIALS_PER_CONDITION}  "
                    f"fwd={'✓' if fwd.correct else '✗'} ({fwd.elapsed_s:.1f}s)  "
                    f"bwd={'✓' if bwd.correct else '✗'} ({bwd.elapsed_s:.1f}s)",
                    flush=True,
                )
            fwd_acc = fwd_correct / TRIALS_PER_CONDITION
            bwd_acc = bwd_correct / TRIALS_PER_CONDITION
            results[(n, "forward")] = fwd_acc
            results[(n, "backward")] = bwd_acc
            print(
                f"N={n:>3}  forward={fwd_acc:.2f}  backward={bwd_acc:.2f}  "
                f"avg_fwd={fwd_elapsed/TRIALS_PER_CONDITION:.1f}s  "
                f"avg_bwd={bwd_elapsed/TRIALS_PER_CONDITION:.1f}s  "
                f"block={time.time()-block_t0:.1f}s",
                flush=True,
            )
    finally:
        fp.close()
    return results


def write_summary(results: dict) -> None:
    with SUMMARY_PATH.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["N", "direction", "n_trials", "n_correct", "accuracy"])
        for n in SET_SIZES:
            for d in ("forward", "backward"):
                acc = results[(n, d)]
                n_correct = round(acc * TRIALS_PER_CONDITION)
                w.writerow([n, d, TRIALS_PER_CONDITION, n_correct, acc])


def compute_thresholds(results: dict) -> dict:
    bwd = {n: results[(n, "backward")] for n in SET_SIZES}
    fwd = {n: results[(n, "forward")] for n in SET_SIZES}
    span90 = None
    thr50 = None
    for n in SET_SIZES:
        if bwd[n] >= 0.9:
            span90 = n
        if bwd[n] >= 0.5:
            thr50 = n
    return {
        "span90_backward": span90,
        "threshold50_backward": thr50,
        "forward_accuracy_by_n": fwd,
        "backward_accuracy_by_n": bwd,
    }


def plot(thresholds: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ns = list(SET_SIZES)
    fwd = [thresholds["forward_accuracy_by_n"][n] for n in ns]
    bwd = [thresholds["backward_accuracy_by_n"][n] for n in ns]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(ns, fwd, marker="o", color="#2c7fb8", label="Forward", linewidth=1.6)
    ax.plot(ns, bwd, marker="o", color="#d7301f", label="Backward", linewidth=1.6)
    ax.set_xlabel("Set Size (N)")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(-0.02, 1.04)
    ax.set_title("Qwen3.6-35B-A3B (reasoning, no tools)\nForward vs Backward Digit Span")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower left")

    s90 = thresholds["span90_backward"]
    t50 = thresholds["threshold50_backward"]
    s90_str = f"{s90}" if s90 is not None else f"<{min(ns)}"
    t50_str = f"{t50}" if t50 is not None else f"<{min(ns)}"
    ax.text(
        0.98, 0.98,
        f"Span$_{{90}}$ = {s90_str}\nN$_{{50}}$ = {t50_str}",
        transform=ax.transAxes,
        ha="right", va="top",
        fontsize=10,
        bbox=dict(facecolor="white", edgecolor="0.7", alpha=0.9),
    )
    fig.tight_layout()
    fig.savefig(PLOT_PATH, dpi=160)
    print(f"plot: {PLOT_PATH}")


def main() -> int:
    print(f"endpoint: {URL}")
    print(f"model:    {MODEL}")
    print(f"set sizes: {SET_SIZES}")
    print(f"trials per condition: {TRIALS_PER_CONDITION}")
    print(f"output dir: {OUT_DIR}")
    print(flush=True)

    results = run_experiment()
    write_summary(results)
    thresholds = compute_thresholds(results)

    print()
    print("=" * 60)
    print(f"Span90 (backward >= 0.9):     N = {thresholds['span90_backward']}")
    print(f"Threshold50 (backward >= 0.5): N = {thresholds['threshold50_backward']}")
    print("=" * 60)
    print()
    plot(thresholds)
    print(f"summary: {SUMMARY_PATH}")
    print(f"trials:  {TRIALS_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
