"""Extension run for the BDS experiment — skip the small sizes already
captured in eval_data/bds_qwen36/trials.jsonl and walk up adaptively.

Goal: find where Qwen3.6's backward span actually collapses. We've already
shown small-N is noisy due to over-verification spirals. This run focuses
on large-N to map the real capacity curve.

Strategy:
  - Open trials.jsonl in APPEND mode (preserves N=2..7 already captured).
  - Walk through SET_SIZES sequentially.
  - At each N: 5 trials per direction (faster; we just want the capacity
    curve, not full statistical power).
  - Early-stop the walk if backward accuracy < ADAPTIVE_STOP_BELOW.
  - Same prompts, same scoring, same proxy as scripts/bds_experiment.py.
"""

from __future__ import annotations

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
PORT = int(os.environ.get("QWEN_PORT", "8000"))
MODEL = os.environ.get("QWEN_MODEL_NAME", "./models/Qwen3.6-35B-A3B-OptiQ-4bit")
URL = f"http://{HOST}:{PORT}/v1/chat/completions"

OUT_DIR = Path(__file__).resolve().parent.parent / "eval_data" / "bds_qwen36"
TRIALS_PATH = OUT_DIR / "trials.jsonl"

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

# Minimal system prompt to give the chat template SOMETHING to anchor
# behavior to. Without one, the user prompt becomes the model's only
# behavioral signal and reasoning-trained models read it with maximum
# severity — which produced the over-verification spirals we observed
# at trivial N (the model debating `8,0` vs `8, 0` for 4 KB of CoT).
# Most commercial APIs supply a default system prompt; the paper didn't
# specify they removed it, so this is a small, well-justified deviation.
SYSTEM_PROMPT = "You are a helpful assistant. Answer concisely."

THINK_END = "</think>"
CONTROL_TOKENS = ("<|im_end|>", "<|im_start|>", "<|endoftext|>")

# Walk: skip 10/12/15/20 per user request — go straight to 30, then up.
SET_SIZES = [2, 20, 30, 50, 75, 100, 150, 200]
TRIALS_PER_CONDITION = 5
# Bumped 12288 → 32768 after the prior run showed 3/5 N=30 backward failures
# all hit max_tokens MID-CoT (no </think> emitted, finish_reason=length, the
# entire reasoning trace leaked into `content` and got mis-scored as a
# 3000-digit answer). Successful runs at N=30 used ~8-10K reasoning chars,
# so 32K gives 3-4× headroom for slightly longer reasoning paths.
MAX_TOKENS = 32768
# Bumped 900 → 1800 to match the larger token budget. At ~50-75 tok/s with
# DFlash speculative decoding, 32K tokens is roughly 7-11 minutes worst case.
REQUEST_TIMEOUT_S = 1800
ADAPTIVE_STOP_BELOW = 0.2   # if backward accuracy drops below 20% at some N,
                            # don't bother walking higher


def fmt_sequence(digits): return ", ".join(str(d) for d in digits)


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
    finish_reason: str | None
    incomplete: bool   # True if model was cut off mid-CoT (no </think>
                       # emitted AND finish_reason='length'). Distinguishes
                       # "wrong answer" from "ran out of tokens before
                       # answering".


def call_model(prompt: str):
    body = json.dumps({
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.0,
        "max_tokens": MAX_TOKENS,
        "stream": False,
    }).encode("utf-8")
    req = urllib.request.Request(
        URL, data=body, headers={"Content-Type": "application/json"}, method="POST",
    )
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT_S) as resp:
            data = json.load(resp)
    except urllib.error.URLError as e:
        return "", 0, time.time() - t0, f"URLError: {e}", None
    except Exception as e:
        return "", 0, time.time() - t0, f"{type(e).__name__}: {e}", None
    choice = data["choices"][0]
    msg = choice["message"]
    content = msg.get("content") or ""
    reasoning = msg.get("reasoning_content") or msg.get("reasoning") or ""
    finish_reason = choice.get("finish_reason")
    return content, len(reasoning), time.time() - t0, None, finish_reason


def run_one(direction, sequence, idx):
    seq_str = fmt_sequence(sequence)
    if direction == "forward":
        prompt = FORWARD_PROMPT.format(sequence=seq_str)
        expected = list(sequence)
    else:
        prompt = BACKWARD_PROMPT.format(sequence=seq_str)
        expected = list(reversed(sequence))
    content, rlen, elapsed, err, finish_reason = call_model(prompt)
    answer = extract_answer_text(content)
    parsed = extract_digit_sequence(answer)
    # Detect "ran out of tokens before answering" — this is a different
    # failure mode from "got the reversal wrong". When the model hits the
    # length cap mid-CoT (a) it never emits a closing </think>, (b) the
    # proxy can't separate reasoning from answer, (c) `content` is the
    # full reasoning trace, and (d) extract_digit_sequence picks up every
    # digit the model wrote during enumeration — guaranteed mismatch.
    # We mark these as incomplete and exclude them from accuracy denominators
    # (vs. counting them as "wrong" which conflates capacity with budget).
    incomplete = (
        err is None
        and finish_reason == "length"
        and rlen == 0  # proxy didn't extract any reasoning_content
        and "</think>" not in content
    )
    correct = (err is None and not incomplete and parsed == expected)
    return Trial(
        n=len(sequence), trial_idx=idx, sequence=list(sequence),
        direction=direction, expected=expected,
        response_content=content, response_reasoning_chars=rlen,
        parsed=parsed, correct=correct,
        elapsed_s=elapsed, error=err,
        finish_reason=finish_reason, incomplete=incomplete,
    )


def write_trial(fp, t: Trial):
    answer = extract_answer_text(t.response_content) if t.response_content else ""
    fp.write(json.dumps({
        "n": t.n, "trial_idx": t.trial_idx, "direction": t.direction,
        "sequence": t.sequence, "expected": t.expected,
        "parsed": t.parsed, "correct": t.correct,
        "elapsed_s": round(t.elapsed_s, 3),
        "reasoning_chars": t.response_reasoning_chars,
        "error": t.error,
        "finish_reason": t.finish_reason,
        "incomplete": t.incomplete,
        "answer_text": answer[:512],
        "content": (t.response_content[:4096] if t.response_content else ""),
    }) + "\n")
    fp.flush()


def main():
    print(f"endpoint: {URL}")
    print(f"model:    {MODEL}")
    print(f"set sizes (adaptive walk): {SET_SIZES}")
    print(f"trials per condition: {TRIALS_PER_CONDITION}")
    print(f"appending to: {TRIALS_PATH}")
    print(flush=True)

    rng = random.Random(20260509 + 1)  # different seed than primary run

    fp = TRIALS_PATH.open("a")
    try:
        for n in SET_SIZES:
            block_t0 = time.time()
            fwd_correct = 0
            bwd_correct = 0
            fwd_incomplete = 0
            bwd_incomplete = 0
            for i in range(TRIALS_PER_CONDITION):
                seq = [rng.randint(0, 9) for _ in range(n)]
                fwd = run_one("forward", seq, i)
                write_trial(fp, fwd)
                bwd = run_one("backward", seq, i)
                write_trial(fp, bwd)
                fwd_correct += int(fwd.correct)
                bwd_correct += int(bwd.correct)
                fwd_incomplete += int(fwd.incomplete)
                bwd_incomplete += int(bwd.incomplete)
                def mark(t):
                    if t.correct: return "✓"
                    if t.incomplete: return "…"  # cut off mid-CoT
                    return "✗"
                print(
                    f"  N={n} trial {i+1}/{TRIALS_PER_CONDITION}  "
                    f"fwd={mark(fwd)} ({fwd.elapsed_s:.1f}s)  "
                    f"bwd={mark(bwd)} ({bwd.elapsed_s:.1f}s)",
                    flush=True,
                )
            # Accuracy reported over COMPLETED trials only — incomplete trials
            # are budget-overflow signals, not wrong answers.
            fwd_completed = TRIALS_PER_CONDITION - fwd_incomplete
            bwd_completed = TRIALS_PER_CONDITION - bwd_incomplete
            fwd_acc = (fwd_correct / fwd_completed) if fwd_completed else 0.0
            bwd_acc = (bwd_correct / bwd_completed) if bwd_completed else 0.0
            print(
                f"N={n:>3}  forward={fwd_acc:.2f} ({fwd_correct}/{fwd_completed})"
                f"{' [' + str(fwd_incomplete) + ' incomplete]' if fwd_incomplete else ''}  "
                f"backward={bwd_acc:.2f} ({bwd_correct}/{bwd_completed})"
                f"{' [' + str(bwd_incomplete) + ' incomplete]' if bwd_incomplete else ''}  "
                f"block={time.time() - block_t0:.1f}s",
                flush=True,
            )
            # Walk-stop: only stop on a real capacity collapse. If the model
            # is succeeding when it has time to think but hitting the budget
            # otherwise, that's a "make max_tokens bigger" signal, not a
            # "the model can't do it" signal.
            if bwd_completed >= 2 and bwd_acc < ADAPTIVE_STOP_BELOW:
                print(
                    f"\n[adaptive stop] backward accuracy at N={n} = {bwd_acc:.2f} "
                    f"on {bwd_completed} completed trials < {ADAPTIVE_STOP_BELOW} — "
                    f"capacity collapse confirmed; not walking higher.",
                    flush=True,
                )
                break
            if bwd_completed == 0:
                print(
                    f"\n[walk-stop] all {TRIALS_PER_CONDITION} backward trials at N={n} "
                    f"hit max_tokens — even with 32K budget the model can't finish "
                    f"reversing this length. Budget is the bottleneck, not capacity per se.",
                    flush=True,
                )
                break
    finally:
        fp.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
