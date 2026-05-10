#!/usr/bin/env python3
"""Micro-benchmark for the local qwen server.

Measures time-to-first-token (TTFT) and decode throughput (tok/s) at
three context sizes, with both cold and warm runs (warm exercises the
prompt cache). Stdlib only.

Usage:
    python scripts/benchmark.py
"""

from __future__ import annotations

import json
import os
import statistics
import time
import urllib.request

HOST = os.environ.get("QWEN_HOST", "127.0.0.1")
if HOST in ("0.0.0.0", ""):
    HOST = "127.0.0.1"
PORT = os.environ.get("QWEN_PORT", "8000")
MODEL = os.environ.get("QWEN_MODEL_NAME", "qwen3.6")
URL = f"http://{HOST}:{PORT}/v1/chat/completions"


def stream_chat(messages: list[dict], max_tokens: int = 128) -> dict:
    body = json.dumps(
        {
            "model": MODEL,
            "messages": messages,
            "max_tokens": max_tokens,
            "stream": True,
            "temperature": 0.0,
        }
    ).encode("utf-8")
    req = urllib.request.Request(
        URL,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    first_token_t: float | None = None
    start = time.time()
    output_chars = 0
    with urllib.request.urlopen(req, timeout=600) as resp:
        for raw in resp:
            line = raw.decode("utf-8", errors="replace").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            try:
                obj = json.loads(payload)
            except json.JSONDecodeError:
                continue
            delta = obj.get("choices", [{}])[0].get("delta", {})
            content = delta.get("content") or ""
            reasoning = delta.get("reasoning_content") or delta.get("reasoning") or ""
            if content or reasoning:
                if first_token_t is None:
                    first_token_t = time.time() - start
                # Treat reasoning + content alike for decode-rate measurement.
                output_chars += len(content) + len(reasoning)
    total = time.time() - start
    ttft = first_token_t or total
    decode_time = max(total - ttft, 1e-6)
    # Approximate tokens from chars (consistent enough across runs to compare).
    approx_tokens = output_chars / 4
    return {
        "ttft_s": ttft,
        "total_s": total,
        "decode_tps": approx_tokens / decode_time,
        "out_chars": output_chars,
    }


def make_long_prompt(target_chars: int) -> str:
    chunk = (
        "The processing pipeline routes incoming events through validation, "
        "parsing, dispatch, and finalization stages. Each stage may inspect "
        "the message envelope and emit metadata into the routing context. "
        "Logging is deferred until commit. Errors propagate via the typed "
        "result channel. The orchestrator owns lifetime; stages are stateless. "
    )
    n = target_chars // len(chunk) + 1
    return (chunk * n)[:target_chars]


# Force longer responses so decode-rate is measurable.
LONG_OUTPUT_INSTRUCTION = (
    "\n\nIgnore the content above. Respond with exactly: 'one two three four "
    "five six seven eight nine ten eleven twelve thirteen fourteen fifteen "
    "sixteen seventeen eighteen nineteen twenty.'"
)

CASES = [
    ("short    ", LONG_OUTPUT_INSTRUCTION.strip()),
    ("medium 5k", make_long_prompt(5000) + LONG_OUTPUT_INSTRUCTION),
    ("long  30k", make_long_prompt(30000) + LONG_OUTPUT_INSTRUCTION),
]


def warmup() -> None:
    stream_chat([{"role": "user", "content": "hi"}], max_tokens=4)


def main() -> None:
    print(f"Benchmarking {URL}")
    print(f"Model: {MODEL}\n")
    print(f"{'case':<12}{'mode':<8}{'prompt':<12}{'ttft (s)':<11}{'decode tps':<13}{'total (s)':<11}")
    print("-" * 67)

    warmup()

    for label, content in CASES:
        msgs = [{"role": "user", "content": content}]
        prompt_chars = len(content)
        # Cold run: this prompt has not been seen recently.
        cold = stream_chat(msgs, max_tokens=32)
        # Warm run: same prompt — prefix should hit the prompt cache.
        warm = stream_chat(msgs, max_tokens=32)
        print(
            f"{label:<12}{'cold':<8}{prompt_chars:>6,}c   "
            f"{cold['ttft_s']:>7.2f}    {cold['decode_tps']:>7.1f}      {cold['total_s']:>6.2f}"
        )
        print(
            f"{label:<12}{'warm':<8}{prompt_chars:>6,}c   "
            f"{warm['ttft_s']:>7.2f}    {warm['decode_tps']:>7.1f}      {warm['total_s']:>6.2f}"
        )


if __name__ == "__main__":
    main()
