#!/usr/bin/env python3
"""Router scaffold for dual-model agent inference.

Routes individual chat-completion requests between two upstream model
servers based on per-turn complexity. Use case: keep the 35B-A3B for
turns that genuinely need its reasoning, dispatch routine continuation
turns (read_file follow-ups, simple bash chains, done(summary)) to a
smaller faster model.

This file is a SCAFFOLD — it does not run today. To activate:

  1. Download a small chat-tuned model with MLX 4-bit weights. Recommended:
     - Qwen3-8B (matched tokenizer; cleanest router because tool_calls
       parse identically), or
     - Llama-3.1-8B-Instruct (more reliable tool-calling adherence in 8B
       class but different tokenizer — agent.py would need to dispatch
       per-model).

  2. Run a SECOND `dflash-serve` (or an `mlx-lm` server) on a different
     port — say 8003 — pointing at the 8B model. The 35B stays on 8002
     served by the existing daemon.

  3. Replace `agent.py`'s URL constant with a function that picks
     LARGE_URL or SMALL_URL per turn using `_route()`. Simplest wiring:
     replace `_do_post`'s URL with `route_url(messages)` from this file.

  4. Run `agent_bench.py` and validate: m1 quality should still pass,
     short-prompt TPS should INCREASE because the 8B is faster, and
     long-prompt TPS may not change much because long turns still need
     the 35B for reasoning.

  5. Iterate the routing heuristic. v1 is dumb (`_route_v1`); v2-v4
     ideas below are commented but not implemented.

Key tradeoff: routing decisions can be wrong. A too-aggressive route to
the 8B may flub structured tool args (the 8B is less reliable on JSON).
Start conservative — route only `tool` follow-ups where the prior
assistant turn produced no `reasoning_content` and the user message was
a tool result, not a fresh user prompt.

Memory check on a 48 GB Mac: 35B (~20 GB) + 8B (~5 GB) + KV cache + OS
= 28-30 GB. Fits with the existing speed knobs (KV quant, draft quant).
Without those knobs, you'd be borderline and the 8B might OOM.
"""

from __future__ import annotations

import os
from typing import Any


# Upstream URLs. The 8B is OPTIONAL — if SMALL_URL is unset/unreachable,
# router falls back to LARGE_URL transparently.
LARGE_URL = os.environ.get("QWEN_LARGE_URL", "http://127.0.0.1:8000/v1/chat/completions")
SMALL_URL = os.environ.get("QWEN_SMALL_URL")  # e.g. "http://127.0.0.1:8003/v1/chat/completions"
LARGE_MODEL = os.environ.get("QWEN_LARGE_MODEL", "./models/Qwen3.6-35B-A3B-OptiQ-4bit")
SMALL_MODEL = os.environ.get("QWEN_SMALL_MODEL", "Qwen3-8B-4bit")  # placeholder

# Routing knobs.
SMALL_FOR_TOOL_FOLLOWUP = os.environ.get("QWEN_ROUTE_SMALL_TOOL", "1") not in ("", "0", "false")
SMALL_REASONING_THRESHOLD = int(os.environ.get("QWEN_ROUTE_REASONING_MAX", "300"))


def _last_assistant_reasoning_len(messages: list[dict]) -> int:
    """Return the char length of the most recent assistant turn's reasoning_content.
    A long reasoning trace means the 35B was thinking hard — keep it on the 35B
    for the next turn to maintain coherence.
    """
    for m in reversed(messages):
        if m.get("role") == "assistant":
            r = m.get("reasoning_content") or m.get("reasoning") or ""
            return len(r)
    return 0


def _last_user_was_tool_result(messages: list[dict]) -> bool:
    """True if the most recent message is a `tool` role result (not a fresh user prompt)."""
    if not messages:
        return False
    return messages[-1].get("role") == "tool"


def _route_v1(messages: list[dict]) -> str:
    """Conservative router v1: only route tool-result follow-ups to the 8B,
    and only when the prior assistant didn't reason heavily (which would
    indicate the 35B was doing real work).

    Returns LARGE_URL or SMALL_URL.
    """
    if SMALL_URL is None:
        return LARGE_URL
    if not SMALL_FOR_TOOL_FOLLOWUP:
        return LARGE_URL
    # Hard route: 35B for first turn (no prior assistant), final summary, or
    # whenever the last assistant turn produced substantial reasoning.
    if not _last_user_was_tool_result(messages):
        return LARGE_URL
    if _last_assistant_reasoning_len(messages) > SMALL_REASONING_THRESHOLD:
        return LARGE_URL
    # Soft route: 8B for routine tool-result follow-ups.
    return SMALL_URL


# Future iterations (not implemented):
#
# v2: route based on prompt complexity (token count + presence of
#     code-block markers + math). Above some token threshold or when the
#     prompt asks for novel reasoning, force 35B.
#
# v3: speculative routing of tool calls. Run the 8B in parallel with the
#     35B for the same prompt; if their tool-call match exactly, accept
#     the 8B's response (saved 35B latency); otherwise drop the 8B.
#     Costs more compute but reduces tail latency.
#
# v4: train a tiny classifier on (messages, model that handled, was_correct)
#     triples from agent traces. Replace heuristic with learned policy.


def route_url(messages: list[dict]) -> str:
    """Public entrypoint. Returns the URL to POST the chat-completion request to."""
    return _route_v1(messages)


def route_model(url: str) -> str:
    """Return the model name to send in the request body for a given URL."""
    return SMALL_MODEL if url == SMALL_URL else LARGE_MODEL


# ---------- agent.py wiring patch (for reference) ----------
# Replace agent.py's `_do_post` body's hardcoded URL/MODEL with:
#
#     from router import route_url, route_model
#     url = route_url(messages)
#     model = route_model(url)
#     body = json.dumps({"model": model, "messages": messages, ...}).encode()
#     req = urllib.request.Request(url, data=body, headers=..., method="POST")
#
# Remember to import os/router. Backwards-compatible: if SMALL_URL is unset,
# all requests go to LARGE_URL (= original behavior).
