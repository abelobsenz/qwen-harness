#!/usr/bin/env python3
"""loop_guard_marker — detect the proxy's loop-guard abort marker in
assistant content, with a high-precision filter against benign mentions.

Background. The proxy emits a `[loop-guard: …]` marker when it aborts
a generation that fell into a repetition loop. Agents that read assistant
content need to detect that marker and inject a course-correction nudge
on the next turn — without the nudge, the model has no signal that its
last turn was truncated and often resumes the same loop.

The naïve detector — substring match on `[loop-guard:` — has a
false-positive risk surfaced in Round 15: a model legitimately
explaining the loop-guard system to the user, or a tool result echoing
the codebase, would trip the detector with no actual abort having
happened.

This module is the canonical detection point. Both `agent.py` and
`agent_graph.py` import from here so the false-positive fix only has
to live in one place. The detection requires BOTH:

  - The literal substring `[loop-guard:` (necessary)
  - One of the proxy's specific abort suffix phrases (sufficient)

The proxy ALWAYS includes one of two suffix phrases in its actual
abort marker:

  - `"output stopped early"` (non-streaming format)
  - `"fell into a repetition loop"` (streaming format)

These are deliberately specific so the marker is self-disambiguating
in plaintext.
"""

from __future__ import annotations

import re

# Marker reason capture: up to em-dash or closing bracket. Hyphens
# inside reason words like `low-churn`, `suffix-dominant` are preserved
# because we use only `—` or `]` as delimiters (not bare `-`).
LOOP_GUARD_RE = re.compile(
    r"\[loop-guard:\s*([^\]\n]+?)\s*(?:—|]\s*)",
    re.IGNORECASE,
)

# The disambiguating suffix. Benign mentions of `[loop-guard:` won't
# include these phrases; the proxy's actual abort always does.
LOOP_GUARD_SUFFIX_RE = re.compile(
    r"output stopped early|fell into a repetition loop",
    re.IGNORECASE,
)


def is_proxy_abort_marker(content: str) -> bool:
    """True iff `content` carries the qwen_proxy's actual abort marker.

    Filters out benign mentions of the literal substring (model
    explanations, grep echoes, code-review references) by requiring
    the proxy's specific suffix phrasing.
    """
    if not content or "[loop-guard:" not in content:
        return False
    return LOOP_GUARD_SUFFIX_RE.search(content) is not None


def extract_reason(content: str) -> str:
    """Extract the human-readable reason from a confirmed abort marker.
    Falls back to a generic phrase if the regex doesn't match the
    captured group (shouldn't happen if `is_proxy_abort_marker(content)`
    returned True, but defensive)."""
    m = LOOP_GUARD_RE.search(content)
    if m:
        return m.group(1).strip()
    return "repetition loop"


def harness_nudge_message(reason: str) -> dict:
    """Build a `[HARNESS]` user-message dict that tells the model its
    previous response was loop-aborted and asks for a different angle.
    Same shape both agent.py and agent_graph.py use."""
    return {
        "role": "user",
        "content": (
            f"[HARNESS] Your previous response was cut off by the proxy's "
            f"loop guard ({reason}). The model fell into a repetition "
            f"loop. Do NOT resume that line of reasoning. Step back, take "
            f"a different angle, and answer the user's question with a "
            f"shorter, more concrete plan. If you genuinely don't know "
            f"the answer, say so directly instead of looping."
        ),
    }


__all__ = [
    "LOOP_GUARD_RE",
    "LOOP_GUARD_SUFFIX_RE",
    "is_proxy_abort_marker",
    "extract_reason",
    "harness_nudge_message",
]
