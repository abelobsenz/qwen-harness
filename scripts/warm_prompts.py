#!/usr/bin/env python3
"""warm_prompts — populate dflash-serve's APC with the catalog of known
system prompts at server boot.

Each prefill-only request (max_tokens=1) costs ~600ms of one-time prefill
but saves ~40ms TTFT on the first user request that hits the same prompt
prefix. Net positive for any long-running daemon that touches multiple
distinct prompt families per lifetime (the agent uses 25+).

Usage:
    python scripts/warm_prompts.py                # warm against :8002 directly
    QWEN_WARM_URL=http://127.0.0.1:8000 ...        # via proxy
    QWEN_WARM_DISABLE=1 ...                        # no-op (for testing)

Invoked automatically by bin/qwen after wait_ready when DFLASH_WARM_PROMPTS=1.
"""
from __future__ import annotations
import json, os, sys, time, urllib.request

URL = os.environ.get("QWEN_WARM_URL", "http://127.0.0.1:8002") + "/v1/chat/completions"
MODEL_URL = os.environ.get("QWEN_WARM_URL", "http://127.0.0.1:8002") + "/v1/models"


def _resolve_model() -> str:
    with urllib.request.urlopen(MODEL_URL, timeout=4) as r:
        items = json.loads(r.read()).get("data") or []
    return items[0].get("id") if items else "qwen3.6"


# Catalog of prompts to pre-warm. Add to this list when new graphs / agents
# ship. Keep entries short — only the FIRST ~512 tokens of each prompt
# matter for cache-hit purposes (APC keys on token prefixes).
def _agent_system_prompt() -> str:
    """Mirrors scripts/agent.py SYSTEM_PROMPT_STATIC. Imported lazily so
    this script runs without the full agent dependencies."""
    p = os.path.join(os.path.dirname(__file__), "agent.py")
    if not os.path.exists(p):
        return ""
    with open(p) as f:
        src = f.read()
    # Extract SYSTEM_PROMPT_STATIC = f"""...""" block via simple parse.
    marker = 'SYSTEM_PROMPT_STATIC = f"""\\\n'
    if marker in src:
        start = src.index(marker) + len(marker)
        end = src.index('"""', start)
        return src[start:end].rstrip()
    return ""


def _chat_default_prompt() -> str:
    """Generic chat default — pre-warm for the chat REPL's empty system
    prompt (no system message at all). The empty case is handled by APC
    naturally; we add a short "You are Qwen..." just in case the user
    sets one via /system."""
    return "You are Qwen, a helpful assistant."


def known_prompts() -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    ap = _agent_system_prompt()
    if ap:
        out.append(("agent_static", ap))
    out.append(("chat_default", _chat_default_prompt()))
    return out


def prefill(model: str, system_prompt: str, timeout: float = 60.0) -> float:
    body = json.dumps({
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "ok"},
        ],
        "max_tokens": 1,
        "temperature": 0.0,
        "stream": False,
    }).encode()
    req = urllib.request.Request(
        URL, data=body, headers={"Content-Type": "application/json"}, method="POST",
    )
    t0 = time.monotonic()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        r.read()
    return time.monotonic() - t0


def main() -> int:
    if os.environ.get("QWEN_WARM_DISABLE"):
        print("warm_prompts: disabled via QWEN_WARM_DISABLE")
        return 0
    try:
        model = _resolve_model()
    except Exception as e:  # noqa: BLE001
        print(f"warm_prompts: cannot reach server ({e}); skipping", file=sys.stderr)
        return 1
    catalog = known_prompts()
    if not catalog:
        print("warm_prompts: empty catalog; nothing to do")
        return 0
    print(f"warm_prompts: model={model}  prompts={len(catalog)}")
    total = 0.0
    for label, prompt in catalog:
        try:
            dt = prefill(model, prompt)
        except Exception as e:  # noqa: BLE001
            print(f"  [{label}] FAIL: {e}")
            continue
        total += dt
        print(f"  [{label}] {dt*1000:.0f}ms  ({len(prompt)} chars)")
    print(f"warm_prompts: total {total*1000:.0f}ms")
    return 0


if __name__ == "__main__":
    sys.exit(main())
