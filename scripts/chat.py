#!/usr/bin/env python3
"""Tiny streaming chat REPL for the local mlx-openai-server.

Connects to the server defined by QWEN_HOST / QWEN_PORT / QWEN_MODEL_NAME
(0.0.0.0 is rewritten to 127.0.0.1). Stdlib-only — no extra deps.
"""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request

try:
    import readline  # noqa: F401  (enables line editing + history)
except ImportError:
    pass

HOST = os.environ.get("QWEN_HOST", "127.0.0.1")
if HOST in ("0.0.0.0", ""):
    HOST = "127.0.0.1"
PORT = os.environ.get("QWEN_PORT", "8000")
MODEL = os.environ.get("QWEN_MODEL_NAME", "qwen3.6")
URL = f"http://{HOST}:{PORT}/v1/chat/completions"

DIM = "\x1b[2m"
RESET = "\x1b[0m"
BOLD = "\x1b[1m"

HELP = """\
Commands:
  /exit, /quit       leave the chat (server keeps running)
  /clear             reset conversation history
  /system <text>     set or replace the system prompt
  /help              show this help
"""


def stream_chat(messages: list[dict]) -> str | None:
    body = json.dumps(
        {"model": MODEL, "messages": messages, "stream": True}
    ).encode("utf-8")
    req = urllib.request.Request(
        URL,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    answer: list[str] = []
    in_reasoning = False
    try:
        with urllib.request.urlopen(req) as resp:
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
                reasoning = delta.get("reasoning_content") or delta.get("reasoning")
                content = delta.get("content")
                # Belt-and-suspenders: even though the proxy strips
                # chat-template control tokens (<|im_end|> etc.) from
                # the SSE stream, we also strip them client-side in
                # case (a) the user is connected directly to a
                # bare dflash-serve (port 8002) without the proxy, or
                # (b) the running proxy hasn't been restarted yet to
                # pick up the strip. Cost: a substring check per
                # chunk; common case (no marker) is one `in` test.
                if reasoning and any(t in reasoning for t in
                                     ("<|im_end|>", "<|im_start|>", "<|endoftext|>")):
                    for t in ("<|im_end|>", "<|im_start|>", "<|endoftext|>"):
                        reasoning = reasoning.replace(t, "")
                if content and any(t in content for t in
                                   ("<|im_end|>", "<|im_start|>", "<|endoftext|>")):
                    for t in ("<|im_end|>", "<|im_start|>", "<|endoftext|>"):
                        content = content.replace(t, "")
                if reasoning:
                    if not in_reasoning:
                        sys.stdout.write(f"{DIM}thinking… ")
                        in_reasoning = True
                    sys.stdout.write(reasoning)
                    sys.stdout.flush()
                if content:
                    if in_reasoning:
                        sys.stdout.write(f"{RESET}\n")
                        in_reasoning = False
                    sys.stdout.write(content)
                    sys.stdout.flush()
                    answer.append(content)
    except urllib.error.URLError as e:
        if in_reasoning:
            sys.stdout.write(RESET)
        print(f"\n[error] cannot reach {URL}: {e}", file=sys.stderr)
        return None
    except KeyboardInterrupt:
        if in_reasoning:
            sys.stdout.write(RESET)
        print("\n[interrupted]", file=sys.stderr)
        return None

    if in_reasoning:
        sys.stdout.write(RESET)
    print()
    return "".join(answer)


def main() -> int:
    print(f"{BOLD}qwen chat{RESET}  model={MODEL}  endpoint={URL}")
    print("type /help for commands, Ctrl-D or /exit to leave\n")

    messages: list[dict] = []

    while True:
        try:
            user = input(f"{BOLD}›{RESET} ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0

        if not user:
            continue
        if user in ("/exit", "/quit"):
            return 0
        if user == "/help":
            print(HELP)
            continue
        if user == "/clear":
            messages = []
            print("[history cleared]")
            continue
        if user.startswith("/system"):
            text = user[len("/system"):].strip()
            messages = [m for m in messages if m["role"] != "system"]
            if text:
                messages.insert(0, {"role": "system", "content": text})
                print("[system prompt set]")
            else:
                print("[system prompt cleared]")
            continue

        messages.append({"role": "user", "content": user})
        reply = stream_chat(messages)
        if reply is None:
            messages.pop()
            continue
        messages.append({"role": "assistant", "content": reply})


if __name__ == "__main__":
    sys.exit(main())
