#!/usr/bin/env python3
"""tool_call_bench — measure tool-call success rate ON vs OFF grammar.

Sends N prompts that should produce a tool call. For each response,
classify:
  - 'good_tool_call': content has parsed tool_calls (server populated them)
                       OR contains valid <tool_call>{...}</tool_call> JSON
  - 'malformed':     contains <tool_call>...</tool_call> but JSON inside fails json.loads
  - 'no_tool_call':  no <tool_call> markers at all
  - 'truncated':     started <tool_call> but no closing tag

Usage:
    python scripts/tool_call_bench.py [--prompts N] [--max-tokens M]
"""
from __future__ import annotations
import argparse, json, re, time, urllib.request

URL = "http://127.0.0.1:8002/v1/chat/completions"
MODEL = "./models/Qwen3.6-35B-A3B-OptiQ-4bit"

# 20 prompts that should each produce a tool call.
PROMPTS = [
    ("read_file", "Read the contents of test.py.", {
        "type": "function", "function": {"name": "read_file",
        "description": "Read a file's contents",
        "parameters": {"type": "object",
                       "properties": {"path": {"type": "string"}},
                       "required": ["path"]}}}),
    ("read_file", "Open config.json please.", None),
    ("read_file", "I need to see what's in main.py.", None),
    ("write_file", "Create a new file at hello.txt with content 'hi'.", {
        "type": "function", "function": {"name": "write_file",
        "description": "Write content to a file",
        "parameters": {"type": "object",
                       "properties": {"path": {"type": "string"},
                                       "content": {"type": "string"}},
                       "required": ["path", "content"]}}}),
    ("grep", "Search for the word 'TODO' in src/.", {
        "type": "function", "function": {"name": "grep",
        "description": "Search files for a pattern",
        "parameters": {"type": "object",
                       "properties": {"pattern": {"type": "string"},
                                       "path": {"type": "string"}},
                       "required": ["pattern", "path"]}}}),
    ("grep", "Find every occurrence of 'def main' in this repo.", None),
    ("list_files", "Show me what's in the docs/ folder.", {
        "type": "function", "function": {"name": "list_files",
        "description": "List files in a directory",
        "parameters": {"type": "object",
                       "properties": {"path": {"type": "string"}},
                       "required": ["path"]}}}),
    ("list_files", "List files in /tmp.", None),
    ("bash", "Run `ls -la` in the project root.", {
        "type": "function", "function": {"name": "bash",
        "description": "Run a shell command",
        "parameters": {"type": "object",
                       "properties": {"cmd": {"type": "string"}},
                       "required": ["cmd"]}}}),
    ("bash", "Execute `git status` for me.", None),
    ("bash", "Tell me the kernel version with `uname -a`.", None),
    ("python_run", "Compute 17 squared using Python.", {
        "type": "function", "function": {"name": "python_run",
        "description": "Run a Python snippet",
        "parameters": {"type": "object",
                       "properties": {"code": {"type": "string"}},
                       "required": ["code"]}}}),
    ("python_run", "Use python to print('hello world').", None),
    ("web_fetch", "Fetch the contents of https://example.com.", {
        "type": "function", "function": {"name": "web_fetch",
        "description": "Fetch a URL",
        "parameters": {"type": "object",
                       "properties": {"url": {"type": "string"}},
                       "required": ["url"]}}}),
    ("web_fetch", "Get the text from https://www.wikipedia.org/.", None),
    ("memory_save", "Remember that my favorite color is blue.", {
        "type": "function", "function": {"name": "memory_save",
        "description": "Save a memory",
        "parameters": {"type": "object",
                       "properties": {"key": {"type": "string"},
                                       "value": {"type": "string"}},
                       "required": ["key", "value"]}}}),
    ("memory_search", "Search my memory for 'project ideas'.", {
        "type": "function", "function": {"name": "memory_search",
        "description": "Search memory",
        "parameters": {"type": "object",
                       "properties": {"query": {"type": "string"}},
                       "required": ["query"]}}}),
    ("apply_patch", "Apply this patch:\n--- a/x.py\n+++ b/x.py\n@@\n-foo\n+bar", {
        "type": "function", "function": {"name": "apply_patch",
        "description": "Apply a unified diff",
        "parameters": {"type": "object",
                       "properties": {"diff": {"type": "string"}},
                       "required": ["diff"]}}}),
    ("done", "I'm finished — call done.", {
        "type": "function", "function": {"name": "done",
        "description": "Signal task complete",
        "parameters": {"type": "object",
                       "properties": {"summary": {"type": "string"}},
                       "required": ["summary"]}}}),
    ("read_file", "Show me the README.md.", None),
]


def _post(prompt_text: str, tool_def: dict | None,
          max_tokens: int = 500, timeout: float = 60.0) -> dict:
    body = json.dumps({
        "model": MODEL,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant. Use tools when appropriate."},
            {"role": "user", "content": prompt_text},
        ],
        "tools": [tool_def] if tool_def else None,
        "tool_choice": "auto" if tool_def else None,
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": False,
    }).encode()
    req = urllib.request.Request(URL, data=body,
                                 headers={"Content-Type": "application/json"},
                                 method="POST")
    t0 = time.monotonic()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        data = json.loads(r.read())
    return {"data": data, "wall": time.monotonic() - t0}


_TC_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
# Qwen3.6 emits XML-style tool calls:
#   <tool_call><function=NAME><parameter=KEY>VAL</parameter>...</function></tool_call>
_XML_FN_RE = re.compile(r"<function\s*=\s*([\w_-]+)>(.*?)</function>", re.DOTALL)
_XML_PARAM_RE = re.compile(r"<parameter\s*=\s*([\w_-]+)>(.*?)</parameter>", re.DOTALL)


def classify(content: str, tool_calls_field: list | None) -> str:
    """Return one of: good_tool_call, malformed, no_tool_call, truncated.

    Recognizes BOTH JSON-style (`{"name": ..., "arguments": ...}`) and
    XML-style (`<function=name><parameter=k>v</parameter></function>`)
    tool calls. Qwen3.6 emits the XML style.
    """
    if tool_calls_field:
        return "good_tool_call"
    if "<tool_call>" not in content:
        return "no_tool_call"
    if "</tool_call>" not in content:
        return "truncated"
    m = _TC_RE.search(content)
    if not m:
        return "malformed"
    inner = m.group(1).strip()
    if not inner:
        return "malformed"
    # Try XML format first (Qwen3.6 native).
    fn_match = _XML_FN_RE.search(inner)
    if fn_match:
        # Has at least one <function=...>...</function> block. Check inside has
        # well-formed parameter blocks (or at least one).
        fn_body = fn_match.group(2)
        params = _XML_PARAM_RE.findall(fn_body)
        # Even an empty function (no params) is valid for some tools.
        if "<function" in inner and "</function>" in inner:
            return "good_tool_call"
        return "malformed"
    # Fallback: try JSON.
    try:
        parsed = json.loads(inner)
        if isinstance(parsed, dict) and (parsed.get("name") or parsed.get("function")):
            return "good_tool_call"
    except Exception:
        pass
    return "malformed"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", type=int, default=len(PROMPTS))
    ap.add_argument("--max-tokens", type=int, default=500)
    ap.add_argument("--label", default="run")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    prompts_to_run = PROMPTS[: args.prompts]
    counts = {"good_tool_call": 0, "malformed": 0,
              "no_tool_call": 0, "truncated": 0}
    total_wall = 0.0

    last_tool = None
    print(f"[{args.label}] running {len(prompts_to_run)} prompts ...")
    for i, (tool_name, prompt, tool_def) in enumerate(prompts_to_run):
        # Carry the tool_def through if not provided per-prompt
        if tool_def is None:
            for tn, _, td in PROMPTS:
                if tn == tool_name and td is not None:
                    tool_def = td
                    break
        try:
            out = _post(prompt, tool_def, max_tokens=args.max_tokens)
        except Exception as e:
            print(f"  [{i:02d} {tool_name:12s}] ERROR: {e}")
            counts["truncated"] += 1
            continue
        d = out["data"]
        msg = (d.get("choices") or [{}])[0].get("message", {})
        content = msg.get("content", "") or ""
        tc = msg.get("tool_calls")
        cls = classify(content, tc)
        counts[cls] += 1
        total_wall += out["wall"]
        if args.verbose:
            snippet = content.split("</think>")[-1][:80].replace("\n", "\\n")
            print(f"  [{i:02d} {tool_name:12s}] {cls:>15s}  wall={out['wall']:.1f}s  → {snippet}")
        else:
            print(f"  [{i:02d} {tool_name:12s}] {cls}")

    n = len(prompts_to_run)
    print(f"\n[{args.label}] summary over {n} prompts ({total_wall:.1f}s wall):")
    for k, v in counts.items():
        pct = 100.0 * v / max(1, n)
        print(f"  {k:>16s}: {v}/{n}  ({pct:.0f}%)")
    print(f"  success_rate (good): {100.0 * counts['good_tool_call'] / max(1, n):.0f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
