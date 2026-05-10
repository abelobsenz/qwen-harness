#!/usr/bin/env python3
"""Static checks for the chat tool tier.

The goal is functional routing, not just shrinking context: keep specialized
tools that prevent wasteful generic calls, and leave genuinely duplicative or
high-risk affordances in the rare tier.
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def _tool(name: str) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": f"{name} description",
            "parameters": {"type": "object", "properties": {}},
        },
    }


def main() -> int:
    os.environ["QWEN_UI_RARE_TOOLS"] = "0"
    import qwen_ui

    failures: list[str] = []

    def check(label: str, ok: bool, detail: str = "") -> None:
        marker = "✓" if ok else "✗"
        suffix = f" — {detail}" if detail else ""
        print(f"  [{marker}] {label}{suffix}")
        if not ok:
            failures.append(label)

    print("== qwen_ui chat tool tier tests ==\n")
    must_keep = {
        "bash", "python_run", "test_run",
        "web_search", "web_fetch", "web_outline",
        "github_repo", "arxiv_search", "arxiv_fetch", "doi_resolve",
        "inspect_data", "agent_graph_list", "agent_graph_run",
    }
    for name in sorted(must_keep):
        check(f"{name} stays in chat keep tier", name in qwen_ui._CHAT_KEEP_TOOLS)

    must_rare = {
        "csv_summary", "python_reset", "explore", "subagent_implement",
        "memory_get", "memory_list", "notebook_edit", "notebook_run",
    }
    for name in sorted(must_rare):
        check(f"{name} stays rare", name in qwen_ui._CHAT_RARE_TOOLS)

    tools = [_tool(n) for n in sorted(must_keep | must_rare | {"mcp_local__ping"})]
    filtered = qwen_ui._chat_tool_tier(tools)
    names = {(t.get("function") or {}).get("name") for t in filtered}
    check("kept tools survive filtering", must_keep <= names)
    check("rare tools are hidden by default", not (must_rare & names), str(must_rare & names))
    check("registered MCP tools survive filtering", "mcp_local__ping" in names)

    terse = qwen_ui._terse_tools([_tool("bash"), _tool("python_run"), _tool("test_run")])
    desc = {t["function"]["name"]: t["function"]["description"] for t in terse}
    check("bash blurb discourages Python overlap", "Python snippets" in desc["bash"])
    check("python_run blurb distinguishes shell", "not shell" in desc["python_run"])
    check("test_run blurb prefers test tool", "prefer over bash" in desc["test_run"])

    class FakeResp:
        def __iter__(self):
            frames = [
                {"choices": [{"delta": {"role": "assistant"}}]},
                {"choices": [{"delta": {"content": "I will inspect the file."}}]},
                {"choices": [{"delta": {"tool_calls": [{
                    "id": "call_0",
                    "type": "function",
                    "function": {
                        "name": "read_file",
                        "arguments": json.dumps({"path": "README.md"}),
                    },
                }]}, "finish_reason": "tool_calls"}]},
            ]
            for frame in frames:
                yield f"data: {json.dumps(frame)}\n\n".encode()
            yield b"data: [DONE]\n\n"

        def close(self):
            return None

    old_post = qwen_ui._post_upstream_stream
    try:
        qwen_ui._post_upstream_stream = lambda *_args, **_kwargs: FakeResp()
        events: list[tuple[str, dict]] = []
        asst, calls = qwen_ui._stream_one_completion(
            [], [], False, lambda event, payload: events.append((event, payload))
        )
        check("structured streamed tool call returned", len(calls) == 1)
        check("structured streamed tool call name", calls[0]["function"]["name"] == "read_file")
        check("assistant history stores structured tool call", asst.get("tool_calls") == calls)
        check("assistant visible content kept", asst.get("content") == "I will inspect the file.")
    finally:
        qwen_ui._post_upstream_stream = old_post

    print(f"\n== {'PASS' if not failures else 'FAIL'} ({len(failures)} failure(s)) ==")
    if failures:
        for f in failures:
            print(f"  - {f}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
