#!/usr/bin/env python3
"""Tests for chunked tool-result condensing.

The condense path should save tokens only for long research-style outputs,
keep task-relevant chunks, and fail open when disabled or when the result is
short. This test uses synthetic text so it is deterministic and fast.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main() -> int:
    from agent_tools import condense_tool_result, triage_tool_result

    old_env = {k: os.environ.get(k) for k in (
        "QWEN_RESULT_CONDENSE",
        "QWEN_CONDENSE_MIN_CHARS",
        "QWEN_CONDENSE_CHUNK_CHARS",
        "QWEN_CONDENSE_TOP_K",
        "QWEN_CONDENSE_LEAD_CHARS",
        "QWEN_TRIAGE_ENABLE",
    )}
    failures: list[str] = []

    def check(label: str, ok: bool, detail: str = "") -> None:
        marker = "✓" if ok else "✗"
        suffix = f" — {detail}" if detail else ""
        print(f"  [{marker}] {label}{suffix}")
        if not ok:
            failures.append(label)

    try:
        os.environ["QWEN_RESULT_CONDENSE"] = "1"
        os.environ["QWEN_CONDENSE_MIN_CHARS"] = "900"
        os.environ["QWEN_CONDENSE_CHUNK_CHARS"] = "520"
        os.environ["QWEN_CONDENSE_TOP_K"] = "2"
        os.environ["QWEN_CONDENSE_LEAD_CHARS"] = "220"
        os.environ["QWEN_TRIAGE_ENABLE"] = "0"

        task = "optimize streaming tool calls repetition penalty and metrics"
        paras = [
            "Title: Runtime Optimization Notes\nThis lead explains scope and provenance.",
        ]
        for i in range(12):
            paras.append(
                f"Unrelated section {i}\n"
                "gardening cooking travel weather music cinema " * 18
            )
        paras.insert(
            7,
            "Conclusion\n"
            "Streaming tool calls should be assembled into structured deltas. "
            "A repetition penalty should adjust logits before greedy argmax. "
            "Metrics should track token savings and loop guard aborts. " * 6,
        )
        result = "\n\n".join(paras)

        condensed, info = condense_tool_result(task, "web_fetch", result)
        print("== tool-result condense tests ==\n")
        check("long web_fetch is condensed", info["verdict"] == "condensed")
        check(
            "condensed output saves at least 25%",
            len(condensed) < len(result) * 0.75,
            f"{len(result)} -> {len(condensed)} chars",
        )
        check("relevant chunk retained", "repetition penalty" in condensed)
        check(
            "not all chunks retained",
            0 < info["chunks_kept"] < info["chunks_in"],
            f"{info['chunks_kept']}/{info['chunks_in']}",
        )

        triaged, tinfo = triage_tool_result(task, "web_fetch", result)
        check("triage entrypoint uses condense first", tinfo["verdict"] == "condensed")
        check("triage condensed text matches", triaged == condensed)

        short, sinfo = condense_tool_result(task, "web_fetch", "short result")
        check("short result is skipped", sinfo["verdict"] == "skipped" and short == "short result")

        os.environ["QWEN_RESULT_CONDENSE"] = "0"
        disabled, dinfo = condense_tool_result(task, "web_fetch", result)
        check("env disable is fail-open", dinfo["verdict"] == "skipped" and disabled == result)

        skipped, skinfo = condense_tool_result(task, "bash", result)
        check("non-research tools are skipped", skinfo["verdict"] == "skipped" and skipped == result)
    finally:
        for k, v in old_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    print(f"\n== {'PASS' if not failures else 'FAIL'} ({len(failures)} failure(s)) ==")
    if failures:
        for f in failures:
            print(f"  - {f}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
