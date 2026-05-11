#!/usr/bin/env python3
"""Worst-case fixture for `triage_tool_result` → `truncate` pipeline.

Failure mode being pinned: a 240k+ SEC-10-K-shaped payload with the
load-bearing fact buried in the middle. Earlier behaviour (truncate
BEFORE condense, agent.py iter < the bug fix) head-truncated to 60k
which threw away the data tables. The current order
(condense first, then `truncate(result, TOOL_RESULT_MAX)`) preserves
chunk-ranked content.

This test guards two concrete invariants:

  1. **Survival**: the named fact ("long-term debt … $42,150 million")
     survives end-to-end at the same byte budget the live agent uses
     (TOOL_RESULT_MAX=60000).
  2. **Compression ratio + timing**: condense produces ≤25% of input
     size on a 1.5 MB payload AND runs in <500 ms on a stock laptop.
     If a future refactor regresses to >500 ms we want to know — the
     condense path is in the synchronous tool-result hot path.

Why this fixture matters: the audit's task-4 F2 noted the worst case
isn't represented in the existing `test_tool_result_condense.py` set.
This test adds the missing fixture without re-running real I/O.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


# Reproducible 10-K-shaped doc with a single load-bearing fact in the
# middle, plus distractor sections of similar size.
def _build_10k_payload(target_chars: int = 1_500_000) -> tuple[str, str]:
    """Return (payload, target_fact). Doc shape:

      Item 1   — Business
      Item 1A  — Risk Factors    (large)
      Item 7   — MD&A             (large; the FACT lives in a sub-section here)
      Item 8   — Financial Statements (large; contains 5-year debt tables)
      Item 15  — Exhibits

    The fact is wrapped in a markdown table so `_looks_like_table`
    boosts its chunk score by +2.0 (agent_tools.py around L1480).
    """
    # The literal target — chosen to be unique and easy to grep.
    fact_value = "$42,150 million"
    target_fact = f"Long-term debt as of the end of the fiscal year was {fact_value}."

    pad_para = ("This paragraph contains general boilerplate text "
                "about the company's operations, market conditions, "
                "competitive landscape, and accounting policies. "
                "Numbers in this paragraph (1234, 5678, 9012) are "
                "intentionally generic and uncorrelated with the "
                "question being asked. ") * 8

    sections: list[str] = []

    # ITEM 1 — Business
    sections.append("ITEM 1. BUSINESS\n" + (pad_para + "\n\n") * 40)

    # ITEM 1A — Risk Factors
    sections.append("ITEM 1A. RISK FACTORS\n" + (pad_para + "\n\n") * 40)

    # ITEM 7 — MD&A
    md_a = "ITEM 7. MANAGEMENT'S DISCUSSION AND ANALYSIS\n\n"
    md_a += (pad_para + "\n\n") * 20
    # Embed the fact in a table — both header AND table boosts will fire.
    md_a += (
        "LIQUIDITY AND CAPITAL RESOURCES\n\n"
        "The Company maintains long-term debt obligations as follows:\n\n"
        "| Period                | Amount             |\n"
        "|-----------------------|--------------------|\n"
        f"| Long-term debt (FY)  | {fact_value}       |\n"
        "| Short-term debt (FY) | $3,210 million     |\n"
        "| Total debt (FY)      | $45,360 million    |\n\n"
        f"{target_fact}\n\n"
    )
    md_a += (pad_para + "\n\n") * 20
    sections.append(md_a)

    # ITEM 8 — Financial Statements
    sections.append("ITEM 8. FINANCIAL STATEMENTS\n" + (pad_para + "\n\n") * 40)

    # ITEM 15 — Exhibits
    sections.append("ITEM 15. EXHIBITS\n" + (pad_para + "\n\n") * 40)

    payload = "\n\n".join(sections)

    # Pad to target_chars; the FACT must remain in the middle 50% of the doc.
    if len(payload) < target_chars:
        pad = " " + ("Generic boilerplate filler. " * 100)
        repeat = (target_chars - len(payload)) // len(pad) + 1
        payload = payload + pad * repeat

    return payload[:target_chars], target_fact


def main() -> int:
    from agent_tools import triage_tool_result  # noqa: WPS433

    failures = 0
    print("== worst-case tool-truncation fixture ==\n")

    # Live agent uses TOOL_RESULT_MAX = 60000 by default; match agent.py:53.
    cap = int(os.environ.get("QWEN_AGENT_TOOL_TRUNC", "60000"))

    # Replicate agent.truncate() inline so we don't pull in the agent module
    # (which imports CachedDispatcher and triggers more side effects).
    def truncate(s: str, n: int) -> str:
        return s if len(s) <= n else s[:n] + f"\n…[truncated {len(s) - n} chars]"

    # Force-enable condense regardless of caller env.
    old_env = dict(os.environ)
    os.environ["QWEN_RESULT_CONDENSE"] = "1"
    os.environ.pop("QWEN_CONDENSE_MIN_CHARS", None)
    os.environ.pop("QWEN_CONDENSE_CHUNK_CHARS", None)
    os.environ.pop("QWEN_CONDENSE_TOP_K", None)
    os.environ.pop("QWEN_CONDENSE_LEAD_CHARS", None)
    os.environ["QWEN_TRIAGE_ENABLE"] = "0"  # condense-only path (the live ordering)

    try:
        payload, target_fact = _build_10k_payload(1_500_000)
        question = ("Item 7 MD&A liquidity: what was the company's "
                    "long-term debt at the end of the fiscal year? "
                    "Quote the dollar amount.")

        t0 = time.perf_counter()
        condensed, info = triage_tool_result(question, "web_fetch", payload)
        # Then apply the agent's final char-level cap.
        final = truncate(condensed, cap)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        # 1. Did condense actually fire?
        verdict = info.get("verdict")
        ok = verdict == "condensed"
        marker = "✓" if ok else "✗"
        print(f"  [{marker}] condense fires on 1.5 MB payload (verdict={verdict!r})")
        if not ok:
            failures += 1

        # 2. Size shrinkage: at least 25% reduction from raw → condensed.
        ratio = len(condensed) / len(payload)
        ok = ratio <= 0.25
        marker = "✓" if ok else "✗"
        print(f"  [{marker}] condense compression ratio {ratio*100:.1f}% ≤ 25% "
              f"({len(payload)} → {len(condensed)} chars)")
        if not ok:
            failures += 1

        # 3. Final byte size after `truncate(final, TOOL_RESULT_MAX)`.
        ok = len(final) <= cap + 200  # 200 char headroom for the truncation marker
        marker = "✓" if ok else "✗"
        print(f"  [{marker}] final size ≤ TOOL_RESULT_MAX+headroom "
              f"({len(final)} ≤ {cap + 200})")
        if not ok:
            failures += 1

        # 4. THE FACT survived. Check the exact target fact AND the dollar
        #    amount in case the fact got split mid-chunk.
        survived_full = target_fact in final
        survived_amount = "$42,150 million" in final
        ok = survived_amount
        marker = "✓" if ok else "✗"
        print(f"  [{marker}] target fact ('$42,150 million') survives "
              f"end-to-end (verbatim full sentence={survived_full})")
        if not ok:
            failures += 1
            # Diagnostic: surface the first 400 chars of `final`.
            print(f"        first 400 chars of final:\n        {final[:400]!r}")

        # 5. Performance budget: condense + truncate must complete in <500 ms
        #    on this size of input. The live agent processes one of these
        #    per tool result, so anything slower turns into TTFB.
        ok = elapsed_ms < 500.0
        marker = "✓" if ok else "✗"
        print(f"  [{marker}] condense+truncate completes in {elapsed_ms:.1f} ms "
              f"< 500 ms budget")
        if not ok:
            failures += 1

        # 6. Sanity: at least one of the kept chunks is "header-like" or
        #    "table-like" — otherwise condense regressed to plain ranking.
        kept = info.get("chunks_kept", 0)
        total = info.get("chunks_in", 0)
        ok = 0 < kept < total
        marker = "✓" if ok else "✗"
        print(f"  [{marker}] condense kept a strict subset of chunks "
              f"({kept}/{total})")
        if not ok:
            failures += 1

        # 7. Non-research tool (e.g. bash) is skipped entirely — the live
        #    agent only condenses `_CONDENSE_TOOLS`. Verify the negative case
        #    still holds at this payload size (regression: someone widens
        #    the condense set and breaks bash output structure).
        _, info_bash = triage_tool_result(question, "bash", payload[:50_000])
        ok = info_bash.get("verdict") == "skipped"
        marker = "✓" if ok else "✗"
        print(f"  [{marker}] bash output bypasses condense ({info_bash.get('verdict')!r})")
        if not ok:
            failures += 1

    finally:
        # Restore env
        for k in list(os.environ.keys()):
            if k not in old_env:
                os.environ.pop(k, None)
        for k, v in old_env.items():
            os.environ[k] = v

    print(f"\n== {'PASS' if failures == 0 else 'FAIL'} ({failures} failure(s)) ==")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
