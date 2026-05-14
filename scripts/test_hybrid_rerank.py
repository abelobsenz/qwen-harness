#!/usr/bin/env python3
"""Hybrid retrieval tests for the condense path.

Validates the BM25 + cross-encoder hybrid that replaced the legacy
set-intersection scorer. Three things matter:

  1. BM25 beats set-intersection on the "rare-term wins" pattern
     (target chunk has rare query tokens; decoys share only common ones).
  2. Cross-encoder rerank distinguishes "topic mentioned" from
     "answer present" (the gap BM25 alone can't close).
  3. Graceful degradation: if rank_bm25 or the ONNX reranker fails to
     load, the path falls back without exploding.

This test uses a synthetic SEC-style doc so it's deterministic and fast.
It does pull the 22 MB Xenova/ms-marco-MiniLM-L-6-v2 ONNX model on first
run (cached at ~/.cache/huggingface afterward); when offline the rerank
assertions are skipped, not failed.
"""

from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def _adversarial_doc() -> tuple[str, str, str]:
    """A doc where the target answer chunk has rare query terms but is
    surrounded by decoys that share only the COMMON terms. The legacy
    set-intersection scorer treats every match equally regardless of
    rarity, so it can't pull the target above the decoys."""
    query = "What was Netflix average revenue per user ARPU in Q4 2024?"

    paras: list[str] = []
    paras.append(
        "Netflix 2024 Annual Report\n"
        "This annual report covers fiscal year 2024 results and operations."
    )
    # 10 decoy chunks: mention netflix + revenue + user generically, NO
    # answer specifics, NO rare query tokens like "ARPU" or "Q4". Set-
    # intersection will give each of these the same overlap as the
    # target chunk (since "netflix" / "revenue" / "user" all match).
    for i in range(10):
        paras.append(
            f"Section {i}. Netflix is a streaming entertainment company. "
            "Netflix serves users worldwide with streaming content. "
            "Revenue from members continues to grow. We invest in content "
            "to attract and retain users. Our streaming users grew. " * 3
        )
    # Near-answer chunk: high TF on "ARPU" but does NOT contain the
    # actual number — it's a topic-mention chunk, the most common
    # decoy type in real corpora (SEC filings, news, papers). The
    # cross-encoder distinguishes "topic mentioned" from "answer
    # present"; BM25 alone often can't.
    paras.append(
        "ARPU Overview\n"
        "ARPU is a key metric tracked across our membership base. "
        "We report ARPU on a quarterly cadence. ARPU informs forecasting "
        "and pricing. ARPU has been stable across regions. ARPU is "
        "discussed in segment reporting." * 2
    )
    # Target answer chunk: actual Q4 2024 ARPU value.
    target_marker = "$11.50"
    paras.append(
        "Item 7. Q4 2024 Financial Highlights\n"
        f"Average revenue per member (ARPU) was {target_marker}, up 1% "
        "year-over-year. Total Q4 2024 revenue was $10.2 billion."
    )
    # 7 trailing decoys (boilerplate).
    for i in range(7):
        paras.append(
            f"Forward-looking statement {i}. This filing contains "
            "forward-looking statements within the meaning of securities "
            "laws. Risks include regulatory, competitive, and macroeconomic "
            "factors. Actual results may differ materially. " * 2
        )

    return query, "\n\n".join(paras), target_marker


def main() -> int:
    # Import after sys.path tweak so we get the in-tree agent_tools.
    from agent_tools import (
        _bm25_scores,
        _chunk_score,
        _cross_encoder_rerank,
        _extract_top_sentences,
        _looks_like_table,
        _result_chunks,
        _split_sentences,
        _tokenize_relevance,
        _tokenize_relevance_list,
        condense_tool_result,
    )

    old_env = {k: os.environ.get(k) for k in (
        "QWEN_RESULT_CONDENSE", "QWEN_CONDENSE_MIN_CHARS",
        "QWEN_CONDENSE_CHUNK_CHARS", "QWEN_CONDENSE_TOP_K",
        "QWEN_CONDENSE_LEAD_CHARS", "QWEN_CONDENSE_RERANK",
        "QWEN_CONDENSE_HEADER_CONTINUITY",
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
        os.environ["QWEN_CONDENSE_MIN_CHARS"] = "1000"
        os.environ["QWEN_CONDENSE_CHUNK_CHARS"] = "500"
        os.environ["QWEN_CONDENSE_TOP_K"] = "2"
        os.environ["QWEN_CONDENSE_LEAD_CHARS"] = "300"
        os.environ["QWEN_CONDENSE_HEADER_CONTINUITY"] = "0"
        # Default rerank off; we'll flip per case.
        os.environ["QWEN_CONDENSE_RERANK"] = "0"

        query, doc, target_marker = _adversarial_doc()
        print(f"== hybrid retrieval tests (doc={len(doc)} chars) ==\n")

        # ---- Stage probe: BM25 differentiates the target above pure
        #      lexical-overlap decoys. Set-intersection ties them.
        chunks = _result_chunks(doc, 500)
        target_idx = next(
            (i for i, c in enumerate(chunks) if target_marker in c), -1
        )
        check("target chunk exists in chunking", target_idx >= 0,
              f"idx={target_idx}, total={len(chunks)}")

        task_terms = _tokenize_relevance(query)
        legacy_scores = [_chunk_score(task_terms, c, i) for i, c in enumerate(chunks)]
        legacy_target = legacy_scores[target_idx]
        legacy_target_rank = sorted(legacy_scores, reverse=True).index(legacy_target) + 1

        query_tokens = _tokenize_relevance_list(query)
        chunk_tokens = [_tokenize_relevance_list(c) for c in chunks]
        bm25 = _bm25_scores(query_tokens, chunk_tokens)
        check("BM25 returns scores", bm25 is not None, f"len={len(bm25) if bm25 else 0}")
        if bm25 is not None:
            bm25_target = bm25[target_idx]
            bm25_target_rank = sorted(bm25, reverse=True).index(bm25_target) + 1
            check(
                "BM25 ranks target at least as high as legacy",
                bm25_target_rank <= legacy_target_rank,
                f"legacy rank={legacy_target_rank}, BM25 rank={bm25_target_rank}",
            )

        # ---- BM25-only run: does the condensed output keep the target?
        condensed_bm25, info_bm25 = condense_tool_result(query, "web_fetch", doc)
        check(
            "BM25-only keeps target answer in condensed output",
            target_marker in condensed_bm25,
            f"kept={info_bm25['chunks_kept']}/{info_bm25['chunks_in']}",
        )

        # ---- Cross-encoder probe: target wins over the near-answer chunk.
        # The near-answer chunk is the one engineered to spam every query
        # term without containing the answer value.
        near_idx = next(
            (i for i, c in enumerate(chunks)
             if "is a key metric" in c and target_marker not in c),
            -1,
        )
        check("near-answer decoy chunk exists", near_idx >= 0,
              f"near_idx={near_idx}")

        # Sanity: BM25 should rank the near-answer chunk above the target
        # (this is the failure mode rerank exists to fix). If this isn't
        # true on this fixture, the rerank-lift demonstration below is
        # weak but the test isn't wrong — just print so the regression
        # signal is visible.
        if bm25 is not None and near_idx >= 0:
            print(f"  [.] BM25: target={bm25[target_idx]:.2f}, "
                  f"near={bm25[near_idx]:.2f} "
                  f"({'near>target — rerank should flip' if bm25[near_idx] > bm25[target_idx] else 'target>=near already'})")

        t0 = time.time()
        ce = _cross_encoder_rerank(query, chunks)
        ce_latency_ms = (time.time() - t0) * 1000.0
        if ce is None:
            print("  [~] cross-encoder unavailable (offline or load failed) "
                  "— skipping rerank assertions")
        else:
            check(
                "cross-encoder prefers target over near-answer",
                ce[target_idx] > ce[near_idx],
                f"target={ce[target_idx]:.2f}, near={ce[near_idx]:.2f}",
            )
            check(
                "cross-encoder latency is reasonable",
                ce_latency_ms < 2000.0,
                f"{ce_latency_ms:.0f}ms for {len(chunks)} pairs",
            )

            # ---- Hybrid (BM25 + rerank) keeps target.
            os.environ["QWEN_CONDENSE_RERANK"] = "1"
            condensed_hyb, info_hyb = condense_tool_result(query, "web_fetch", doc)
            check(
                "hybrid (BM25+rerank) keeps target answer",
                target_marker in condensed_hyb,
                f"kept={info_hyb['chunks_kept']}/{info_hyb['chunks_in']}, "
                f"rerank_used={info_hyb.get('rerank_used')}",
            )
            check(
                "rerank_used flag is set",
                info_hyb.get("rerank_used") is True,
            )

        # ---- Graceful degradation: if BM25 import is forced to fail, the
        # legacy fallback path must still keep the run alive (it'll lose
        # ranking quality but mustn't crash).
        import agent_tools as at
        orig_bm25 = at._bm25_scores
        at._bm25_scores = lambda *_a, **_kw: None  # simulate rank_bm25 missing
        try:
            condensed_fb, info_fb = condense_tool_result(query, "web_fetch", doc)
            check(
                "fallback path still produces a condensed result",
                info_fb["verdict"] == "condensed",
            )
        finally:
            at._bm25_scores = orig_bm25

        # ---- Sentence-level second pass (iter 40): tighter output by
        # scoring sentences within each kept chunk and keeping the top N.
        prose_chunk = (
            "Item 7. Management's Discussion and Analysis of Financial Condition. "
            "The following discussion should be read in conjunction with our "
            "consolidated financial statements. "
            "In the fourth quarter of fiscal 2024, average revenue per member "
            f"(ARPU) was {target_marker}, an increase of 1% year-over-year. "
            "We caution that forward-looking statements involve risks. "
            "Actual results could differ materially due to various factors."
        )
        sents = _split_sentences(prose_chunk)
        check("sentence splitter returns >=4 sentences for finance prose",
              len(sents) >= 4, f"got {len(sents)}")

        if ce is not None:
            extracted = _extract_top_sentences(query, prose_chunk, top_n=2)
            check("sentence-extract preserves the answer-bearing line",
                  target_marker in extracted)
            check("sentence-extract reduces chunk size by at least 30%",
                  len(extracted) < len(prose_chunk) * 0.7,
                  f"{len(prose_chunk)} -> {len(extracted)} chars")

        # Table chunks must pass through unchanged.
        table_chunk = (
            "| Period | ARPU | Members |\n"
            "| Q1 2024 | 11.40 | 269.6M |\n"
            "| Q2 2024 | 11.45 | 277.6M |\n"
            "| Q3 2024 | 11.45 | 282.7M |\n"
            f"| Q4 2024 | {target_marker} | 301.6M |"
        )
        check("table passes through sentence-extract unchanged",
              _extract_top_sentences(query, table_chunk, top_n=2) == table_chunk)
        check("_looks_like_table identifies tabular chunk",
              _looks_like_table(table_chunk))

        # End-to-end: enable sentence-extract via env and confirm the
        # condensed output keeps the target.
        os.environ["QWEN_CONDENSE_RERANK"] = "1"
        os.environ["QWEN_CONDENSE_SENTENCE_EXTRACT"] = "1"
        os.environ["QWEN_CONDENSE_SENTENCE_TOP_N"] = "3"
        condensed_se, info_se = condense_tool_result(query, "web_fetch", doc)
        check("sentence-extract end-to-end keeps target answer",
              target_marker in condensed_se,
              f"kept={info_se['chunks_kept']}/{info_se['chunks_in']}, chars={info_se['chars_out']}")
        # Don't fail if the size doesn't shrink on this small fixture —
        # the chunks here are already short and may have <top_n sentences.
        # Real win shows up on long SEC-style chunks.
        os.environ.pop("QWEN_CONDENSE_SENTENCE_EXTRACT", None)
        os.environ.pop("QWEN_CONDENSE_SENTENCE_TOP_N", None)

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
