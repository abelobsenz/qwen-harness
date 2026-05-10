#!/usr/bin/env python3
"""loop_guard: detect and stop pathological generation loops.

Background. dflash_mlx uses pure greedy argmax decoding (see
runtime.greedy_tokens_with_mask). With no temperature, no top_p, no
repetition_penalty, greedy decoding is mathematically guaranteed to
loop once it enters a low-loss n-gram cycle. This is exactly the
14k-character "I will use make_table now. Then the Mermaid code…"
collapse the user reported.

Strategy. Two complementary detectors that the proxy calls per chunk:

  1. Suffix-repeat detector. Any literal suffix of length >= MIN_LEN that
     repeats >= MIN_REPEATS times consecutively at the tail of the
     stream is a confirmed loop. This catches the textbook
     "ABCABCABCABC…" failure mode and the "sentence-pair" failure mode
     ("X. Y. X. Y. X. Y.").

  2. N-gram churn detector. Sliding-window count of distinct n-grams
     versus total n-grams. When the ratio drops under FLOOR over a
     WINDOW-sized tail, the model is recycling phrases even if the
     repeat isn't perfectly periodic. This catches near-loops
     ("paraphrase A; paraphrase A'; paraphrase A''; A again").

Both detectors are pure Python, allocation-cheap, and bounded by the
window size — safe to call on every streamed token chunk.

Defaults. Picked from the failure pattern:

  - SUFFIX_MIN_LEN  = 24 chars   ("I will use make_table now." is 26)
  - SUFFIX_REPEATS  = 4          (4× repetition is unambiguous)
  - NGRAM_WINDOW    = 600 chars  (plenty for thinking blocks)
  - NGRAM_N         = 6
  - NGRAM_FLOOR     = 0.45       (45% distinct = collapse)
  - MIN_TEXT        = 200 chars  (skip checks on short outputs)

These are tuned for English prose + JSON + code. Numerical sequences
naturally repeat tokens at low n-gram size so the floor-based detector
deliberately uses n=6 to avoid false positives on lists like
"1, 2, 3, 4, 5, 6, 7…".

The detector reports loops via a structured tuple so the proxy can log
the reason ("suffix" vs "low-churn") and adjust thresholds via env vars
without rebuilding.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class LoopGuardConfig:
    """Tunable thresholds. All env-overridable so we can adjust without
    redeploying."""

    suffix_min_len: int = int(os.environ.get("LOOP_GUARD_SUFFIX_MIN_LEN", "24"))
    suffix_repeats: int = int(os.environ.get("LOOP_GUARD_SUFFIX_REPEATS", "4"))
    ngram_window: int = int(os.environ.get("LOOP_GUARD_NGRAM_WINDOW", "600"))
    ngram_n: int = int(os.environ.get("LOOP_GUARD_NGRAM_N", "6"))
    ngram_floor: float = float(os.environ.get("LOOP_GUARD_NGRAM_FLOOR", "0.45"))
    min_text: int = int(os.environ.get("LOOP_GUARD_MIN_TEXT", "200"))
    enabled: bool = os.environ.get("LOOP_GUARD_DISABLE", "0") in ("", "0", "false", "False")
    # Phrase-repeat detector (catches interspersed near-verbatim sentences
    # in stuck thinking blocks). The char-level ngram detector misses these
    # because the variations between repetitions keep distinct/total above
    # the floor. We split on sentence boundaries and count exact phrase
    # occurrences. Window is intentionally bigger than ngram_window because
    # stuck-thinking loops emit lots of novel-looking text BETWEEN
    # repetitions, so a small window doesn't see enough copies of any one
    # phrase to trigger.
    phrase_window: int = int(os.environ.get("LOOP_GUARD_PHRASE_WINDOW", "4000"))
    # Bumped 10 → 25 after observing false positives: short common phrases
    # like "I need to ", "Let me ", "Let's start " are 10-15 chars and
    # naturally repeat 4× during normal thinking, causing the proxy to
    # abort streams that weren't actually looping. 25 chars requires a
    # whole repeated phrase fragment (e.g. "I need to find Q4 2024" =
    # 23 chars), which only happens in genuine stuck-thinking loops.
    phrase_min_len: int = int(os.environ.get("LOOP_GUARD_PHRASE_MIN_LEN", "25"))
    phrase_max_len: int = int(os.environ.get("LOOP_GUARD_PHRASE_MAX_LEN", "200"))
    phrase_min_repeats: int = int(os.environ.get("LOOP_GUARD_PHRASE_REPEATS", "4"))


@dataclass(frozen=True)
class LoopReport:
    triggered: bool
    reason: str = ""
    detail: str = ""
    repeated_chunk: str = ""


def _check_suffix_repeat(text: str, cfg: LoopGuardConfig) -> Optional[LoopReport]:
    """Return a positive LoopReport if a literal suffix of length L repeats
    R times consecutively at the tail. Tries successively shorter L starting
    from len(text)//R down to suffix_min_len. First match wins.

    Why scan downward: a long repeat (e.g. a full paragraph) is more
    informative than the shortest-cycle interpretation ("e.g. just the
    final period repeats").
    """
    n = len(text)
    if n < cfg.suffix_min_len * cfg.suffix_repeats:
        return None
    max_l = min(n // cfg.suffix_repeats, n // 2)
    if max_l < cfg.suffix_min_len:
        return None
    for L in range(max_l, cfg.suffix_min_len - 1, -1):
        chunk = text[n - L : n]
        # All R copies present at the tail?
        ok = True
        for k in range(1, cfg.suffix_repeats):
            start = n - L * (k + 1)
            end = start + L
            if start < 0 or text[start:end] != chunk:
                ok = False
                break
        if ok:
            preview = chunk if len(chunk) <= 80 else chunk[:80] + "…"
            return LoopReport(
                triggered=True,
                reason="suffix",
                detail=f"suffix len={L} repeats×{cfg.suffix_repeats}",
                repeated_chunk=preview,
            )
    return None


def _check_ngram_churn(text: str, cfg: LoopGuardConfig) -> Optional[LoopReport]:
    """Return a positive LoopReport if the trailing window has less than
    `ngram_floor` distinct/total n-grams. Catches near-loops where the
    paraphrase isn't byte-identical."""
    if len(text) < cfg.ngram_window:
        return None
    tail = text[-cfg.ngram_window :]
    n = cfg.ngram_n
    total = len(tail) - n + 1
    if total < 50:  # statistically unstable
        return None
    seen: dict[str, int] = {}
    for i in range(total):
        gram = tail[i : i + n]
        seen[gram] = seen.get(gram, 0) + 1
    distinct = len(seen)
    ratio = distinct / total
    if ratio >= cfg.ngram_floor:
        return None
    # Find the most-repeated n-gram for the report (debug aid).
    worst_gram, worst_count = max(seen.items(), key=lambda kv: kv[1])
    preview = worst_gram if len(worst_gram) <= 80 else worst_gram[:80] + "…"
    return LoopReport(
        triggered=True,
        reason="low-churn",
        detail=f"distinct/total={distinct}/{total} ratio={ratio:.2f} "
        f"top-gram×{worst_count}",
        repeated_chunk=preview,
    )


_SENTENCE_SPLIT_RE = None  # lazy: avoid re import at module load


def _check_phrase_repeat(text: str, cfg: LoopGuardConfig) -> Optional[LoopReport]:
    """Catch interspersed repetition of short imperative phrases. The
    char-level ngram detector misses these because near-verbatim variation
    keeps distinct/total just above the floor (e.g. 0.49 vs floor 0.45).

    Splits the trailing window on sentence boundaries (`.!?\\n`) and counts
    EXACT phrase occurrences (whitespace-collapsed). Triggers when any
    phrase between phrase_min_len and phrase_max_len chars appears
    `phrase_min_repeats` or more times.

    The min_len floor (10 chars) avoids catching boilerplate like "yes"
    or punctuation. The max_len ceiling (200 chars) avoids matching
    long content like a code snippet that legitimately appears once.
    """
    global _SENTENCE_SPLIT_RE
    if _SENTENCE_SPLIT_RE is None:
        import re as _re
        _SENTENCE_SPLIT_RE = _re.compile(r"[.!?\n]+")
    tail = text[-cfg.phrase_window:] if len(text) > cfg.phrase_window else text
    if len(tail) < cfg.phrase_min_len * cfg.phrase_min_repeats:
        return None
    counts: dict[str, int] = {}
    for p in _SENTENCE_SPLIT_RE.split(tail):
        # Collapse whitespace and strip — small variations like extra
        # spaces / newlines shouldn't count as different phrases.
        p = " ".join(p.split())
        L = len(p)
        if L < cfg.phrase_min_len or L > cfg.phrase_max_len:
            continue
        # Skip XML/HTML tags — legit tool-call fanouts emit identical
        # `<function=...>`/`</function>` lines, and these aren't loops.
        # Also skip phrases that contain a close-tag fragment (`</`)
        # because the sentence splitter can break tags across phrases
        # (e.g. `path>src/main.py</parameter>` splits at the `.` in `.py`,
        # leaving `py</parameter>` as a "phrase").
        if p.startswith("<") or p.startswith("&") or "</" in p:
            continue
        # Only count phrases that are PROSE-LIKE. Filter out JSON/code/
        # structured-data chunks where identical repetition is legitimate
        # (e.g. duplicate JSON array entries). Heuristic: at least 70% of
        # chars must be ASCII letters or spaces (digits and underscores
        # excluded — JSON values often have many digits, code identifiers
        # use underscores). Thinking-loop phrases like
        # "Let's run web_search" hit ~0.78; JSON entries hit ~0.50-0.60.
        prose_chars = sum(1 for c in p if c.isalpha() or c == " ")
        if prose_chars / L < 0.70:
            continue
        counts[p] = counts.get(p, 0) + 1
    if not counts:
        return None
    top, top_count = max(counts.items(), key=lambda kv: kv[1])
    if top_count >= cfg.phrase_min_repeats:
        preview = top if len(top) <= 80 else top[:80] + "…"
        return LoopReport(
            triggered=True,
            reason="phrase-repeat",
            detail=f"phrase ×{top_count}, len={len(top)}",
            repeated_chunk=preview,
        )
    return None


_DIGIT_LIST_SEPS = frozenset(", \t\n.;:-()[]{}")


def _looks_like_diverse_numerical_sequence(text: str) -> bool:
    """True if `text` is dominantly digits + delimiters AND has real
    digit diversity. Used to short-circuit the loop-guard on legitimate
    numerical outputs (backward-digit-span answers, JSON numeric arrays,
    statistical samples), where the literal `, ` delimiter pattern repeats
    indefinitely and trips the phrase / suffix / churn detectors even when
    the digits themselves are perfectly varied.

    Threshold tuned to fire on:
      - "3, 5, 8, 1, 0, 7, ..." (BDS / random digit list)             → True
      - "1, 2, 3, ..., 199" (ascending list, existing self-test)      → True
      - "[3, 5, 8]" (JSON array)                                       → True
    But NOT on:
      - "0, 0, 0, 0, 0, ..." (stuck single-digit loop)                 → False  (digit_set < 3)
      - "Here are some digits: 3, 5, 8" (mostly prose)                 → False  (numeric chars < 85%)
      - "Step 1. Step 2. Step 3. ..." (loop with numbers)              → False  (alpha chars dominate)
    """
    n = len(text)
    if n == 0:
        return False
    digits = 0
    seps = 0
    digit_set: set[str] = set()
    for c in text:
        if c.isdigit():
            digits += 1
            digit_set.add(c)
        elif c in _DIGIT_LIST_SEPS:
            seps += 1
    if (digits + seps) / n < 0.85:
        return False
    # Diversity floor: a stuck 1- or 2-digit loop ("0, 0, 0, ..." or
    # "0, 1, 0, 1, ...") is a real failure mode and must still fire.
    return digits >= 10 and len(digit_set) >= 3


def check_text(text: str, cfg: Optional[LoopGuardConfig] = None) -> LoopReport:
    """Top-level: run both detectors and combine the verdict.

    Trigger logic — designed to avoid false positives on legitimate
    structurally-repetitive outputs (XML tool-call fanouts, markdown with
    repeating section templates, JSON arrays, etc.):

        - SUFFIX-repeat alone is NOT enough — many legitimate outputs have
          high boilerplate ratios that the byte-identical detector picks
          up (e.g. ten distinct read_file calls share the XML wrapper).
        - LOW-CHURN alone is NOT enough — wordy prose can naturally have
          churn ratios in the 0.4-0.5 band without being a true loop.
        - BOTH together = a real pathological loop.

    Special case: an EXTREME suffix repeat (suffix_repeats × 1.5 or more
    occurrences of a chunk longer than `confident_suffix_len`) is enough
    on its own — at that point you don't need the second opinion.

    Cheap-to-call: skip when disabled or under min_text.
    """
    if cfg is None:
        cfg = LoopGuardConfig()
    if not cfg.enabled or len(text) < cfg.min_text:
        return LoopReport(triggered=False)
    # Numerical-sequence short-circuit. A diverse digit list (BDS answer,
    # JSON numeric array, statistical sample) repeats its delimiter
    # `, ` indefinitely — that's structural to the content, not a
    # generation loop. The phrase/suffix/churn detectors otherwise mistake
    # the delimiter pattern for repetition. A stuck single-digit loop
    # ("0, 0, 0, ...") still fires because diversity falls below the floor.
    if _looks_like_diverse_numerical_sequence(text):
        return LoopReport(triggered=False)
    # Phrase-repeat detector runs FIRST — it's specifically designed for
    # the "stuck thinking" failure mode where interspersed near-verbatim
    # sentences slip past the ngram floor.
    phrase_rep = _check_phrase_repeat(text, cfg)
    if phrase_rep is not None:
        return phrase_rep
    suffix_rep = _check_suffix_repeat(text, cfg)
    churn_rep = _check_ngram_churn(text, cfg)
    # Combined trigger: both detectors agree.
    if suffix_rep is not None and churn_rep is not None:
        return LoopReport(
            triggered=True,
            reason="combined",
            detail=f"suffix({suffix_rep.detail}) + churn({churn_rep.detail})",
            repeated_chunk=suffix_rep.repeated_chunk,
        )
    # Suffix-only triggers — for cases where the text is shorter than the
    # churn window or has a near-100%-boilerplate ratio that the churn
    # detector wouldn't classify as loopy.
    if suffix_rep is not None:
        # (a) Dominant-repeat: the literal repeat takes up most of the text.
        #     If suffix_len × repeats / total_len >= 0.80, the model has
        #     produced almost no novel content. Boilerplate-fanout cases
        #     have lots of *unique* bytes between anchors, so ratio < 0.80.
        if " repeats×" in suffix_rep.detail and "len=" in suffix_rep.detail:
            try:
                # Parse "suffix len=L repeats×R"
                lpart = suffix_rep.detail.split("len=")[1]
                L = int(lpart.split(" ")[0])
                R = int(suffix_rep.detail.split("repeats×")[1])
                covered = L * R
                if covered >= len(text) * 0.80:
                    return LoopReport(
                        triggered=True,
                        reason="suffix-dominant",
                        detail=(f"{suffix_rep.detail} covers "
                                f"{covered}/{len(text)} ({100*covered/len(text):.0f}%) of text"),
                        repeated_chunk=suffix_rep.repeated_chunk,
                    )
            except (IndexError, ValueError):
                pass
        # (b) Extreme-repeat: same chunk many more times than the basic
        #     threshold — unambiguous regardless of context.
        confident_repeat = _check_confident_suffix(text, cfg)
        if confident_repeat is not None:
            return confident_repeat
    return LoopReport(triggered=False)


def _check_confident_suffix(text: str, cfg: LoopGuardConfig) -> Optional[LoopReport]:
    """Stricter check for the "this is unambiguously a loop" case: a
    significantly longer chunk repeats more times than the basic threshold.
    Lets us fire on extreme repetition even when churn alone wouldn't
    trip (e.g. 30× exact repeats of a single line).
    """
    confident_repeats = max(cfg.suffix_repeats * 2, 8)
    confident_min_len = max(cfg.suffix_min_len, 30)
    n = len(text)
    if n < confident_min_len * confident_repeats:
        return None
    max_l = min(n // confident_repeats, n // 2)
    if max_l < confident_min_len:
        return None
    for L in range(max_l, confident_min_len - 1, -1):
        chunk = text[n - L : n]
        ok = True
        for k in range(1, confident_repeats):
            start = n - L * (k + 1)
            end = start + L
            if start < 0 or text[start:end] != chunk:
                ok = False
                break
        if ok:
            preview = chunk if len(chunk) <= 80 else chunk[:80] + "…"
            return LoopReport(
                triggered=True,
                reason="suffix-extreme",
                detail=f"suffix len={L} repeats×{confident_repeats}",
                repeated_chunk=preview,
            )
    return None


class StreamingLoopGuard:
    """Stateful wrapper for streaming generation. Holds a sliding window
    of recent text and only re-runs detection every CHECK_EVERY chars to
    keep per-token overhead negligible.

    Usage in streaming proxy:

        guard = StreamingLoopGuard()
        for chunk in upstream_stream:
            report = guard.observe(chunk)
            if report.triggered:
                yield abort_marker()
                break
            yield chunk

    The window is bounded by NGRAM_WINDOW + 1 KB headroom; memory is
    constant regardless of total stream length.
    """

    def __init__(self, cfg: Optional[LoopGuardConfig] = None,
                 check_every: int = 128) -> None:
        # Default check_every=128: at ~50-100 TPS that's ~32 tokens or
        # ~320 ms of decoding between checks — keeps per-token overhead
        # near zero while still catching short-input loops. Round 33
        # briefly tried 64 to catch loops faster, but doubled the
        # measured per-request overhead (2.5ms → 5ms) AND broke the
        # 1000-req sequential memory test (cumulative slowdown crossed
        # the 10s per-request timeout). Reverted to 128. The real-world
        # case that motivated the change turned out to be a daemon-not-
        # restarted issue, not a detection-latency issue.
        self.cfg = cfg or LoopGuardConfig()
        self.check_every = check_every
        self._buf: list[str] = []
        self._buf_len = 0
        self._last_check_at = 0
        # Need to keep enough history for the WIDEST detector window —
        # phrase detector uses up to phrase_window chars. Add 1 KB headroom.
        self._max_window = max(self.cfg.ngram_window, self.cfg.phrase_window) + 1024

    def observe(self, chunk: str) -> LoopReport:
        if not chunk:
            return LoopReport(triggered=False)
        self._buf.append(chunk)
        self._buf_len += len(chunk)
        # Compact periodically so total memory is bounded.
        if self._buf_len > self._max_window * 2:
            joined = "".join(self._buf)[-self._max_window :]
            self._buf = [joined]
            self._buf_len = len(joined)
        # Don't run detection on every token — too expensive.
        if self._buf_len - self._last_check_at < self.check_every:
            return LoopReport(triggered=False)
        self._last_check_at = self._buf_len
        # Build a single string view for the detector. We always check the
        # tail (last max_window chars) because that's where the repeat tail
        # would appear.
        tail = "".join(self._buf)[-self._max_window :]
        return check_text(tail, self.cfg)

    def finalize(self) -> LoopReport:
        """Run one final detector pass on the entire buffered tail. Call
        this when the upstream stream has ended — without it, loops that
        develop in the trailing < check_every chars would be missed.
        Idempotent: safe to call after a triggered observe()."""
        tail = "".join(self._buf)[-self._max_window :]
        return check_text(tail, self.cfg)

    def text(self) -> str:
        """Return the (windowed) accumulated text — useful for emitting
        an abort marker that includes the current tail in logs."""
        return "".join(self._buf)


# ---------- self-test entry point: run with `python -m loop_guard` ----------

def _run_self_tests() -> int:
    """Embedded smoke tests. Not pytest because the project intentionally
    runs without it for the inference-side scripts.

    Returns exit code: 0 = all pass, 1 = any failure.
    """
    failures = 0

    def expect_loop(label: str, text: str, cfg: Optional[LoopGuardConfig] = None) -> None:
        nonlocal failures
        rep = check_text(text, cfg)
        ok = rep.triggered
        print(f"  [{'✓' if ok else '✗'}] {label}: {rep.reason or 'NO LOOP'} "
              f"({rep.detail})")
        if not ok:
            failures += 1

    def expect_clean(label: str, text: str, cfg: Optional[LoopGuardConfig] = None) -> None:
        nonlocal failures
        rep = check_text(text, cfg)
        ok = not rep.triggered
        print(f"  [{'✓' if ok else '✗'}] {label}: "
              f"{'clean' if ok else 'FALSE POSITIVE: ' + rep.reason}")
        if not ok:
            failures += 1

    print("== Loop guard self-tests ==")

    # Reproduce the exact loopiness pattern from the user's example.
    print("\n[1] User-reported loop pattern:")
    user_pattern = (
        "Then the Mermaid code. I will not use any other tools. I will just "
        "output the text. Wait, I need to use `make_table` for the table. "
        "I will do that now. One thing: The user asked for "
        "\"contributions of each\". I will list Frey, Ribet, Wiles. "
        "I will use `make_table` now."
    )
    looped = (user_pattern + "\n\n") * 5
    expect_loop("user-reported make_table loop", looped)

    # 2026-05-08: stuck-thinking loop with interspersed near-verbatim
    # sentences. The char-level ngram churn detector misses these (ratio
    # stays just above 0.45). Caught by the phrase-repeat detector.
    # Fixture content kept as-is to preserve detector test coverage.
    rephrase_loop = (
        "I'll search for Netflix Q4 2024 share repurchases. Let's run web_search.\n"
        "Actually, let me search for the 10-K instead. Let's run web_search.\n"
        "Wait, the 10-K is for fiscal year. Let's run web_search.\n"
        "Hmm, let me search for the 10-Q. Let's run web_search.\n"
        "I'll do it. Let's run web_search.\n"
    ) * 3
    expect_loop("rephrase-style thinking loop", rephrase_loop)

    # Exact periodic ABCABC pattern. Each input MUST be > min_text (200 chars)
    # to clear the no-fire-on-short-output guard.
    print("\n[2] Synthetic periodic loops:")
    expect_loop("ABCDE×30 (300ch)", "abcdefghij" * 30)
    # Use ×10 not ×6: the dominant-suffix threshold is 80%, and
    # the detector finds the latest-4-repeats so coverage = 4×40/text_len.
    # ×6 (66%) is under threshold; ×10 (160%) saturates, ×20 (320%) easy.
    expect_loop("paragraph×20 (~800ch)",
                ("This is a long paragraph that repeats. " * 20))

    # Near-loop (paraphrases) — churn detector territory.
    print("\n[3] Near-loops:")
    near = ("The model said yes. The model said yeah. The model said yep. " * 12)
    expect_loop("paraphrase loop", near)

    # Healthy outputs that should NOT trigger.
    print("\n[4] Clean prose (should NOT trigger):")
    expect_clean("short healthy answer", "Hello! Here is a short clean reply.")
    expect_clean(
        "varied technical text",
        "Greedy decoding picks the most probable token at each step, which "
        "tends to amplify any local minima in the logit distribution. To "
        "mitigate this, we introduce sampling temperature, top-k, top-p, "
        "and an n-gram repetition penalty that adjusts logits based on "
        "previously generated tokens. Together these break out of "
        "degenerate cycles without harming coherence.",
    )
    expect_clean(
        "numerical sequence",
        ", ".join(str(i) for i in range(1, 200)),
    )
    expect_clean(
        "JSON dict",
        "{" + ", ".join(f'"{k}": {k}' for k in range(50)) + "}",
    )
    # Backward-Digit-Span style outputs — random single digits joined by
    # ", ". The repeating delimiter previously tripped the phrase / suffix
    # / churn detectors even though the digit values are uniformly random.
    import random as _rnd
    bds_rng = _rnd.Random(0)
    expect_clean(
        "BDS random digit list (N=200)",
        ", ".join(str(bds_rng.randint(0, 9)) for _ in range(200)),
    )
    # JSON array of small ints — same delimiter-repetition pattern.
    expect_clean(
        "JSON array of small ints",
        "[" + ", ".join(str(bds_rng.randint(0, 9)) for _ in range(150)) + "]",
    )
    # Stuck-digit loops MUST still fire — diversity floor protects us.
    expect_loop(
        "stuck single-digit loop (0, 0, 0, ...)",
        ", ".join("0" for _ in range(150)),
    )

    # Streaming-mode test.
    print("\n[5] Streaming detection:")
    guard = StreamingLoopGuard()
    big_loop = ("X" * 30 + " ") * 50
    triggered_at = -1
    rep = LoopReport(triggered=False)
    for i in range(0, len(big_loop), 32):
        rep = guard.observe(big_loop[i : i + 32])
        if rep.triggered:
            triggered_at = i
            break
    if rep.triggered and triggered_at < len(big_loop) // 2:
        print(f"  [✓] streaming: detected loop at offset {triggered_at} "
              f"(reason={rep.reason})")
    else:
        print(f"  [✗] streaming: missed loop or detected too late "
              f"(at={triggered_at})")
        failures += 1

    # Boundary: exactly threshold repeats should fire, threshold-1 should not.
    print("\n[6] Boundary conditions:")
    cfg = LoopGuardConfig()
    base = "ABCDEFGHIJKLMNOPQRSTUVWX"  # exactly 24 chars
    # 24 × 12 = 288 chars (clears the 200-char min_text floor)
    expect_loop(f"suffix×{cfg.suffix_repeats} 288ch",
                base * cfg.suffix_repeats * 3)
    # Below min_text — should never trigger regardless of repeat count.
    expect_clean(f"suffix×{cfg.suffix_repeats - 1} below min_text",
                 base * (cfg.suffix_repeats - 1))

    # Non-trigger: a pathological-looking but legitimate code-gen output
    # (recurring imports etc).
    code_sample = (
        "import json\nimport os\nimport sys\nimport re\nimport time\n\n"
        "def foo():\n    return 1\n\ndef bar():\n    return 2\n\n"
        "def baz():\n    return 3\n\nif __name__ == '__main__':\n    foo()\n"
    )
    expect_clean("python code with repetitive imports", code_sample)

    print(f"\n== {'PASS' if failures == 0 else 'FAIL'} "
          f"({failures} failure(s)) ==")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    import sys
    sys.exit(_run_self_tests())
