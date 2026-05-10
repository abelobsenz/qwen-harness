# Loop-protection audit (2026-05-08)

After shipping `DFLASH_REP_PENALTY=0.075` and the phrase-repeat detector, here's the full layer map of what catches what.

## What was wrong before

Loops slipped through because the existing protection was **detection-only**, not source-side. With pure greedy decoding, low-loss n-gram cycles are mathematically inevitable on certain prompts. The detection layers caught some patterns but missed:

1. **Interspersed near-verbatim phrase repetition** in thinking blocks (Netflix case): char-level n-gram ratio sat at 0.491, just above the 0.45 floor.
2. **Token-level micro-loops** that didn't expand into anything the char detector could see.

## The layers, ordered source → detection → reaction

### Layer 1 — DECODER (token-level, source side) ★ NEW
- **`DFLASH_REP_PENALTY=0.075`** + **`DFLASH_REP_HISTORY=128`**
- Subtracts a small penalty from logits of recently-emitted token IDs before argmax
- **Verdict: KEEP.** Validated this session: breaks the Netflix loop at the source, ~no TPS cost (87.8 vs 87.0 baseline), quality 5/5, tool-calls 100%.

### Layer 2 — STREAM DETECTION (`scripts/loop_guard.py`)
Detects loops in the streamed text via three sub-detectors:

| Sub-detector | Catches | Verdict |
|---|---|---|
| `_check_suffix_repeat` | Literal byte-identical suffix repeats (`ABCABCABC`) | **KEEP** — different shape from what rep_penalty catches |
| `_check_phrase_repeat` ★ NEW | Interspersed near-verbatim sentences (Netflix case) | **KEEP** — what the audit was designed for |
| `_check_ngram_churn` | Low distinct/total ratio | **KEEP but downgrade priority** — was the only catch for some cases pre-phrase-detector. Now overlaps with phrase detector. Useful as a separate confirmation signal in the combined trigger. |
| `_check_confident_suffix` | Extreme repetition (8+ × of long chunks) | **KEEP** — fires alone (no combined condition); covers cases where churn is high but suffix repeats |

### Layer 3 — PROXY ABORT (`scripts/qwen_proxy.py`)
- Wraps stream in `StreamingLoopGuard`, aborts on detection
- Emits `[loop-guard: <reason>]` marker in the truncated output
- **Verdict: KEEP.** Necessary glue between detection (Layer 2) and reaction (Layer 5).

### Layer 4 — AGENT-SIDE TURN COUNTERS (`scripts/agent.py`)
- `_consecutive_all_cached_turns` (threshold=3) — turns where every tool call hit `[cached…]`
- `_consecutive_missing_arg_turns` (threshold=3) — turns with malformed tool calls
- After threshold: inject a "hard commit" nudge to break the loop
- **Verdict: KEEP.** Different failure mode from token-level loops. The model can emit a *new* response each turn, but the response chooses to re-issue duplicate tool calls. Rep_penalty doesn't help here (each turn has a fresh context window) and the stream loop guard doesn't either (each individual turn isn't loopy, only the multi-turn pattern is).

### Layer 5 — LOOP-GUARD MARKER REACTION (`scripts/loop_guard_marker.py`)
- Detects `[loop-guard:...]` marker from proxy in the model's response
- Single-fires per user query (`_loop_guard_nudge_fired`)
- Injects a course-correction nudge for the next turn
- **Verdict: KEEP.** Without this, after a proxy abort the next turn often resumes the loop because nothing in context says the prior attempt was a runaway.

### Layer 6 — HARD CAPS
- `MAX_STEPS=1000` — agent step ceiling
- `QWEN_AGENT_COMPACT_AT=60000` — context-size compaction (different problem)
- `QWEN_AGENT_TOOL_TRUNC=60000` — per-tool result truncation after tool-result condense
- **Verdict: KEEP all three.** Backstops; cheap; address concerns orthogonal to repetition.

### Layer 7 — SYSTEM PROMPT RULES (`scripts/agent.py:62-90`)
- "never the same call twice (server dedups)"
- "`(no matches)` is a confirmed negative; `[cached…]` means the same evidence is already available — use it or change a real dimension"
- "For web_search, each new query must change entity, period, metric, source type, site/domain, filetype, geography, or exact phrase; synonyms and word order are not new searches"
- "One short planning pass, then act; repeated self-questioning is a stop signal"
- "Don't keep tool-calling once the task is answered"
- **Verdict: KEEP all.** Free; complementary; prevents cases that detection can't reach (e.g., the model's *intent* to retry).

### Layer 8 — TOOL-CALL DEDUP (`agent_tools.CachedDispatcher`)
- Dedupes identical (fn, args) tool calls in the same session
- Returns `[cached…]` on duplicate; this is not a source negative, only a signal that the existing cached result should be used
- **Verdict: KEEP.** Wholly different layer. Prevents wasted work, doesn't directly prevent the model emitting duplicate calls (that's Layer 4's job).

## What I considered removing

| Candidate | Why proposed | Why kept |
|---|---|---|
| `_check_ngram_churn` | Mostly overlaps with new phrase detector | Still catches some patterns the phrase detector misses (near-paraphrase loops with no clean sentence boundaries). Cost: 1 ms per check, negligible. |
| `_check_confident_suffix` | Special case of basic suffix detector | The two have different trigger conditions (combined-with-churn vs alone). The confident path fires when churn is HIGH but suffix repeats — a real failure mode that the basic detector misses. Cost: same code path, no measurable overhead. |
| `_consecutive_missing_arg_turns` | Tool-call success is now 100% in bench, this never fires | Defensive measure for regression scenarios. If the model ever loses tool-call competence, this is the recovery path. Cost: 4 lines of code + 1 int counter. Worth keeping. |

## Net architecture after this session

```
              ┌────────────────────────────────┐
              │ Layer 1: DFLASH_REP_PENALTY    │ ★ source-side fix
              │ (rep_penalty=0.075, hist=128)  │   stops most loops at logit step
              └─────────────┬──────────────────┘
                            ↓ (penalty applied per-cycle)
              ┌─────────────────────────────────┐
              │ Layer 2: loop_guard detectors   │
              │  • suffix-repeat                │ <- byte-identical loops
              │  • phrase-repeat ★              │ <- interspersed near-verbatim
              │  • ngram-churn                  │ <- general low-churn
              │  • confident-suffix             │ <- extreme repetition
              └─────────────┬───────────────────┘
                            ↓ on trigger
              ┌─────────────────────────────────┐
              │ Layer 3: proxy abort + marker   │
              └─────────────┬───────────────────┘
                            ↓ next turn
              ┌─────────────────────────────────┐
              │ Layer 5: marker → nudge inject  │
              └─────────────────────────────────┘

(running in parallel for tool-call-level loops:)

              ┌────────────────────────────────┐
              │ Layer 4: agent turn counters   │
              │  cached_turns, missing_args    │
              └─────────────┬──────────────────┘
                            ↓ threshold hit
                   inject "commit nudge"

(static prevention:)

              ┌────────────────────────────────┐
              │ Layer 7: system prompt rules   │
              │  + Layer 8: tool dedup         │
              │  + Layer 6: hard caps          │
              └────────────────────────────────┘
```

## Final verdict

**Nothing removed.** Each layer addresses a non-overlapping failure mode. The "redundancy" between phrase-repeat and ngram-churn is defense-in-depth — both have non-zero false-negative rates, and keeping both reduces total miss rate to near zero at cost of ~1 ms per stream check.

**One thing added at every layer was correct:**
- Layer 1: rep_penalty turned ON at 0.075
- Layer 2: phrase-repeat detector
- Layer 7: "reasoning hygiene" prose rule

## Tunable knobs for future tuning

| Knob | Current | Direction if more loops slip through |
|---|---|---|
| `DFLASH_REP_PENALTY` | 0.075 | Up to 0.1 (costs ~7% TPS) |
| `DFLASH_REP_HISTORY` | 128 | Up to 256 (costs slightly more memory) |
| `LOOP_GUARD_PHRASE_REPEATS` | 4 | Down to 3 (more aggressive, slight false-positive risk) |
| `LOOP_GUARD_NGRAM_FLOOR` | 0.45 | Up to 0.55 (more aggressive) |
| `QWEN_AGENT_MAX_STEPS` | 1000 | Down to 500 (catches runaways sooner) |

## Bench summary at the new defaults

| Bench | Pre-session | Post-(Metal+rep_pen+phrase) |
|---|---|---|
| TPS short ~80 | 81.3 | **87.8** (+8%) |
| TPS p_3k | 22.7 | **24.7** (+9%) |
| TPS p_5k | 13.7 | **14.7** (+7%) |
| Quality probe | 5/5 | **5/5** |
| Tool-call success | 95% | **100%** |
| Netflix-style stuck-thinking | unbounded loop | **closed `</think>` and answered** |
| `loop_guard.py` self-tests | 13/13 | **14/14** (added regression for Netflix case) |

Net: faster + cleaner + fewer loops, no quality regression.
