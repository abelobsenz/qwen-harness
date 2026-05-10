# Constrained tool-call decoding — FINAL findings (2026-05-08)

## TL;DR
**Don't ship the JSON grammar.** It actively harms tool-call output on Qwen3.6.

| Metric | Grammar OFF (baseline) | Grammar ON (JSON) |
|---|---|---|
| good_tool_call | **95% ✓** | **0% ✗** |
| malformed | 0% | 50% |
| truncated | 5% | 50% |

20-prompt bench, max_tokens=250. The grammar takes a model that's already highly competent at tool calls and breaks it.

## Why the audit recommendation was wrong for this model

The audit recommended "constrained decoding for tool-call JSON." That assumed:
1. The model emits JSON tool calls
2. The model is failing to emit valid JSON

Both assumptions are false for **Qwen3.6**:

1. **Qwen3.6 emits XML-style tool calls**, not JSON:
   ```
   <tool_call>
   <function=read_file>
   <parameter=path>test.py</parameter>
   </function>
   </tool_call>
   ```
   This is the format the model was trained on. Most other models (GPT, Claude, Llama) use JSON. Qwen3.6 is the exception.

2. **The baseline success rate is 95%**, not low. The model is highly reliable. Imposing constraints ON TOP OF an already-reliable output stream creates artificial failure modes (truncation when the constraint forces `<|im_end|>` early; malformation when the constraint targets a different format than the model wants to emit).

## What was actually shipped from this thread

Even though the grammar isn't useful for this model, the work produced real artifacts:

### 1. PEARL prefetch is empirically dead
`/tmp/mlx_stream_overlap_test.py` proves MLX streams don't overlap on Metal (0.96× speedup). No PEARL, no point.

### 2. Architectural surveys
- Sliding window — model already hybrid-attention; SW would degrade global-view layers
- mx.compile — already shipped at hot path (`@mx.compile` on argmax/match-acceptance)
- Three audit items invalidated against current model

### 3. Working char-FSM grammar framework
`scripts/json_grammar.py` (470 lines) — char-level JSON FSM. Validates 21/21 fused-token tool calls correctly. Could be used for a different model that DOES emit JSON. Standalone, no MLX deps in the core.

### 4. Working runtime hook framework
`scripts/json_grammar_runtime.py` (220 lines) — bridges char-FSM into dflash via per-cycle hook. Verified to fire correctly in real spec-decode loop. Pre-commit truncation, transition-aware behavior, edge-case termination — all implemented.

### 5. Hard-won architectural insights documented
- Spec decode × per-token constraints fundamentally conflict (commits multiple tokens per cycle)
- Cache rollback is required to handle bad-staged_first cleanly
- The transition-into-constraint cycle has carry-over from unconstrained drafting that can't be undone without rollback

### 6. Tool-call success-rate harness
`scripts/tool_call_bench.py` — 20 diverse tool-using prompts, classifies output as good/malformed/no_tool/truncated. Recognizes BOTH JSON and XML formats. Now runnable as a regression test for any future change.

## The right next step (not this session)

**Build an XML grammar.** Qwen3.6's tool calls follow:
```
<tool_call> ws? <function=NAME> ws? (<parameter=KEY> CONTENT </parameter> ws?)* </function> ws? </tool_call>
```

This is grammatically simpler than JSON and matches what the model wants to emit. If the goal is "guarantee well-formed tool calls under all conditions," an XML FSM would do that without breaking the model's natural behavior.

But even this is a marginal win — the 95% baseline means there's only 5% headroom, and most of that 5% is `truncated` (max_tokens hit), not malformed.

**Better use of optimization time:** the 5% truncation could be addressed by raising max_tokens, not by constrained decoding.

## Recommendation

1. Leave `DFLASH_JSON_GRAMMAR` env-gated and OFF by default. ✓ already shipped
2. Document this finding in the audit doc as a cautionary tale.
3. Don't ship XML grammar unless tool-call quality drops significantly in production.
4. Move optimization budget to higher-leverage items.

## Higher-leverage items remaining

From earlier session work, these are still on the table:

1. **KV fragments Path B+C** (1 day) — disk cache for compactor outputs + bm25 relevance scoring. No runtime changes. Addresses 80% of compaction pain.
2. **Update audit doc** (~15 min) — three of seven items are stale; document what's actually true now.
3. **Mask compute optimization** — irrelevant now since grammar doesn't help.

## Daemon state
Stopped. Patches in `runtime.py` remain but env-gated. To revert completely:
```
cp /tmp/runtime_pre_enchilada.py venv/lib/python3.14/site-packages/dflash_mlx/runtime.py
```

## Bench data files
- `/tmp/bench_off_v2.txt` — baseline (95% success)
- `/tmp/bench_on_v2.txt` — grammar ON (0% success)
- `/tmp/qwen36_token_classification.json` — tokenizer cache (still useful if XML grammar built)

## Time spent this session
~20 minutes of focused work for a clear no-go finding. Saved future sessions the work of building an XML grammar and only realizing afterward that the model was already 95% reliable.
